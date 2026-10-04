//! The live voice changer on audio devices.
//!
//! Threads:
//! * **input callback** (driver thread, priority raised): downmix → the
//!   [`Processor`] → push processed mono into one SPSC ring per output.
//!   Control messages arrive through a bounded lock-free channel; chains that
//!   finished cross-fading go back through another so nothing is freed on the
//!   audio thread.
//! * **output / monitor callbacks**: [`DriftReader`] resamples the ring at the
//!   device clock (drift + rate conversion), applies the monitor gain and
//!   upmixes. They can only read processed audio.
//! * **supervisor** (normal priority): owns the cpal streams, builds chains,
//!   emits events (levels ~10 Hz, xruns, device loss), and reopens devices
//!   after a disconnect (retry every second).

use super::chain::{Chain, DspMsg, FrontEnd, Processor};
use super::devices;
use super::dsp::{db_to_lin, denormals_off, lin_to_db};
use super::presets::Preset;
use super::ring::{ring, Consumer, DriftReader, Producer};
use crate::{Error, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How to run the voice changer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceConfig {
    /// Microphone by name (`None` = system default).
    pub input: Option<String>,
    /// Which input channel to use (`None` = average all channels).
    pub input_channel: Option<u16>,
    /// Output device by name (`None` = system default). Ignored when `virtual_mic` is on.
    pub output: Option<String>,
    /// Send the voice to the virtual microphone: Windows/macOS — the detected
    /// virtual cable; Linux — a PipeWire/PulseAudio mic Nekotone creates.
    pub virtual_mic: bool,
    /// Also play the voice here (headphones). `Some("")` = system default output.
    pub monitor: Option<String>,
    pub preset: Preset,
    pub front_end: FrontEnd,
    pub output_gain_db: f32,
    pub monitor_gain_db: f32,
    /// Limiter ceiling (dBFS).
    pub ceiling_db: f32,
    /// Requested device buffer in frames (128–1024; the OS may use its own).
    pub block_frames: u32,
    /// Preferred processing rate (the input device's rate is used if it cannot do this).
    pub sample_rate: u32,
    /// Start with the output muted (silence from the first sample, no fade):
    /// used when the app restarts the engine after a device change while
    /// the user had muted it.
    pub start_muted: bool,
    /// The speaker's saved calibration: voices with targets start from it
    /// (and keep learning). `None` = learn from neutral as you speak.
    pub profile: Option<super::Profile>,
    /// How far the voice ducks while a sound pad plays (dB).
    pub pad_duck_db: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        VoiceConfig {
            input: None,
            input_channel: None,
            output: None,
            virtual_mic: false,
            monitor: None,
            preset: super::presets::preset_by_id("female").expect("built-in"),
            front_end: FrontEnd::default(),
            output_gain_db: 0.0,
            monitor_gain_db: -6.0,
            ceiling_db: -1.0,
            block_frames: 256,
            sample_rate: 48000,
            start_muted: false,
            profile: None,
            pad_duck_db: 10.0,
        }
    }
}

/// Events for the UI (delivered on the supervisor thread).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum VoiceEvent {
    Started { input: String, output: String, monitor: Option<String>, sample_rate: u32, latency_ms: f32 },
    /// About ten times a second.
    Level { input_db: f32, output_db: f32, gate_open: bool, pitch_hz: f32, out_pitch_hz: f32, limiter_db: f32 },
    /// Buffer underrun/overflow count increased on `stream` ("input", "output", "monitor").
    Xrun { stream: &'static str, count: u64 },
    /// A device went away; the engine keeps retrying and outputs nothing meanwhile.
    DeviceLost { role: &'static str, device: String, message: String },
    /// Devices are working again after `DeviceLost`.
    Recovered { latency_ms: f32 },
    Error { message: String },
    Stopped,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LatencyBreakdown {
    /// Capture buffer (device → input callback).
    pub input_ms: f32,
    /// Voice stage + limiter lookahead.
    pub processing_ms: f32,
    /// Queued in the ring (drift-controlled).
    pub ring_ms: f32,
    /// Output callback → speaker/cable.
    pub output_ms: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VoiceStatus {
    pub running: bool,
    pub muted: bool,
    /// Mic-to-output latency (sum of `latency`).
    pub latency_ms: f32,
    pub latency: LatencyBreakdown,
    pub input_level_db: f32,
    pub output_level_db: f32,
    pub gate_open: bool,
    /// Bit `id % 32` set for each sound pad playing.
    pub pads_playing: u32,
    pub gate_threshold_db: f32,
    pub noise_floor_db: f32,
    /// Your pitch (0 when unvoiced).
    pub pitch_hz: f32,
    /// The changed voice's pitch (0 when unvoiced or no voice stage).
    pub out_pitch_hz: f32,
    /// Input-callback processing time / real time (0.05 = 5 % of one core).
    pub cpu_load: f32,
    pub xruns: u64,
    /// Clock correction of the output reader in parts per million.
    pub drift_ppm: f32,
    pub sample_rate: u32,
    pub input_device: String,
    pub output_device: String,
    pub monitor_device: Option<String>,
    pub preset_id: String,
    /// Who is speaking, as measured so far (voices with targets adapt to it).
    pub profile: super::Profile,
}

#[derive(Default)]
struct AF32(AtomicU32);
impl AF32 {
    fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
    fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn max(&self, v: f32) {
        let _ = self.0.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| (v > f32::from_bits(b)).then_some(v.to_bits()));
    }
    fn take(&self) -> f32 {
        f32::from_bits(self.0.swap(0, Ordering::Relaxed))
    }
}

#[derive(Default)]
struct Shared {
    running: AtomicBool,
    muted: AtomicBool,
    in_peak: AF32,
    out_peak: AF32,
    in_peak_ui: AF32,
    out_peak_ui: AF32,
    gate_open: AtomicBool,
    pads_playing: AtomicU32,
    gate_thr: AF32,
    noise_floor: AF32,
    pitch: AF32,
    out_pitch: AF32,
    prof_f0: AF32,
    prof_spread: AF32,
    prof_tract: AF32,
    prof_secs: AF32,
    limiter: AF32,
    cpu: AF32,
    in_lat: AF32,
    out_lat: AF32,
    ring_s: AF32,
    drift: AF32,
    proc_lat: AF32,
    overflows: AtomicU64,
    out_xruns: AtomicU64,
    mon_xruns: AtomicU64,
    in_max_frames: AtomicU32,
    monitor_gain: AF32,
    stream_failed: AtomicBool,
    fail_msg: Mutex<Option<(&'static str, String)>>,
    rate: AtomicU32,
    names: Mutex<(String, String, Option<String>)>,
}

enum Ctl {
    Preset,
    Param { block: usize, index: usize, value: f32 },
    Mute(bool),
    OutputGain(f32),
    FrontEnd(FrontEnd),
    Ceiling(f32),
    Profile(Option<super::Profile>),
    PlayPad { id: u32, clip: Arc<[f32]>, gain_db: f32 },
    StopPads(Option<u32>),
    PadDuck(f32),
    Stop,
}

type OnEvent = Arc<dyn Fn(VoiceEvent) + Send + Sync>;

/// A running voice changer. Dropping it stops it.
pub struct VoiceEngine {
    ctl: Sender<Ctl>,
    shared: Arc<Shared>,
    state: Arc<Mutex<VoiceConfig>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl VoiceEngine {
    /// Open the devices and start. Fails (with a sentence for the user) when
    /// the devices cannot be opened at all; later device loss is reported
    /// through `on_event` and recovered from automatically.
    pub fn start(config: VoiceConfig, on_event: impl Fn(VoiceEvent) + Send + Sync + 'static) -> Result<VoiceEngine> {
        let mut config = config;
        config.preset.sanitize();
        let shared = Arc::new(Shared::default());
        shared.monitor_gain.set(db_to_lin(config.monitor_gain_db));
        shared.muted.store(config.start_muted, Ordering::Relaxed);
        let state = Arc::new(Mutex::new(config));
        let (ctl_tx, ctl_rx) = bounded::<Ctl>(256);
        let (ready_tx, ready_rx) = bounded::<Result<()>>(1);
        let on_event: OnEvent = Arc::new(on_event);
        let th = {
            let shared = shared.clone();
            let state = state.clone();
            std::thread::Builder::new()
                .name("nekotone-voice".into())
                .spawn(move || supervisor(state, shared, ctl_rx, ready_tx, on_event))
                .map_err(Error::Io)?
        };
        match ready_rx.recv_timeout(Duration::from_secs(15)) {
            Ok(Ok(())) => Ok(VoiceEngine { ctl: ctl_tx, shared, state, thread: Some(th) }),
            Ok(Err(e)) => {
                let _ = th.join();
                Err(e)
            }
            Err(_) => {
                let _ = ctl_tx.send(Ctl::Stop);
                Err(Error::Output("the audio devices did not start within 15 seconds".into()))
            }
        }
    }

    /// Switch preset (40 ms cross-fade).
    pub fn set_preset(&self, preset: Preset) {
        let mut p = preset;
        p.sanitize();
        self.state.lock().preset = p;
        let _ = self.ctl.try_send(Ctl::Preset);
    }

    /// The current preset including parameter changes (save it as a user preset).
    pub fn preset(&self) -> Preset {
        self.state.lock().preset.clone()
    }

    /// Change one parameter of the running preset: `block_id` is the slot id
    /// (e.g. "voice"), `name` the parameter (e.g. "pitch_st"). Values are clamped.
    pub fn set_param(&self, block_id: &str, name: &str, value: f32) -> Result<()> {
        let mut st = self.state.lock();
        let preset_name = st.preset.name.clone();
        let Some(block) = st.preset.blocks.iter().position(|b| b.id == block_id) else {
            return Err(Error::Output(format!("preset \"{preset_name}\" has no block \"{block_id}\"")));
        };
        let spec = &mut st.preset.blocks[block].block;
        let Some(index) = spec.param_index(name) else {
            return Err(Error::Output(format!("block \"{block_id}\" ({}) has no parameter \"{name}\"", spec.kind())));
        };
        spec.set_index(index, value);
        let value = spec.get(name).unwrap_or(value);
        drop(st);
        self.ctl.try_send(Ctl::Param { block, index, value }).map_err(|_| Error::Output("the voice engine is busy; try again".into()))
    }

    pub fn set_front_end(&self, f: FrontEnd) {
        self.state.lock().front_end = f;
        let _ = self.ctl.try_send(Ctl::FrontEnd(f));
    }

    /// Play a sound pad: `clip` is mono at [`VoiceStatus::sample_rate`]
    /// (see `pads::prepare_clip`). The caller must keep its own reference
    /// to the clip while the engine runs, so the audio thread never frees
    /// it. Ignored while the devices are being reopened.
    pub fn play_pad(&self, id: u32, clip: Arc<[f32]>, gain_db: f32) {
        let _ = self.ctl.try_send(Ctl::PlayPad { id, clip, gain_db });
    }

    /// Stop one sound pad, or all of them (`None`).
    pub fn stop_pads(&self, id: Option<u32>) {
        let _ = self.ctl.try_send(Ctl::StopPads(id));
    }

    /// How far the voice ducks while a pad plays (dB, 0–40).
    pub fn set_pad_duck_db(&self, db: f32) {
        let db = db.clamp(0.0, 40.0);
        self.state.lock().pad_duck_db = db;
        let _ = self.ctl.try_send(Ctl::PadDuck(db));
    }

    pub fn set_output_gain_db(&self, db: f32) {
        self.state.lock().output_gain_db = db;
        let _ = self.ctl.try_send(Ctl::OutputGain(db));
    }

    pub fn set_monitor_gain_db(&self, db: f32) {
        self.state.lock().monitor_gain_db = db;
        self.shared.monitor_gain.set(db_to_lin(db));
    }

    pub fn set_ceiling_db(&self, db: f32) {
        self.state.lock().ceiling_db = db;
        let _ = self.ctl.try_send(Ctl::Ceiling(db));
    }

    /// Panic button: `true` outputs silence (5 ms fade) until `false`.
    pub fn set_bypass_mute(&self, mute: bool) {
        self.shared.muted.store(mute, Ordering::Relaxed);
        let _ = self.ctl.try_send(Ctl::Mute(mute));
    }

    pub fn status(&self) -> VoiceStatus {
        let s = &self.shared;
        let lat = LatencyBreakdown {
            input_ms: s.in_lat.get() * 1000.0,
            processing_ms: s.proc_lat.get() * 1000.0,
            ring_ms: s.ring_s.get() * 1000.0,
            output_ms: s.out_lat.get() * 1000.0,
        };
        let names = s.names.lock().clone();
        // One lock for both fields: a guard taken inside the struct literal
        // lives to the end of the expression, so locking `state` twice there
        // deadlocked (Go live hung once the devices were open).
        let (preset_id, style) = {
            let st = self.state.lock();
            (st.preset.id.clone(), st.profile.and_then(|p| p.style))
        };
        VoiceStatus {
            running: s.running.load(Ordering::Relaxed),
            muted: s.muted.load(Ordering::Relaxed),
            latency_ms: lat.input_ms + lat.processing_ms + lat.ring_ms + lat.output_ms,
            latency: lat,
            input_level_db: lin_to_db(s.in_peak_ui.get()),
            output_level_db: lin_to_db(s.out_peak_ui.get()),
            gate_open: s.gate_open.load(Ordering::Relaxed),
            pads_playing: s.pads_playing.load(Ordering::Relaxed),
            gate_threshold_db: s.gate_thr.get(),
            noise_floor_db: s.noise_floor.get(),
            pitch_hz: s.pitch.get(),
            out_pitch_hz: s.out_pitch.get(),
            cpu_load: s.cpu.get(),
            xruns: s.overflows.load(Ordering::Relaxed) + s.out_xruns.load(Ordering::Relaxed) + s.mon_xruns.load(Ordering::Relaxed),
            drift_ppm: s.drift.get() * 1e6,
            sample_rate: s.rate.load(Ordering::Relaxed),
            input_device: names.0,
            output_device: names.1,
            monitor_device: names.2,
            preset_id,
            // the style layer comes from the calibration (not measured live)
            profile: super::Profile { f0_hz: s.prof_f0.get(), spread_st: s.prof_spread.get(), tract: s.prof_tract.get(), voiced_secs: s.prof_secs.get(), style }.sanitized(),
        }
    }

    /// Start the speaker profile again from `profile` (a new calibration,
    /// or `None`: learn from neutral). Also used after device restarts.
    pub fn set_profile(&self, profile: Option<super::Profile>) {
        self.state.lock().profile = profile;
        let _ = self.ctl.try_send(Ctl::Profile(profile));
    }

    /// Stop and release the devices (also removes the Linux virtual mic).
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        let _ = self.ctl.send(Ctl::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for VoiceEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ───────────────────────────── supervisor ─────────────────────────────

struct Streams {
    _input: cpal::Stream,
    _output: cpal::Stream,
    _monitor: Option<cpal::Stream>,
    dsp: Sender<DspMsg>,
    trash: Receiver<Box<Chain>>,
    rate: f32,
    latency_samples: usize,
    #[cfg(target_os = "linux")]
    _vmic: Option<super::virtual_mic::linux::VirtualMic>,
}

fn supervisor(state: Arc<Mutex<VoiceConfig>>, shared: Arc<Shared>, ctl: Receiver<Ctl>, ready: Sender<Result<()>>, on_event: OnEvent) {
    let mut streams: Option<Streams>;
    // copy the config first: a guard in the match scrutinee is held through the arms
    let cfg = state.lock().clone();
    match open_all(&cfg, &shared, &on_event) {
        Ok(s) => {
            let lat = s.latency_samples as f32 / s.rate;
            shared.proc_lat.set(lat);
            let names = shared.names.lock().clone();
            streams = Some(s);
            shared.running.store(true, Ordering::Relaxed);
            let _ = ready.send(Ok(()));
            on_event(VoiceEvent::Started {
                input: names.0,
                output: names.1,
                monitor: names.2,
                sample_rate: shared.rate.load(Ordering::Relaxed),
                latency_ms: lat * 1000.0,
            });
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    }
    let mut last_level = Instant::now();
    let mut retry_at: Option<Instant> = None;
    let mut xr = [0u64; 3];
    let mut lost_reported = false;
    loop {
        let msg = ctl.recv_timeout(Duration::from_millis(25));
        match msg {
            Ok(Ctl::Stop) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            Ok(m) => {
                if let Some(s) = streams.as_ref() {
                    let dm = match m {
                        Ctl::Preset => {
                            let preset = state.lock().preset.clone();
                            Some(DspMsg::SwapChain(Box::new(Chain::new(&preset.blocks, s.rate))))
                        }
                        Ctl::Param { block, index, value } => Some(DspMsg::Param { block, index, value }),
                        Ctl::Mute(v) => Some(DspMsg::Mute(v)),
                        Ctl::OutputGain(v) => Some(DspMsg::OutputGainDb(v)),
                        Ctl::FrontEnd(f) => Some(DspMsg::FrontEnd(f)),
                        Ctl::Ceiling(v) => Some(DspMsg::CeilingDb(v)),
                        Ctl::Profile(p) => Some(DspMsg::Profile(p)),
                        Ctl::PlayPad { id, clip, gain_db } => Some(DspMsg::PlayPad { id, clip, gain_db }),
                        Ctl::StopPads(id) => Some(DspMsg::StopPads(id)),
                        Ctl::PadDuck(db) => Some(DspMsg::PadDuck(db)),
                        Ctl::Stop => None,
                    };
                    if let Some(dm) = dm {
                        if s.dsp.send_timeout(dm, Duration::from_millis(200)).is_err() {
                            on_event(VoiceEvent::Error { message: "the audio thread is not responding".into() });
                        }
                    }
                }
                // otherwise the change is already in `state` and applies on reopen
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
        }
        if let Some(s) = streams.as_ref() {
            for t in s.trash.try_iter() {
                drop(t);
            }
        }
        // device failure → drop streams, retry every second
        if shared.stream_failed.swap(false, Ordering::Relaxed) {
            let (role, message) = shared.fail_msg.lock().take().unwrap_or(("audio", "stream error".into()));
            let names = shared.names.lock().clone();
            let device = match role {
                "input" => names.0,
                "monitor" => names.2.unwrap_or_default(),
                _ => names.1,
            };
            streams = None;
            shared.running.store(false, Ordering::Relaxed);
            on_event(VoiceEvent::DeviceLost { role, device, message });
            lost_reported = true;
            retry_at = Some(Instant::now() + Duration::from_millis(500));
        }
        if streams.is_none() {
            if let Some(t) = retry_at {
                if Instant::now() >= t {
                    let cfg = state.lock().clone();
                    match open_all(&cfg, &shared, &on_event) {
                        Ok(s) => {
                            let lat = s.latency_samples as f32 / s.rate;
                            shared.proc_lat.set(lat);
                            let mute = shared.muted.load(Ordering::Relaxed);
                            let _ = s.dsp.try_send(DspMsg::Mute(mute));
                            streams = Some(s);
                            shared.running.store(true, Ordering::Relaxed);
                            retry_at = None;
                            if lost_reported {
                                on_event(VoiceEvent::Recovered { latency_ms: lat * 1000.0 });
                                lost_reported = false;
                            }
                        }
                        Err(_) => retry_at = Some(Instant::now() + Duration::from_secs(1)),
                    }
                }
            }
        }
        if last_level.elapsed() >= Duration::from_millis(100) {
            last_level = Instant::now();
            let ip = shared.in_peak.take();
            let op = shared.out_peak.take();
            shared.in_peak_ui.set(ip);
            shared.out_peak_ui.set(op);
            if streams.is_some() {
                on_event(VoiceEvent::Level {
                    input_db: lin_to_db(ip),
                    output_db: lin_to_db(op),
                    gate_open: shared.gate_open.load(Ordering::Relaxed),
                    pitch_hz: shared.pitch.get(),
                    out_pitch_hz: shared.out_pitch.get(),
                    limiter_db: lin_to_db(shared.limiter.get().max(1e-6)),
                });
            }
            let now = [
                shared.overflows.load(Ordering::Relaxed),
                shared.out_xruns.load(Ordering::Relaxed),
                shared.mon_xruns.load(Ordering::Relaxed),
            ];
            for (i, name) in ["input", "output", "monitor"].iter().enumerate() {
                if now[i] > xr[i] {
                    on_event(VoiceEvent::Xrun { stream: name, count: now[i] });
                }
            }
            xr = now;
        }
    }
    drop(streams);
    shared.running.store(false, Ordering::Relaxed);
    on_event(VoiceEvent::Stopped);
}

fn error_cb(shared: &Arc<Shared>, role: &'static str) -> impl FnMut(cpal::Error) + Send + 'static {
    let shared = shared.clone();
    move |e: cpal::Error| {
        let fatal = matches!(
            e.kind(),
            cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::StreamInvalidated | cpal::ErrorKind::BackendError
        );
        if matches!(e.kind(), cpal::ErrorKind::Xrun) {
            shared.overflows.fetch_add(1, Ordering::Relaxed);
        }
        if fatal {
            *shared.fail_msg.lock() = Some((role, e.to_string()));
            shared.stream_failed.store(true, Ordering::Relaxed);
        }
    }
}

fn open_all(cfg: &VoiceConfig, shared: &Arc<Shared>, on_event: &OnEvent) -> Result<Streams> {
    // where the voice goes
    #[cfg(target_os = "linux")]
    let mut vmic = None;
    #[cfg(target_os = "linux")]
    let before = if cfg.virtual_mic { super::virtual_mic::linux::VirtualMic::our_sink_inputs() } else { Vec::new() };
    let out_name: Option<String> = if cfg.virtual_mic {
        #[cfg(target_os = "linux")]
        {
            vmic = Some(super::virtual_mic::linux::VirtualMic::create()?);
            None // default device; the new stream is moved into the null sink below
        }
        #[cfg(not(target_os = "linux"))]
        {
            let st = super::virtual_mic::virtual_mic_status();
            match st.output_device {
                Some(d) => Some(d),
                None => return Err(Error::Output(st.instructions)),
            }
        }
    } else {
        cfg.output.clone()
    };

    let block = cfg.block_frames.clamp(64, 2048);
    let (out_dev, out_dev_name) = devices::find(false, out_name.as_deref())?;
    let (out_fmt, out_cfg) = devices::choose_config(&out_dev, false, Some(cfg.sample_rate), block).map_err(|e| Error::Output(format!("{out_dev_name}: {e}")))?;
    let (in_dev, in_dev_name) = devices::find(true, cfg.input.as_deref())?;
    let (in_fmt, in_cfg) = devices::choose_config(&in_dev, true, Some(out_cfg.sample_rate), block).map_err(|e| Error::Output(format!("{in_dev_name}: {e}")))?;
    let rate = in_cfg.sample_rate as f32;

    let mut processor = Box::new(Processor::new(rate, block as usize, cfg.front_end, &cfg.preset.blocks, cfg.output_gain_db, cfg.ceiling_db).with_profile(cfg.profile));
    processor.handle(DspMsg::PadDuck(cfg.pad_duck_db));
    let latency_samples = processor.latency();
    let ring_cap = (rate as usize) / 2;
    let (main_p, main_c) = ring(ring_cap);
    let monitor_target = cfg.monitor.as_ref().map(|m| if m.is_empty() { None } else { Some(m.clone()) });
    let (mon_p, mon_c) = if monitor_target.is_some() {
        let (p, c) = ring(ring_cap);
        (Some(p), Some(c))
    } else {
        (None, None)
    };
    let (dsp_tx, dsp_rx) = bounded::<DspMsg>(64);
    let (trash_tx, trash_rx) = bounded::<Box<Chain>>(8);
    let mute = shared.muted.load(Ordering::Relaxed);
    let _ = dsp_tx.try_send(DspMsg::Mute(mute));
    shared.rate.store(in_cfg.sample_rate, Ordering::Relaxed);
    for c in [&shared.overflows, &shared.out_xruns, &shared.mon_xruns] {
        c.store(0, Ordering::Relaxed);
    }

    // output first (so on Linux its stream can be moved into the null sink)
    let output = build_output(&out_dev, out_fmt, out_cfg, main_c, rate, shared.clone(), false).map_err(|e| Error::Output(format!("{out_dev_name}: could not open the output ({e})")))?;
    output.play().map_err(|e| Error::Output(format!("{out_dev_name}: could not start the output ({e})")))?;
    #[cfg(target_os = "linux")]
    if cfg.virtual_mic {
        let mut moved = 0;
        for _ in 0..20 {
            moved = super::virtual_mic::linux::VirtualMic::capture_new_streams(&before);
            if moved > 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if moved == 0 {
            on_event(VoiceEvent::Error {
                message: "could not route the voice into the Nekotone_Voice microphone (no PulseAudio/PipeWire stream found for this process)".into(),
            });
        }
    }
    let _ = on_event;
    let (monitor, mon_name) = match (monitor_target, mon_c) {
        (Some(target), Some(c)) => {
            let (dev, name) = devices::find(false, target.as_deref())?;
            let (fmt, mcfg) = devices::choose_config(&dev, false, Some(in_cfg.sample_rate), block)?;
            let s = build_output(&dev, fmt, mcfg, c, rate, shared.clone(), true).map_err(|e| Error::Output(format!("{name}: could not open the monitor ({e})")))?;
            s.play().map_err(|e| Error::Output(format!("{name}: could not start the monitor ({e})")))?;
            (Some(s), Some(name))
        }
        _ => (None, None),
    };
    let input = build_input(&in_dev, in_fmt, in_cfg, cfg.input_channel, processor, main_p, mon_p, dsp_rx, trash_tx, shared.clone())
        .map_err(|e| Error::Output(format!("{in_dev_name}: could not open the microphone ({e})")))?;
    input.play().map_err(|e| Error::Output(format!("{in_dev_name}: could not start the microphone ({e})")))?;
    let shown_out = if cfg.virtual_mic {
        let st = super::virtual_mic::virtual_mic_status();
        st.mic_name.map(|m| format!("{out_dev_name} → {m}")).unwrap_or(out_dev_name)
    } else {
        out_dev_name
    };
    *shared.names.lock() = (in_dev_name, shown_out, mon_name);
    Ok(Streams {
        _input: input,
        _output: output,
        _monitor: monitor,
        dsp: dsp_tx,
        trash: trash_rx,
        rate,
        latency_samples,
        #[cfg(target_os = "linux")]
        _vmic: vmic,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_input(
    dev: &cpal::Device,
    fmt: cpal::SampleFormat,
    cfg: cpal::StreamConfig,
    channel: Option<u16>,
    processor: Box<Processor>,
    main: Producer,
    mon: Option<Producer>,
    rx: Receiver<DspMsg>,
    trash: Sender<Box<Chain>>,
    shared: Arc<Shared>,
) -> std::result::Result<cpal::Stream, String> {
    macro_rules! go {
        ($t:ty) => {
            input_stream::<$t>(dev, cfg, channel, processor, main, mon, rx, trash, shared).map_err(|e| e.to_string())
        };
    }
    match fmt {
        cpal::SampleFormat::F32 => go!(f32),
        cpal::SampleFormat::I16 => go!(i16),
        cpal::SampleFormat::I32 => go!(i32),
        cpal::SampleFormat::U16 => go!(u16),
        cpal::SampleFormat::F64 => go!(f64),
        cpal::SampleFormat::U8 => go!(u8),
        cpal::SampleFormat::I8 => go!(i8),
        other => Err(format!("sample format {other} is not supported")),
    }
}

#[allow(clippy::too_many_arguments)]
fn input_stream<T>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    channel: Option<u16>,
    mut processor: Box<Processor>,
    mut main: Producer,
    mut mon: Option<Producer>,
    rx: Receiver<DspMsg>,
    trash: Sender<Box<Chain>>,
    shared: Arc<Shared>,
) -> std::result::Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let ch = cfg.channels.max(1) as usize;
    let pick = channel.map(|c| (c as usize).min(ch - 1));
    let rate = cfg.sample_rate as f32;
    let mut mono = vec![0.0f32; 4096];
    let mut first = true;
    let mut pending: Option<Box<Chain>> = None;
    let err = error_cb(&shared, "input");
    dev.build_input_stream::<T, _, _>(
        cfg,
        move |data: &[T], info: &cpal::InputCallbackInfo| {
            if first {
                first = false;
                crate::threads::raise_current_thread_priority();
            }
            denormals_off();
            let t0 = Instant::now();
            while let Ok(m) = rx.try_recv() {
                processor.handle(m);
            }
            let frames = data.len() / ch;
            shared.in_max_frames.fetch_max(frames as u32, Ordering::Relaxed);
            let mut done = 0;
            while done < frames {
                let n = (frames - done).min(mono.len());
                let src = &data[done * ch..(done + n) * ch];
                for (m, f) in mono[..n].iter_mut().zip(src.chunks_exact(ch)) {
                    *m = match pick {
                        Some(c) => <f32 as cpal::FromSample<T>>::from_sample_(f[c]),
                        None => f.iter().map(|s| <f32 as cpal::FromSample<T>>::from_sample_(*s)).sum::<f32>() / ch as f32,
                    };
                }
                processor.process(&mut mono[..n]);
                if main.push(&mono[..n]) < n {
                    shared.overflows.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(m) = mon.as_mut() {
                    m.push(&mono[..n]);
                }
                done += n;
            }
            if pending.is_none() {
                pending = processor.take_trash();
            }
            if let Some(t) = pending.take() {
                if let Err(TrySendError::Full(t)) = trash.try_send(t) {
                    pending = Some(t);
                }
            }
            let m = &mut processor.meters;
            shared.in_peak.max(m.in_peak);
            shared.out_peak.max(m.out_peak);
            m.in_peak = 0.0;
            m.out_peak = 0.0;
            shared.gate_open.store(m.gate_open, Ordering::Relaxed);
            shared.pads_playing.store(m.pads_playing, Ordering::Relaxed);
            shared.pitch.set(m.pitch_hz);
            shared.out_pitch.set(m.out_pitch_hz);
            shared.prof_f0.set(m.profile.f0_hz);
            shared.prof_spread.set(m.profile.spread_st);
            shared.prof_tract.set(m.profile.tract);
            shared.prof_secs.set(m.profile.voiced_secs);
            shared.limiter.set(m.limiter_gain);
            shared.gate_thr.set(processor.gate_threshold_db());
            shared.noise_floor.set(processor.noise_floor_db());
            let ts = info.timestamp();
            if let Some(d) = ts.callback.checked_duration_since(ts.capture) {
                shared.in_lat.set(d.as_secs_f32());
            }
            let load = t0.elapsed().as_secs_f32() / (frames.max(1) as f32 / rate);
            let prev = shared.cpu.get();
            shared.cpu.set(prev + (load - prev) * 0.05);
        },
        err,
        None,
    )
}

fn build_output(
    dev: &cpal::Device,
    fmt: cpal::SampleFormat,
    cfg: cpal::StreamConfig,
    ring: Consumer,
    in_rate: f32,
    shared: Arc<Shared>,
    monitor: bool,
) -> std::result::Result<cpal::Stream, String> {
    macro_rules! go {
        ($t:ty) => {
            output_stream::<$t>(dev, cfg, ring, in_rate, shared, monitor).map_err(|e| e.to_string())
        };
    }
    match fmt {
        cpal::SampleFormat::F32 => go!(f32),
        cpal::SampleFormat::I16 => go!(i16),
        cpal::SampleFormat::I32 => go!(i32),
        cpal::SampleFormat::U16 => go!(u16),
        cpal::SampleFormat::F64 => go!(f64),
        cpal::SampleFormat::U8 => go!(u8),
        cpal::SampleFormat::I8 => go!(i8),
        other => Err(format!("sample format {other} is not supported")),
    }
}

fn output_stream<T>(
    dev: &cpal::Device,
    cfg: cpal::StreamConfig,
    mut ring: Consumer,
    in_rate: f32,
    shared: Arc<Shared>,
    monitor: bool,
) -> std::result::Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let ch = cfg.channels.max(1) as usize;
    let out_rate = cfg.sample_rate as f64;
    let mut reader = DriftReader::new(in_rate as f64, out_rate, 0.003, 4096);
    let mut mono = vec![0.0f32; 4096];
    let mut first = true;
    let err = error_cb(&shared, if monitor { "monitor" } else { "output" });
    dev.build_output_stream::<T, _, _>(
        cfg,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            if first {
                first = false;
                crate::threads::raise_current_thread_priority();
            }
            denormals_off();
            let frames = data.len() / ch;
            // hold one input burst plus one output burst (in input samples) + 1 ms
            let in_max = shared.in_max_frames.load(Ordering::Relaxed) as f64;
            let want = in_max + frames as f64 * (in_rate as f64 / out_rate) + 0.001 * in_rate as f64;
            if reader.target < want {
                reader.target = want;
            }
            let g = if monitor { shared.monitor_gain.get() } else { 1.0 };
            let mut done = 0;
            while done < frames {
                let n = (frames - done).min(mono.len());
                reader.read(&mut ring, &mut mono[..n]);
                for (f, m) in data[done * ch..(done + n) * ch].chunks_exact_mut(ch).zip(mono[..n].iter()) {
                    let v = (m * g).clamp(-1.0, 1.0);
                    for o in f.iter_mut() {
                        *o = T::from_sample(v);
                    }
                }
                done += n;
            }
            if monitor {
                shared.mon_xruns.store(reader.xruns, Ordering::Relaxed);
            } else {
                shared.out_xruns.store(reader.xruns, Ordering::Relaxed);
                shared.ring_s.set(reader.fill_seconds() as f32);
                shared.drift.set(reader.correction() as f32);
                let ts = info.timestamp();
                if let Some(d) = ts.playback.checked_duration_since(ts.callback) {
                    shared.out_lat.set(d.as_secs_f32());
                }
            }
        },
        err,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the real engine on the default devices for 10 s (needs a mic and speakers).
    /// `cargo test -p nekotone-core voice::engine -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn runs_on_default_devices_for_ten_seconds() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        let cfg = VoiceConfig { preset: crate::voice::preset_by_id("robot").unwrap(), ..Default::default() };
        let eng = VoiceEngine::start(cfg, move |e| ev.lock().push(e)).expect("start");
        std::thread::sleep(Duration::from_secs(4));
        eng.set_preset(crate::voice::preset_by_id("female").unwrap());
        eng.set_param("voice", "pitch_st", 5.0).unwrap();
        std::thread::sleep(Duration::from_secs(3));
        eng.set_bypass_mute(true);
        std::thread::sleep(Duration::from_secs(3));
        let st = eng.status();
        println!("{st:#?}");
        assert!(st.running);
        assert!(st.latency_ms > 0.0 && st.latency_ms < 200.0);
        assert!(st.cpu_load < 0.5);
        eng.stop();
        let ev = events.lock();
        assert!(matches!(ev.first(), Some(VoiceEvent::Started { .. })));
        assert!(matches!(ev.last(), Some(VoiceEvent::Stopped)));
        assert!(!ev.iter().any(|e| matches!(e, VoiceEvent::DeviceLost { .. })));
    }
}
