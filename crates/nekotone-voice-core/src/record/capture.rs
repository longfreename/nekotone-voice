//! Where the audio comes from: a [`CaptureSource`] pushes timestamped
//! chunks into a [`CaptureSink`]. The real sources are cpal input streams
//! (microphones) and loopback streams (what the computer plays); tests use
//! [`FakeCapture`], which plays a prepared signal on a virtual clock.
//!
//! ## System audio (loopback)
//!
//! - **Windows**: WASAPI loopback. cpal opens an *input* stream on an
//!   *output* device and sets `AUDCLNT_STREAMFLAGS_LOOPBACK`. WASAPI sends
//!   nothing while nothing plays; the recorder fills those gaps with
//!   silence by timestamp, so the tracks stay aligned.
//! - **Linux** (PulseAudio or PipeWire with pipewire-pulse): every output
//!   has a `.monitor` source. cpal's ALSA host reaches it through the
//!   `pulse` ALSA device with `PULSE_SOURCE=@DEFAULT_MONITOR@` (set for
//!   this process when the system source is opened), or through an input
//!   device whose name contains "monitor". Without the pulse ALSA plugin
//!   (`alsa-plugins-pulseaudio` / `pipewire-alsa`) no loopback is listed.
//! - **macOS**: no built-in loopback; a virtual device such as BlackHole
//!   shows up as an input and can be chosen as the system source.

use super::SourceRole;
use crate::{Error, Result};
use crossbeam_channel::{Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// The recorder's master clock, in seconds since the recording started.
pub trait Clock: Send + Sync {
    fn now(&self) -> f64;
}

/// The system's monotonic clock.
#[derive(Debug)]
pub struct RealClock(Instant);

impl RealClock {
    pub fn new() -> RealClock {
        RealClock(Instant::now())
    }
}

impl Default for RealClock {
    fn default() -> Self {
        RealClock::new()
    }
}

impl Clock for RealClock {
    fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

/// A clock that tests (and [`FakeCapture`]) move by hand.
#[derive(Debug, Default)]
pub struct ManualClock(AtomicU64);

impl ManualClock {
    pub fn set(&self, secs: f64) {
        // Never goes backwards.
        let mut cur = self.0.load(Ordering::Relaxed);
        loop {
            if f64::from_bits(cur) >= secs {
                return;
            }
            match self.0.compare_exchange_weak(cur, secs.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }
}

impl Clock for ManualClock {
    fn now(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

/// Native format of a capture source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureFormat {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// What travels from a capture callback to the recorder's engine.
///
/// `source` and `time` are set for every source so a multi-source consumer
/// can tell chunks apart and align them; the single-source voice pipeline
/// ignores both.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum Chunk {
    Audio { source: usize, samples: Vec<f32>, time: f64 },
    Failed { source: usize, message: String },
    /// The source reconnected after a failure (`name`: the device now used).
    Restored { source: usize, name: String },
}

/// Handed to a [`CaptureSource`] on start: push audio here. Never blocks
/// in the real-time path (a full queue drops the chunk and counts it).
#[derive(Clone)]
pub struct CaptureSink {
    pub(crate) source: usize,
    pub(crate) tx: Sender<Chunk>,
    pub(crate) pool: Receiver<Vec<f32>>,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) dropped: Arc<AtomicU64>,
    /// Block instead of dropping (fake sources in tests only).
    pub(crate) blocking: bool,
}

impl CaptureSink {
    /// Push interleaved samples. `latency` is how long ago the first frame
    /// was captured (from the driver's timestamps; 0 when unknown).
    pub fn push(&self, samples: &[f32], latency_secs: f64) {
        let time = self.clock.now() - latency_secs.max(0.0);
        self.push_at(samples, time);
    }

    /// Push with an explicit capture time of the first frame (master clock).
    pub fn push_at(&self, samples: &[f32], time: f64) {
        let mut v = self.pool.try_recv().unwrap_or_default();
        v.clear();
        v.extend_from_slice(samples);
        let chunk = Chunk::Audio { source: self.source, samples: v, time };
        if self.blocking {
            let _ = self.tx.send(chunk);
        } else if self.tx.try_send(chunk).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Report that the device failed (unplugged, driver error).
    pub fn fail(&self, message: String) {
        let _ = self.tx.try_send(Chunk::Failed { source: self.source, message });
    }

    /// Report that a failed device is delivering again.
    pub fn restored(&self, name: String) {
        let _ = self.tx.try_send(Chunk::Restored { source: self.source, name });
    }

    /// The master clock (fake sources move a `ManualClock` themselves).
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }
}

/// Converts interleaved audio from a reopened device to the format the
/// consumer was given: channels mapped (mono duplicated, many averaged to
/// mono, else the first ones), rate changed by linear interpolation (good
/// enough for speech; only used after a device swap).
pub(crate) struct Adapt {
    nc: usize,
    tc: usize,
    step: f64,
    pos: f64,
    prev: Vec<f32>,
    mapped: Vec<f32>,
}

impl Adapt {
    pub(crate) fn new(in_channels: usize, in_rate: u32, out_channels: usize, out_rate: u32) -> Adapt {
        let (nc, tc) = (in_channels.max(1), out_channels.max(1));
        Adapt { nc, tc, step: in_rate as f64 / out_rate.max(1) as f64, pos: 1.0, prev: vec![0.0; tc], mapped: Vec::new() }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.nc == self.tc && self.step == 1.0
    }

    pub(crate) fn run(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let (nc, tc) = (self.nc, self.tc);
        self.mapped.clear();
        for f in input.chunks_exact(nc) {
            for c in 0..tc {
                let v = if nc == tc {
                    f[c]
                } else if nc == 1 {
                    f[0]
                } else if tc == 1 {
                    f.iter().sum::<f32>() / nc as f32
                } else {
                    f[c.min(nc - 1)]
                };
                self.mapped.push(v);
            }
        }
        out.clear();
        let n = self.mapped.len() / tc;
        if self.step == 1.0 {
            out.extend_from_slice(&self.mapped);
            return;
        }
        if n == 0 {
            return;
        }
        // frame 0 is the last frame of the previous call, frame k is mapped[k-1]
        // (the first call starts at frame 1: no delay)
        let frame = |k: usize, c: usize, prev: &[f32], m: &[f32]| if k == 0 { prev[c] } else { m[(k - 1) * tc + c] };
        while self.pos < n as f64 {
            let i = self.pos.floor() as usize;
            let fr = (self.pos - i as f64) as f32;
            for c in 0..tc {
                let a = frame(i, c, &self.prev, &self.mapped);
                let b = frame(i + 1, c, &self.prev, &self.mapped);
                out.push(a + (b - a) * fr);
            }
            self.pos += self.step;
        }
        self.pos -= n as f64;
        self.prev.copy_from_slice(&self.mapped[(n - 1) * tc..n * tc]);
    }
}

/// A source of audio for the recorder.
pub trait CaptureSource: Send {
    fn format(&self) -> CaptureFormat;
    /// Start delivering to `sink` (from any thread). Errors are sentences.
    fn start(&mut self, sink: CaptureSink) -> Result<()>;
    /// Stop delivering; after this returns no more chunks arrive.
    fn stop(&mut self);
}

// ---------------------------------------------------------------------------
// Fake capture (tests, demos)
// ---------------------------------------------------------------------------

/// Plays a prepared signal as if it were captured live, on a virtual clock.
///
/// `clock_ratio` simulates a device crystal that is off: 1.001 delivers the
/// signal 0.1 % faster than its nominal rate. `gaps` are (start, end)
/// master-clock seconds during which nothing is delivered (a silent
/// loopback device), and the samples of that time are skipped.
pub struct FakeCapture {
    pub format: CaptureFormat,
    pub signal: Arc<Vec<f32>>,
    pub chunk_frames: usize,
    pub clock_ratio: f64,
    pub jitter_secs: f64,
    pub gaps: Vec<(f64, f64)>,
    /// Real seconds to sleep per chunk (0 = as fast as the engine takes it).
    pub pace: f64,
    pub clock: Arc<ManualClock>,
    /// Advance the manual clock (one fake source should own the clock).
    pub drives_clock: bool,
    /// Master time of the first frame.
    pub start_at: f64,
    pub done: Arc<AtomicBool>,
    /// Master time this source has delivered up to (f64 bits; +inf when done).
    pub position: Arc<AtomicU64>,
    /// The clock owner waits for these followers (their `position`), so a
    /// follower never falls seconds behind the clock as no real device would.
    pub wait_for: Vec<Arc<AtomicU64>>,
    /// The longest single push into the recorder (nanoseconds): how long the
    /// capture callback was held up, which a real device cannot afford.
    pub max_push_ns: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl FakeCapture {
    pub fn new(name: &str, sample_rate: u32, channels: u16, signal: Vec<f32>, clock: Arc<ManualClock>) -> FakeCapture {
        FakeCapture {
            format: CaptureFormat { name: name.to_string(), sample_rate, channels },
            signal: Arc::new(signal),
            chunk_frames: (sample_rate / 100).max(1) as usize,
            clock_ratio: 1.0,
            jitter_secs: 0.0,
            gaps: vec![],
            pace: 0.0,
            clock,
            drives_clock: true,
            start_at: 0.0,
            done: Arc::new(AtomicBool::new(false)),
            position: Arc::new(AtomicU64::new(0f64.to_bits())),
            wait_for: vec![],
            max_push_ns: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
}

impl CaptureSource for FakeCapture {
    fn format(&self) -> CaptureFormat {
        self.format.clone()
    }

    fn start(&mut self, mut sink: CaptureSink) -> Result<()> {
        sink.blocking = true;
        let signal = self.signal.clone();
        let ch = self.format.channels.max(1) as usize;
        let rate = self.format.sample_rate as f64 * self.clock_ratio;
        let chunk = self.chunk_frames;
        let jitter = self.jitter_secs;
        let gaps = self.gaps.clone();
        let pace = self.pace;
        let clock = self.clock.clone();
        let drives = self.drives_clock;
        let start_at = self.start_at;
        let stop = self.stop.clone();
        let done = self.done.clone();
        let position = self.position.clone();
        let wait_for = self.wait_for.clone();
        let max_push = self.max_push_ns.clone();
        self.thread = Some(std::thread::spawn(move || {
            let frames = signal.len() / ch;
            let mut k = 0usize;
            let mut rng = 0x1234_5678_9abc_def1u64;
            while k < frames && !stop.load(Ordering::Relaxed) {
                let n = chunk.min(frames - k);
                let t = start_at + k as f64 / rate;
                let t_end = start_at + (k + n) as f64 / rate;
                if drives {
                    // Lockstep: never run more than 50 ms ahead of a follower.
                    for f in &wait_for {
                        while f64::from_bits(f.load(Ordering::Relaxed)) + 0.05 < t && !stop.load(Ordering::Relaxed) {
                            std::thread::yield_now();
                        }
                    }
                    clock.set(t_end);
                }
                if !gaps.iter().any(|&(a, b)| t >= a && t < b) {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let j = ((rng >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * jitter;
                    let t0 = std::time::Instant::now();
                    sink.push_at(&signal[k * ch..(k + n) * ch], t + j);
                    max_push.fetch_max(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                } else if !drives {
                    // Let the owner of the clock move on.
                }
                k += n;
                position.store(t_end.to_bits(), Ordering::Relaxed);
                if pace > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(pace));
                } else if !drives {
                    // Stay roughly in step with the clock owner.
                    let mut spins = 0;
                    while clock.now() + 0.05 < t_end && !stop.load(Ordering::Relaxed) && spins < 20_000 {
                        std::thread::yield_now();
                        spins += 1;
                    }
                }
            }
            position.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
            done.store(true, Ordering::Relaxed);
        }));
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// A device that can be recorded from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    /// Name to pass in `Sources` (as the system shows it).
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
    /// Mic = an input; System = captures what an output plays.
    pub role: SourceRole,
}

/// Name of the synthetic Linux entry that means "monitor of the default
/// output through the pulse ALSA device".
pub const LINUX_DEFAULT_MONITOR: &str = "Default output (monitor)";

#[cfg(feature = "player")]
mod real {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    fn dev_name(d: &cpal::Device) -> Option<String> {
        d.description().ok().map(|x| x.name().to_string())
    }

    pub fn input_devices() -> Vec<DeviceInfo> {
        let host = cpal::default_host();
        let default = host.default_input_device().and_then(|d| dev_name(&d));
        let mut out = vec![];
        if let Ok(devs) = host.input_devices() {
            for d in devs {
                let Some(name) = dev_name(&d) else { continue };
                if is_monitor_name(&name) {
                    continue;
                }
                let Ok(cfg) = d.default_input_config() else { continue };
                out.push(DeviceInfo { is_default: Some(&name) == default.as_ref(), name, sample_rate: cfg.sample_rate(), channels: cfg.channels(), role: SourceRole::Mic });
            }
        }
        out
    }

    fn is_monitor_name(name: &str) -> bool {
        let l = name.to_ascii_lowercase();
        l.contains("monitor") || l.contains("blackhole") || l.contains("loopback") || l.contains("soundflower")
    }

    pub fn loopback_devices() -> Vec<DeviceInfo> {
        let host = cpal::default_host();
        let mut out = vec![];
        #[cfg(windows)]
        {
            let default = host.default_output_device().and_then(|d| dev_name(&d));
            if let Ok(devs) = host.output_devices() {
                for d in devs {
                    let Some(name) = dev_name(&d) else { continue };
                    let Ok(cfg) = d.default_output_config() else { continue };
                    out.push(DeviceInfo { is_default: Some(&name) == default.as_ref(), name, sample_rate: cfg.sample_rate(), channels: cfg.channels(), role: SourceRole::System });
                }
            }
        }
        #[cfg(not(windows))]
        {
            let mut has_pulse = false;
            if let Ok(devs) = host.input_devices() {
                for d in devs {
                    let Some(name) = dev_name(&d) else { continue };
                    let l = name.to_ascii_lowercase();
                    if l == "pulse" || l.starts_with("pulse") || l == "pipewire" {
                        has_pulse = true;
                    }
                    if !is_monitor_name(&name) {
                        continue;
                    }
                    let Ok(cfg) = d.default_input_config() else { continue };
                    out.push(DeviceInfo { is_default: false, name, sample_rate: cfg.sample_rate(), channels: cfg.channels(), role: SourceRole::System });
                }
            }
            if has_pulse && cfg!(target_os = "linux") {
                out.insert(0, DeviceInfo { name: LINUX_DEFAULT_MONITOR.into(), is_default: true, sample_rate: 48_000, channels: 2, role: SourceRole::System });
            }
        }
        out
    }

    /// Open and start a capture stream delivering `want`'s format to `sink`;
    /// stream errors go to `fail`. Returns the device's name.
    fn open_stream(role: SourceRole, device: Option<&str>, want: &CaptureFormat, sink: &CaptureSink, fail: &Sender<String>) -> Result<(cpal::Stream, String)> {
        let (dev, cfg, name) = find(role, device)?;
        let format = cfg.sample_format();
        let config: cpal::StreamConfig = cfg.config();
        let fail = fail.clone();
        let err_name = name.clone();
        // only a lost or broken stream is reopened: an overrun (Xrun) is a
        // glitch and DeviceChanged means it already followed the new device.
        // (0.4.5 reopened on every error, and a Focusrite and a Sonic Studio
        // mic, which report overruns, never stayed open.)
        let on_err = move |e: cpal::Error| {
            if matches!(e.kind(), cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::StreamInvalidated | cpal::ErrorKind::BackendError | cpal::ErrorKind::HostUnavailable) {
                let _ = fail.try_send(format!("{err_name}: {e}"));
            }
        };
        let mut adapt = Adapt::new(cfg.channels() as usize, cfg.sample_rate(), want.channels as usize, want.sample_rate);
        macro_rules! build {
            ($t:ty) => {{
                let sink = sink.clone();
                let mut buf: Vec<f32> = Vec::new();
                let mut conv: Vec<f32> = Vec::new();
                dev.build_input_stream::<$t, _, _>(
                    config,
                    move |data: &[$t], info: &cpal::InputCallbackInfo| {
                        use cpal::Sample;
                        buf.clear();
                        buf.extend(data.iter().map(|s| s.to_sample::<f32>()));
                        let ts = info.timestamp();
                        let latency = ts.callback.duration_since(ts.capture).as_secs_f64();
                        if adapt.is_identity() {
                            sink.push(&buf, latency);
                        } else {
                            adapt.run(&buf, &mut conv);
                            sink.push(&conv, latency);
                        }
                    },
                    on_err,
                    None,
                )
            }};
        }
        let stream = match format {
            cpal::SampleFormat::F32 => build!(f32),
            cpal::SampleFormat::I16 => build!(i16),
            cpal::SampleFormat::U16 => build!(u16),
            cpal::SampleFormat::I32 => build!(i32),
            cpal::SampleFormat::F64 => build!(f64),
            cpal::SampleFormat::U8 => build!(u8),
            cpal::SampleFormat::I8 => build!(i8),
            other => return Err(Error::Output(format!("sample format {other} is not supported"))),
        }
        .map_err(|e| Error::Output(format!("could not open the recording stream ({e})")))?;
        stream.play().map_err(|e| Error::Output(format!("could not start recording ({e})")))?;
        Ok((stream, name))
    }

    /// Find the device and the config to open it with.
    fn find(role: SourceRole, name: Option<&str>) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String)> {
        let host = cpal::default_host();
        let what = match role {
            SourceRole::Mic => "microphone",
            SourceRole::System => "system audio device",
        };
        match role {
            SourceRole::Mic => {
                let dev = match name {
                    None => host.default_input_device().ok_or_else(|| Error::Output("no microphone is available; plug one in or pick a device in Settings".into()))?,
                    Some(want) => host
                        .input_devices()
                        .ok()
                        .and_then(|mut it| it.find(|d| dev_name(d).as_deref() == Some(want)))
                        .ok_or_else(|| Error::Output(format!("no {what} called \"{want}\"; pick another one")))?,
                };
                let n = dev_name(&dev).unwrap_or_else(|| "Microphone".into());
                let cfg = dev.default_input_config().map_err(|e| Error::Output(format!("{n}: no usable recording format ({e})")))?;
                Ok((dev, cfg, n))
            }
            SourceRole::System => {
                #[cfg(windows)]
                {
                    let dev = match name {
                        None => host.default_output_device().ok_or_else(|| Error::Output("no output device to record the system audio from".into()))?,
                        Some(want) => host
                            .output_devices()
                            .ok()
                            .and_then(|mut it| it.find(|d| dev_name(d).as_deref() == Some(want)))
                            .ok_or_else(|| Error::Output(format!("no {what} called \"{want}\"; pick another one")))?,
                    };
                    let n = dev_name(&dev).unwrap_or_else(|| "System audio".into());
                    let cfg = dev.default_output_config().map_err(|e| Error::Output(format!("{n}: no usable format for loopback ({e})")))?;
                    Ok((dev, cfg, format!("{n} (loopback)")))
                }
                #[cfg(not(windows))]
                {
                    let want = name.filter(|n| *n != LINUX_DEFAULT_MONITOR);
                    let devs: Vec<cpal::Device> = host.input_devices().map(|it| it.collect()).unwrap_or_default();
                    let dev = match want {
                        Some(w) => devs.into_iter().find(|d| dev_name(d).as_deref() == Some(w)),
                        None => {
                            // The pulse ALSA device reading the default output's monitor.
                            // SAFETY-ish: set before the device is opened; libpulse reads it on connect.
                            std::env::set_var("PULSE_SOURCE", "@DEFAULT_MONITOR@");
                            devs.into_iter().find(|d| {
                                let l = dev_name(d).unwrap_or_default().to_ascii_lowercase();
                                l == "pulse" || l.starts_with("pulse") || l.contains("monitor")
                            })
                        }
                    }
                    .ok_or_else(|| {
                        Error::Output(
                            "no system-audio source found: on Linux install the PulseAudio/PipeWire ALSA plugin (alsa-plugins-pulseaudio or pipewire-alsa) so the output's monitor can be recorded; on macOS install a loopback device such as BlackHole".into(),
                        )
                    })?;
                    let n = dev_name(&dev).unwrap_or_else(|| "System audio".into());
                    let cfg = dev.default_input_config().map_err(|e| Error::Output(format!("{n}: no usable recording format ({e})")))?;
                    Ok((dev, cfg, format!("{n} (monitor)")))
                }
            }
        }
    }

    /// A cpal input (or loopback) stream, owned by its own thread (cpal
    /// streams are not `Send` on every platform).
    pub struct CpalCapture {
        role: SourceRole,
        device: Option<String>,
        format: CaptureFormat,
        stop: Option<Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl CpalCapture {
        pub fn open(role: SourceRole, device: Option<&str>) -> Result<CpalCapture> {
            let (_, cfg, name) = find(role, device)?;
            Ok(CpalCapture {
                role,
                device: device.map(str::to_string),
                format: CaptureFormat { name, sample_rate: cfg.sample_rate(), channels: cfg.channels() },
                stop: None,
                thread: None,
            })
        }
    }

    impl CaptureSource for CpalCapture {
        fn format(&self) -> CaptureFormat {
            self.format.clone()
        }

        /// Opens the device and keeps it open: when it fails (unplugged, the
        /// mic changed in Windows, a driver reset) the sink hears `fail`, the
        /// device is reopened every second (the chosen one, then after 3 s the
        /// default one) and `restored` follows. A reopened device with
        /// another rate or channel count is converted to the first format,
        /// which the consumer was told.
        fn start(&mut self, sink: CaptureSink) -> Result<()> {
            let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(1);
            let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<()>>(1);
            let role = self.role;
            let device = self.device.clone();
            let want = self.format.clone();
            self.thread = Some(std::thread::spawn(move || {
                let (fail_tx, fail_rx) = crossbeam_channel::bounded::<String>(4);
                let mut stream = match open_stream(role, device.as_deref(), &want, &sink, &fail_tx) {
                    Ok((s, _)) => {
                        let _ = ready_tx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                loop {
                    crossbeam_channel::select! {
                        recv(stop_rx) -> _ => break,
                        recv(fail_rx) -> m => {
                            drop(stream);
                            sink.fail(m.unwrap_or_default());
                            let mut tries = 0u32;
                            let reopened = loop {
                                match stop_rx.recv_timeout(std::time::Duration::from_secs(1)) {
                                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                                    _ => break None,
                                }
                                while fail_rx.try_recv().is_ok() {}
                                tries += 1;
                                let name = if tries > 3 { None } else { device.as_deref() };
                                if let Ok(r) = open_stream(role, name, &want, &sink, &fail_tx) {
                                    break Some(r);
                                }
                            };
                            match reopened {
                                Some((s, name)) => {
                                    stream = s;
                                    sink.restored(name);
                                }
                                None => return,
                            }
                        }
                    }
                }
                drop(stream);
            }));
            self.stop = Some(stop_tx);
            match ready_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(r) => r,
                Err(_) => Err(Error::Output("the audio device did not start within 10 s".into())),
            }
        }

        fn stop(&mut self) {
            if let Some(s) = self.stop.take() {
                let _ = s.send(());
            }
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    impl Drop for CpalCapture {
        fn drop(&mut self) {
            self.stop();
        }
    }
}

#[cfg(feature = "player")]
pub use real::CpalCapture;

/// Microphones and other inputs (not loopback/monitor devices).
pub fn input_devices() -> Vec<DeviceInfo> {
    #[cfg(feature = "player")]
    {
        real::input_devices()
    }
    #[cfg(not(feature = "player"))]
    {
        vec![]
    }
}

/// Devices whose playback can be recorded (Windows: every output device;
/// Linux: the default output's monitor and any "monitor" input; macOS:
/// virtual loopback inputs such as BlackHole).
pub fn loopback_devices() -> Vec<DeviceInfo> {
    #[cfg(feature = "player")]
    {
        real::loopback_devices()
    }
    #[cfg(not(feature = "player"))]
    {
        vec![]
    }
}

#[cfg(test)]
mod adapt_tests {
    use super::Adapt;

    #[test]
    fn a_swapped_device_is_converted_to_the_first_format() {
        // a 44.1 kHz stereo mic replaces a 48 kHz mono one: a 440 Hz tone
        // comes out at 48 kHz mono, same pitch, same length, no clicks
        let (rin, rout) = (44_100u32, 48_000u32);
        let secs = 1.0;
        let n = (rin as f32 * secs) as usize;
        let stereo: Vec<f32> = (0..n).flat_map(|i| {
            let v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rin as f32).sin() * 0.5;
            [v, v]
        }).collect();
        let mut a = Adapt::new(2, rin, 1, rout);
        assert!(!a.is_identity());
        let (mut out, mut part) = (Vec::new(), Vec::new());
        for c in stereo.chunks(441 * 2) {
            a.run(c, &mut part);
            out.extend_from_slice(&part);
        }
        assert!((out.len() as i64 - rout as i64).abs() <= 2, "{} samples", out.len());
        let worst = out.iter().skip(1).enumerate().map(|(i, v)| {
            let want = (2.0 * std::f32::consts::PI * 440.0 * (i + 1) as f32 / rout as f32).sin() * 0.5;
            (v - want).abs()
        }).fold(0.0f32, f32::max);
        assert!(worst < 0.01, "worst error {worst}");
        assert!(Adapt::new(1, 48_000, 1, 48_000).is_identity());
    }
}
