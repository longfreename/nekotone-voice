//! Voice Studio: the real-time voice changer behind the Voice view.
//!
//! Every command here is `async`; anything that touches devices, disk or the
//! DSP runs in `spawn_blocking`, so the window thread never waits (Tauri runs
//! sync commands on it). Registered in lib.rs inside the marked voice block.
//!
//! * One [`VoiceEngine`] at a time lives in [`LIVE`]. Its events go to the
//!   page through the `Channel` given to `voice_start`; the page polls
//!   `voice_status` (~15 Hz) for meters.
//! * The engine is stopped when the main window is destroyed (app exit), via
//!   a window-event hook installed on the first start.
//! * Previews are offline renders of a short sample (the user's 5-second
//!   recording, or a built-in synthetic voice) through the same DSP as the
//!   live engine, cached as WAV under `<data dir>/cache/voice-previews`,
//!   keyed by the sample file and the exact preset chain.
//! * User presets live in `<data dir>/voice-presets` (the app's data folder,
//!   so UI tests with their own data folder never touch the user's presets).

use crate::commands::{any, err, CmdError, CmdResult};
use crate::AppState;
use nekotone_voice_core::models::ModelId;
use nekotone_voice_core::stt::{Transcriber, WhisperOnnx};
use nekotone_voice_core::tts::accent::Accent;
use nekotone_voice_core::tts::Tts;
use nekotone_voice_core::tts::chatterbox::{self, Chatterbox};
use nekotone_voice_core::voice::speak::{is_clone_voice, CloneSynth, SpeakConfig, SpeakEngine, SpeakEvent, SpeakStatus, Synth, VoiceRouter};
use nekotone_voice_core::voice::{self, AudioDevice, BlockType, FrontEnd, Preset, VirtualMicStatus, VoiceConfig, VoiceEngine, VoiceEvent, VoiceStatus};
use parking_lot::Mutex;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager};

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> CmdResult<T> + Send + 'static) -> CmdResult<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(any)?
}

struct Live {
    engine: VoiceEngine,
    /// Last limiter gain reduction from the Level events (dB, ≤ 0).
    limiter_db: Arc<AtomicU32>,
    /// Set while a device is lost (role and message), cleared on recovery.
    lost: Arc<Mutex<Option<String>>>,
    /// Sound pad clips sent to the engine, by (file, rate). Kept for the
    /// whole session: the engine is stopped before this is dropped, so the
    /// audio thread never frees a clip (see `voice::pads`).
    pads: Mutex<std::collections::HashMap<(String, u32), Arc<[f32]>>>,
}

static LIVE: Mutex<Option<Live>> = parking_lot::const_mutex(None);
/// Serialises start/stop so two quick presses never open devices twice.
static STARTING: Mutex<()> = parking_lot::const_mutex(());
static EXIT_HOOK: std::sync::Once = std::sync::Once::new();

fn data_dir(app: &AppHandle) -> PathBuf {
    app.state::<AppState>().data_dir.clone()
}
fn presets_dir(app: &AppHandle) -> PathBuf {
    data_dir(app).join("voice-presets")
}
fn previews_dir(app: &AppHandle) -> PathBuf {
    data_dir(app).join("cache").join("voice-previews")
}
fn sample_path(app: &AppHandle) -> PathBuf {
    data_dir(app).join("voice-sample.wav")
}
fn profile_path(app: &AppHandle) -> PathBuf {
    data_dir(app).join("voice-profile.json")
}
/// The saved calibration (None when there is none or it cannot be read).
fn saved_profile(app: &AppHandle) -> Option<voice::Profile> {
    let text = std::fs::read_to_string(profile_path(app)).ok()?;
    serde_json::from_str::<voice::Profile>(&text).ok().map(|p| p.sanitized())
}

fn with_live<T>(f: impl FnOnce(&Live) -> CmdResult<T>) -> CmdResult<T> {
    match LIVE.lock().as_ref() {
        Some(l) => f(l),
        None => Err(CmdError { kind: "not-found", message: "the voice changer is not running".into() }),
    }
}

// ───────────────────────── devices, presets ─────────────────────────

#[derive(Serialize)]
pub struct Devices {
    pub inputs: Vec<AudioDevice>,
    pub outputs: Vec<AudioDevice>,
    pub virtual_mic: VirtualMicStatus,
}

/// Microphones, outputs and the virtual-microphone status (lists devices; ~50 ms).
#[tauri::command]
pub async fn voice_devices() -> CmdResult<Devices> {
    blocking(|| Ok(Devices { inputs: voice::input_devices(), outputs: voice::output_devices(), virtual_mic: voice::virtual_mic_status() })).await
}

#[derive(Serialize)]
pub struct PresetList {
    pub builtin: Vec<Preset>,
    pub user: Vec<Preset>,
    /// User preset files that could not be read ("file: reason").
    pub problems: Vec<String>,
    pub dir: String,
}

#[tauri::command]
pub async fn voice_presets(app: AppHandle) -> CmdResult<PresetList> {
    blocking(move || {
        let dir = presets_dir(&app);
        let (user, problems) = voice::load_user_presets(&dir);
        Ok(PresetList { builtin: voice::builtin_presets(), user, problems, dir: dir.display().to_string() })
    })
    .await
}

#[derive(Serialize)]
pub struct Catalogue {
    pub blocks: Vec<BlockType>,
    pub front_end: FrontEnd,
}

/// Every block type with its parameter table (sliders), and the default mic clean-up.
#[tauri::command]
pub async fn voice_block_types() -> CmdResult<Catalogue> {
    Ok(Catalogue { blocks: voice::block_types(), front_end: FrontEnd::default() })
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "preset".into()
    } else {
        out
    }
}

/// Files in the presets folder with the preset id each one holds.
fn user_files(dir: &Path) -> Vec<(PathBuf, String)> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("json")) == Some(true))
        .filter_map(|p| {
            let s = std::fs::read_to_string(&p).ok()?;
            let pr = Preset::from_json(&s).ok()?;
            Some((p, pr.id))
        })
        .collect()
}

/// Save a user preset. With `replace` = the id of an existing user preset,
/// that one is overwritten (and removed first when the new name gives a new
/// id: rename). Otherwise a new id is made from the name, never clashing
/// with a built-in or another user preset. Returns the saved preset.
#[tauri::command]
pub async fn voice_save_preset(app: AppHandle, preset: Preset, replace: Option<String>) -> CmdResult<Preset> {
    blocking(move || {
        let dir = presets_dir(&app);
        let mut p = preset;
        p.builtin = false;
        let name = p.name.trim().to_string();
        if name.is_empty() {
            return Err(any("give the preset a name"));
        }
        p.name = name;
        let files = user_files(&dir);
        let builtin: Vec<String> = voice::builtin_presets().into_iter().map(|b| b.id).collect();
        let base = slug(&p.name);
        let taken = |id: &str| builtin.iter().any(|b| b == id) || files.iter().any(|(_, f)| f == id && Some(f.as_str()) != replace.as_deref());
        let mut id = if builtin.contains(&base) { format!("my-{base}") } else { base.clone() };
        let mut n = 2;
        while taken(&id) {
            id = format!("{}-{n}", if builtin.contains(&base) { format!("my-{base}") } else { base.clone() });
            n += 1;
        }
        if let Some(old) = replace.as_deref() {
            if old != id {
                for (path, fid) in &files {
                    if fid == old {
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
        p.id = id;
        voice::save_user_preset(&dir, &p).map_err(err)?;
        p.sanitize();
        Ok(p)
    })
    .await
}

/// Delete a user preset by id. True when a file was removed.
#[tauri::command]
pub async fn voice_delete_preset(app: AppHandle, id: String) -> CmdResult<bool> {
    blocking(move || {
        let mut removed = false;
        for (path, fid) in user_files(&presets_dir(&app)) {
            if fid == id {
                std::fs::remove_file(&path).map_err(any)?;
                removed = true;
            }
        }
        Ok(removed)
    })
    .await
}

// ───────────────────────── the live engine ─────────────────────────

fn install_exit_hook(app: &AppHandle) {
    EXIT_HOOK.call_once(|| {
        if let Some(w) = app.get_webview_window("main") {
            w.on_window_event(|e| {
                if let tauri::WindowEvent::Destroyed = e {
                    let live = LIVE.lock().take();
                    if let Some(l) = live {
                        crate::log_file("voice: stopping with the window");
                        l.engine.stop();
                    }
                    stop_speak();
                }
            });
        }
    });
}

/// Start (or restart) the voice changer. Returns once the devices run, or
/// with a sentence saying why they could not be opened.
#[tauri::command]
pub async fn voice_start(app: AppHandle, mut config: VoiceConfig, on_event: Channel<VoiceEvent>) -> CmdResult<VoiceStatus> {
    install_exit_hook(&app);
    blocking(move || {
        // voices with targets start from the saved calibration
        if config.profile.is_none() {
            config.profile = saved_profile(&app);
        }
        let _g = STARTING.lock();
        let old = LIVE.lock().take();
        if let Some(l) = old {
            l.engine.stop();
        }
        if stop_speak() {
            crate::log_file("speak: stopped for the voice changer");
        }
        let limiter_db = Arc::new(AtomicU32::new(0f32.to_bits()));
        let lost: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let (lim, lo) = (limiter_db.clone(), lost.clone());
        let preset = config.preset.id.clone();
        let engine = VoiceEngine::start(config, move |ev| {
            match &ev {
                VoiceEvent::Level { limiter_db, .. } => {
                    lim.store(limiter_db.to_bits(), Ordering::Relaxed);
                    return; // the page polls voice_status for meters
                }
                VoiceEvent::DeviceLost { role, device, message } => {
                    *lo.lock() = Some(format!("{role}: {device}: {message}"));
                    crate::log_file(&format!("voice: {role} device lost ({device}): {message}"));
                }
                VoiceEvent::Recovered { .. } => *lo.lock() = None,
                VoiceEvent::Error { message } => crate::log_file(&format!("voice: {message}")),
                _ => {}
            }
            let _ = on_event.send(ev);
        })
        .map_err(err)?;
        let st = engine.status();
        crate::log_file(&format!("voice: started {preset} ({} → {}, {:.0} ms)", st.input_device, st.output_device, st.latency_ms));
        *LIVE.lock() = Some(Live { engine, limiter_db, lost, pads: Mutex::new(Default::default()) });
        Ok(st)
    })
    .await
}

/// Stop and release the devices. False when it was not running.
#[tauri::command]
pub async fn voice_stop(app: AppHandle) -> CmdResult<bool> {
    blocking(move || {
        let _g = STARTING.lock();
        let live = LIVE.lock().take();
        match live {
            Some(l) => {
                let measured = l.engine.status().profile;
                l.engine.stop();
                crate::log_file("voice: stopped");
                learn_from_session(&app, measured);
                Ok(true)
            }
            None => Ok(false),
        }
    })
    .await
}

/// Continuous learning (Settings: "Keep learning my voice", off by
/// default): fold what this session measured into the saved identity, when
/// it was clearly the same person (see `voice::learn_identity`). Stays on
/// this PC, in voice-profile.json, like the calibration.
fn learn_from_session(app: &AppHandle, measured: voice::Profile) {
    let on = app.state::<crate::AppState>().settings.lock().voice_learn;
    if !on {
        return;
    }
    match voice::learn_identity(saved_profile(app), measured) {
        Some(p) => {
            if write_profile(app, &p).is_ok() {
                crate::log_file(&format!("voice: learnt from the session ({:.0} s of speech): {:.0} Hz, tract {:.2}", measured.voiced_secs, p.f0_hz, p.tract));
            }
        }
        None => crate::log_file(&format!("voice: not learnt from the session ({:.0} s of speech at {:.0} Hz: too little, or not the saved voice)", measured.voiced_secs, measured.f0_hz)),
    }
}

/// Save the identity atomically (a temporary file, then a rename).
fn write_profile(app: &AppHandle, p: &voice::Profile) -> CmdResult<()> {
    let path = profile_path(app);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(p).map_err(any)?).map_err(any)?;
    std::fs::rename(&tmp, &path).map_err(any)?;
    Ok(())
}

// ───────────────────────── sound pads (stream deck) ─────────────────────────

/// Play the file at `path` as sound pad `id` into the voice output (the
/// voice ducks while it plays). The first press decodes it (up to 20 s);
/// later presses start at once.
#[tauri::command]
pub async fn voice_pad_play(id: u32, path: String, gain_db: f32) -> CmdResult<()> {
    blocking(move || {
        let (rate, cached) = {
            let g = LIVE.lock();
            let Some(l) = g.as_ref() else {
                return Err(CmdError { kind: "not-found", message: "go live first: pads play into the voice changer's output".into() });
            };
            let rate = l.engine.status().sample_rate;
            let cached = l.pads.lock().get(&(path.clone(), rate)).cloned();
            (rate, cached)
        };
        let clip = match cached {
            Some(c) => c,
            None => {
                // decode outside the lock (it can take a moment)
                let decoded = nekotone_voice_core::audio::decode(std::path::Path::new(&path)).map_err(err)?;
                let decoded = if decoded.sample_rate != rate { nekotone_voice_core::audio::resample(&decoded, rate).map_err(err)? } else { decoded };
                let clip = voice::pads::prepare_clip(&decoded.samples, rate as f32);
                let g = LIVE.lock();
                let Some(l) = g.as_ref() else { return Ok(()) };
                let mut cache = l.pads.lock();
                if cache.len() >= voice::pads::MAX_PADS * 2 {
                    return Err(any(format!("too many different sounds this session (up to {}): stop and start the voice changer to free them", voice::pads::MAX_PADS * 2)));
                }
                cache.entry((path.clone(), rate)).or_insert(clip).clone()
            }
        };
        with_live(|l| {
            l.engine.play_pad(id, clip, gain_db);
            Ok(())
        })
    })
    .await
}

/// Stop one pad, or all of them (`id` null).
#[tauri::command]
pub async fn voice_pad_stop(id: Option<u32>) -> CmdResult<()> {
    with_live(|l| {
        l.engine.stop_pads(id);
        Ok(())
    })
}

/// How far the voice ducks while a pad plays (dB).
#[tauri::command]
pub async fn voice_pad_duck(db: f32) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_pad_duck_db(db);
        Ok(())
    })
}

/// Apply the voice controls (each -1..1) to a preset as saved; the page
/// sends the result's changed parameters (or the whole preset when a tone
/// block was added).
#[tauri::command]
pub fn voice_apply_controls(preset: voice::Preset, controls: voice::VoiceControls) -> voice::Preset {
    voice::apply_controls(&preset, &controls)
}

#[derive(Serialize)]
pub struct LiveStatus {
    #[serde(flatten)]
    pub status: VoiceStatus,
    /// Limiter gain reduction (dB, ≤ 0).
    pub limiter_db: f32,
    /// A device is lost and being retried ("role: device: message").
    pub lost: Option<String>,
}

/// Levels, gate, pitch, latency, CPU… of the running engine; null when stopped.
#[tauri::command]
pub async fn voice_status() -> CmdResult<Option<LiveStatus>> {
    Ok(LIVE.lock().as_ref().map(|l| LiveStatus {
        status: l.engine.status(),
        limiter_db: f32::from_bits(l.limiter_db.load(Ordering::Relaxed)),
        lost: l.lost.lock().clone(),
    }))
}

/// Switch the running preset (40 ms cross-fade).
#[tauri::command]
pub async fn voice_set_preset(preset: Preset) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_preset(preset);
        Ok(())
    })
}

/// The running preset with every live parameter change (null when stopped).
#[tauri::command]
pub async fn voice_current_preset() -> CmdResult<Option<Preset>> {
    Ok(LIVE.lock().as_ref().map(|l| l.engine.preset()))
}

/// Change one parameter of the running preset; returns the clamped value.
#[tauri::command]
pub async fn voice_set_param(block: String, name: String, value: f32) -> CmdResult<f32> {
    with_live(|l| {
        l.engine.set_param(&block, &name, value).map_err(err)?;
        let p = l.engine.preset();
        Ok(p.block(&block).and_then(|(_, b)| b.block.get(&name)).unwrap_or(value))
    })
}

/// Panic button: true = output silence until false.
#[tauri::command]
pub async fn voice_set_mute(mute: bool) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_bypass_mute(mute);
        Ok(())
    })
}

#[tauri::command]
pub async fn voice_set_output_gain(db: f32) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_output_gain_db(db.clamp(-24.0, 24.0));
        Ok(())
    })
}

#[tauri::command]
pub async fn voice_set_monitor_gain(db: f32) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_monitor_gain_db(db.clamp(-60.0, 12.0));
        Ok(())
    })
}

#[tauri::command]
pub async fn voice_set_front_end(front_end: FrontEnd) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_front_end(front_end);
        Ok(())
    })
}

#[tauri::command]
pub async fn voice_set_ceiling(db: f32) -> CmdResult<()> {
    with_live(|l| {
        l.engine.set_ceiling_db(db.clamp(-24.0, 0.0));
        Ok(())
    })
}

// ───────────────────────── samples and previews ─────────────────────────

#[derive(Serialize)]
pub struct SampleInfo {
    pub path: String,
    pub seconds: f32,
    pub peaks: Vec<f32>,
}

fn peaks(samples: &[f32], bins: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; bins];
    }
    let per = samples.len().div_ceil(bins).max(1);
    samples.chunks(per).map(|c| c.iter().fold(0.0f32, |a, v| a.max(v.abs())).min(1.0)).collect()
}

fn sample_info(path: &Path) -> CmdResult<SampleInfo> {
    let clip = nekotone_voice_core::audio::decode(path).map_err(err)?;
    Ok(SampleInfo { path: path.display().to_string(), seconds: clip.duration_secs(), peaks: peaks(&clip.samples, 120) })
}

/// The user's recorded sample, if there is one.
#[tauri::command]
pub async fn voice_sample(app: AppHandle) -> CmdResult<Option<SampleInfo>> {
    blocking(move || {
        let p = sample_path(&app);
        if !p.is_file() {
            return Ok(None);
        }
        sample_info(&p).map(Some)
    })
    .await
}

/// Forget the recorded sample (previews use the built-in voice again).
#[tauri::command]
pub async fn voice_delete_sample(app: AppHandle) -> CmdResult<()> {
    blocking(move || {
        let p = sample_path(&app);
        if p.is_file() {
            std::fs::remove_file(&p).map_err(any)?;
        }
        Ok(())
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileInfo {
    /// The saved calibration voices start from.
    pub saved: Option<voice::Profile>,
    /// What the running voice changer has measured (None when stopped).
    pub live: Option<voice::Profile>,
}

/// The speaker profile: the saved calibration and the live measurement.
#[tauri::command]
pub async fn voice_profile(app: AppHandle) -> CmdResult<ProfileInfo> {
    blocking(move || Ok(ProfileInfo { saved: saved_profile(&app), live: LIVE.lock().as_ref().map(|l| l.engine.status().profile) })).await
}

/// "Calibrate": measure your voice from the recorded sample and save it, so
/// voices with targets are right from your first word (they keep learning).
#[tauri::command]
pub async fn voice_calibrate(app: AppHandle) -> CmdResult<voice::Profile> {
    blocking(move || {
        let rec = sample_path(&app);
        if !rec.is_file() {
            return Err(CmdError { kind: "not-found", message: "record a sample of your voice first".into() });
        }
        let clip = nekotone_voice_core::audio::decode(&rec).map_err(err)?;
        // the full identity: pitch, range and tract, and the style layer
        let p = voice::measure_identity(&clip.samples, clip.sample_rate as f32);
        if p.voiced_secs < 2.0 {
            return Err(any(format!("only {:.1} s of clear speech in the sample: record a sentence or two in your normal voice and try again", p.voiced_secs)));
        }
        write_profile(&app, &p)?;
        if let Some(l) = LIVE.lock().as_ref() {
            l.engine.set_profile(Some(p));
        }
        crate::log_file(&format!("voice: calibrated {:.0} Hz, tract {:.2} ({:.1} s of speech)", p.f0_hz, p.tract, p.voiced_secs));
        Ok(p)
    })
    .await
}

/// Forget the calibration (voices learn your voice from scratch as you speak).
#[tauri::command]
pub async fn voice_forget_profile(app: AppHandle) -> CmdResult<()> {
    blocking(move || {
        let path = profile_path(&app);
        if path.is_file() {
            std::fs::remove_file(&path).map_err(any)?;
        }
        if let Some(l) = LIVE.lock().as_ref() {
            l.engine.set_profile(None);
        }
        Ok(())
    })
    .await
}

fn store_normalized_sample(app: &AppHandle, clip: nekotone_voice_core::audio::Clip, source: &str) -> CmdResult<SampleInfo> {
    let dest = sample_path(app);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(any)?;
    }
    let mut samples = clip.samples;
    let peak = samples.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if peak < 1e-4 {
        return Err(any("the recording is silent: check the microphone and try again"));
    }
    let gain = (0.5 / peak).min(20.0);
    samples.iter_mut().for_each(|v| *v *= gain);
    nekotone_voice_core::audio::write_wav(&dest, &samples, clip.sample_rate, 1).map_err(err)?;
    crate::log_file(&format!(
        "voice: stored a {:.1} s sample from {source} (peak {:.1} dBFS before normalising)",
        samples.len() as f32 / clip.sample_rate as f32,
        20.0 * peak.log10()
    ));
    sample_info(&dest)
}

/// Import an audio file as the saved sample.
#[tauri::command]
pub async fn voice_import_sample(app: AppHandle, path: String) -> CmdResult<SampleInfo> {
    blocking(move || {
        let clip = nekotone_voice_core::audio::decode(Path::new(&path)).map_err(err)?;
        store_normalized_sample(&app, clip, "an imported file")
    })
    .await
}

/// Store a browser-recorded WAV as the saved sample.
#[tauri::command]
pub async fn voice_store_sample(app: AppHandle, bytes: Vec<u8>) -> CmdResult<SampleInfo> {
    blocking(move || {
        let dir = data_dir(&app).join("capture-cache");
        std::fs::create_dir_all(&dir).map_err(any)?;
        let incoming = dir.join("incoming-sample.wav");
        std::fs::write(&incoming, bytes).map_err(any)?;
        let clip = nekotone_voice_core::audio::decode(&incoming).map_err(err)?;
        let _ = std::fs::remove_file(&incoming);
        store_normalized_sample(&app, clip, "the browser recorder")
    })
    .await
}

#[derive(Serialize)]
pub struct Preview {
    /// The rendered WAV (mono, 16-bit).
    pub path: String,
    pub seconds: f32,
    pub peaks: Vec<f32>,
    /// True when it came from the cache.
    pub cached: bool,
    /// "recorded" (your sample) or "synthetic" (the built-in voice).
    pub sample: &'static str,
    pub render_ms: f32,
}

/// 64-bit FNV-1a (stable across runs and Rust versions, for cache names).
fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn file_key(path: &Path) -> String {
    let meta = std::fs::metadata(path).ok();
    let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let modified = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}|{len}|{modified}", path.display())
}

/// Bump when the synthetic voice or the render settings change.
const PREVIEW_VERSION: &str = "voice-preview-3"; // 3: voices with targets, timbre match

/// Render `preset` over the recorded sample (`use_recorded` and one exists)
/// or the built-in synthetic voice; cached.
#[tauri::command]
pub async fn voice_preview(app: AppHandle, preset: Preset, use_recorded: bool) -> CmdResult<Preview> {
    blocking(move || {
        let dir = previews_dir(&app);
        std::fs::create_dir_all(&dir).map_err(any)?;
        let rec = sample_path(&app);
        let (input, kind) = if use_recorded && rec.is_file() {
            (rec, "recorded")
        } else {
            // A female-register sample shows "Male" best; everything else starts from a male voice.
            let high = preset.id == "male";
            let p = dir.join(if high { "synthetic-high.wav" } else { "synthetic-low.wav" });
            if !p.is_file() {
                let s = synthetic_voice(if high { 205.0 } else { 112.0 }, 4.2);
                let tmp = p.with_extension("tmp.wav");
                nekotone_voice_core::audio::write_wav(&tmp, &s, SYN_RATE, 1).map_err(err)?;
                std::fs::rename(&tmp, &p).map_err(any)?;
            }
            (p, "synthetic")
        };
        let src_key = file_key(&input);
        let chain = serde_json::to_string(&preset.blocks).map_err(any)?;
        let mut h = fnv(PREVIEW_VERSION.as_bytes(), 0xcbf29ce484222325);
        h = fnv(src_key.as_bytes(), h);
        h = fnv(chain.as_bytes(), h);
        let out = dir.join(format!("{h:016x}.wav"));
        let t0 = std::time::Instant::now();
        let (samples, rate, cached) = if out.is_file() {
            let c = nekotone_voice_core::audio::decode(&out).map_err(err)?;
            (c.samples, c.sample_rate, true)
        } else {
            let clip = nekotone_voice_core::audio::decode(&input).map_err(err)?;
            // start from this clip's own speaker profile: the preview is on
            // target from its first word (derived from the clip: the cache key holds)
            let profile = Some(voice::measure_profile(&clip.samples, clip.sample_rate as f32));
            let o = voice::RenderOptions { tail_secs: 1.2, profile, ..Default::default() };
            let mut p = preset.clone();
            p.sanitize();
            let y = voice::render_with(&clip.samples, clip.sample_rate, &p, &o);
            let tmp = out.with_extension("tmp.wav");
            nekotone_voice_core::audio::write_wav(&tmp, &y, clip.sample_rate, 1).map_err(err)?;
            std::fs::rename(&tmp, &out).map_err(any)?;
            prune_previews(&dir, 80);
            (y, clip.sample_rate, false)
        };
        Ok(Preview {
            path: out.display().to_string(),
            seconds: samples.len() as f32 / rate as f32,
            peaks: peaks(&samples, 120),
            cached,
            sample: kind,
            render_ms: t0.elapsed().as_secs_f32() * 1000.0,
        })
    })
    .await
}

/// Keep the newest `keep` rendered previews.
fn prune_previews(dir: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).map(|n| n.len() == 20 && n.ends_with(".wav")).unwrap_or(false))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    if files.len() <= keep {
        return;
    }
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    for (_, p) in files.into_iter().skip(keep) {
        let _ = std::fs::remove_file(p);
    }
}

// ───────────────────────── the built-in synthetic voice ─────────────────────────

const SYN_RATE: u32 = 48_000;

/// A two-pole resonator with unity DC gain (one formant).
struct Resonator {
    c1: f32,
    c2: f32,
    g: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn new(f: f32, bw: f32) -> Self {
        let rate = SYN_RATE as f32;
        let r = (-std::f32::consts::PI * bw / rate).exp();
        let c1 = 2.0 * r * (std::f32::consts::TAU * f / rate).cos();
        let c2 = -r * r;
        Resonator { c1, c2, g: 1.0 - c1 - c2, y1: 0.0, y2: 0.0 }
    }
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.g * x + self.c1 * self.y1 + self.c2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Speech-like material for previews when there is no recording: a
/// Rosenberg glottal source through vowel formants (a e i o u), phrase
/// intonation with declination, syllable envelopes, "s"/"sh" noise and
/// pauses. Deterministic. `f0` sets the register (112 Hz male, 205 Hz female).
pub fn synthetic_voice(f0: f32, secs: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    const V: [[(f32, f32); 4]; 5] = [
        [(730.0, 90.0), (1090.0, 110.0), (2440.0, 160.0), (3400.0, 250.0)], // a
        [(530.0, 70.0), (1840.0, 100.0), (2480.0, 150.0), (3500.0, 250.0)], // e
        [(270.0, 60.0), (2290.0, 100.0), (3010.0, 150.0), (3700.0, 250.0)], // i
        [(570.0, 80.0), (840.0, 90.0), (2410.0, 150.0), (3400.0, 250.0)],   // o
        [(300.0, 60.0), (870.0, 90.0), (2240.0, 150.0), (3300.0, 250.0)],   // u
    ];
    let scale = if f0 > 160.0 { 1.17 } else { 1.0 }; // a smaller vocal tract for the high voice
    let rate = SYN_RATE as f32;
    let total = (secs * rate) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(total + 4800);
    let mut seed = 0x1234_5678u32;
    let mut noise = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    // Syllable plan: (vowel, length s, sibilant after, pause after)
    let plan: [(usize, f32, bool, f32); 14] = [
        (0, 0.20, false, 0.02),
        (2, 0.16, true, 0.03),
        (3, 0.26, false, 0.02),
        (1, 0.18, false, 0.22),
        (4, 0.17, true, 0.02),
        (0, 0.24, false, 0.03),
        (2, 0.15, false, 0.02),
        (3, 0.30, true, 0.30),
        (1, 0.19, false, 0.02),
        (0, 0.22, true, 0.03),
        (4, 0.16, false, 0.02),
        (2, 0.20, false, 0.02),
        (3, 0.34, true, 0.20),
        (0, 0.28, false, 0.25),
    ];
    let mut phase = 0.0f32;
    let mut prev = 0.0f32;
    let mut k = 0usize;
    while out.len() < total {
        let (vi, dur, sib, pause) = plan[k % plan.len()];
        let n = (dur * rate) as usize;
        let mut res: Vec<Resonator> = V[vi].iter().map(|&(f, bw)| Resonator::new(f * scale, bw)).collect();
        let mut seg = Vec::with_capacity(n);
        for i in 0..n {
            let tp = (out.len() + i) as f32 / rate; // time in the take
            let phrase = (tp % 1.9) / 1.9; // declination over each phrase
            let f = f0 * (1.0 + 0.10 * (2.0 * PI * 0.9 * tp).sin() - 0.12 * phrase + 0.004 * noise());
            phase += f / rate;
            phase -= phase.floor();
            let (a, b) = (0.4, 0.16);
            let g = if phase < a {
                0.5 * (1.0 - (PI * phase / a).cos())
            } else if phase < a + b {
                (PI * (phase - a) / (2.0 * b)).cos()
            } else {
                0.0
            };
            let mut x = g - prev + 0.01 * noise(); // radiation + breath
            prev = g;
            for r in res.iter_mut() {
                x = r.tick(x);
            }
            let t = i as f32 / n as f32;
            seg.push(x * (PI * t).sin().powf(0.5));
        }
        let m = seg.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        out.extend(seg.iter().map(|v| v / m * 0.45));
        if sib {
            let n = (0.09 * rate) as usize;
            let (mut lp, mut hp_prev_x, mut hp_y) = (0.0f32, 0.0f32, 0.0f32);
            for i in 0..n {
                let x = noise();
                // crude band: one-pole high-pass then low-pass
                hp_y = 0.55 * (hp_y + x - hp_prev_x);
                hp_prev_x = x;
                lp += 0.6 * (hp_y - lp);
                out.push(lp * 0.16 * (PI * i as f32 / n as f32).sin());
            }
        }
        out.extend(std::iter::repeat_n(0.0, (pause * rate) as usize));
        k += 1;
    }
    out.truncate(total);
    out
}

// ───────────────────────── Speak for me ─────────────────────────
//
// Your words in a Kokoro voice (voice/speak.rs in core): the microphone goes
// only to the voice-activity detector and Whisper; the output carries only
// synthesised speech. One engine at a time, and never together with the
// voice changer (each start stops the other). Models are loaded once and
// kept (Kokoro ~1 s, Whisper ~1 s).

struct SpeakModels {
    tts: Option<Arc<Tts>>,
    stt: Option<(String, Arc<dyn Transcriber>)>,
    /// Chatterbox with the saved voice prints (your own voice); loaded on first use.
    clone: Option<Arc<CloneSynth>>,
}

static SPEAK: Mutex<Option<SpeakEngine>> = parking_lot::const_mutex(None);
static SPEAK_MODELS: Mutex<SpeakModels> = parking_lot::const_mutex(SpeakModels { tts: None, stt: None, clone: None });
/// Serialises loading Chatterbox (seconds, ~1 GB) so two quick clicks load it once.
static CLONE_LOADING: Mutex<()> = parking_lot::const_mutex(());

/// The first voice's print name (`clone:mine`); more voices get ids from their names.
const MY_VOICE: &str = "mine";

fn clone_dir(app: &AppHandle) -> PathBuf {
    data_dir(app).join("voice-clone")
}

/// The compute server from Settings (kept while its address and token stay
/// the same, so it remembers which voice prints it already has).
static COMPUTE: Mutex<Option<(String, String, Arc<nekotone_voice_core::remote::ComputeServer>)>> = parking_lot::const_mutex(None);

fn compute_server(app: &AppHandle) -> Option<Arc<nekotone_voice_core::remote::ComputeServer>> {
    let (url, token) = {
        let s = app.state::<AppState>();
        let s = s.settings.lock();
        (s.compute_server.trim().to_string(), s.compute_token.trim().to_string())
    };
    if url.is_empty() {
        return None;
    }
    let mut g = COMPUTE.lock();
    if let Some((u, t, s)) = g.as_ref() {
        if *u == url && *t == token {
            return Some(s.clone());
        }
    }
    let s = Arc::new(nekotone_voice_core::remote::ComputeServer::new(&url, Some(token.clone())).ok()?);
    *g = Some((url, token, s.clone()));
    Some(s)
}

/// The voice clone for Speak for me: on the compute server when one is set
/// and answers (this PC loads Chatterbox only if the server fails), else here.
fn speak_clone_synth(app: &AppHandle) -> CmdResult<Arc<dyn Synth>> {
    if let Some(server) = compute_server(app) {
        match server.info() {
            Ok(i) => {
                crate::log_file(&format!("speak: voice clone on {} ({}, {})", i.host, i.backend, server.url()));
                let app2 = app.clone();
                let mode = app.state::<AppState>().settings.lock().compute_mode.clone();
                let rc = nekotone_voice_core::remote::RemoteClone::new(server, clone_dir(app), move || {
                    speak_clone(&app2).map(|c| c as Arc<dyn Synth>).map_err(|e| nekotone_voice_core::Error::Model(e.message))
                })
                .with_rails(nekotone_voice_core::remote::Rails::from_name(&mode));
                // this PC's engine loads in the background, so it can take
                // over at once when the server is busy
                rc.preload();
                return Ok(Arc::new(rc));
            }
            Err(e) => crate::log_file(&format!("speak: compute server not used ({e}); the voice clone runs on this PC")),
        }
    }
    Ok(speak_clone(app)? as Arc<dyn Synth>)
}

// ───────────────────────── share this PC (serve) ─────────────────────────

/// The server Neko Player runs for other computers: (port, stopper, backend).
static SERVING: Mutex<Option<(u16, nekotone_voice_core::remote::Stopper, String)>> = parking_lot::const_mutex(None);
static SERVE_ERROR: Mutex<Option<String>> = parking_lot::const_mutex(None);

#[derive(Serialize)]
pub struct ServeStatus {
    pub running: bool,
    pub port: u16,
    /// "192.168.1.50:8199" for other computers' Settings → Compute server.
    pub address: Option<String>,
    /// Where the voice clone runs here ("directml", "cpu").
    pub backend: Option<String>,
    pub error: Option<String>,
}

fn serve_status(port: u16) -> ServeStatus {
    let s = SERVING.lock();
    let ip = nekotone_voice_core::remote::lan_address().map(|i| i.to_string());
    match s.as_ref() {
        Some((p, _, b)) => ServeStatus { running: true, port: *p, address: Some(format!("{}:{p}", ip.unwrap_or_else(|| "this-pc".into()))), backend: Some(b.clone()), error: None },
        None => ServeStatus { running: false, port, address: None, backend: None, error: SERVE_ERROR.lock().clone() },
    }
}

fn stop_serving() {
    if let Some((_, stop, _)) = SERVING.lock().take() {
        stop.stop();
        crate::log_file("serve: stopped");
    }
}

/// Start (or restart) serving with the Settings' port and token; loads the
/// voice clone here if it is not loaded yet.
pub(crate) fn start_serving(app: &AppHandle) -> CmdResult<()> {
    stop_serving();
    let (port, token) = {
        let s = app.state::<AppState>();
        let s = s.settings.lock();
        (s.serve_port, s.serve_token.trim().to_string())
    };
    let r = (|| {
        let listener = nekotone_voice_core::remote::bind(&format!("0.0.0.0:{port}")).map_err(err)?;
        let synth = speak_clone(app)?;
        let engine = Arc::new(nekotone_voice_core::remote::ChatterboxEngine::new(synth.engine().clone()));
        // what other computers are told (the graphics side when any part runs there)
        let backend = nekotone_voice_core::remote::CloneEngine::backend(&*engine);
        let stop = listener.stopper();
        let prints = data_dir(app).join("server-prints");
        std::thread::Builder::new()
            .name("nekotone-serve".into())
            .spawn(move || {
                let _ = listener.run(engine, &prints, Some(token).filter(|t| !t.is_empty()), 2, None);
            })
            .map_err(any)?;
        *SERVING.lock() = Some((port, stop, backend.clone()));
        crate::log_file(&format!("serve: sharing the voice clone on port {port} ({backend})"));
        Ok(())
    })();
    *SERVE_ERROR.lock() = r.as_ref().err().map(|e: &CmdError| e.message.clone());
    r
}

/// Share this PC on or off (saved in Settings by the page).
#[tauri::command]
pub async fn compute_serve(app: AppHandle, on: bool) -> CmdResult<ServeStatus> {
    blocking(move || {
        let port = app.state::<AppState>().settings.lock().serve_port;
        if on {
            start_serving(&app)?;
        } else {
            stop_serving();
            *SERVE_ERROR.lock() = None;
        }
        Ok(serve_status(port))
    })
    .await
}

#[tauri::command]
pub async fn compute_serve_status(app: AppHandle) -> CmdResult<ServeStatus> {
    let port = app.state::<AppState>().settings.lock().serve_port;
    Ok(serve_status(port))
}

/// Is the compute server at `url` there, and what does it offer?
#[tauri::command]
pub async fn compute_server_test(url: String, token: Option<String>) -> CmdResult<nekotone_voice_core::remote::ServerInfo> {
    blocking(move || nekotone_voice_core::remote::ComputeServer::new(&url, token).and_then(|s| s.info()).map_err(err)).await
}

/// Chatterbox with this app's voice prints (loads it the first time: a few seconds).
fn speak_clone(app: &AppHandle) -> CmdResult<Arc<CloneSynth>> {
    if let Some(c) = &SPEAK_MODELS.lock().clone {
        return Ok(c.clone());
    }
    let _g = CLONE_LOADING.lock();
    if let Some(c) = &SPEAK_MODELS.lock().clone {
        return Ok(c.clone());
    }
    let models = app.state::<AppState>().models.clone();
    let t0 = std::time::Instant::now();
    let cb = Arc::new(Chatterbox::load(&models).map_err(err)?);
    crate::log_file(&format!("speak: Chatterbox loaded in {:.2} s ({:?})", t0.elapsed().as_secs_f32(), cb.engines()));
    let c = Arc::new(CloneSynth::new(cb, clone_dir(app)));
    SPEAK_MODELS.lock().clone = Some(c.clone());
    Ok(c)
}

fn speak_tts(app: &AppHandle) -> CmdResult<Arc<Tts>> {
    let mut m = SPEAK_MODELS.lock();
    if let Some(t) = &m.tts {
        return Ok(t.clone());
    }
    let models = app.state::<AppState>().models.clone();
    let t0 = std::time::Instant::now();
    let tts = Arc::new(Tts::load(&models).map_err(err)?);
    crate::log_file(&format!("speak: Kokoro loaded in {:.2} s on {}", t0.elapsed().as_secs_f32(), tts.device()));
    m.tts = Some(tts.clone());
    Ok(tts)
}

fn speak_stt(app: &AppHandle) -> CmdResult<(Arc<dyn Transcriber>, String)> {
    let want = app.state::<AppState>().settings.lock().whisper_model.clone();
    let id = ModelId::from_name(&want).filter(|m| m.is_whisper()).unwrap_or(ModelId::WhisperBase);
    let mut m = SPEAK_MODELS.lock();
    if let Some((name, s)) = &m.stt {
        if name == id.name() {
            return Ok((s.clone(), name.clone()));
        }
    }
    let models = app.state::<AppState>().models.clone();
    let t0 = std::time::Instant::now();
    let s: Arc<dyn Transcriber> = Arc::new(WhisperOnnx::load(&models, id).map_err(err)?);
    crate::log_file(&format!("speak: {} loaded in {:.2} s", id.name(), t0.elapsed().as_secs_f32()));
    m.stt = Some((id.name().to_string(), s.clone()));
    Ok((s, id.name().to_string()))
}

fn with_speak<T>(f: impl FnOnce(&SpeakEngine) -> CmdResult<T>) -> CmdResult<T> {
    match SPEAK.lock().as_ref() {
        Some(e) => f(e),
        None => Err(CmdError { kind: "not-found", message: "Speak for me is not running".into() }),
    }
}

/// Stop Speak for me (used by voice_start and at exit). True when it ran.
pub fn stop_speak() -> bool {
    let e = SPEAK.lock().take();
    match e {
        Some(e) => {
            e.stop();
            crate::log_file("speak: stopped");
            true
        }
        None => false,
    }
}

pub fn shutdown() {
    if let Some(live) = LIVE.lock().take() {
        live.engine.stop();
    }
    stop_speak();
    stop_serving();
}

#[derive(Serialize)]
pub struct SpeakInfo {
    /// Kokoro (and its lexicons and voices) are downloaded.
    pub installed: bool,
    /// Download size of the voice model when it is not installed.
    pub download_bytes: u64,
    pub license: &'static str,
    pub voices: Vec<nekotone_voice_core::tts::VoiceInfo>,
    /// The Whisper model Speak for me listens with (Settings → transcription).
    pub whisper_model: String,
    pub whisper_installed: bool,
    pub running: bool,
    /// Accents the voice can speak in, for the picker.
    pub accents: Vec<AccentInfo>,
}

#[derive(Serialize)]
pub struct AccentInfo {
    pub id: &'static str,
    pub label: &'static str,
}

/// What Speak for me needs and has (fast: no model is loaded).
#[tauri::command]
pub async fn speak_info(app: AppHandle) -> CmdResult<SpeakInfo> {
    blocking(move || {
        let st = app.state::<AppState>();
        let info = nekotone_voice_core::models::info(ModelId::TtsKokoro);
        let whisper = st.settings.lock().whisper_model.clone();
        let wid = ModelId::from_name(&whisper).filter(|m| m.is_whisper()).unwrap_or(ModelId::WhisperBase);
        Ok(SpeakInfo {
            installed: st.models.path(ModelId::TtsKokoro).is_some(),
            download_bytes: info.total_bytes(),
            license: info.license,
            voices: nekotone_voice_core::tts::voices(),
            whisper_model: wid.name().to_string(),
            whisper_installed: st.models.path(wid).is_some(),
            running: SPEAK.lock().is_some(),
            accents: Accent::ALL.iter().map(|a| AccentInfo { id: a.id(), label: a.label() }).collect(),
        })
    })
    .await
}

/// Start (or restart) Speak for me. Stops the voice changer first. Loads
/// Kokoro and (when listening) Whisper on first use.
#[tauri::command]
pub async fn speak_start(app: AppHandle, config: SpeakConfig, on_event: Channel<SpeakEvent>) -> CmdResult<SpeakStatus> {
    install_exit_hook(&app);
    blocking(move || {
        let _g = STARTING.lock();
        let old = LIVE.lock().take();
        if let Some(l) = old {
            l.engine.stop();
            crate::log_file("voice: stopped for Speak for me");
        }
        stop_speak();
        // Kokoro whenever it is installed (switching back to a stock voice
        // while on air needs it); Chatterbox only when your voice is chosen
        // or Change my voice is on (it takes seconds and ~1 GB to load)
        let mine = is_clone_voice(&config.voice);
        let need_clone = mine || config.convert;
        let kokoro_installed = app.state::<AppState>().models.path(ModelId::TtsKokoro).is_some();
        let kokoro: Option<Arc<dyn Synth>> = if !mine || kokoro_installed { Some(speak_tts(&app)? as Arc<dyn Synth>) } else { None };
        let clone: Option<Arc<dyn Synth>> = if need_clone { Some(speak_clone_synth(&app)?) } else { SPEAK_MODELS.lock().clone.clone().map(|c| c as Arc<dyn Synth>) };
        let tts = Arc::new(VoiceRouter { kokoro, clone, clone_first: mine });
        let stt = if config.listen { Some(speak_stt(&app)?) } else { None };
        let voice = config.voice.clone();
        let engine = SpeakEngine::start(config, tts, stt, move |ev| {
            match &ev {
                SpeakEvent::Level { .. } => return, // the page polls speak_status for meters
                SpeakEvent::Error { message } => crate::log_file(&format!("speak: {message}")),
                SpeakEvent::DeviceLost { role, message } => crate::log_file(&format!("speak: {role} device lost: {message}")),
                SpeakEvent::Speaking { latency_ms: Some(l), .. } => crate::log_file(&format!("speak: latency {l:.0} ms")),
                _ => {}
            }
            let _ = on_event.send(ev);
        })
        .map_err(err)?;
        let st = engine.core().status();
        crate::log_file(&format!("speak: started {voice} ({:?} → {}, TTS on {}, {})", st.input_device, st.output_device, st.tts_device, if st.stt_model.is_empty() { "type only" } else { &st.stt_model }));
        *SPEAK.lock() = Some(engine);
        Ok(st)
    })
    .await
}

#[tauri::command]
pub async fn speak_stop() -> CmdResult<bool> {
    blocking(|| {
        let _g = STARTING.lock();
        Ok(stop_speak())
    })
    .await
}

/// Levels, queue, latency… of the running engine; null when stopped.
#[tauri::command]
pub async fn speak_status() -> CmdResult<Option<SpeakStatus>> {
    Ok(SPEAK.lock().as_ref().map(|e| e.core().status()))
}

/// Type to speak: queue a line. Returns its id (null for an empty line).
#[tauri::command]
pub async fn speak_say(text: String) -> CmdResult<Option<u64>> {
    with_speak(|e| Ok(e.core().say(&text)))
}

#[tauri::command]
pub async fn speak_skip() -> CmdResult<()> {
    with_speak(|e| {
        e.core().skip();
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_clear() -> CmdResult<()> {
    with_speak(|e| {
        e.core().clear();
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_mute(mute: bool) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_mute(mute);
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_listening(on: bool) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_listening(on);
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_voice(voice: String, speed: f32) -> CmdResult<()> {
    if is_clone_voice(&voice) {
        // the running engine must have Chatterbox loaded (the page restarts it otherwise)
        if SPEAK_MODELS.lock().clone.is_none() {
            return Err(CmdError { kind: "error", message: "your voice is not loaded yet: go on air again".into() });
        }
    } else if nekotone_voice_core::tts::voice(&voice).is_none() {
        return Err(any(format!("there is no voice called \"{voice}\"")));
    }
    with_speak(|e| {
        e.core().set_voice(&voice);
        e.core().set_speed(speed);
        Ok(())
    })
}

/// Accent of the synthetic voice ("voice" = the voice's own); applies from the next line.
#[tauri::command]
pub async fn speak_set_accent(accent: String) -> CmdResult<()> {
    let a = Accent::from_id(&accent).ok_or_else(|| any(format!("there is no accent called \"{accent}\"")))?;
    with_speak(|e| {
        e.core().set_accent(a);
        Ok(())
    })
}

/// Effect on the synthetic voice (null = clean); applies from the next line.
#[tauri::command]
pub async fn speak_set_effect(effect: Option<Preset>) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_effect(effect);
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_translate(translate: bool, language: Option<String>) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_translate(translate, language);
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_options(trim_fillers: bool, mask_profanity: bool, half_duplex: bool) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_text_options(trim_fillers, mask_profanity, half_duplex);
        Ok(())
    })
}

#[tauri::command]
pub async fn speak_set_gains(output_db: f32, monitor_db: f32) -> CmdResult<()> {
    with_speak(|e| {
        e.core().set_output_gain_db(output_db);
        e.core().set_monitor_gain_db(monitor_db);
        Ok(())
    })
}

/// Bump when the preview line or rendering changes.
const SPEAK_PREVIEW_VERSION: &str = "speak-preview-3"; // 3: accents

/// A voice saying a sample line (optionally through an effect); cached WAV.
#[tauri::command]
pub async fn speak_preview(app: AppHandle, voice: String, text: Option<String>, speed: Option<f32>, effect: Option<Preset>, accent: Option<String>) -> CmdResult<Preview> {
    if is_clone_voice(&voice) {
        return clone_preview(app, voice, text, effect).await;
    }
    blocking(move || {
        let info = nekotone_voice_core::tts::voice(&voice).ok_or_else(|| any(format!("there is no voice called \"{voice}\"")))?;
        let line = text.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| format!("Hi, I'm {}. This is how I sound when I speak for you.", info.name));
        let speed = speed.unwrap_or(1.0).clamp(0.5, 2.0);
        let accent = accent.as_deref().and_then(Accent::from_id).unwrap_or_default();
        let dir = previews_dir(&app);
        std::fs::create_dir_all(&dir).map_err(any)?;
        let chain = effect.as_ref().map(|p| serde_json::to_string(&p.blocks).unwrap_or_default()).unwrap_or_default();
        let mut h = fnv(SPEAK_PREVIEW_VERSION.as_bytes(), 0xcbf29ce484222325);
        for part in [voice.as_str(), line.as_str(), &format!("{speed:.2}"), chain.as_str(), accent.id()] {
            h = fnv(part.as_bytes(), h);
            h = fnv(b"|", h);
        }
        let out = dir.join(format!("{h:016x}.wav"));
        let t0 = std::time::Instant::now();
        let (samples, rate, cached) = if out.is_file() {
            let c = nekotone_voice_core::audio::decode(&out).map_err(err)?;
            (c.samples, c.sample_rate, true)
        } else {
            let tts = speak_tts(&app)?;
            let a = tts.synthesize_accent(&line, &voice, speed, accent).map_err(err)?;
            let (y, rate) = match effect {
                Some(mut p) => {
                    p.sanitize();
                    let x = nekotone_voice_core::audio::resample(&nekotone_voice_core::audio::Clip { samples: a, sample_rate: nekotone_voice_core::tts::SAMPLE_RATE, source_channels: 1 }, 48_000).map_err(err)?;
                    let profile = Some(voice::measure_profile(&x.samples, 48_000.0));
                    let o = voice::RenderOptions { tail_secs: 1.2, profile, ..Default::default() };
                    (voice::render_with(&x.samples, 48_000, &p, &o), 48_000)
                }
                None => (a, nekotone_voice_core::tts::SAMPLE_RATE),
            };
            let tmp = out.with_extension("tmp.wav");
            nekotone_voice_core::audio::write_wav(&tmp, &y, rate, 1).map_err(err)?;
            std::fs::rename(&tmp, &out).map_err(any)?;
            prune_previews(&dir, 80);
            (y, rate, false)
        };
        Ok(Preview {
            path: out.display().to_string(),
            seconds: samples.len() as f32 / rate as f32,
            peaks: peaks(&samples, 120),
            cached,
            sample: "tts",
            render_ms: t0.elapsed().as_secs_f32() * 1000.0,
        })
    })
    .await
}

// ───────────────────────── your own voice (Chatterbox) ─────────────────────────
//
// "My voices": a few seconds of you (the Voice changer's recorded sample, or
// a file) → Chatterbox's speech encoder → a named voice print saved as
// `<data dir>/voice-clone/<id>.nkvoice` (the first is `mine`). Encoded once;
// Speak for me then speaks as `clone:<id>`. Several can be kept (different
// mics, moods, recordings) and named. Only your own voice, or a voice whose
// owner agreed (the page asks before making one). Stock prints
// (`stock-<kokoro id>`, for Change my voice into a Kokoro voice) live in the
// same folder and are not listed.

#[derive(Serialize)]
pub struct CloneVoice {
    /// Voice id for Speak for me ("clone:mine").
    pub id: String,
    pub name: String,
    pub reference_secs: f32,
    /// Unix seconds.
    pub created: u64,
    pub source: String,
}

#[derive(Serialize)]
pub struct CloneInfo {
    /// The voice-clone model (tts-chatterbox) is downloaded.
    pub installed: bool,
    pub download_bytes: u64,
    pub license: &'static str,
    /// Your first saved voice, if any (kept for older pages; see `voices`).
    pub voice: Option<CloneVoice>,
    /// Every voice you saved, oldest first.
    pub voices: Vec<CloneVoice>,
    /// Seconds of the Voice changer's recorded sample (0 = none).
    pub sample_secs: f32,
    /// Chatterbox is loaded, and where its language model runs ("cpu", "directml"…).
    pub loaded_on: Option<String>,
    /// Loading would put the language model on a GPU (DirectML / TensorRT allowed and a GPU build).
    pub gpu_possible: bool,
    /// Seconds the reference must exceed.
    pub min_secs: f32,
}

fn clone_entry(id: &str, p: &chatterbox::VoicePrint) -> CloneVoice {
    CloneVoice { id: format!("{}{id}", chatterbox::CLONE_PREFIX), name: p.info.name.clone(), reference_secs: p.info.reference_secs, created: p.info.created, source: p.info.source.clone() }
}

/// Your saved voices (not the stock prints), oldest first.
fn clone_voices(app: &AppHandle) -> Vec<CloneVoice> {
    let Ok(rd) = std::fs::read_dir(clone_dir(app)) else { return Vec::new() };
    let mut out: Vec<CloneVoice> = rd
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|x| x.to_str()) != Some(chatterbox::PRINT_EXT) {
                return None;
            }
            let id = path.file_stem()?.to_str()?.to_string();
            if id.starts_with("stock-") {
                return None;
            }
            chatterbox::VoicePrint::load(&path).ok().map(|p| clone_entry(&id, &p))
        })
        .collect();
    out.sort_by_key(|v| (v.created, v.id.clone()));
    out
}

/// A print id from a name: lower-case letters, digits and '-', unique in the folder.
fn new_voice_id(app: &AppHandle, name: &str) -> String {
    let mut base: String = name.trim().to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    base = base.split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-");
    // the default name keeps the first voice's old id, so saved choices still find it
    if base.is_empty() || base.starts_with("stock") || base == "my-voice" {
        base = MY_VOICE.into();
    }
    base.truncate(40);
    let dir = clone_dir(app);
    if !chatterbox::print_path(&dir, &base).exists() {
        return base;
    }
    (2..).map(|n| format!("{base}-{n}")).find(|id| !chatterbox::print_path(&dir, id).exists()).expect("a free id")
}

/// `clone:<id>` or `<id>` → `<id>` (letters, digits, '-', '_' only).
fn voice_id_arg(id: &str) -> CmdResult<String> {
    let id = id.strip_prefix(chatterbox::CLONE_PREFIX).unwrap_or(id);
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(CmdError { kind: "invalid", message: format!("\"{id}\" is not a saved voice") });
    }
    Ok(id.to_string())
}

/// What "My voice" needs and has (fast: no model is loaded).
#[tauri::command]
pub async fn speak_clone_info(app: AppHandle) -> CmdResult<CloneInfo> {
    blocking(move || {
        let st = app.state::<AppState>();
        let info = nekotone_voice_core::models::info(ModelId::TtsChatterbox);
        let sample = sample_path(&app);
        let sample_secs = if sample.is_file() { nekotone_voice_core::audio::decode(&sample).map(|c| c.duration_secs()).unwrap_or(0.0) } else { 0.0 };
        Ok(CloneInfo {
            installed: st.models.path(ModelId::TtsChatterbox).is_some(),
            download_bytes: info.total_bytes(),
            license: info.license,
            voice: clone_voices(&app).into_iter().next(),
            voices: clone_voices(&app),
            sample_secs,
            loaded_on: SPEAK_MODELS.lock().clone.as_ref().map(|c| c.device()),
            gpu_possible: nekotone_voice_core::models::gpu_supported() && nekotone_voice_core::models::accelerator() != nekotone_voice_core::models::Accelerator::Cpu,
            min_secs: chatterbox::MIN_REFERENCE_SECS,
        })
    })
    .await
}

/// Make a voice from `source` (a file path, or None for the Voice changer's
/// recorded sample). With `id`, that voice is made again (keeping its name
/// unless `name` is given); without, a new voice called `name` (default "My
/// voice") is saved. Loads Chatterbox on first use.
#[tauri::command]
pub async fn speak_clone_make(app: AppHandle, source: Option<String>, name: Option<String>, id: Option<String>) -> CmdResult<CloneVoice> {
    blocking(move || {
        let (path, label) = match source.filter(|s| !s.trim().is_empty()) {
            Some(p) => {
                let pb = PathBuf::from(&p);
                let label = pb.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(p);
                (pb, label)
            }
            None => (sample_path(&app), "recorded sample".to_string()),
        };
        if !path.is_file() {
            return Err(CmdError { kind: "not-found", message: "record a sample of your voice first (10 seconds of you talking)".into() });
        }
        let name = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
        let (vid, name) = match id {
            Some(id) => {
                let vid = voice_id_arg(&id)?;
                let old = chatterbox::VoicePrint::load(&chatterbox::print_path(&clone_dir(&app), &vid)).ok().map(|p| p.info.name);
                let name = name.or(old).unwrap_or_else(|| "My voice".into());
                (vid, name)
            }
            None => {
                let name = name.unwrap_or_else(|| "My voice".into());
                (new_voice_id(&app, &name), name)
            }
        };
        let clip = nekotone_voice_core::audio::decode(&path).map_err(err)?;
        let synth = speak_clone(&app)?;
        let t0 = std::time::Instant::now();
        let print = synth.engine().encode_voice(&clip.samples, clip.sample_rate, &name, &label).map_err(err)?;
        let dest = chatterbox::print_path(&clone_dir(&app), &vid);
        print.save(&dest).map_err(err)?;
        synth.set_print(&vid, print.clone());
        crate::log_file(&format!("speak: made voice \"{name}\" ({vid}) from {label} ({:.1} s) in {:.2} s", print.info.reference_secs, t0.elapsed().as_secs_f32()));
        Ok(clone_entry(&vid, &print))
    })
    .await
}

/// Delete a saved voice (default: the first, `mine`). The recording stays;
/// see voice_delete_sample.
#[tauri::command]
pub async fn speak_clone_delete(app: AppHandle, id: Option<String>) -> CmdResult<()> {
    blocking(move || {
        let vid = voice_id_arg(id.as_deref().unwrap_or(MY_VOICE))?;
        let p = chatterbox::print_path(&clone_dir(&app), &vid);
        if p.is_file() {
            std::fs::remove_file(&p).map_err(any)?;
        }
        if let Some(c) = SPEAK_MODELS.lock().clone.as_ref() {
            c.forget(&vid);
        }
        Ok(())
    })
    .await
}

/// Rename a saved voice (its id, and so any saved choice of it, stays).
#[tauri::command]
pub async fn speak_clone_rename(app: AppHandle, id: String, name: String) -> CmdResult<CloneVoice> {
    blocking(move || {
        let vid = voice_id_arg(&id)?;
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(CmdError { kind: "invalid", message: "a voice needs a name".into() });
        }
        let path = chatterbox::print_path(&clone_dir(&app), &vid);
        let mut print = chatterbox::VoicePrint::load(&path).map_err(err)?;
        print.info.name = name;
        print.save(&path).map_err(err)?;
        if let Some(c) = SPEAK_MODELS.lock().clone.as_ref() {
            c.set_print(&vid, print.clone());
        }
        Ok(clone_entry(&vid, &print))
    })
    .await
}

/// Bump when the clone preview line or its rendering changes.
const CLONE_PREVIEW_VERSION: &str = "clone-preview-1";

/// Your voice saying a sample line (optionally through an effect); cached per print.
async fn clone_preview(app: AppHandle, voice: String, text: Option<String>, effect: Option<Preset>) -> CmdResult<Preview> {
    blocking(move || {
        let name = voice.strip_prefix(chatterbox::CLONE_PREFIX).unwrap_or(&voice).to_string();
        let print_file = chatterbox::print_path(&clone_dir(&app), &name);
        if !print_file.is_file() {
            return Err(CmdError { kind: "not-found", message: "make your voice first: record a sample, then Make my voice".into() });
        }
        let line = text.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "Hi, this is my own voice. This is how I sound when I speak for you.".into());
        let dir = previews_dir(&app);
        std::fs::create_dir_all(&dir).map_err(any)?;
        let chain = effect.as_ref().map(|p| serde_json::to_string(&p.blocks).unwrap_or_default()).unwrap_or_default();
        let mut h = fnv(CLONE_PREVIEW_VERSION.as_bytes(), 0xcbf29ce484222325);
        let print_key = file_key(&print_file);
        for part in [print_key.as_str(), line.as_str(), chain.as_str()] {
            h = fnv(part.as_bytes(), h);
            h = fnv(b"|", h);
        }
        let out = dir.join(format!("{h:016x}.wav"));
        let t0 = std::time::Instant::now();
        let (samples, rate, cached) = if out.is_file() {
            let c = nekotone_voice_core::audio::decode(&out).map_err(err)?;
            (c.samples, c.sample_rate, true)
        } else {
            // on the compute server when there is one (no local model load), else here
            let remote = compute_server(&app).filter(|s| s.usable()).and_then(|s| {
                let mut a = Vec::new();
                for piece in chatterbox::plan(&line, chatterbox::FIRST_PIECE_CHARS) {
                    a.extend(s.synth(&piece.phonemes, &print_file).ok()?);
                }
                Some(a)
            });
            let a = match remote {
                Some(a) => a,
                None => {
                    let synth = speak_clone(&app)?;
                    let print = chatterbox::VoicePrint::load(&print_file).map_err(err)?;
                    synth.engine().synthesize(&line, &print, &chatterbox::Sampling::default()).map_err(err)?
                }
            };
            let (y, rate) = match effect {
                Some(mut p) => {
                    p.sanitize();
                    let x = nekotone_voice_core::audio::resample(&nekotone_voice_core::audio::Clip { samples: a, sample_rate: chatterbox::SAMPLE_RATE, source_channels: 1 }, 48_000).map_err(err)?;
                    let profile = Some(voice::measure_profile(&x.samples, 48_000.0));
                    let o = voice::RenderOptions { tail_secs: 1.2, profile, ..Default::default() };
                    (voice::render_with(&x.samples, 48_000, &p, &o), 48_000)
                }
                None => (a, chatterbox::SAMPLE_RATE),
            };
            let tmp = out.with_extension("tmp.wav");
            nekotone_voice_core::audio::write_wav(&tmp, &y, rate, 1).map_err(err)?;
            std::fs::rename(&tmp, &out).map_err(any)?;
            prune_previews(&dir, 80);
            (y, rate, false)
        };
        Ok(Preview {
            path: out.display().to_string(),
            seconds: samples.len() as f32 / rate as f32,
            peaks: peaks(&samples, 120),
            cached,
            sample: "clone",
            render_ms: t0.elapsed().as_secs_f32() * 1000.0,
        })
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_voice_is_speechlike_and_renders() {
        let s = synthetic_voice(112.0, 3.0);
        assert_eq!(s.len(), 3 * SYN_RATE as usize);
        let peak = s.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(peak > 0.2 && peak < 1.0, "{peak}");
        let y = voice::render(&s, SYN_RATE, &voice::preset_by_id("dragon").unwrap());
        assert_eq!(y.len(), s.len());
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(y.iter().fold(0.0f32, |a, v| a.max(v.abs())) > 0.05);
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("My Dragon!"), "my-dragon");
        assert_eq!(slug("  "), "preset");
    }
}
