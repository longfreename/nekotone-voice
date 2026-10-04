//! Real-time voice changer: a microphone-to-virtual-microphone effect engine
//! (owner: the voice package).
//!
//! ```text
//! mic (cpal input) ─ mono ─ HPF ─ gate ─ de-esser ─ preset chain ─ gain ─ limiter ─ mute ─┬─ SPSC ring ─ drift resampler ─ output (virtual cable / null sink)
//!                                                                                         └─ SPSC ring ─ drift resampler ─ monitor (headphones)
//! ```
//!
//! * [`VoiceEngine`] runs it on devices (feature `player`, which brings cpal).
//! * [`render`] runs the *same* [`chain::Processor`] offline, so previews,
//!   tests and the live mic sound identical.
//! * Presets are chains of [`BlockSpec`]s ([`presets()`], JSON user presets).
//!   A flat chain can hold parallel layers (`split` / `layer` / `merge`,
//!   see [`dsp::route`]): the creature voices mix a main voice with a darker
//!   sub layer that has its own saturation. The chain aligns the latency of
//!   both sides of a merge.
//! * [`virtual_mic::virtual_mic_status`] explains how other programs get the
//!   processed voice as a microphone on this OS.
//!
//! Guarantees ("your real voice never breaks through"):
//! 1. There is no dry path: the output callbacks only read the processed ring.
//! 2. Voice presets re-synthesise the voice (LP-PSOLA, see
//!    [`dsp::shifter`]): voiced sound is rebuilt from grains at the new
//!    period and unvoiced sound from shaped noise, so the original f0 does not
//!    survive. Every path through a layered preset must pass a voice stage
//!    ([`chain::every_path_resynthesises`], checked for all built-ins).
//!    Presets with `disguise == false` (Cathedral, Radio, Studio…)
//!    are marked so the UI can say "keeps your voice".
//! 3. A gate with hysteresis/hold silences everything below the threshold.
//! 4. Mute ([`VoiceEngine::set_bypass_mute`]) outputs silence (5 ms fade).
//! 5. A lookahead brick-wall limiter is last; the output never exceeds its ceiling.
//! 6. Preset changes cross-fade (40 ms, equal power) between two fully
//!    processed chains padded to the same latency.
//! 7. If processing overloads, the output underruns into silence — never dry.

pub mod blocks;
pub mod chain;
pub mod dsp;
pub mod identity;
pub mod pads;
pub mod params;
pub mod presets;
pub mod ring;
pub mod speak;
pub mod virtual_mic;

#[cfg(feature = "player")]
pub mod devices;
#[cfg(feature = "player")]
pub mod engine;

#[cfg(test)]
mod lab;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod adaptive_tests;

pub use blocks::{block_types, BlockSpec, BlockType};
pub use chain::{FrontEnd, Processor};
pub use dsp::profile::{learn as learn_identity, measure as measure_profile, measure_identity, Profile};
pub use dsp::style::VoiceStyle;
pub use identity::{apply_controls, VoiceControls};
pub use params::ParamDef;
pub use presets::{builtin_presets, load_user_presets, preset_by_id, presets, save_user_preset, user_presets_dir, BlockSlot, Preset};
pub use virtual_mic::{virtual_mic_status, VirtualMicKind, VirtualMicStatus};

#[cfg(feature = "player")]
pub use devices::{input_devices, output_devices, AudioDevice};
#[cfg(feature = "player")]
pub use engine::{LatencyBreakdown, VoiceConfig, VoiceEngine, VoiceEvent, VoiceStatus};

use crate::Result;
use std::path::Path;

/// Options for offline rendering.
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    pub front_end: FrontEnd,
    pub output_gain_db: f32,
    pub ceiling_db: f32,
    /// Processing block size (the engine uses its device block size; results
    /// are identical up to float rounding).
    pub block: usize,
    /// Remove the processing latency so the output lines up with the input.
    pub align: bool,
    /// Extra seconds rendered after the input (reverb/echo tails).
    pub tail_secs: f32,
    /// The speaker profile to start from (a saved calibration); `None`
    /// measures from neutral as the input plays.
    pub profile: Option<dsp::profile::Profile>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions { front_end: FrontEnd::default(), output_gain_db: 0.0, ceiling_db: -1.0, block: 128, align: true, tail_secs: 0.0, profile: None }
    }
}

/// Render `input` (mono, `rate` Hz) through `preset` with the real-time DSP.
/// Output has the same length as the input and is latency-aligned.
pub fn render(input: &[f32], rate: u32, preset: &Preset) -> Vec<f32> {
    render_with(input, rate, preset, &RenderOptions::default())
}

/// [`render`] with options.
pub fn render_with(input: &[f32], rate: u32, preset: &Preset, o: &RenderOptions) -> Vec<f32> {
    let mut p = Processor::new(rate as f32, o.block, o.front_end, &preset.blocks, o.output_gain_db, o.ceiling_db).with_profile(o.profile);
    let lat = if o.align { p.latency() } else { 0 };
    let tail = (o.tail_secs.max(0.0) * rate as f32) as usize;
    let total = input.len() + tail + lat;
    let mut out = Vec::with_capacity(total);
    let mut buf = vec![0.0f32; o.block.max(1)];
    let mut pos = 0;
    while pos < total {
        let n = o.block.min(total - pos);
        for (k, b) in buf[..n].iter_mut().enumerate() {
            *b = input.get(pos + k).copied().unwrap_or(0.0);
        }
        p.process(&mut buf[..n]);
        out.extend_from_slice(&buf[..n]);
        drop(p.take_trash());
        pos += n;
    }
    out.drain(..lat);
    out
}

/// Result of [`render_file`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct RenderReport {
    pub seconds: f32,
    pub sample_rate: u32,
    pub latency_ms: f32,
    /// Processing time / audio duration (0.02 = 2 % of one core).
    pub cpu_fraction: f32,
}

/// Decode any audio file Nekotone reads, render it through `preset` (with a
/// tail of `tail_secs`) and write a mono 16-bit WAV. Used by the CLI.
pub fn render_file(input: &Path, output: &Path, preset: &Preset, tail_secs: f32) -> Result<RenderReport> {
    let clip = crate::audio::decode(input)?;
    let rate = clip.sample_rate;
    let o = RenderOptions { tail_secs, ..Default::default() };
    let t0 = std::time::Instant::now();
    let out = render_with(&clip.samples, rate, preset, &o);
    let el = t0.elapsed().as_secs_f32();
    crate::audio::write_wav(output, &out, rate, 1)?;
    let secs = clip.samples.len() as f32 / rate as f32;
    let lat = Processor::new(rate as f32, 128, o.front_end, &preset.blocks, 0.0, -1.0).latency();
    Ok(RenderReport { seconds: secs, sample_rate: rate, latency_ms: lat as f32 * 1000.0 / rate as f32, cpu_fraction: el / secs.max(1e-3) })
}
