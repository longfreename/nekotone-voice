//! Built-in voice presets and JSON user presets.
//!
//! A preset is an ordered list of blocks with ids (for `set_param`). User
//! presets are JSON files (one preset per file) in a folder, by default
//! `%LOCALAPPDATA%\Nekotone\voice-presets`.

use super::blocks::BlockSpec;
use super::dsp::character::*;
use super::dsp::delay::*;
use super::dsp::dynamics::*;
use super::dsp::filters::*;
use super::dsp::modulation::*;
use super::dsp::reverb::*;
use super::dsp::route::*;
use super::dsp::shifter::*;
use super::dsp::timbre::*;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockSlot {
    /// Stable id within the preset, used by `set_param` (e.g. "voice", "verb").
    pub id: String,
    pub block: BlockSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    /// Stable identifier (slug), e.g. "female".
    pub id: String,
    pub name: String,
    /// One line for the picker.
    pub description: String,
    /// "Gender & age", "Creatures", "Fun", "Spaces & devices", "Clean", or anything for user presets.
    pub category: String,
    /// True when the preset re-synthesises the voice with a different
    /// pitch/formant (so your natural voice is not recognisable). The UI
    /// should mark presets where this is false ("keeps your voice").
    #[serde(default)]
    pub disguise: bool,
    /// True for presets shipped with Nekotone.
    #[serde(default)]
    pub builtin: bool,
    pub blocks: Vec<BlockSlot>,
}

impl Preset {
    /// The block with this id.
    pub fn block(&self, id: &str) -> Option<(usize, &BlockSlot)> {
        self.blocks.iter().enumerate().find(|(_, b)| b.id == id)
    }

    /// True when every path from input to output re-synthesises the voice
    /// (layers included), i.e. the dry voice cannot reach the output.
    pub fn every_path_resynthesises(&self) -> bool {
        super::chain::every_path_resynthesises(&self.blocks)
    }

    /// Clamp every parameter; fill empty ids.
    pub fn sanitize(&mut self) {
        for (i, b) in self.blocks.iter_mut().enumerate() {
            b.block.sanitize();
            if b.id.trim().is_empty() {
                b.id = format!("{}{}", b.block.kind(), i + 1);
            }
        }
        if self.id.trim().is_empty() {
            self.id = slug(&self.name);
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn from_json(s: &str) -> Result<Preset> {
        let mut p: Preset = serde_json::from_str(s).map_err(|e| Error::Other(anyhow::anyhow!("not a voice preset: {e}")))?;
        p.sanitize();
        Ok(p)
    }
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

fn slot(id: &str, block: BlockSpec) -> BlockSlot {
    BlockSlot { id: id.into(), block }
}

fn preset(id: &str, name: &str, category: &str, disguise: bool, description: &str, blocks: Vec<BlockSlot>) -> Preset {
    Preset {
        id: id.into(),
        name: name.into(),
        description: description.into(),
        category: category.into(),
        disguise,
        builtin: true,
        blocks,
    }
}

fn voice(pitch_st: f32, formant: f32) -> VoiceParams {
    VoiceParams { pitch_st, formant, ..Default::default() }
}

/// A voice defined by where it should end up, whoever speaks: a speaking
/// pitch (median) and a vocal-tract scale (1 ≈ average man, 1.17 average
/// woman). The shift adapts to the measured speaker (see `dsp::profile`).
fn target(hz: f32, tract: f32) -> VoiceParams {
    VoiceParams { target_hz: hz, target_tract: tract, ..Default::default() }
}

/// Timbre match towards a voice's body and brightness (dB re the mids), at a common speech level.
fn timbre(body_db: f32, bright_db: f32) -> BlockSpec {
    BlockSpec::Timbre(TimbreParams { body_db, bright_db, ..Default::default() })
}

fn comp(threshold_db: f32, ratio: f32, makeup_db: f32) -> BlockSpec {
    BlockSpec::Compressor(CompressorParams { threshold_db, ratio, makeup_db, attack_ms: 4.0, release_ms: 120.0, knee_db: 6.0 })
}

fn eq(low_hz: f32, low_db: f32, mid1: (f32, f32, f32), mid2: (f32, f32, f32), high_hz: f32, high_db: f32) -> BlockSpec {
    BlockSpec::Eq(EqParams {
        low_hz,
        low_db,
        mid1_hz: mid1.0,
        mid1_db: mid1.1,
        mid1_q: mid1.2,
        mid2_hz: mid2.0,
        mid2_db: mid2.1,
        mid2_q: mid2.2,
        high_hz,
        high_db,
        gain_db: 0.0,
    })
}

const AGE: &str = "Gender & age";
const CREATURES: &str = "Creatures";
const FUN: &str = "Fun";
const SPACES: &str = "Spaces & devices";
const CLEAN: &str = "Clean";
const SPEAKERS: &str = "Speakers";

/// A speaker built on your voice: where they speak (pitch, tract), how
/// (melody, breath, irregularity, fry, tremor) and their colour (timbre
/// targets, an optional character EQ), whoever is talking.
struct Speaker {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    hz: f32,
    tract: f32,
    expr: f32,
    breath: f32,
    jitter_cents: f32,
    growl: f32,
    tremor_st: f32,
    body_db: f32,
    bright_db: f32,
    eq: Option<BlockSpec>,
    trim_db: f32,
    /// Live accent colour (`dsp::accent::ACCENTS` index; 0 = the speaker's own).
    accent: f32,
}

fn speaker(sp: Speaker) -> Preset {
    let mut blocks = vec![
        slot(
            "voice",
            BlockSpec::Voice(VoiceParams {
                expr: sp.expr,
                breath: sp.breath,
                jitter_cents: sp.jitter_cents,
                growl: sp.growl,
                // an older voice's tremor: slow and shallow
                vibrato_hz: if sp.tremor_st > 0.0 { 5.5 } else { 5.0 },
                vibrato_st: sp.tremor_st,
                accent: sp.accent,
                ..target(sp.hz, sp.tract)
            }),
        ),
        slot("match", timbre(sp.body_db, sp.bright_db)),
    ];
    if let Some(eq) = sp.eq {
        blocks.push(slot("eq", eq));
    }
    blocks.push(slot("comp", comp(-24.0, 2.5, 3.0)));
    blocks.push(slot("trim", BlockSpec::Gain(GainParams { db: sp.trim_db })));
    preset(sp.id, sp.name, SPEAKERS, true, sp.description, blocks)
}

/// The cast: men and women of different ages and colours, each landing on
/// the same voice whoever speaks (adaptive targets).
fn speakers() -> Vec<Preset> {
    let base = Speaker { id: "", name: "", description: "", hz: 0.0, tract: 1.0, expr: 1.0, breath: 0.0, jitter_cents: 0.0, growl: 0.0, tremor_st: 0.0, body_db: 0.0, bright_db: -12.0, eq: None, trim_db: 4.8, accent: 0.0 };
    [
        Speaker { id: "emma", name: "Emma", description: "Young woman, clear and bright (about 220 Hz).", hz: 220.0, tract: 1.2, expr: 1.1, breath: 0.08, body_db: -3.0, bright_db: -10.0, ..base },
        Speaker { id: "olivia", name: "Olivia", description: "Woman with a warm alto (about 180 Hz), relaxed.", hz: 180.0, tract: 1.13, breath: 0.12, body_db: 0.0, bright_db: -12.0, ..base },
        Speaker { id: "chloe", name: "Chloe", description: "Teenage girl, light and lively (about 255 Hz).", hz: 255.0, tract: 1.26, expr: 1.2, breath: 0.1, body_db: -4.0, bright_db: -9.0, ..base },
        Speaker { id: "margaret", name: "Margaret", description: "Older woman, soft and breathy with a slight tremor (about 195 Hz).", hz: 195.0, tract: 1.12, expr: 0.9, breath: 0.5, jitter_cents: 16.0, tremor_st: 0.18, body_db: -2.0, bright_db: -14.0, ..base },
        Speaker { id: "nina", name: "Nina", description: "Woman with a husky, smoky low voice (about 165 Hz).", hz: 165.0, tract: 1.1, expr: 0.95, breath: 0.45, growl: 0.08, body_db: 1.0, bright_db: -15.0, ..base },
        Speaker { id: "liam", name: "Liam", description: "Young man, light tenor (about 135 Hz).", hz: 135.0, tract: 1.03, expr: 1.05, breath: 0.05, body_db: 0.0, bright_db: -12.0, ..base },
        Speaker { id: "james", name: "James", description: "Man with a warm baritone (about 105 Hz).", hz: 105.0, tract: 0.97, expr: 0.95, body_db: 3.0, bright_db: -13.0, ..base },
        Speaker { id: "victor", name: "Victor", description: "Man with a deep, steady bass (about 80 Hz).", hz: 80.0, tract: 0.9, expr: 0.85, body_db: 5.0, bright_db: -15.0, ..base },
        Speaker { id: "walter", name: "Walter", description: "Older man, gravelly with a slight tremor (about 112 Hz).", hz: 112.0, tract: 0.97, expr: 0.9, breath: 0.3, jitter_cents: 20.0, growl: 0.18, tremor_st: 0.15, body_db: 2.0, bright_db: -14.0, ..base },
        Speaker { id: "tyler", name: "Tyler", description: "Teenage boy, a livelier melody (about 165 Hz).", hz: 165.0, tract: 1.1, expr: 1.15, breath: 0.05, body_db: -1.0, bright_db: -11.0, ..base },
        Speaker {
            id: "neil",
            name: "Neil",
            description: "Man with a thin, nasal voice (about 125 Hz).",
            hz: 125.0,
            tract: 1.0,
            body_db: -3.0,
            bright_db: -11.0,
            // nasal: a resonance near 1.3 kHz, less chest
            eq: Some(eq(200.0, -4.0, (1300.0, 8.0, 2.0), (2600.0, 2.0, 1.5), 8000.0, 0.0)),
            ..base
        },
        // the same cast idea with a live accent colour (see dsp::accent)
        Speaker { id: "sophie", name: "Sophie", description: "Woman with a British (RP) colour: falling phrases, no r (about 205 Hz).", hz: 205.0, tract: 1.16, expr: 0.95, breath: 0.08, body_db: -2.0, bright_db: -11.0, accent: 1.0, ..base },
        Speaker { id: "henry", name: "Henry", description: "Man with a British (RP) colour, measured baritone (about 108 Hz).", hz: 108.0, tract: 0.97, expr: 0.9, body_db: 2.0, bright_db: -13.0, accent: 1.0, ..base },
        Speaker { id: "mia", name: "Mia", description: "Woman with an Australian colour: rising phrases, fronted vowels (about 215 Hz).", hz: 215.0, tract: 1.18, expr: 1.1, breath: 0.06, body_db: -3.0, bright_db: -10.0, accent: 2.0, ..base },
        Speaker { id: "jack", name: "Jack", description: "Man with an Australian colour, easy-going (about 120 Hz).", hz: 120.0, tract: 1.0, expr: 1.05, body_db: 1.0, bright_db: -12.0, accent: 2.0, ..base },
        Speaker { id: "siobhan", name: "Siobhan", description: "Woman with an Irish colour: a lilting melody (about 200 Hz).", hz: 200.0, tract: 1.15, expr: 1.1, breath: 0.1, body_db: -1.0, bright_db: -11.0, accent: 3.0, ..base },
        Speaker { id: "callum", name: "Callum", description: "Man with a Scottish colour: r kept, fronted oo (about 115 Hz).", hz: 115.0, tract: 0.98, expr: 1.0, body_db: 2.0, bright_db: -13.0, accent: 4.0, ..base },
        Speaker { id: "daisy", name: "Daisy", description: "Woman with a Southern US colour: wide, drawled melody (about 195 Hz).", hz: 195.0, tract: 1.14, expr: 1.15, breath: 0.12, body_db: -1.0, bright_db: -12.0, accent: 5.0, ..base },
        Speaker { id: "arjun", name: "Arjun", description: "Man with an Indian English colour: a syllable-rate lilt (about 130 Hz).", hz: 130.0, tract: 1.02, expr: 1.1, body_db: 0.0, bright_db: -11.0, accent: 6.0, ..base },
        Speaker { id: "rhys", name: "Rhys", description: "Man with a Welsh colour: a sing-song melody (about 125 Hz).", hz: 125.0, tract: 1.0, expr: 1.1, body_db: 1.0, bright_db: -12.0, accent: 7.0, ..base },
    ]
    .into_iter()
    .map(speaker)
    .collect()
}

const PERSONAS: &str = "You";

/// Your own voice with a twist: personas built from the voice controls
/// (`identity`) on your voice as it is (no targets), so each is *you*,
/// more confident, calmer, younger… rather than someone else.
fn personas() -> Vec<Preset> {
    use super::identity::{apply_controls, VoiceControls};
    let you = |id: &str, name: &str, description: &str| {
        // your level evened out towards a normal speaking level (at most
        // 9 dB either way); your tone kept. No compressor: you are already
        // the right voice. Measured on the owner's quiet mic (speech at
        // -52 dB): the old 18 dB lift and a -24 dB compressor raised the room
        // noise 20 dB and took the voice's clarity from 0.665 to 0.569
        // (periodicity, 0.3-1.5 kHz) and HNR from 9.3 to 7.4 dB; the
        // owner heard it as messy and unclear.
        let level = BlockSpec::Timbre(TimbreParams { tone: 0.0, range_db: 9.0, ..Default::default() });
        preset(id, name, PERSONAS, false, description, vec![slot("voice", BlockSpec::Voice(VoiceParams::default())), slot("level", level)])
    };
    let c = VoiceControls::default();
    let mut out = vec![
        apply_controls(&you("confident-me", "Confident me", "You, more confident: a little lower, a wider melody, steadier, more presence."), &VoiceControls { confidence: 0.8, clarity: 0.3, ..c }),
        apply_controls(&you("relaxed-me", "Relaxed me", "You, calmer: a flatter melody and a warmer tone."), &VoiceControls { energy: -0.6, warmth: 0.6, confidence: -0.2, ..c }),
        apply_controls(&you("storyteller-me", "Storyteller me", "You, telling a story: a lively melody and a warm, close tone."), &VoiceControls { energy: 0.8, warmth: 0.4, ..c }),
        apply_controls(&you("professional-me", "Professional me", "You on a call that matters: clear, even and confident."), &VoiceControls { clarity: 0.6, confidence: 0.4, energy: -0.1, ..c }),
        apply_controls(&you("younger-me", "Younger me", "You, younger: a little higher and a smaller vocal tract."), &VoiceControls { age: -0.7, ..c }),
        apply_controls(&you("older-me", "Older me", "You, older: lower, breathier, with a slight roughness and tremor."), &VoiceControls { age: 0.8, ..c }),
    ];
    // Anonymous me: still sounds like a person of your kind, but moved off
    // your identity: a different pitch and tract size, your personal
    // intonation flattened, your own breath and irregularity replaced
    let mut anon = apply_controls(&you("anonymous-me", "Anonymous me", "Harder to recognise, still natural: a different pitch and head size, your melody evened out. Not a full disguise; for that, use a Speaker."), &VoiceControls { clarity: 0.3, warmth: -0.2, ..c });
    for b in anon.blocks.iter_mut() {
        if let BlockSpec::Voice(v) = &mut b.block {
            v.pitch_st = -1.5;
            v.formant = 0.92;
            v.expr = 0.7;
        }
    }
    out.push(anon);
    out
}

/// The built-in presets, in display order.
pub fn builtin_presets() -> Vec<Preset> {
    let mut all = personas();
    all.extend(speakers());
    all.extend(classic_presets());
    all
}

fn classic_presets() -> Vec<Preset> {
    vec![
        preset("female", "Female", AGE, true,
            "A woman's voice whoever speaks: about 205 Hz and a 17 % smaller vocal tract than an average man, a little breath and air.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { breath: 0.12, ..target(205.0, 1.17) })),
                slot("match", timbre(-2.0, -10.0)),
                slot("eq", eq(180.0, -3.0, (900.0, -1.0, 1.0), (3200.0, 1.5, 1.0), 7500.0, 2.5)),
                slot("comp", comp(-24.0, 2.5, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 4.9 })),
            ]),
        preset("male", "Male", AGE, true,
            "A man's voice whoever speaks: about 110 Hz and an average man's vocal tract, fuller lows.",
            vec![
                slot("voice", BlockSpec::Voice(target(110.0, 1.0))),
                slot("match", timbre(2.0, -13.0)),
                slot("eq", eq(140.0, 2.0, (350.0, 0.5, 1.0), (2500.0, -0.5, 1.0), 7000.0, -1.5)),
                slot("comp", comp(-24.0, 2.5, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 4.6 })),
            ]),
        preset("child", "Child", AGE, true,
            "A child whoever speaks: about 280 Hz, a 30 % smaller vocal tract, a livelier melody; light and airy.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { breath: 0.1, jitter_cents: 5.0, expr: 1.15, ..target(280.0, 1.3) })),
                slot("match", timbre(-3.0, -10.0)),
                slot("eq", eq(250.0, -4.0, (1500.0, 1.0, 1.0), (4000.0, 1.5, 1.0), 8000.0, 1.0)),
                slot("comp", comp(-24.0, 2.5, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.0 })),
            ]),
        preset("mouse", "Mouse", AGE, true,
            "Tiny and squeaky: about 470 Hz through a tract half the size of a man's.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { expr: 1.2, ..target(470.0, 1.55) })),
                slot("match", timbre(-4.0, -9.0)),
                slot("eq", eq(350.0, -6.0, (1200.0, 0.0, 1.0), (5000.0, 1.0, 1.0), 9000.0, 0.5)),
                slot("comp", comp(-24.0, 3.0, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 6.0 })),
            ]),
        preset("giant", "Giant", AGE, true,
            "Huge and slow: about 62 Hz through a tract 40 % larger than a man's, a steadier melody, clean pulse source, a stone hall kept short.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { pulse: 0.4, expr: 0.8, ..target(62.0, 0.72) })),
                slot("match", timbre(4.0, -16.0)),
                slot("eq", eq(110.0, 3.0, (300.0, -1.5, 1.0), (1800.0, 1.5, 1.0), 5000.0, -4.0)),
                slot("comp", comp(-24.0, 3.0, 3.0)),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 1.3, decay_s: 1.4, predelay_ms: 25.0, damping: 0.65, diffusion: 0.7, early: 0.5, modulation: 0.4, lowcut_hz: 90.0, highcut_hz: 5000.0, mix: 0.14 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.9 })),
            ]),
        preset("narrator", "Deep Narrator", AGE, true,
            "Movie-trailer warmth: about 88 Hz, a slightly larger tract than an average man, rich lows, firm compression.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { expr: 0.9, ..target(88.0, 0.93) })),
                slot("match", timbre(3.0, -13.0)),
                slot("eq", eq(140.0, 3.0, (400.0, -2.0, 1.2), (3500.0, 1.5, 1.0), 9000.0, 1.5)),
                slot("sat", BlockSpec::Saturation(SaturationParams { drive_db: 4.0, asym: 0.2, tone_hz: 12000.0, mix: 0.4, gain_db: 0.0 })),
                slot("comp", comp(-26.0, 3.5, 5.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 4.8 })),
            ]),
        preset("dragon", "Dragon", CREATURES, true,
            "Main voice at about 72 Hz through a large tract with growl; a sub layer an octave below it (darker, tube-saturated, voiced only) 9 dB under; low shelf, high cut, short dark room. Lands there whoever speaks.",
            vec![
                slot("split", BlockSpec::Split(SplitParams { bus: 1.0 })),
                slot("voice", BlockSpec::Voice(VoiceParams { growl: 0.25, pulse: 0.3, expr: 0.8, ..target(72.0, 0.8) })),
                slot("layer", BlockSpec::Layer(LayerParams { bus: 1.0 })),
                slot("layer_voice", BlockSpec::Voice(VoiceParams { growl: 0.35, pulse: 0.6, noise_db: -60.0, expr: 0.8, ..target(36.0, 0.7) })),
                slot("sat", BlockSpec::Saturation(SaturationParams { drive_db: 10.0, asym: 0.35, tone_hz: 1500.0, mix: 1.0, gain_db: 0.0 })),
                slot("merge", BlockSpec::Merge(MergeParams { bus: 1.0, layer_db: -9.0, main_db: 0.0 })),
                slot("match", timbre(4.0, -16.0)),
                slot("rumble", BlockSpec::HighPass(PassParams { hz: 45.0, q: 0.707, stages: 1.0 })),
                slot("eq", eq(120.0, 1.5, (450.0, -2.0, 1.0), (2200.0, 2.5, 1.0), 6000.0, -4.0)),
                slot("comp", comp(-22.0, 3.0, 3.0)),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 0.8, decay_s: 0.9, predelay_ms: 12.0, damping: 0.7, diffusion: 0.7, early: 0.5, modulation: 0.3, lowcut_hz: 100.0, highcut_hz: 5000.0, mix: 0.14 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.0 })),
            ]),
        preset("demon", "Demon", CREATURES, true,
            "Two layers: a voice at about 85 Hz doubled with a detuned copy, plus a sub an octave and more below (-6 dB) with growl and distortion; ring grit; dark cavern with pre-delay. Lands there whoever speaks.",
            vec![
                slot("split", BlockSpec::Split(SplitParams { bus: 1.0 })),
                slot("voice", BlockSpec::Voice(VoiceParams { v1_st: 0.18, v1_db: -4.0, jitter_cents: 6.0, pulse: 0.2, expr: 0.85, ..target(85.0, 0.82) })),
                slot("sat", BlockSpec::Saturation(SaturationParams { drive_db: 8.0, asym: 0.25, tone_hz: 6000.0, mix: 0.4, gain_db: 0.0 })),
                slot("layer", BlockSpec::Layer(LayerParams { bus: 1.0 })),
                slot("layer_voice", BlockSpec::Voice(VoiceParams { growl: 0.4, pulse: 0.7, noise_db: -60.0, expr: 0.85, ..target(34.0, 0.66) })),
                slot("sat2", BlockSpec::Saturation(SaturationParams { drive_db: 18.0, asym: 0.4, tone_hz: 1100.0, mix: 1.0, gain_db: 0.0 })),
                slot("merge", BlockSpec::Merge(MergeParams { bus: 1.0, layer_db: -6.0, main_db: 0.0 })),
                slot("match", timbre(4.0, -16.0)),
                slot("ring", BlockSpec::RingMod(RingModParams { hz: 42.0, lfo_depth_hz: 6.0, lfo_hz: 0.3, mix: 0.12, gain_db: 0.0 })),
                slot("rumble", BlockSpec::HighPass(PassParams { hz: 40.0, q: 0.707, stages: 1.0 })),
                slot("eq", eq(100.0, 1.5, (500.0, -3.0, 1.0), (1800.0, 2.5, 1.2), 5000.0, -6.0)),
                slot("comp", comp(-22.0, 3.0, 3.0)),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 1.6, decay_s: 2.6, predelay_ms: 45.0, damping: 0.72, diffusion: 0.75, early: 0.5, modulation: 0.4, lowcut_hz: 110.0, highcut_hz: 4500.0, mix: 0.2 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.2 })),
            ]),
        preset("alien", "Alien", CREATURES, true,
            "About 185 Hz through a small tract with slow pitch/formant drift; a ring-modulated copy (420 Hz) blended 8 dB under; flanger shimmer; transmission band.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { vibrato_hz: 0.9, vibrato_st: 0.35, formant_lfo_hz: 0.6, formant_lfo_depth: 0.08, ..target(185.0, 1.28) })),
                slot("match", timbre(0.0, -11.0)),
                slot("split", BlockSpec::Split(SplitParams { bus: 1.0 })),
                slot("layer", BlockSpec::Layer(LayerParams { bus: 1.0 })),
                slot("ring", BlockSpec::RingMod(RingModParams { hz: 420.0, lfo_depth_hz: 40.0, lfo_hz: 0.4, mix: 1.0, gain_db: 0.0 })),
                slot("merge", BlockSpec::Merge(MergeParams { bus: 1.0, layer_db: -8.0, main_db: 0.0 })),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 1.0, rate_hz: 0.25, delay_ms: 1.5, depth_ms: 1.0, feedback: 0.5, mix: 0.35 })),
                slot("radio", BlockSpec::Radio(RadioParams { low_hz: 200.0, high_hz: 7000.0, honk_hz: 2500.0, honk_db: 2.0, drive_db: 3.0, noise_db: -60.0, gain_db: 0.0 })),
                slot("comp", comp(-24.0, 2.5, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.1 })),
            ]),
        preset("ghost", "Ghost", CREATURES, true,
            "Breathy spirit: mostly whispered at about 175 Hz, drifting chorus, a soft echo and a long but controlled reverb.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { whisper: 0.8, ..target(175.0, 1.1) })),
                slot("match", timbre(0.0, -11.0)),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 2.0, rate_hz: 0.25, delay_ms: 18.0, depth_ms: 5.0, feedback: 0.0, mix: 0.35 })),
                slot("echo", BlockSpec::Echo(EchoParams { time_ms: 380.0, feedback: 0.35, damping_hz: 3000.0, lowcut_hz: 200.0, wow: 0.4, mix: 0.18 })),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 1.6, decay_s: 4.0, predelay_ms: 60.0, damping: 0.5, diffusion: 0.75, early: 0.3, modulation: 0.6, lowcut_hz: 150.0, highcut_hz: 7000.0, mix: 0.3 })),
                slot("comp", comp(-26.0, 3.0, 5.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 4.8 })),
            ]),
        preset("robot", "Robot", CREATURES, true,
            "Classic vocoder robot: monotone 110 Hz from clean pulses through your vocal tract, a light metallic comb.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { flatten: 1.0, fixed_hz: 110.0, pulse: 1.0, ..voice(0.0, 1.0) })),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 1.0, rate_hz: 0.1, delay_ms: 3.2, depth_ms: 0.0, feedback: 0.35, mix: 0.22 })),
                slot("eq", eq(150.0, -1.0, (1000.0, 1.0, 1.0), (3000.0, 2.0, 1.0), 8000.0, 0.0)),
                slot("comp", comp(-24.0, 3.0, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 3.0 })),
            ]),
        preset("wuuwuu", "WuuWuu", FUN, true,
            "Playful warble: wobbling pitch and mouth around 200 Hz, shimmering chorus.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { vibrato_hz: 5.5, vibrato_st: 1.0, formant_lfo_hz: 3.0, formant_lfo_depth: 0.15, ..target(200.0, 1.12) })),
                slot("match", timbre(0.0, -11.0)),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 3.0, rate_hz: 1.2, delay_ms: 12.0, depth_ms: 3.0, feedback: 0.0, mix: 0.4 })),
                slot("wobble", BlockSpec::Wobble(WobbleParams { rate_hz: 3.0, min_hz: 500.0, max_hz: 3500.0, q: 2.0, band: 0.0, mix: 0.3 })),
                slot("comp", comp(-24.0, 3.0, 3.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.6 })),
            ]),
        preset("underwater", "Underwater", FUN, true,
            "Bottom of the pool: muffled at 1.3 kHz, slow warble, bubbling resonance, a small wet room.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { vibrato_hz: 1.2, vibrato_st: 0.35, ..voice(-2.0, 0.9) })),
                slot("lowpass", BlockSpec::LowPass(PassParams { hz: 1300.0, q: 0.8, stages: 2.0 })),
                slot("wobble", BlockSpec::Wobble(WobbleParams { rate_hz: 0.8, min_hz: 300.0, max_hz: 900.0, q: 3.5, band: 1.0, mix: 0.4 })),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 2.0, rate_hz: 0.4, delay_ms: 20.0, depth_ms: 6.0, feedback: 0.0, mix: 0.45 })),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 0.6, decay_s: 1.0, predelay_ms: 5.0, damping: 0.8, diffusion: 0.7, early: 0.4, modulation: 0.5, lowcut_hz: 80.0, highcut_hz: 2500.0, mix: 0.25 })),
                slot("comp", comp(-24.0, 3.0, 4.0)),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.1 })),
            ]),
        preset("choir", "Choir", FUN, false,
            "You, harmonised: a third, a fifth and octaves, spread and in a hall.",
            vec![
                slot("voice", BlockSpec::Voice(VoiceParams { jitter_cents: 12.0, main_db: -2.0, v1_st: 4.0, v1_db: -6.0, v2_st: 7.0, v2_db: -5.0, v3_st: -12.0, v3_db: -7.0, v4_st: 12.0, v4_db: -11.0, ..voice(0.0, 1.0) })),
                slot("chorus", BlockSpec::Chorus(ChorusParams { voices: 3.0, rate_hz: 0.6, delay_ms: 16.0, depth_ms: 4.0, feedback: 0.0, mix: 0.45 })),
                slot("comp", comp(-24.0, 3.0, 2.0)),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 1.6, decay_s: 3.5, predelay_ms: 30.0, damping: 0.45, diffusion: 0.75, early: 0.4, modulation: 0.5, lowcut_hz: 120.0, highcut_hz: 8000.0, mix: 0.32 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 5.4 })),
            ]),
        preset("cathedral", "Cathedral", SPACES, false,
            "Your own voice in a huge stone cathedral: 6 s tail, smooth and modulated, words kept in front.",
            vec![
                slot("eq", eq(120.0, -1.0, (500.0, 0.0, 1.0), (3000.0, 1.0, 1.0), 9000.0, 1.0)),
                slot("comp", comp(-24.0, 2.5, 3.0)),
                slot("verb", BlockSpec::Reverb(ReverbParams { size: 2.0, decay_s: 6.0, predelay_ms: 50.0, damping: 0.4, diffusion: 0.8, early: 0.45, modulation: 0.6, lowcut_hz: 100.0, highcut_hz: 7000.0, mix: 0.42 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 4.0 })),
            ]),
        preset("radio", "Radio", SPACES, false,
            "AM radio / walkie-talkie: 450 Hz-4 kHz, driven small speaker, a little hiss.",
            vec![
                slot("comp", comp(-28.0, 6.0, 8.0)),
                slot("radio", BlockSpec::Radio(RadioParams { low_hz: 450.0, high_hz: 4000.0, honk_hz: 1800.0, honk_db: 3.0, drive_db: 10.0, noise_db: -40.0, gain_db: 2.0 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 1.3 })),
            ]),
        preset("telephone", "Telephone", SPACES, false,
            "An old phone line: 300-3400 Hz, 8 kHz codec.",
            vec![
                slot("comp", comp(-26.0, 4.0, 6.0)),
                slot("radio", BlockSpec::Radio(RadioParams { low_hz: 300.0, high_hz: 3400.0, honk_hz: 1500.0, honk_db: 2.0, drive_db: 4.0, noise_db: -52.0, gain_db: 2.0 })),
                slot("crush", BlockSpec::Bitcrush(BitcrushParams { bits: 12.0, rate_hz: 8000.0, mix: 1.0 })),
                slot("lowpass", BlockSpec::LowPass(PassParams { hz: 3600.0, q: 0.707, stages: 2.0 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 3.0 })),
            ]),
        preset("megaphone", "Megaphone", SPACES, false,
            "A loud-hailer: honky horn resonance, overdrive and a street slap.",
            vec![
                slot("comp", comp(-26.0, 6.0, 6.0)),
                slot("radio", BlockSpec::Radio(RadioParams { low_hz: 600.0, high_hz: 4500.0, honk_hz: 2000.0, honk_db: 5.0, drive_db: 14.0, noise_db: -60.0, gain_db: 0.0 })),
                slot("echo", BlockSpec::Echo(EchoParams { time_ms: 90.0, feedback: 0.12, damping_hz: 3000.0, lowcut_hz: 300.0, wow: 0.0, mix: 0.12 })),
                slot("trim", BlockSpec::Gain(GainParams { db: 2.2 })),
            ]),
        preset("studio", "Studio", CLEAN, false,
            "Your own voice, polished: gentle EQ and compression (does not disguise you).",
            vec![
                slot("eq", eq(100.0, -1.0, (300.0, -1.5, 1.0), (4000.0, 1.5, 1.0), 10000.0, 2.0)),
                slot("comp", comp(-22.0, 2.5, 3.0)),
            ]),
    ]
}

/// Built-in presets, with any user presets appended (see `load_user_presets`).
pub fn presets() -> Vec<Preset> {
    builtin_presets()
}

/// A built-in preset by id.
pub fn preset_by_id(id: &str) -> Option<Preset> {
    builtin_presets().into_iter().find(|p| p.id.eq_ignore_ascii_case(id) || p.name.eq_ignore_ascii_case(id))
}

/// Default folder for user presets.
pub fn user_presets_dir() -> PathBuf {
    crate::data_dir().join("voice-presets")
}

/// Load every `*.json` preset in `dir`. Files that fail to parse are
/// returned in the second list as "file: reason" sentences.
pub fn load_user_presets(dir: &Path) -> (Vec<Preset>, Vec<String>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return (ok, bad);
    };
    let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        if p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("json")) != Some(true) {
            continue;
        }
        match std::fs::read_to_string(&p).map_err(Error::from).and_then(|s| Preset::from_json(&s)) {
            Ok(mut pr) => {
                pr.builtin = false;
                ok.push(pr);
            }
            Err(e) => bad.push(format!("{}: {e}", p.display())),
        }
    }
    (ok, bad)
}

/// Save a preset as `<dir>/<id>.json` (creating the folder). Returns the path.
pub fn save_user_preset(dir: &Path, preset: &Preset) -> Result<PathBuf> {
    let mut p = preset.clone();
    p.builtin = false;
    p.sanitize();
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", slug(&p.id)));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, p.to_json())?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_are_complete_and_unique() {
        let ps = builtin_presets();
        assert!(ps.len() >= 19);
        let mut ids: Vec<&str> = ps.iter().map(|p| p.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), ps.len());
        for want in [
            "female", "male", "dragon", "demon", "mouse", "wuuwuu", "cathedral", "robot", "radio", "telephone", "ghost", "giant",
            "child", "alien", "underwater", "megaphone", "choir", "narrator",
        ] {
            assert!(preset_by_id(want).is_some(), "{want}");
        }
        for p in &ps {
            assert!(!p.description.is_empty());
            // disguising presets must re-synthesise on every path (layers too)
            if p.disguise {
                assert!(p.every_path_resynthesises(), "{}", p.id);
            }
            let mut ids: Vec<&str> = p.blocks.iter().map(|b| b.id.as_str()).collect();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), p.blocks.len(), "{} has duplicate block ids", p.id);
        }
    }

    #[test]
    fn user_presets_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = preset_by_id("dragon").unwrap();
        p.id = "My Dragon!".into();
        p.name = "My Dragon".into();
        let path = save_user_preset(dir.path(), &p).unwrap();
        assert!(path.ends_with("my-dragon.json"));
        std::fs::write(dir.path().join("broken.json"), "{nope").unwrap();
        let (ok, bad) = load_user_presets(dir.path());
        assert_eq!(ok.len(), 1);
        assert_eq!(bad.len(), 1);
        assert_eq!(ok[0].blocks, p.blocks);
        assert!(!ok[0].builtin);
        // out-of-range values are clamped on load
        let j = r#"{"id":"x","name":"X","description":"","category":"Mine","blocks":[{"id":"v","block":{"type":"voice","pitch_st":99}}]}"#;
        let q = Preset::from_json(j).unwrap();
        assert_eq!(q.blocks[0].block.get("pitch_st"), Some(24.0));
    }
}
