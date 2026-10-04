//! Voice controls: six sliders that move a voice's *identity*, not its
//! audio. Each control is -1..1 (0 = unchanged) and maps onto the
//! parameters a person's voice is made of: speaking pitch and vocal-tract
//! size (physiology), intonation range (behaviour), breath, irregularity
//! and tremor (style), and a tone EQ (warmth, clarity):
//!
//! | control    | pitch        | tract  | intonation | breath / jitter / tremor | tone EQ                          | level |
//! |------------|--------------|--------|------------|--------------------------|----------------------------------|-------|
//! | confidence | −0.8 st      |        | ×1.25      | less breath, steadier    | +1.5 dB presence                 | +1.5  |
//! | warmth     |              |        |            |                          | +4 dB below 300 Hz, −3 dB above 5 kHz |       |
//! | clarity    |              |        |            |                          | +4 dB presence, −1.5 dB boxiness, +2.5 dB air |  |
//! | age (+ older, − younger) | ∓1.5 st | ∓6 % | older: ×0.9 | older: breath, jitter, fry, tremor | older: −1 dB air        |       |
//! | gender (+ feminine, − masculine) | ±4 st | ±8 % | ×1.1 feminine | a little breath feminine | ∓1.5 dB body, ±1.5 dB air | |
//! | energy     |              |        | ×1.35      |                          | +1 dB presence                   | +1.5  |
//!
//! (values at ±1; each scales linearly). On a voice with targets (the
//! adaptive voices) the pitch and tract move the *targets*; on a voice
//! relative to yours they move the shift. [`apply_controls`] always starts
//! from the preset as saved, so moving a slider back and forth never
//! accumulates.

use super::blocks::BlockSpec;
use super::dsp::filters::EqParams;
use super::presets::{BlockSlot, Preset};
use serde::{Deserialize, Serialize};

/// The six controls, each -1..1 (0 = the voice as it is).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct VoiceControls {
    pub confidence: f32,
    pub warmth: f32,
    pub clarity: f32,
    /// + older, − younger.
    pub age: f32,
    /// + more feminine, − more masculine.
    pub gender: f32,
    pub energy: f32,
}

impl VoiceControls {
    /// Every control inside -1..1 (non-finite values become 0).
    pub fn sanitized(self) -> VoiceControls {
        let c = |v: f32| if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
        VoiceControls { confidence: c(self.confidence), warmth: c(self.warmth), clarity: c(self.clarity), age: c(self.age), gender: c(self.gender), energy: c(self.energy) }
    }

    pub fn is_neutral(&self) -> bool {
        *self == VoiceControls::default()
    }
}

/// The block id of the tone EQ the controls add (when a voice has none).
pub const TONE_ID: &str = "tone";

/// Pitch (semitones) and tract (log ratio) moves of the controls.
fn physiology(c: &VoiceControls) -> (f32, f32) {
    let st = 4.0 * c.gender - 0.8 * c.confidence - 1.5 * c.age;
    let ln_tract = 0.08 * c.gender - 0.06 * c.age;
    (st, ln_tract)
}

/// `base` with the controls applied. Needs a Voice block (the first one is
/// the main voice); without one only the tone and level change. Adds a tone
/// EQ block (id [`TONE_ID`]) after the voice stage when the controls need
/// one and the voice has none.
pub fn apply_controls(base: &Preset, controls: &VoiceControls) -> Preset {
    let c = controls.sanitized();
    let mut p = base.clone();
    if c.is_neutral() {
        return p;
    }
    let (st, ln_tract) = physiology(&c);
    let mut voice_at = None;
    for (i, slot) in p.blocks.iter_mut().enumerate() {
        if let BlockSpec::Voice(v) = &mut slot.block {
            if v.target_hz > 0.0 {
                v.target_hz = (v.target_hz * 2f32.powf(st / 12.0)).clamp(40.0, 600.0);
            } else {
                v.pitch_st = (v.pitch_st + st).clamp(-24.0, 24.0);
            }
            if v.target_tract > 0.0 {
                v.target_tract = (v.target_tract * ln_tract.exp()).clamp(0.5, 2.0);
            } else {
                v.formant = (v.formant * ln_tract.exp()).clamp(0.5, 2.0);
            }
            let older = c.age.max(0.0);
            let expr = v.expr * (1.0 + 0.25 * c.confidence) * (1.0 + 0.35 * c.energy) * (1.0 - 0.1 * older) * (1.0 + 0.1 * c.gender.max(0.0));
            v.expr = expr.clamp(0.0, 2.0);
            v.breath = (v.breath + 0.35 * older + 0.06 * c.gender.max(0.0) - 0.1 * c.confidence.max(0.0)).clamp(0.0, 1.0);
            v.jitter_cents = ((v.jitter_cents + 20.0 * older) * (1.0 - 0.5 * c.confidence.max(0.0))).clamp(0.0, 100.0);
            // a little vocal fry comes with age
            v.growl = (v.growl + 0.08 * older).clamp(0.0, 1.0);
            if older > 0.0 {
                if v.vibrato_st <= 0.0 {
                    // an older voice's tremor: slow and shallow
                    v.vibrato_hz = 5.5;
                }
                v.vibrato_st = (v.vibrato_st + 0.12 * older).clamp(0.0, 2.0);
            }
            voice_at = Some(i);
            break;
        }
    }
    // tone: warmth (body), clarity (presence, less box, air), the rest a little
    let low_db = 4.0 * c.warmth - 1.5 * c.gender;
    let presence_db = 4.0 * c.clarity + 1.5 * c.confidence + 1.0 * c.energy;
    let box_db = -1.5 * c.clarity.max(0.0);
    let air_db = 2.5 * c.clarity - 3.0 * c.warmth + 1.5 * c.gender - 1.0 * c.age.max(0.0);
    let level_db = 1.5 * c.confidence + 1.5 * c.energy;
    let tone = |e: &mut EqParams| {
        e.low_hz = 300.0;
        e.low_db = (e.low_db + low_db).clamp(-24.0, 24.0);
        e.mid1_hz = 3000.0;
        e.mid1_q = 1.0;
        e.mid1_db = (e.mid1_db + presence_db).clamp(-24.0, 24.0);
        e.mid2_hz = 500.0;
        e.mid2_q = 1.2;
        e.mid2_db = (e.mid2_db + box_db).clamp(-24.0, 24.0);
        e.high_hz = 5000.0;
        e.high_db = (e.high_db + air_db).clamp(-24.0, 24.0);
        e.gain_db = (e.gain_db + level_db).clamp(-24.0, 24.0);
    };
    if let Some(slot) = p.blocks.iter_mut().find(|b| b.id == TONE_ID) {
        if let BlockSpec::Eq(e) = &mut slot.block {
            tone(e);
            return p;
        }
    }
    let mut e = EqParams::default();
    tone(&mut e);
    // after the voice stage and its timbre match, before dynamics and effects
    let at = match voice_at {
        Some(i) => {
            let mut j = i + 1;
            while j < p.blocks.len() && matches!(p.blocks[j].block, BlockSpec::Timbre(_)) {
                j += 1;
            }
            j
        }
        None => 0,
    };
    p.blocks.insert(at, BlockSlot { id: TONE_ID.into(), block: BlockSpec::Eq(e) });
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::dsp::shifter::VoiceParams;

    fn you() -> Preset {
        Preset {
            id: "me".into(),
            name: "Me".into(),
            description: String::new(),
            category: String::new(),
            disguise: false,
            builtin: false,
            blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(VoiceParams::default()) }],
        }
    }

    fn voice(p: &Preset) -> VoiceParams {
        p.blocks.iter().find_map(|b| if let BlockSpec::Voice(v) = b.block { Some(v) } else { None }).unwrap()
    }

    fn tone(p: &Preset) -> EqParams {
        p.blocks.iter().find_map(|b| if let BlockSpec::Eq(e) = b.block { Some(e) } else { None }).unwrap()
    }

    #[test]
    fn neutral_controls_change_nothing() {
        let base = you();
        assert_eq!(apply_controls(&base, &VoiceControls::default()), base);
    }

    #[test]
    fn controls_move_the_identity_and_never_accumulate() {
        let base = you();
        let fem = apply_controls(&base, &VoiceControls { gender: 1.0, ..Default::default() });
        let v = voice(&fem);
        assert!((v.pitch_st - 4.0).abs() < 1e-4 && (v.formant - 0.08f32.exp()).abs() < 1e-4);
        // applied to the saved preset again: the same result, not twice the move
        assert_eq!(apply_controls(&base, &VoiceControls { gender: 1.0, ..Default::default() }), fem);
        // exactly one tone block, after the voice
        assert_eq!(fem.blocks.iter().filter(|b| b.id == TONE_ID).count(), 1);
        assert_eq!(fem.blocks[1].id, TONE_ID);
        // an adaptive voice moves its targets instead
        let mut adaptive = you();
        if let BlockSpec::Voice(v) = &mut adaptive.blocks[0].block {
            v.target_hz = 110.0;
            v.target_tract = 1.0;
        }
        let v = voice(&apply_controls(&adaptive, &VoiceControls { gender: -1.0, ..Default::default() }));
        assert!((v.target_hz - 110.0 * 2f32.powf(-4.0 / 12.0)).abs() < 0.01 && v.pitch_st == 0.0);
        assert!((v.target_tract - (-0.08f32).exp()).abs() < 1e-4 && v.formant == 1.0);
    }

    #[test]
    fn each_control_does_what_its_name_says() {
        let base = you();
        let at = |c: VoiceControls| apply_controls(&base, &c);
        let conf = at(VoiceControls { confidence: 1.0, ..Default::default() });
        assert!(voice(&conf).expr > 1.2 && voice(&conf).pitch_st < 0.0 && tone(&conf).gain_db > 1.0);
        let warm = at(VoiceControls { warmth: 1.0, ..Default::default() });
        assert!(tone(&warm).low_db > 3.0 && tone(&warm).high_db < -2.0);
        let clear = at(VoiceControls { clarity: 1.0, ..Default::default() });
        assert!(tone(&clear).mid1_db > 3.5 && tone(&clear).high_db > 2.0 && tone(&clear).mid2_db < 0.0);
        let old = at(VoiceControls { age: 1.0, ..Default::default() });
        let v = voice(&old);
        assert!(v.jitter_cents > 10.0 && v.vibrato_st > 0.1 && v.breath > 0.15 && v.pitch_st < 0.0 && v.formant < 1.0);
        let young = voice(&at(VoiceControls { age: -1.0, ..Default::default() }));
        assert!(young.pitch_st > 1.0 && young.formant > 1.0 && young.jitter_cents == 0.0);
        let lively = voice(&at(VoiceControls { energy: 1.0, ..Default::default() }));
        let calm = voice(&at(VoiceControls { energy: -1.0, ..Default::default() }));
        assert!(lively.expr > 1.3 && calm.expr < 0.7);
    }

    #[test]
    fn out_of_range_and_nonsense_values_are_clamped() {
        let c = VoiceControls { confidence: f32::NAN, warmth: 9.0, clarity: -9.0, age: f32::INFINITY, gender: 0.5, energy: 0.0 }.sanitized();
        assert_eq!(c, VoiceControls { confidence: 0.0, warmth: 1.0, clarity: -1.0, age: 0.0, gender: 0.5, energy: 0.0 });
    }
}
