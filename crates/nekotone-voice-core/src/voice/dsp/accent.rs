//! Accent colour for a live voice: what can be moved in real time without
//! knowing the words.
//!
//! An accent is (1) where the vowels sit (F1/F2 of each vowel), (2) whether
//! r colours vowels (an r-coloured vowel has F3 pulled down near F2), and
//! (3) the melody (how phrases rise or fall, how much the pitch moves, a
//! syllable-rate lilt). Each frame's formants are classified by their
//! position in a speaker-normalised vowel space (Hz of an average adult
//! male: divide by the speaker's tract scale) and moved by the accent's
//! rules; the melody is shaped on every grain. This colours your voice
//! towards the accent; it cannot change which vowel a word uses (that needs
//! the words: Speak for me).
//!
//! | accent      | vowels | r | melody |
//! |-------------|--------|---|--------|
//! | British RP  | LOT backer, GOAT fronter | removed | falling phrases, narrower range |
//! | Australian  | TRAP raised, GOOSE/FOOT fronted, FLEECE onset lowered | removed | rising phrases (uptalk), livelier |
//! | Irish       | STRUT backed, TRAP backer | kept | lilt, rising |
//! | Scottish    | GOOSE strongly fronted, low vowels central | kept | gently rising |
//! | Southern US | TRAP raised, GOOSE fronted | kept | drawled glides, wide range, falling |
//! | Indian      | TRAP raised, STRUT centralised | kept | syllable-rate lilt, livelier |
//! | Welsh       | GOOSE a little fronted | removed | strong sing-song lilt, wide range |

/// Accent labels, in the order of the Voice block's `accent` values (0 = none).
pub const ACCENTS: [&str; 8] = ["None", "British RP", "Australian", "Irish", "Scottish", "Southern US", "Indian", "Welsh"];

/// The accent for a parameter value (rounded; 0 = none).
pub fn accent_index(v: f32) -> usize {
    (v.round().max(0.0) as usize).min(ACCENTS.len() - 1)
}

/// Formant ratios for one frame. `f` holds F1..F3 in Hz (the first `n`
/// valid), `tract` the speaker's tract scale, `amount` 0..1 scales the move.
pub fn vowel_ratios(accent: usize, f: &[f32; 3], n: usize, tract: f32, amount: f32) -> [f32; 3] {
    let mut r = [1.0f32; 3];
    if accent == 0 || n < 2 || amount <= 0.0 {
        return r;
    }
    let t = tract.clamp(0.7, 1.6);
    let (f1, f2) = (f[0] / t, f[1] / t);
    let f3 = if n >= 3 { f[2] / t } else { 2500.0 };
    let high_back = f1 < 450.0 && f2 < 1300.0;
    let low_front = f1 > 620.0 && f2 > 1450.0;
    let low = f1 > 620.0;
    let low_back = f1 > 600.0 && f2 < 1300.0;
    let mid_central = (500.0..750.0).contains(&f1) && (1100.0..1450.0).contains(&f2);
    let goat_like = (400.0..600.0).contains(&f1) && (900.0..1400.0).contains(&f2);
    let high_front = f1 < 400.0 && f2 > 1900.0;
    // r-coloured: F3 pulled down close to F2
    let rhotic = n >= 3 && f3 < 2000.0 && f3 - f2 < 800.0;
    let non_rhotic = matches!(accent, 1 | 2 | 7);
    match accent {
        1 => {
            if low_back {
                r[1] = 0.9;
            }
            if goat_like {
                r[1] = 1.12;
            }
        }
        2 => {
            if low_front {
                r[0] = 0.86;
                r[1] = 1.04;
            }
            if high_back {
                r[1] = 1.35;
            }
            if high_front {
                r[0] = 1.08;
            }
        }
        3 => {
            if mid_central {
                r[1] = 0.85;
            }
            if low_front {
                r[1] = 0.93;
            }
        }
        4 => {
            if high_back {
                r[1] = 1.5;
            }
            if low {
                r[1] = (1350.0 / f2).clamp(0.8, 1.25).powf(0.7);
            }
        }
        5 => {
            if low_front {
                r[0] = 0.9;
                r[1] = 1.05;
            }
            if high_back {
                r[1] = 1.2;
            }
        }
        6 => {
            if low_front {
                r[0] = 0.85;
                r[1] = 1.05;
            }
            if mid_central {
                r[0] = 0.95;
            }
        }
        7 => {
            if high_back {
                r[1] = 1.1;
            }
        }
        _ => {}
    }
    if non_rhotic && rhotic {
        // take the r out: F3 back to where an uncoloured vowel has it
        r[2] = (2450.0 / f3).max(1.0);
    }
    for v in r.iter_mut() {
        *v = v.powf(amount);
    }
    r
}

/// The accent's melody.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Melody {
    /// Multiplies the voice's `expr` (how far the pitch moves around its median).
    pub range: f32,
    /// Pitch slope along a phrase (semitones per second, after `delay_s`).
    pub slope_st_s: f32,
    pub delay_s: f32,
    /// Largest phrase rise (or fall) in semitones.
    pub cap_st: f32,
    /// Lilt: a sine of `lilt_st` semitones at `lilt_hz` along the phrase.
    pub lilt_hz: f32,
    pub lilt_st: f32,
}

pub fn melody(accent: usize, amount: f32) -> Melody {
    let m = match accent {
        1 => Melody { range: 0.9, slope_st_s: -1.2, delay_s: 0.2, cap_st: 2.5, lilt_hz: 0.0, lilt_st: 0.0 },
        2 => Melody { range: 1.1, slope_st_s: 2.0, delay_s: 0.4, cap_st: 3.0, lilt_hz: 0.0, lilt_st: 0.0 },
        3 => Melody { range: 1.15, slope_st_s: 0.8, delay_s: 0.3, cap_st: 2.0, lilt_hz: 2.5, lilt_st: 0.7 },
        4 => Melody { range: 1.05, slope_st_s: 0.6, delay_s: 0.3, cap_st: 1.5, lilt_hz: 0.0, lilt_st: 0.0 },
        5 => Melody { range: 1.2, slope_st_s: -0.6, delay_s: 0.3, cap_st: 1.5, lilt_hz: 1.5, lilt_st: 0.5 },
        6 => Melody { range: 1.15, slope_st_s: 0.3, delay_s: 0.2, cap_st: 1.0, lilt_hz: 4.0, lilt_st: 0.8 },
        7 => Melody { range: 1.3, slope_st_s: 0.0, delay_s: 0.0, cap_st: 0.0, lilt_hz: 3.0, lilt_st: 1.0 },
        _ => Melody { range: 1.0, slope_st_s: 0.0, delay_s: 0.0, cap_st: 0.0, lilt_hz: 0.0, lilt_st: 0.0 },
    };
    let a = amount.clamp(0.0, 1.0);
    Melody { range: 1.0 + (m.range - 1.0) * a, slope_st_s: m.slope_st_s * a, cap_st: m.cap_st * a, lilt_st: m.lilt_st * a, ..m }
}

impl Melody {
    /// Pitch offset (semitones) `t` seconds into a phrase.
    pub fn offset(&self, t: f32) -> f32 {
        let slope = (self.slope_st_s * (t - self.delay_s).max(0.0)).clamp(-self.cap_st, self.cap_st);
        let lilt = if self.lilt_st > 0.0 { self.lilt_st * (2.0 * std::f32::consts::PI * self.lilt_hz * t).sin() } else { 0.0 };
        slope + lilt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_move_the_vowels_they_name_and_nothing_else() {
        let goose = [300.0, 870.0, 2240.0];
        assert!(vowel_ratios(2, &goose, 3, 1.0, 1.0)[1] > 1.3, "Australian fronts GOOSE");
        assert!(vowel_ratios(4, &goose, 3, 1.0, 1.0)[1] > 1.45, "Scottish fronts GOOSE more");
        // the same vowel from a woman (tract 1.17) is recognised the same way
        let goose_w = goose.map(|f| f * 1.17);
        assert!(vowel_ratios(2, &goose_w, 3, 1.17, 1.0)[1] > 1.3);
        let fleece_mid = [530.0, 1840.0, 2480.0];
        assert_eq!(vowel_ratios(2, &fleece_mid, 3, 1.0, 1.0), [1.0, 1.0, 1.0], "DRESS is left alone");
        assert_eq!(vowel_ratios(0, &goose, 3, 1.0, 1.0), [1.0, 1.0, 1.0], "no accent, no change");
        assert_eq!(vowel_ratios(2, &goose, 3, 1.0, 0.0), [1.0, 1.0, 1.0], "amount 0, no change");
        let half = vowel_ratios(2, &goose, 3, 1.0, 0.5)[1];
        assert!((half - 1.35f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn non_rhotic_accents_take_the_r_out() {
        let nurse_r = [490.0, 1350.0, 1690.0];
        assert!(vowel_ratios(1, &nurse_r, 3, 1.0, 1.0)[2] * 1690.0 >= 2400.0, "RP removes r-colour");
        assert!(vowel_ratios(2, &nurse_r, 3, 1.0, 1.0)[2] > 1.4, "Australian too");
        assert_eq!(vowel_ratios(4, &nurse_r, 3, 1.0, 1.0)[2], 1.0, "Scottish keeps its r");
        let plain = [490.0, 1350.0, 2500.0];
        assert_eq!(vowel_ratios(1, &plain, 3, 1.0, 1.0)[2], 1.0, "uncoloured vowels keep F3");
    }

    #[test]
    fn melodies_rise_fall_and_lilt() {
        let au = melody(2, 1.0);
        assert!(au.offset(1.5) > 1.9 && au.offset(0.2) == 0.0, "uptalk rises after 0.4 s");
        assert!(au.offset(10.0) <= 3.0, "and is capped");
        assert!(melody(1, 1.0).offset(1.5) < -1.0, "RP falls");
        let welsh = melody(7, 1.0);
        let swing = (0..100).map(|i| welsh.offset(i as f32 * 0.01)).fold(f32::MIN, f32::max);
        assert!(swing > 0.9, "Welsh lilts");
        assert_eq!(melody(0, 1.0).offset(2.0), 0.0);
        assert_eq!(melody(2, 0.0).offset(2.0), 0.0);
        assert_eq!(accent_index(2.4), 2);
        assert_eq!(accent_index(99.0), ACCENTS.len() - 1);
    }
}
