//! The style layer of a speaker's identity: how you sound beyond pitch and
//! vocal-tract size (those are in [`super::profile`]). Measured once, from
//! the calibration sample, and kept with the profile:
//!
//! | measure     | what it is                                                       |
//! |-------------|------------------------------------------------------------------|
//! | `level_db`  | speech level (RMS while speaking, dBFS)                           |
//! | `body_db`   | 100–400 Hz energy re 400–2500 Hz (as the Timbre block measures)   |
//! | `bright_db` | 2.5–8 kHz energy re 400–2500 Hz                                    |
//! | `hnr_db`    | harmonics-to-noise ratio at 1.5–5 kHz on voiced frames: lower is breathier or rougher |
//! | `rate_syl_s`| articulation rate: syllable nuclei per second of speech (pauses excluded) |
//!
//! Offline only (not on the audio thread): it allocates.

use super::filters::{Biquad, Kind};
use super::pitch::PitchTracker;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct VoiceStyle {
    pub level_db: f32,
    pub body_db: f32,
    pub bright_db: f32,
    pub hnr_db: f32,
    pub rate_syl_s: f32,
}

impl Default for VoiceStyle {
    fn default() -> Self {
        VoiceStyle { level_db: -20.0, body_db: 0.0, bright_db: -12.0, hnr_db: 10.0, rate_syl_s: 4.5 }
    }
}

impl VoiceStyle {
    /// Clamp into plausible ranges (a hand-edited or corrupt saved file).
    pub fn sanitized(self) -> VoiceStyle {
        let d = VoiceStyle::default();
        let ok = |v: f32, lo: f32, hi: f32, def: f32| if v.is_finite() { v.clamp(lo, hi) } else { def };
        VoiceStyle {
            level_db: ok(self.level_db, -90.0, 0.0, d.level_db),
            body_db: ok(self.body_db, -40.0, 30.0, d.body_db),
            bright_db: ok(self.bright_db, -60.0, 20.0, d.bright_db),
            hnr_db: ok(self.hnr_db, -20.0, 40.0, d.hnr_db),
            rate_syl_s: ok(self.rate_syl_s, 0.0, 12.0, d.rate_syl_s),
        }
    }
}

/// The style of the speech in `x`, or None when there is too little voiced speech (< 1 s).
pub fn measure_style(x: &[f32], rate: f32) -> Option<VoiceStyle> {
    let (level_db, body_db, bright_db) = super::timbre::balance(x, rate);
    let (hnr_db, voiced_frames) = hnr_high(x, rate);
    if voiced_frames as f32 * 0.04 < 1.0 {
        return None;
    }
    Some(VoiceStyle { level_db, body_db, bright_db, hnr_db, rate_syl_s: syllable_rate(x, rate) }.sanitized())
}

/// Median harmonics-to-noise ratio (dB) at 1.5–5 kHz over voiced 40 ms
/// frames (the normalised autocorrelation of the band at the pitch period,
/// r / (1 − r)), and how many frames were voiced.
pub fn hnr_high(y: &[f32], r: f32) -> (f32, usize) {
    let mut bp = [Biquad::new(Kind::HighPass, 1500.0, 0.54, 0.0, r), Biquad::new(Kind::HighPass, 1500.0, 1.31, 0.0, r), Biquad::new(Kind::LowPass, 5000.0, 0.54, 0.0, r), Biquad::new(Kind::LowPass, 5000.0, 1.31, 0.0, r)];
    let hb: Vec<f32> = y.iter().map(|v| bp.iter_mut().fold(*v, |a, b| b.tick(a))).collect();
    let mut t = PitchTracker::new(r);
    let frame = (0.04 * r) as usize;
    let mut out = Vec::new();
    for (i, v) in y.iter().enumerate() {
        t.push(*v);
        if i % frame != frame - 1 || i < 2 * frame {
            continue;
        }
        let e = t.analyze();
        if !e.voiced {
            continue;
        }
        let lag = e.period.round() as usize;
        let s = i + 1 - frame;
        if s < lag + 2 {
            continue;
        }
        // best lag within ±2 samples (the period is not an integer)
        let mut best = -1.0f64;
        for d in lag.saturating_sub(2)..=lag + 2 {
            let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
            for k in s..=i {
                let (a, b) = (hb[k] as f64, hb[k - d] as f64);
                xy += a * b;
                xx += a * a;
                yy += b * b;
            }
            best = best.max(xy / (xx * yy).sqrt().max(1e-30));
        }
        let rr = best.clamp(1e-4, 0.9999);
        out.push(10.0 * (rr / (1.0 - rr)).log10() as f32);
    }
    out.sort_by(|a, b| a.total_cmp(b));
    (out.get(out.len() / 2).copied().unwrap_or(0.0), out.len())
}

/// Articulation rate: peaks of the 300–2500 Hz loudness contour (smoothed
/// to ~10 Hz, at least 3 dB above the dips around them and within 25 dB of
/// the loudest speech, at least 100 ms apart) per second of speech, where
/// speech is every 10 ms frame within 30 dB of the loudest and pauses over
/// 250 ms are left out.
pub fn syllable_rate(x: &[f32], rate: f32) -> f32 {
    let mut band = [Biquad::new(Kind::HighPass, 300.0, 0.71, 0.0, rate), Biquad::new(Kind::LowPass, 2500.0, 0.71, 0.0, rate)];
    let hop = (rate * 0.01) as usize;
    let mut env = Vec::with_capacity(x.len() / hop + 1);
    let mut acc = 0.0f64;
    for (i, v) in x.iter().enumerate() {
        let y = band.iter_mut().fold(*v, |a, b| b.tick(a));
        acc += (y * y) as f64;
        if i % hop == hop - 1 {
            env.push(10.0 * (acc / hop as f64).max(1e-20).log10() as f32);
            acc = 0.0;
        }
    }
    if env.len() < 10 {
        return 0.0;
    }
    // ~10 Hz smoothing: a centred 5-frame (50 ms) triangle, twice
    let smooth = |e: &[f32]| -> Vec<f32> {
        (0..e.len())
            .map(|i| {
                let (mut s, mut w) = (0.0, 0.0);
                for d in -2i32..=2 {
                    let j = i as i32 + d;
                    if j >= 0 && (j as usize) < e.len() {
                        let k = (3 - d.abs()) as f32;
                        s += e[j as usize] * k;
                        w += k;
                    }
                }
                s / w
            })
            .collect()
    };
    let env = smooth(&smooth(&env));
    let mut sorted = env.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let top = sorted[(sorted.len() * 98 / 100).min(sorted.len() - 1)];
    // speech time: frames within 30 dB of the loudest; pauses > 250 ms dropped
    let active: Vec<bool> = env.iter().map(|e| *e > top - 30.0).collect();
    let mut speech = 0usize;
    let mut gap = 0usize;
    for a in &active {
        if *a {
            speech += 1 + if gap <= 25 { gap } else { 0 };
            gap = 0;
        } else {
            gap += 1;
        }
    }
    // syllable nuclei
    let mut peaks = 0usize;
    let mut last = -100i64;
    for i in 1..env.len() - 1 {
        let v = env[i];
        if !(v > env[i - 1] && v >= env[i + 1] && v > top - 25.0) || (i as i64 - last) < 10 {
            continue;
        }
        let lo = i.saturating_sub(15);
        let hi = (i + 15).min(env.len() - 1);
        let left = env[lo..i].iter().cloned().fold(f32::MAX, f32::min);
        let right = env[i + 1..=hi].iter().cloned().fold(f32::MAX, f32::min);
        if v - left.max(right) >= 3.0 {
            peaks += 1;
            last = i as i64;
        }
    }
    if speech == 0 {
        0.0
    } else {
        peaks as f32 / (speech as f32 * 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Syllables: 120 Hz pulse vowels of `syl_ms` each, `rate_hz` of them a
    /// second, with a 600 ms pause every eight.
    fn syllables(rate: f32, per_s: f32, secs: f32, noise: f32) -> Vec<f32> {
        let n = (secs * rate) as usize;
        let mut rng = super::super::Rng::new(3);
        let period = rate / per_s;
        let (mut ph, mut out) = (0.0f32, Vec::with_capacity(n));
        let (mut y1, mut y2) = (0.0f32, 0.0f32);
        let (rr, c) = {
            let r = (-std::f32::consts::PI * 120.0 / rate).exp();
            (r, 2.0 * r * (2.0 * std::f32::consts::PI * 700.0 / rate).cos())
        };
        let mut t = 0.0f32;
        let mut count = 0;
        for _ in 0..n {
            t += 1.0;
            if t >= period {
                t -= period;
                count += 1;
            }
            let in_pause = count % 8 == 7;
            let pos = t / period;
            // each syllable: a raised-cosine swell over 60 % of its slot
            let amp = if in_pause || pos > 0.6 { 0.0 } else { (std::f32::consts::PI * pos / 0.6).sin().powi(2) };
            ph += 120.0 / rate;
            let pulse = if ph >= 1.0 {
                ph -= 1.0;
                1.0
            } else {
                0.0
            };
            let src = pulse + noise * rng.bipolar();
            let y = src + c * y1 - rr * rr * y2;
            y2 = y1;
            y1 = y;
            out.push(0.05 * amp * y);
        }
        out
    }

    #[test]
    fn syllable_rate_follows_the_talker() {
        for per_s in [3.0f32, 5.0, 7.0] {
            let x = syllables(48000.0, per_s, 12.0, 0.0);
            let got = syllable_rate(&x, 48000.0);
            println!("{per_s} syl/s -> {got:.2}");
            assert!((got / per_s - 1.0).abs() < 0.15, "{per_s} syl/s measured as {got:.2}");
        }
    }

    #[test]
    fn style_tells_level_and_breathiness_apart() {
        let clean = syllables(48000.0, 5.0, 8.0, 0.0);
        let breathy = syllables(48000.0, 5.0, 8.0, 0.03);
        let quiet: Vec<f32> = clean.iter().map(|v| v * 0.1).collect();
        let (a, b, q) = (measure_style(&clean, 48000.0).unwrap(), measure_style(&breathy, 48000.0).unwrap(), measure_style(&quiet, 48000.0).unwrap());
        println!("clean {a:?}\nbreathy {b:?}\nquiet {q:?}");
        assert!((a.level_db - q.level_db - 20.0).abs() < 0.5, "level follows the gain");
        assert!(b.hnr_db < a.hnr_db - 3.0, "breath lowers the HNR");
        assert!(measure_style(&vec![0.0; 48000], 48000.0).is_none(), "silence has no style");
    }
}
