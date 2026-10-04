//! Timbre match: slow, bounded corrections of level, body and brightness
//! towards a voice's targets, so a preset sounds the same whether you
//! mumble or project, and whatever microphone or voice feeds it.
//!
//! ```text
//!  x ─┬──────────────────────────────── low shelf (body) ─ high shelf (brightness) ─ gain (level) ─ y
//!  └─ bands: body 100–400 Hz │ mid 400–2500 Hz │ bright 2.5–8 kHz, power while speaking
//!       ─▶ shelves = (target − measured) / k        level (after the shelves) ─▶ gain = target − measured
//! ```
//!
//! Feed-forward: the balances are measured on the input (long power
//! averages while speaking, 24 dB/octave bands) and each shelf is set to
//! (target − measured) / k, where k is how much that shelf moves its band
//! against the mids per dB (computed from its response); the level is
//! measured after the shelves and corrected by a flat gain. Nothing is fed
//! back, so nothing can oscillate or wind up on bursty speech (a closed
//! loop with a seconds-long measurement did). It learns only while there is
//! speech (short-term level above −65 dBFS; the front-end gate has already
//! silenced the room), holds in pauses, and glides so it never clicks.

use super::filters::{Biquad, Kind};
use super::{coef, db_to_lin, Block, Ctx};
use crate::voice::params::params;

params! {
    /// Timbre match: level, body and brightness moved towards targets while you speak.
    pub struct TimbreParams {
        /// How far the corrections may go (0 = off, 1 = the full range).
        amount: [0.0, 1.0, 1.0, ""],
        /// Target speech level (RMS while speaking).
        level_db: [-40.0, -6.0, -20.0, "dB"],
        /// Target body: 100–400 Hz energy relative to 400–2500 Hz.
        body_db: [-24.0, 12.0, 0.0, "dB"],
        /// Target brightness: 2.5–8 kHz energy relative to 400–2500 Hz.
        bright_db: [-40.0, 6.0, -12.0, "dB"],
        /// Largest correction of each kind.
        range_db: [0.0, 18.0, 15.0, "dB"],
        /// How fast the measurements follow the voice (phrase-length: faster
        /// would work like a compressor and flatten your phrasing).
        speed_s: [1.0, 10.0, 3.0, "s"],
        /// How much of the body and brightness correction to apply (0 = level
        /// only: your own tone is kept, as the voices relative to you do).
        tone: [0.0, 1.0, 1.0, ""],
    }
}

/// Short-term level above which there is speech (the front-end gate has already silenced the room).
const ACTIVE_DB: f32 = -65.0;
/// Shelf corners, at the edges of the measured bands.
const BODY_HZ: f32 = 350.0;
const BRIGHT_HZ: f32 = 2200.0;

/// A 24 dB/octave band (two high-pass and two low-pass biquads).
#[derive(Clone, Copy)]
struct Band([Biquad; 4]);

impl Band {
    fn new(lo: f32, hi: f32, rate: f32) -> Band {
        let hp = |q| Biquad::new(Kind::HighPass, lo, q, 0.0, rate);
        let lp = |q| Biquad::new(Kind::LowPass, hi, q, 0.0, rate);
        // Butterworth pairs (Q 0.54 and 1.31)
        Band([hp(0.54), hp(1.31), lp(0.54), lp(1.31)])
    }
    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let mut y = x;
        for b in self.0.iter_mut() {
            y = b.tick(y);
        }
        y
    }
}

pub struct Timbre {
    p: TimbreParams,
    rate: f32,
    // analysis of the shelved signal (what the listener gets, before the level gain)
    body_b: Band,
    mid_b: Band,
    bright_b: Band,
    e_full: f32,
    e_body: f32,
    e_mid: f32,
    e_bright: f32,
    e_level: f32,
    e_a: f32,
    /// Long-term mean power while speaking (linear, like an active speech
    /// level: averaging dB would read several dB under the real loudness).
    p_full: f32,
    p_body: f32,
    p_mid: f32,
    p_bright: f32,
    seen: bool,
    /// Level of the shelved signal while speaking (the flat gain after it is exact).
    p_level: f32,
    /// dB the band-to-mids balance moves per dB of shelf gain (see `coverage`).
    k_body: f32,
    k_bright: f32,
    // correction: shelves driven by slow integrators, level feed-forward
    low: Biquad,
    high: Biquad,
    g_body: f32,
    g_bright: f32,
    g_level: f32,
    applied: (f32, f32),
    glide_a: f32,
    count: usize,
    /// Samples per analysis step (1 ms).
    step: usize,
}

impl Timbre {
    pub fn new(p: TimbreParams, rate: f32) -> Self {
        let top = (rate * 0.45).min(8000.0);
        Timbre {
            p,
            rate,
            body_b: Band::new(100.0, 400.0, rate),
            mid_b: Band::new(400.0, 2500.0, rate),
            bright_b: Band::new(2500.0, top, rate),
            e_full: 0.0,
            e_body: 0.0,
            e_mid: 0.0,
            e_bright: 0.0,
            e_level: 0.0,
            e_a: coef(0.03, rate),
            p_full: 0.0,
            p_body: 0.0,
            p_mid: 0.0,
            p_bright: 0.0,
            seen: false,
            p_level: 0.0,
            k_body: coverage(Kind::LowShelf, BODY_HZ, (100.0, 400.0), rate),
            k_bright: coverage(Kind::HighShelf, BRIGHT_HZ, (2500.0, top), rate),
            low: Biquad::new(Kind::LowShelf, BODY_HZ, 0.707, 0.0, rate),
            high: Biquad::new(Kind::HighShelf, BRIGHT_HZ, 0.707, 0.0, rate),
            g_body: 0.0,
            g_bright: 0.0,
            g_level: 0.0,
            applied: (0.0, 0.0),
            glide_a: coef(0.25, rate),
            count: 0,
            step: ((rate * 0.001).round() as usize).max(1),
        }
    }

    /// Current corrections in dB: (level, body, brightness).
    pub fn corrections(&self) -> (f32, f32, f32) {
        (self.g_level, self.g_body, self.g_bright)
    }

    /// What the block measures of its shelved signal while speaking, in dB:
    /// (level, body re mids, brightness re mids); None before any speech.
    pub fn measured(&self) -> Option<(f32, f32, f32)> {
        let mid = Self::db(self.p_mid);
        self.seen.then(|| (Self::db(self.p_full), Self::db(self.p_body) - mid, Self::db(self.p_bright) - mid))
    }

    fn db(e: f32) -> f32 {
        10.0 * e.max(1e-12).log10()
    }

    /// One analysis step: learn while speaking, move the corrections.
    fn control(&mut self, e_level: f32) {
        let speaking = Self::db(e_level) > ACTIVE_DB && self.e_mid > 1e-12;
        if speaking {
            if !self.seen {
                (self.p_full, self.p_body, self.p_mid, self.p_bright, self.p_level, self.seen) = (self.e_full, self.e_body, self.e_mid, self.e_bright, e_level, true);
            } else {
                let a = (coef(self.p.speed_s, self.rate) * self.step as f32).min(1.0);
                self.p_full += (self.e_full - self.p_full) * a;
                self.p_body += (self.e_body - self.p_body) * a;
                self.p_mid += (self.e_mid - self.p_mid) * a;
                self.p_bright += (self.e_bright - self.p_bright) * a;
                self.p_level += (e_level - self.p_level) * a;
            }
        }
        let Some((_, body, bright)) = self.measured() else { return };
        let range = self.p.range_db * self.p.amount;
        // Feed-forward from the input's balance: a shelf of g dB moves the
        // band's balance by k·g dB (k < 1: a shelf does not cover its whole
        // band), so g = (target − measured) / k. Nothing is fed back, so
        // nothing can oscillate or wind up on bursty speech.
        let tone = self.p.tone.clamp(0.0, 1.0);
        let want_body = ((self.p.body_db - body) / self.k_body).clamp(-range, range) * tone;
        let want_bright = ((self.p.bright_db - bright) / self.k_bright).clamp(-range, range) * tone;
        let g = (self.glide_a * self.step as f32).min(1.0);
        self.g_body += (want_body - self.g_body) * g;
        self.g_bright += (want_bright - self.g_bright) * g;
        // level: of the shelved signal, then a flat gain (exact)
        let want_level = (self.p.level_db - Self::db(self.p_level)).clamp(-range, range);
        self.g_level += (want_level - self.g_level) * g;
        if (self.g_body - self.applied.0).abs() > 0.05 || (self.g_bright - self.applied.1).abs() > 0.05 {
            self.low.set(Kind::LowShelf, BODY_HZ, 0.707, self.g_body, self.rate);
            self.high.set(Kind::HighShelf, BRIGHT_HZ, 0.707, self.g_bright, self.rate);
            self.applied = (self.g_body, self.g_bright);
        }
    }
}

/// How many dB the balance of `band` against the mids (400–2500 Hz) moves
/// per dB of a shelf's gain: the shelf's mean dB response over the band
/// minus over the mids (log-spaced points), for a 6 dB shelf, divided by 6.
fn coverage(kind: Kind, hz: f32, band: (f32, f32), rate: f32) -> f32 {
    let mut b = Biquad::new(kind, hz, 0.707, 6.0, rate);
    b.set(kind, hz, 0.707, 6.0, rate);
    let mean_db = |lo: f32, hi: f32| {
        let n = 24;
        (0..n).map(|i| 20.0 * b.magnitude(lo * (hi / lo).powf((i as f32 + 0.5) / n as f32), rate).log10()).sum::<f32>() / n as f32
    };
    ((mean_db(band.0, band.1) - mean_db(400.0, 2500.0)) / 6.0).abs().clamp(0.3, 1.0)
}

impl Block for Timbre {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        if self.p.amount <= 0.0 {
            return;
        }
        for x in buf.iter_mut() {
            // the balance is measured on the input (feed-forward), the level on the shelved signal
            let (b, m, h) = (self.body_b.tick(*x), self.mid_b.tick(*x), self.bright_b.tick(*x));
            let v = self.high.tick(self.low.tick(*x));
            self.e_level += (v * v - self.e_level) * self.e_a;
            self.e_full += (*x * *x - self.e_full) * self.e_a;
            self.e_body += (b * b - self.e_body) * self.e_a;
            self.e_mid += (m * m - self.e_mid) * self.e_a;
            self.e_bright += (h * h - self.e_bright) * self.e_a;
            self.count += 1;
            if self.count % self.step == 0 {
                self.control(self.e_level);
            }
            *x = v * db_to_lin(self.g_level);
        }
    }

    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

/// Exact speech balance of `x` as the block defines it: power means over
/// the samples where the short-term level says someone is speaking.
pub fn balance(x: &[f32], rate: f32) -> (f32, f32, f32) {
    let (mut b, mut m, mut h) = (Band::new(100.0, 400.0, rate), Band::new(400.0, 2500.0, rate), Band::new(2500.0, 8000.0, rate));
    let a = coef(0.03, rate);
    let (mut e, mut sf, mut sb, mut sm, mut sh, mut na) = (0.0f32, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    for v in x {
        let (yb, ym, yh) = (b.tick(*v), m.tick(*v), h.tick(*v));
        e += (v * v - e) * a;
        if Timbre::db(e) > ACTIVE_DB {
            sf += (v * v) as f64;
            na += 1;
            sb += (yb * yb) as f64;
            sm += (ym * ym) as f64;
            sh += (yh * yh) as f64;
        }
    }
    let db = |p: f64| 10.0 * p.max(1e-30).log10() as f32;
    (db(sf / na.max(1) as f64), db(sb) - db(sm), db(sh) - db(sm))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) use super::balance;

    fn talk(rate: f32, gain: f32, tilt: f32) -> Vec<f32> {
        // pulse-train "voice" with a spectral tilt, in words and pauses
        let mut v = crate::voice::dsp::profile::tests::talker(rate, 140.0, 1.08, 14.0, 4);
        let mut lp = Biquad::new(Kind::HighShelf, 2000.0, 0.707, tilt, rate);
        v.iter_mut().for_each(|s| *s = lp.tick(*s) * gain);
        v
    }

    #[test]
    fn quiet_and_loud_dull_and_bright_voices_land_on_the_targets() {
        let rate = 48000.0;
        let ctx = &mut Ctx { rate, ..Default::default() };
        // targets inside what a pulse-train voice can reach (its crest factor is high)
        let p = TimbreParams { level_db: -42.0, body_db: -3.0, bright_db: -14.0, ..Default::default() };
        let mut before = Vec::new();
        let mut after = Vec::new();
        // 16 dB between the quietest and the loudest, 12 dB of tilt
        for (gain, tilt) in [(0.1, -6.0), (0.6, 0.0), (0.25, 6.0)] {
            let x = talk(rate, gain, tilt);
            let mut y = x.clone();
            let mut t = Timbre::new(p, rate);
            t.process(&mut y, ctx);
            // the settled output (from 6 s)
            let from = (6.0 * rate) as usize;
            before.push(balance(&x[from..], rate));
            after.push(balance(&y[from..], rate));
        }
        let spread = |v: &[(f32, f32, f32)], f: fn(&(f32, f32, f32)) -> f32| {
            let xs: Vec<f32> = v.iter().map(f).collect();
            xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(spread(&before, |m| m.0) > 12.0 && spread(&before, |m| m.2) > 6.0, "inputs {before:?}");
        // the point: whatever came in, the same comes out (level and body
        // within 1.5 dB; brightness, which this talker's vowels swing most,
        // within 2.5 dB of a 10 dB input spread: about what can be heard)
        let pick: [(fn(&(f32, f32, f32)) -> f32, f32); 3] = [(|m| m.0, 1.5), (|m| m.1, 1.5), (|m| m.2, 2.5)];
        for (f, tol) in pick {
            assert!(spread(&after, f) < tol, "outputs differ: {after:?} (inputs {before:?})");
        }
        for m in &after {
            // (the quietest input needs 12 dB; its faintest frames fall under the speech threshold)
            assert!((m.0 - p.level_db).abs() < 2.0, "level {after:?}");
            assert!((m.1 - p.body_db).abs() < 2.0, "body {after:?}");
            // a shelf covers its band only partly; k compensates on average
            assert!((m.2 - p.bright_db).abs() < 4.0, "brightness {after:?}");
        }
    }

    #[test]
    fn tone_zero_levels_the_voice_and_keeps_its_colour() {
        let rate = 48000.0;
        let ctx = &mut Ctx { rate, ..Default::default() };
        // (inputs well above the -65 dB speech threshold of `balance`, so the
        // frames it averages are the same before and after)
        for (gain, tilt, lift) in [(0.4, -6.0, 10.0), (0.9, 6.0, -8.0)] {
            let x = talk(rate, gain, tilt);
            let from = (6.0 * rate) as usize;
            let bi = balance(&x[from..], rate);
            // a target `lift` dB from the input's level, inside the correction range
            let p = TimbreParams { level_db: bi.0 + lift, tone: 0.0, ..Default::default() };
            let mut y = x.clone();
            Timbre::new(p, rate).process(&mut y, ctx);
            let ai = balance(&y[from..], rate);
            assert!((ai.0 - p.level_db).abs() < 1.0, "levelled: {bi:?} -> {ai:?}, want {}", p.level_db);
            assert!((ai.1 - bi.1).abs() < 0.3 && (ai.2 - bi.2).abs() < 0.3, "colour kept: {bi:?} -> {ai:?}");
        }
    }

    #[test]
    fn silence_stays_silent_and_off_is_transparent() {
        let rate = 48000.0;
        let mut t = Timbre::new(TimbreParams::default(), rate);
        let mut z = vec![0.0f32; 48000];
        t.process(&mut z, &mut Ctx { rate, ..Default::default() });
        assert!(z.iter().all(|v| *v == 0.0));
        let mut off = Timbre::new(TimbreParams { amount: 0.0, ..Default::default() }, rate);
        let x = talk(rate, 0.3, 0.0);
        let mut y = x.clone();
        off.process(&mut y, &mut Ctx { rate, ..Default::default() });
        assert_eq!(x, y);
    }
}

#[cfg(test)]
mod calibrate {
    /// Speech balance of real recordings (NEKOTONE_BALANCE_WAVS=a.wav;b.wav), for choosing targets.
    #[test]
    #[ignore]
    fn natural_balance_of_recordings() {
        let list = std::env::var("NEKOTONE_BALANCE_WAVS").expect("set NEKOTONE_BALANCE_WAVS");
        for f in list.split(';') {
            let clip = crate::audio::decode(std::path::Path::new(f)).unwrap();
            let clip = if clip.sample_rate != 48000 { crate::audio::resample(&clip, 48000).unwrap() } else { clip };
            println!("{f}: {:?}", super::tests::balance(&clip.samples, 48000.0));
        }
    }
}
