//! Biquad filters (RBJ cookbook, transposed direct form II) and the filter
//! blocks: high-pass, low-pass, band-pass, a 4-band EQ and the "radio /
//! telephone / megaphone" band-limited device model.

use super::{coef, db_to_lin, Block, Ctx, Envelope, Rng};
use crate::voice::params::params;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    LowPass,
    HighPass,
    BandPass,
    Peak,
    LowShelf,
    HighShelf,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn new(kind: Kind, hz: f32, q: f32, gain_db: f32, rate: f32) -> Self {
        let mut b = Biquad::default();
        b.set(kind, hz, q, gain_db, rate);
        b
    }

    /// Identity filter.
    pub fn pass() -> Self {
        Biquad { b0: 1.0, ..Default::default() }
    }

    pub fn set(&mut self, kind: Kind, hz: f32, q: f32, gain_db: f32, rate: f32) {
        let hz = hz.clamp(5.0, rate * 0.49);
        let q = q.max(0.05);
        let w0 = 2.0 * PI * hz / rate;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * q);
        let a = 10f32.powf(gain_db / 40.0);
        let (b0, b1, b2, a0, a1, a2) = match kind {
            Kind::LowPass => ((1.0 - cs) / 2.0, 1.0 - cs, (1.0 - cs) / 2.0, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            Kind::HighPass => ((1.0 + cs) / 2.0, -(1.0 + cs), (1.0 + cs) / 2.0, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            Kind::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cs, 1.0 - alpha),
            Kind::Peak => (1.0 + alpha * a, -2.0 * cs, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cs, 1.0 - alpha / a),
            Kind::LowShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cs + s),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cs),
                    a * ((a + 1.0) - (a - 1.0) * cs - s),
                    (a + 1.0) + (a - 1.0) * cs + s,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cs),
                    (a + 1.0) + (a - 1.0) * cs - s,
                )
            }
            Kind::HighShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cs + s),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cs),
                    a * ((a + 1.0) + (a - 1.0) * cs - s),
                    (a + 1.0) - (a - 1.0) * cs + s,
                    2.0 * ((a - 1.0) - (a + 1.0) * cs),
                    (a + 1.0) - (a - 1.0) * cs - s,
                )
            }
        };
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Magnitude response at `hz` (for tests and UI curves).
    pub fn magnitude(&self, hz: f32, rate: f32) -> f32 {
        let w = 2.0 * PI * hz / rate;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }
}

/// Up to four cascaded identical biquads (12..48 dB/octave).
#[derive(Debug, Clone, Copy)]
pub struct Cascade {
    s: [Biquad; 4],
    n: usize,
}

impl Cascade {
    pub fn new(kind: Kind, hz: f32, q: f32, stages: usize, rate: f32) -> Self {
        let b = Biquad::new(kind, hz, q, 0.0, rate);
        Cascade { s: [b; 4], n: stages.clamp(1, 4) }
    }
    pub fn set(&mut self, kind: Kind, hz: f32, q: f32, stages: usize, rate: f32) {
        self.n = stages.clamp(1, 4);
        for b in self.s.iter_mut() {
            b.set(kind, hz, q, 0.0, rate);
        }
    }
    #[inline]
    pub fn tick(&mut self, mut x: f32) -> f32 {
        for b in self.s[..self.n].iter_mut() {
            x = b.tick(x);
        }
        x
    }
}

// ───────────────────────────── blocks ─────────────────────────────

params! {
    /// High-pass or low-pass filter.
    pub struct PassParams {
        /// Corner frequency.
        hz: [20.0, 20000.0, 1000.0, "Hz"],
        /// Resonance (0.707 = flat Butterworth).
        q: [0.3, 8.0, 0.707, ""],
        /// Number of 12 dB/octave stages.
        stages: [1.0, 4.0, 1.0, ""],
    }
}

pub struct Pass {
    p: PassParams,
    kind: Kind,
    f: Cascade,
    rate: f32,
}

impl Pass {
    pub fn new(kind: Kind, p: PassParams, rate: f32) -> Self {
        Pass { p, kind, f: Cascade::new(kind, p.hz, p.q, p.stages.round() as usize, rate), rate }
    }
}

impl Block for Pass {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            *x = self.f.tick(*x);
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.f.set(self.kind, self.p.hz, self.p.q, self.p.stages.round() as usize, self.rate);
        }
    }
}

params! {
    /// Four-band equaliser: low shelf, two peaks, high shelf.
    pub struct EqParams {
        /// Low shelf corner.
        low_hz: [30.0, 1000.0, 150.0, "Hz"],
        /// Low shelf gain.
        low_db: [-24.0, 24.0, 0.0, "dB"],
        /// First peak centre.
        mid1_hz: [80.0, 12000.0, 500.0, "Hz"],
        /// First peak gain.
        mid1_db: [-24.0, 24.0, 0.0, "dB"],
        /// First peak Q.
        mid1_q: [0.2, 10.0, 1.0, ""],
        /// Second peak centre.
        mid2_hz: [200.0, 16000.0, 3000.0, "Hz"],
        /// Second peak gain.
        mid2_db: [-24.0, 24.0, 0.0, "dB"],
        /// Second peak Q.
        mid2_q: [0.2, 10.0, 1.0, ""],
        /// High shelf corner.
        high_hz: [1000.0, 18000.0, 6000.0, "Hz"],
        /// High shelf gain.
        high_db: [-24.0, 24.0, 0.0, "dB"],
        /// Output gain.
        gain_db: [-24.0, 24.0, 0.0, "dB"],
    }
}

pub struct Eq {
    p: EqParams,
    b: [Biquad; 4],
    g: f32,
    rate: f32,
}

impl Eq {
    pub fn new(p: EqParams, rate: f32) -> Self {
        let mut e = Eq { p, b: [Biquad::pass(); 4], g: 1.0, rate };
        e.update();
        e
    }
    fn update(&mut self) {
        let p = self.p;
        let r = self.rate;
        self.b[0].set(Kind::LowShelf, p.low_hz, 0.707, p.low_db, r);
        self.b[1].set(Kind::Peak, p.mid1_hz, p.mid1_q, p.mid1_db, r);
        self.b[2].set(Kind::Peak, p.mid2_hz, p.mid2_q, p.mid2_db, r);
        self.b[3].set(Kind::HighShelf, p.high_hz, 0.707, p.high_db, r);
        self.g = db_to_lin(p.gain_db);
    }
}

impl Block for Eq {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let active = [self.p.low_db != 0.0, self.p.mid1_db != 0.0, self.p.mid2_db != 0.0, self.p.high_db != 0.0];
        for x in buf.iter_mut() {
            let mut y = *x;
            for (b, on) in self.b.iter_mut().zip(active) {
                if on {
                    y = b.tick(y);
                }
            }
            *x = y * self.g;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

params! {
    /// Band-pass with its own resonance, e.g. a formant or wah.
    pub struct BandParams {
        /// Centre frequency.
        hz: [40.0, 16000.0, 1000.0, "Hz"],
        /// Bandwidth Q.
        q: [0.2, 20.0, 1.0, ""],
        /// Wet level (the filtered signal).
        mix: [0.0, 1.0, 1.0, ""],
    }
}

pub struct Band {
    p: BandParams,
    f: [Biquad; 2],
    rate: f32,
}

impl Band {
    pub fn new(p: BandParams, rate: f32) -> Self {
        let b = Biquad::new(Kind::BandPass, p.hz, p.q, 0.0, rate);
        Band { p, f: [b; 2], rate }
    }
}

impl Block for Band {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let m = self.p.mix;
        for x in buf.iter_mut() {
            let y = { let s0 = self.f[0].tick(*x); self.f[1].tick(s0) };
            *x = *x * (1.0 - m) + y * m;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            for f in self.f.iter_mut() {
                f.set(Kind::BandPass, self.p.hz, self.p.q, 0.0, self.rate);
            }
        }
    }
}

params! {
    /// A band-limited device: radio, telephone, megaphone, walkie-talkie.
    pub struct RadioParams {
        /// Low cut (24 dB/octave).
        low_hz: [50.0, 2000.0, 400.0, "Hz"],
        /// High cut (24 dB/octave).
        high_hz: [1000.0, 12000.0, 3800.0, "Hz"],
        /// Resonant "honk" of the small speaker.
        honk_hz: [300.0, 6000.0, 1800.0, "Hz"],
        /// Honk gain.
        honk_db: [-6.0, 18.0, 3.0, "dB"],
        /// Overdrive into the speaker.
        drive_db: [0.0, 36.0, 6.0, "dB"],
        /// Static hiss level relative to the voice (follows the voice, silent in silence).
        noise_db: [-80.0, -10.0, -40.0, "dB"],
        /// Output gain.
        gain_db: [-24.0, 24.0, 0.0, "dB"],
    }
}

pub struct Radio {
    p: RadioParams,
    hp: Cascade,
    lp: Cascade,
    honk: Biquad,
    nhp: Biquad,
    env: Envelope,
    rng: Rng,
    drive: f32,
    comp: f32,
    noise: f32,
    gain: f32,
    rate: f32,
}

impl Radio {
    pub fn new(p: RadioParams, rate: f32) -> Self {
        let mut r = Radio {
            p,
            hp: Cascade::new(Kind::HighPass, p.low_hz, 0.707, 2, rate),
            lp: Cascade::new(Kind::LowPass, p.high_hz, 0.707, 2, rate),
            honk: Biquad::pass(),
            nhp: Biquad::new(Kind::HighPass, 1500.0, 0.707, 0.0, rate),
            env: Envelope::new(0.002, 0.08, rate),
            rng: Rng::new(0x5eed_4ad1),
            drive: 1.0,
            comp: 1.0,
            noise: 0.0,
            gain: 1.0,
            rate,
        };
        r.update();
        r
    }
    fn update(&mut self) {
        let p = self.p;
        self.hp.set(Kind::HighPass, p.low_hz, 0.707, 2, self.rate);
        self.lp.set(Kind::LowPass, p.high_hz.max(p.low_hz * 1.5), 0.707, 2, self.rate);
        self.honk.set(Kind::Peak, p.honk_hz, 1.4, p.honk_db, self.rate);
        self.drive = db_to_lin(p.drive_db);
        self.comp = 1.0 / self.drive.sqrt();
        self.noise = db_to_lin(p.noise_db) * 3f32.sqrt();
        self.gain = db_to_lin(p.gain_db);
    }
}

impl Block for Radio {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            let e = self.env.tick(*x);
            let mut y = self.hp.tick(*x);
            y = self.honk.tick(y);
            y = (y * self.drive).tanh() * self.comp;
            // hiss rides on the voice envelope, so silence stays silent
            y += self.nhp.tick(self.rng.bipolar()) * self.noise * e;
            y = self.lp.tick(y);
            *x = y * self.gain;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

/// DC blocker / gentle high-pass used by the front end.
#[derive(Debug, Clone, Copy)]
pub struct DcBlock {
    x1: f32,
    y1: f32,
    r: f32,
}

impl DcBlock {
    pub fn new(hz: f32, rate: f32) -> Self {
        DcBlock { x1: 0.0, y1: 0.0, r: 1.0 - coef(1.0 / (2.0 * PI * hz), rate) }
    }
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biquad_responses() {
        let r = 48000.0;
        let lp = Biquad::new(Kind::LowPass, 1000.0, 0.707, 0.0, r);
        assert!((lp.magnitude(100.0, r) - 1.0).abs() < 0.01);
        assert!((lp.magnitude(1000.0, r) - 0.707).abs() < 0.01);
        assert!(lp.magnitude(8000.0, r) < 0.03);
        let hp = Biquad::new(Kind::HighPass, 1000.0, 0.707, 0.0, r);
        assert!(hp.magnitude(100.0, r) < 0.02);
        let pk = Biquad::new(Kind::Peak, 2000.0, 1.0, 6.0, r);
        assert!((pk.magnitude(2000.0, r) - 2.0).abs() < 0.02);
        let hs = Biquad::new(Kind::HighShelf, 4000.0, 0.707, -6.0, r);
        assert!((hs.magnitude(15000.0, r) - 0.5).abs() < 0.03);
        assert!((hs.magnitude(100.0, r) - 1.0).abs() < 0.01);
    }

    #[test]
    fn radio_is_band_limited_and_silent_in_silence() {
        let rate = 48000.0;
        let mut r = Radio::new(RadioParams::default(), rate);
        let mut z = vec![0.0f32; 4800];
        r.process(&mut z, &mut Ctx::default());
        assert!(z.iter().all(|v| *v == 0.0));
        let tone = |hz: f32| {
            let mut r = Radio::new(RadioParams { drive_db: 0.0, noise_db: -80.0, ..Default::default() }, rate);
            let mut b: Vec<f32> = (0..9600).map(|i| 0.1 * (2.0 * PI * hz * i as f32 / rate).sin()).collect();
            r.process(&mut b, &mut Ctx::default());
            (b[4800..].iter().map(|v| v * v).sum::<f32>() / 4800.0).sqrt()
        };
        assert!(tone(1500.0) > 0.03);
        assert!(tone(100.0) < 0.003);
        assert!(tone(10000.0) < 0.003);
    }
}
