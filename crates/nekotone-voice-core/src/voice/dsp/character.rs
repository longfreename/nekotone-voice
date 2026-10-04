//! Character effects: saturation (anti-aliased tanh), bitcrusher, octaver
//! (analog-style sub-octave divider), growl (sub-harmonic amplitude
//! modulation with rasp) and plain gain.

use super::filters::{Biquad, Kind};
use super::{coef, db_to_lin, Block, Ctx, Drift, Envelope, Rng, Smoothed};
use crate::voice::params::params;

params! {
    /// Saturation / distortion.
    pub struct SaturationParams {
        /// Drive into the waveshaper.
        drive_db: [0.0, 48.0, 12.0, "dB"],
        /// Asymmetry (even harmonics, "tube" warmth).
        asym: [0.0, 0.9, 0.1, ""],
        /// Tone: low-pass after the shaper.
        tone_hz: [500.0, 20000.0, 8000.0, "Hz"],
        /// Wet level.
        mix: [0.0, 1.0, 1.0, ""],
        /// Output gain.
        gain_db: [-36.0, 24.0, 0.0, "dB"],
    }
}

/// ln(cosh(x)) without overflow: the antiderivative of tanh.
#[inline]
fn log_cosh(x: f32) -> f32 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p() - std::f32::consts::LN_2
}

pub struct Saturation {
    p: SaturationParams,
    rate: f32,
    x1: f32,
    f1: f32,
    tone: Biquad,
    drive: f32,
    bias: f32,
    comp: Smoothed,
}

impl Saturation {
    pub fn new(p: SaturationParams, rate: f32) -> Self {
        let mut s = Saturation {
            p,
            rate,
            x1: 0.0,
            f1: log_cosh(0.0),
            tone: Biquad::pass(),
            drive: 1.0,
            bias: 0.0,
            comp: Smoothed::new(1.0, 0.02, rate),
        };
        s.update();
        s.comp.cur = s.comp.target;
        s.x1 = s.bias;
        s.f1 = log_cosh(s.bias);
        s
    }
    fn update(&mut self) {
        self.drive = db_to_lin(self.p.drive_db);
        self.bias = self.p.asym * 0.5;
        self.tone.set(Kind::LowPass, self.p.tone_hz, 0.707, 0.0, self.rate);
        self.comp.set(db_to_lin(self.p.gain_db) / self.drive.sqrt());
    }
    /// First-order antiderivative anti-aliasing (ADAA) tanh.
    #[inline]
    fn shape(&mut self, x: f32) -> f32 {
        let f = log_cosh(x);
        let d = x - self.x1;
        let y = if d.abs() > 1e-4 { (f - self.f1) / d } else { (0.5 * (x + self.x1)).tanh() };
        self.x1 = x;
        self.f1 = f;
        y
    }
}

impl Block for Saturation {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let b = self.bias;
        let tb = b.tanh();
        for x in buf.iter_mut() {
            let y = self.shape(*x * self.drive + b) - tb;
            let y = self.tone.tick(y) * self.comp.next();
            *x = *x * (1.0 - self.p.mix) + y * self.p.mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

params! {
    /// Bitcrusher: fewer bits and a lower sample rate.
    pub struct BitcrushParams {
        /// Bit depth.
        bits: [2.0, 16.0, 8.0, ""],
        /// Held sample rate.
        rate_hz: [500.0, 48000.0, 8000.0, "Hz"],
        /// Wet level.
        mix: [0.0, 1.0, 1.0, ""],
    }
}

pub struct Bitcrush {
    p: BitcrushParams,
    rate: f32,
    phase: f32,
    held: f32,
}

impl Bitcrush {
    pub fn new(p: BitcrushParams, rate: f32) -> Self {
        Bitcrush { p, rate, phase: 1.0, held: 0.0 }
    }
}

impl Block for Bitcrush {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let step = 2.0 / 2f32.powf(self.p.bits.round());
        let inc = (self.p.rate_hz / self.rate).min(1.0);
        for x in buf.iter_mut() {
            self.phase += inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                // mid-tread quantiser: zero stays zero
                self.held = (*x / step).round() * step;
            }
            *x = *x * (1.0 - self.p.mix) + self.held * self.p.mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

params! {
    /// Octaver: an analog-style divider adds a tone one octave below (and a rectified octave above).
    pub struct OctaverParams {
        /// Level of the sub-octave.
        sub: [0.0, 2.0, 0.5, ""],
        /// Level of the octave up.
        up: [0.0, 2.0, 0.0, ""],
        /// Tone of the sub (low-pass).
        tone_hz: [100.0, 3000.0, 600.0, "Hz"],
        /// Level of the input.
        dry: [0.0, 1.0, 1.0, ""],
    }
}

pub struct Octaver {
    p: OctaverParams,
    rate: f32,
    track: [Biquad; 2],
    tone: [Biquad; 2],
    uphp: Biquad,
    env: Envelope,
    flip: f32,
    last: f32,
    hyst: f32,
}

impl Octaver {
    pub fn new(p: OctaverParams, rate: f32) -> Self {
        Octaver {
            p,
            rate,
            track: [Biquad::new(Kind::LowPass, 500.0, 0.707, 0.0, rate); 2],
            tone: [Biquad::new(Kind::LowPass, p.tone_hz, 0.707, 0.0, rate); 2],
            uphp: Biquad::new(Kind::HighPass, 120.0, 0.707, 0.0, rate),
            env: Envelope::new(0.003, 0.05, rate),
            flip: 1.0,
            last: 0.0,
            hyst: 0.0,
        }
    }
}

impl Block for Octaver {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            let t = { let s0 = self.track[0].tick(*x); self.track[1].tick(s0) };
            let e = self.env.tick(t);
            self.hyst = e * 0.1;
            // flip on each rising zero crossing (with hysteresis) → square at f/2
            if self.last <= -self.hyst && t > self.hyst {
                self.flip = -self.flip;
            }
            if t.abs() > self.hyst {
                self.last = t;
            }
            let sub = { let s0 = self.tone[0].tick(self.flip * e * 1.4); self.tone[1].tick(s0) };
            let up = self.uphp.tick(x.abs());
            *x = *x * self.p.dry + sub * self.p.sub + up * self.p.up;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            for b in self.tone.iter_mut() {
                b.set(Kind::LowPass, self.p.tone_hz, 0.707, 0.0, self.rate);
            }
        }
    }
}

params! {
    /// Growl: sub-harmonic amplitude modulation plus rasp, for monsters.
    pub struct GrowlParams {
        /// Depth of the sub-harmonic (period-doubling) modulation.
        depth: [0.0, 1.0, 0.5, ""],
        /// Irregular rough modulation depth.
        roughness: [0.0, 1.0, 0.4, ""],
        /// Speed of the rough modulation.
        rough_hz: [5.0, 120.0, 45.0, "Hz"],
        /// Breathy rasp (noise riding the voice around 1.5–4 kHz).
        rasp: [0.0, 1.0, 0.2, ""],
    }
}

pub struct Growl {
    p: GrowlParams,
    rate: f32,
    track: [Biquad; 2],
    env: Envelope,
    flip: f32,
    sm: f32,
    last: f32,
    drift: Drift,
    rng: Rng,
    rasp_bp: [Biquad; 2],
}

impl Growl {
    pub fn new(p: GrowlParams, rate: f32) -> Self {
        Growl {
            p,
            rate,
            track: [Biquad::new(Kind::LowPass, 450.0, 0.707, 0.0, rate); 2],
            env: Envelope::new(0.002, 0.06, rate),
            flip: 1.0,
            sm: 0.0,
            last: 0.0,
            drift: Drift::new(77),
            rng: Rng::new(0xdead_beef),
            rasp_bp: [
                Biquad::new(Kind::HighPass, 1500.0, 0.707, 0.0, rate),
                Biquad::new(Kind::LowPass, 4000.0, 0.707, 0.0, rate),
            ],
        }
    }
}

impl Block for Growl {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let sm_a = coef(0.0008, self.rate);
        let dr_a = coef(1.0 / (2.0 * std::f32::consts::PI * self.p.rough_hz), self.rate);
        for x in buf.iter_mut() {
            let t = { let s0 = self.track[0].tick(*x); self.track[1].tick(s0) };
            let e = self.env.tick(*x);
            let h = e * 0.05;
            if self.last <= -h && t > h {
                self.flip = -self.flip;
            }
            if t.abs() > h {
                self.last = t;
            }
            // smoothed square at f0/2 in [0, 1]
            self.sm += (0.5 * (1.0 + self.flip) - self.sm) * sm_a;
            let rough = 0.5 * (1.0 + self.drift.tick(dr_a));
            let m = 1.0 - self.p.depth * 0.7 * self.sm - self.p.roughness * 0.5 * rough;
            let rasp = { let s0 = self.rasp_bp[0].tick(self.rng.bipolar()); self.rasp_bp[1].tick(s0) } * e * self.p.rasp * 1.5;
            *x = *x * m.max(0.0) + rasp;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

params! {
    /// Plain gain.
    pub struct GainParams {
        /// Gain.
        db: [-60.0, 36.0, 0.0, "dB"],
    }
}

pub struct Gain {
    p: GainParams,
    g: Smoothed,
}

impl Gain {
    pub fn new(p: GainParams, rate: f32) -> Self {
        Gain { p, g: Smoothed::new(db_to_lin(p.db), 0.02, rate) }
    }
}

impl Block for Gain {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            *x *= self.g.next();
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.g.set(db_to_lin(self.p.db));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn tone(hz: f32, n: usize, rate: f32) -> Vec<f32> {
        (0..n).map(|i| 0.4 * (2.0 * PI * hz * i as f32 / rate).sin()).collect()
    }
    fn dft(b: &[f32], hz: f32, rate: f32) -> f32 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, v) in b.iter().enumerate() {
            let w = 2.0 * PI * hz * i as f32 / rate;
            re += v * w.cos();
            im += v * w.sin();
        }
        (re * re + im * im).sqrt() / b.len() as f32
    }

    #[test]
    fn saturation_adds_harmonics_and_keeps_zero() {
        let rate = 48000.0;
        let mut s = Saturation::new(SaturationParams { drive_db: 24.0, asym: 0.0, tone_hz: 20000.0, ..Default::default() }, rate);
        let mut z = vec![0.0f32; 1000];
        s.process(&mut z, &mut Ctx::default());
        assert!(z.iter().all(|v| v.abs() < 1e-7));
        let mut b = tone(200.0, 48000, rate);
        s.process(&mut b, &mut Ctx::default());
        assert!(dft(&b, 600.0, rate) > 0.05 * dft(&b, 200.0, rate));
        assert!(b.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bitcrusher_quantises() {
        let mut c = Bitcrush::new(BitcrushParams { bits: 3.0, rate_hz: 48000.0, mix: 1.0 }, 48000.0);
        let mut b = tone(100.0, 4800, 48000.0);
        c.process(&mut b, &mut Ctx::default());
        let step = 2.0 / 8.0;
        assert!(b.iter().all(|v| ((v / step) - (v / step).round()).abs() < 1e-4));
    }

    #[test]
    fn octaver_makes_a_sub_octave() {
        let rate = 48000.0;
        let mut o = Octaver::new(OctaverParams { sub: 1.0, dry: 0.0, ..Default::default() }, rate);
        let mut b = tone(220.0, 48000, rate);
        o.process(&mut b, &mut Ctx::default());
        let tail = &b[24000..];
        assert!(dft(tail, 110.0, rate) > 3.0 * dft(tail, 220.0, rate), "{} vs {}", dft(tail, 110.0, rate), dft(tail, 220.0, rate));
    }

    #[test]
    fn growl_adds_subharmonic_sidebands() {
        let rate = 48000.0;
        let mut g = Growl::new(GrowlParams { depth: 1.0, roughness: 0.0, rasp: 0.0, ..Default::default() }, rate);
        let mut b = tone(200.0, 48000, rate);
        let before = dft(&b[24000..], 100.0, rate);
        g.process(&mut b, &mut Ctx::default());
        let after = dft(&b[24000..], 100.0, rate) + dft(&b[24000..], 300.0, rate);
        assert!(after > 0.02 && after > 10.0 * before, "{after}");
    }
}
