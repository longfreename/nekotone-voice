//! Modulation effects: chorus/flanger, vibrato, tremolo, the LFO filter
//! "wobble", and the ring modulator.

use super::{db_to_lin, Block, Ctx, DelayLine, Lfo, Smoothed};
use crate::voice::params::params;
use std::f32::consts::PI;

params! {
    /// Chorus (several slowly modulated delayed copies); with one voice, a short delay and feedback it is a flanger.
    pub struct ChorusParams {
        /// Number of delayed voices.
        voices: [1.0, 4.0, 3.0, ""],
        /// Modulation rate.
        rate_hz: [0.02, 10.0, 0.8, "Hz"],
        /// Base delay.
        delay_ms: [0.5, 40.0, 14.0, "ms"],
        /// Modulation depth.
        depth_ms: [0.0, 15.0, 3.0, "ms"],
        /// Feedback (flanger resonance).
        feedback: [-0.95, 0.95, 0.0, ""],
        /// Wet level.
        mix: [0.0, 1.0, 0.5, ""],
    }
}

pub struct Chorus {
    p: ChorusParams,
    rate: f32,
    line: DelayLine,
    lfos: [Lfo; 4],
    fb: f32,
}

impl Chorus {
    pub fn new(p: ChorusParams, rate: f32) -> Self {
        Chorus {
            p,
            rate,
            line: DelayLine::new((rate * 0.06) as usize),
            lfos: [Lfo::new(0.0), Lfo::new(0.25), Lfo::new(0.5), Lfo::new(0.75)],
            fb: 0.0,
        }
    }
}

impl Block for Chorus {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let n = self.p.voices.round().clamp(1.0, 4.0) as usize;
        let base = self.p.delay_ms * 0.001 * self.rate;
        let depth = self.p.depth_ms * 0.001 * self.rate;
        let norm = 1.0 / (n as f32).sqrt();
        for x in buf.iter_mut() {
            self.line.push(*x + self.fb * self.p.feedback);
            let mut wet = 0.0;
            for (k, lfo) in self.lfos[..n].iter_mut().enumerate() {
                // each voice at a slightly different rate so they never lock
                let r = self.p.rate_hz * (1.0 + 0.13 * k as f32);
                let d = base + depth * 0.5 * (1.0 + lfo.tick(r, self.rate));
                let v = self.line.read_cubic(d.max(1.0));
                if k == 0 {
                    self.fb = v;
                }
                wet += v;
            }
            wet *= norm;
            *x = *x * (1.0 - 0.5 * self.p.mix) + wet * self.p.mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

params! {
    /// Vibrato: pitch wobble by a modulated delay (no dry signal).
    pub struct VibratoParams {
        /// Rate.
        rate_hz: [0.1, 15.0, 5.5, "Hz"],
        /// Depth as delay swing.
        depth_ms: [0.0, 8.0, 1.5, "ms"],
    }
}

pub struct Vibrato {
    p: VibratoParams,
    rate: f32,
    line: DelayLine,
    lfo: Lfo,
}

impl Vibrato {
    pub fn new(p: VibratoParams, rate: f32) -> Self {
        Vibrato { p, rate, line: DelayLine::new((rate * 0.02) as usize), lfo: Lfo::new(0.0) }
    }
}

impl Block for Vibrato {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let depth = self.p.depth_ms * 0.001 * self.rate;
        for x in buf.iter_mut() {
            self.line.push(*x);
            let d = 2.0 + depth * 0.5 * (1.0 + self.lfo.tick(self.p.rate_hz, self.rate));
            *x = self.line.read_cubic(d);
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
    fn latency(&self) -> usize {
        (2.0 + self.p.depth_ms * 0.0005 * self.rate) as usize
    }
}

params! {
    /// Tremolo: amplitude LFO.
    pub struct TremoloParams {
        /// Rate.
        rate_hz: [0.1, 30.0, 6.0, "Hz"],
        /// Depth (1 = fully to silence).
        depth: [0.0, 1.0, 0.5, ""],
        /// Shape: 0 = sine, 1 = square-ish.
        shape: [0.0, 1.0, 0.0, ""],
    }
}

pub struct Tremolo {
    p: TremoloParams,
    rate: f32,
    lfo: Lfo,
}

impl Tremolo {
    pub fn new(p: TremoloParams, rate: f32) -> Self {
        Tremolo { p, rate, lfo: Lfo::new(0.0) }
    }
}

impl Block for Tremolo {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let sharp = 1.0 + 9.0 * self.p.shape;
        let norm = sharp.tanh();
        for x in buf.iter_mut() {
            let s = (self.lfo.tick(self.p.rate_hz, self.rate) * sharp).tanh() / norm;
            *x *= 1.0 - self.p.depth * 0.5 * (1.0 - s);
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

params! {
    /// Wobble: a resonant filter swept by an LFO (auto-wah, "wub").
    pub struct WobbleParams {
        /// LFO rate.
        rate_hz: [0.05, 15.0, 3.0, "Hz"],
        /// Lowest cutoff.
        min_hz: [80.0, 8000.0, 400.0, "Hz"],
        /// Highest cutoff.
        max_hz: [200.0, 16000.0, 3000.0, "Hz"],
        /// Resonance.
        q: [0.5, 12.0, 2.5, ""],
        /// 0 = low-pass, 1 = band-pass.
        band: [0.0, 1.0, 0.0, ""],
        /// Wet level.
        mix: [0.0, 1.0, 0.5, ""],
    }
}

/// Topology-preserving state-variable filter (Simper), stable under fast modulation.
#[derive(Default, Clone, Copy)]
pub struct Svf {
    ic1: f32,
    ic2: f32,
}

impl Svf {
    /// Returns (low, band).
    #[inline]
    pub fn tick(&mut self, x: f32, hz: f32, q: f32, rate: f32) -> (f32, f32) {
        let g = (PI * (hz / rate).min(0.49)).tan();
        let k = 1.0 / q;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = x - self.ic2;
        let v1 = a1 * self.ic1 + a2 * v3;
        let v2 = self.ic2 + a2 * self.ic1 + a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        (v2, v1)
    }
}

pub struct Wobble {
    p: WobbleParams,
    rate: f32,
    lfo: Lfo,
    f: Svf,
}

impl Wobble {
    pub fn new(p: WobbleParams, rate: f32) -> Self {
        Wobble { p, rate, lfo: Lfo::new(0.75), f: Svf::default() }
    }
}

impl Block for Wobble {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let lo = self.p.min_hz.ln();
        let hi = self.p.max_hz.max(self.p.min_hz).ln();
        for x in buf.iter_mut() {
            let s = 0.5 * (1.0 + self.lfo.tick(self.p.rate_hz, self.rate));
            let hz = (lo + (hi - lo) * s).exp();
            let (l, b) = self.f.tick(*x, hz, self.p.q, self.rate);
            let wet = l * (1.0 - self.p.band) + b * self.p.band;
            *x = *x * (1.0 - self.p.mix) + wet * self.p.mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        self.p.set(i, v);
    }
}

params! {
    /// Ring modulator: multiplies the voice with a sine carrier (metallic, alien, dalek).
    pub struct RingModParams {
        /// Carrier frequency.
        hz: [1.0, 3000.0, 60.0, "Hz"],
        /// Carrier frequency swing.
        lfo_depth_hz: [0.0, 1000.0, 0.0, "Hz"],
        /// Swing rate.
        lfo_hz: [0.0, 20.0, 0.5, "Hz"],
        /// Wet level.
        mix: [0.0, 1.0, 0.5, ""],
        /// Output gain.
        gain_db: [-24.0, 24.0, 0.0, "dB"],
    }
}

pub struct RingMod {
    p: RingModParams,
    rate: f32,
    phase: f32,
    lfo: Lfo,
    g: Smoothed,
}

impl RingMod {
    pub fn new(p: RingModParams, rate: f32) -> Self {
        RingMod { p, rate, phase: 0.0, lfo: Lfo::new(0.0), g: Smoothed::new(db_to_lin(p.gain_db), 0.02, rate) }
    }
}

impl Block for RingMod {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            let f = (self.p.hz + self.p.lfo_depth_hz * self.lfo.tick(self.p.lfo_hz, self.rate)).max(0.0);
            let c = (2.0 * PI * self.phase).sin();
            self.phase += f / self.rate;
            self.phase -= self.phase.floor();
            let g = self.g.next();
            *x = (*x * (1.0 - self.p.mix) + *x * c * self.p.mix * 1.414) * g;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.g.set(db_to_lin(self.p.gain_db));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_mod_moves_energy_to_sidebands() {
        let rate = 48000.0;
        let mut r = RingMod::new(RingModParams { hz: 200.0, mix: 1.0, ..Default::default() }, rate);
        let mut b: Vec<f32> = (0..48000).map(|i| (2.0 * PI * 1000.0 * i as f32 / rate).sin()).collect();
        r.process(&mut b, &mut Ctx::default());
        let dft = |hz: f32| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, v) in b.iter().enumerate() {
                let w = 2.0 * PI * hz * i as f32 / rate;
                re += v * w.cos();
                im += v * w.sin();
            }
            (re * re + im * im).sqrt() / b.len() as f32
        };
        assert!(dft(1000.0) < 0.01);
        assert!(dft(800.0) > 0.3 && dft(1200.0) > 0.3);
    }

    #[test]
    fn tremolo_depth() {
        let mut t = Tremolo::new(TremoloParams { depth: 1.0, rate_hz: 5.0, shape: 0.0 }, 48000.0);
        let mut b = vec![1.0f32; 48000];
        t.process(&mut b, &mut Ctx::default());
        let mn = b.iter().cloned().fold(f32::MAX, f32::min);
        let mx = b.iter().cloned().fold(f32::MIN, f32::max);
        assert!(mn < 0.01 && mx > 0.99);
    }

    #[test]
    fn chorus_and_wobble_are_finite_and_silent_in_silence() {
        let rate = 48000.0;
        let mut c = Chorus::new(ChorusParams { feedback: 0.9, ..Default::default() }, rate);
        let mut w = Wobble::new(WobbleParams { q: 12.0, ..Default::default() }, rate);
        let mut v = Vibrato::new(VibratoParams::default(), rate);
        let mut z = vec![0.0f32; 4800];
        c.process(&mut z, &mut Ctx::default());
        w.process(&mut z, &mut Ctx::default());
        v.process(&mut z, &mut Ctx::default());
        assert!(z.iter().all(|x| *x == 0.0));
        let mut s: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.05).sin()).collect();
        c.process(&mut s, &mut Ctx::default());
        w.process(&mut s, &mut Ctx::default());
        assert!(s.iter().all(|x| x.is_finite() && x.abs() < 20.0));
    }
}
