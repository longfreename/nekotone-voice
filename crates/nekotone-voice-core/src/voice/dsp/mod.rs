//! DSP building blocks for the voice changer. Everything here is mono,
//! allocation-free after construction and safe to run on the audio thread.

pub mod character;
pub mod delay;
pub mod dynamics;
pub mod filters;
pub mod lpc;
pub mod modulation;
pub mod accent;
pub mod pitch;
pub mod profile;
pub mod reverb;
pub mod route;
pub mod shifter;
pub mod style;
pub mod tamer;
pub mod timbre;

use std::f32::consts::PI;

/// Per-call context shared by the blocks of one chain.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ctx {
    /// Sample rate in Hz.
    pub rate: f32,
    /// Latest input f0 estimate from a pitch stage in Hz (0 = unvoiced/unknown).
    pub pitch_hz: f32,
    /// Latest output f0 of a pitch stage in Hz (0 = unvoiced/unknown).
    pub out_pitch_hz: f32,
    /// Who is speaking (measured by the processor before the chain); voice
    /// stages with a target move this speaker to it.
    pub profile: profile::Profile,
}

/// A processing stage. `process` works in place on a block of mono samples.
pub trait Block: Send {
    fn process(&mut self, buf: &mut [f32], ctx: &mut Ctx);
    /// Change parameter `index` (the index in the block's `PARAMS` table).
    fn set_param(&mut self, index: usize, value: f32);
    /// Constant delay this block adds, in samples.
    fn latency(&self) -> usize {
        0
    }
    /// Routing blocks (see [`route`]) name the layer bus they use; the chain
    /// then calls [`Block::route`] instead of [`Block::process`].
    fn bus(&self) -> Option<usize> {
        None
    }
    /// Work on the chain's signal and one layer bus (routing blocks only).
    fn route(&mut self, _work: &mut [f32], _bus: &mut [f32]) {}
}

#[inline]
pub fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[inline]
pub fn lin_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-12).log10()
}

/// One-pole smoothing coefficient for a time constant in seconds.
#[inline]
pub fn coef(tau_secs: f32, rate: f32) -> f32 {
    if tau_secs <= 0.0 {
        1.0
    } else {
        1.0 - (-1.0 / (tau_secs * rate)).exp()
    }
}

/// A value that glides towards its target (one-pole), for click-free gain changes.
#[derive(Debug, Clone, Copy)]
pub struct Smoothed {
    pub cur: f32,
    pub target: f32,
    a: f32,
}

impl Smoothed {
    pub fn new(v: f32, tau_secs: f32, rate: f32) -> Self {
        Smoothed { cur: v, target: v, a: coef(tau_secs, rate) }
    }
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> f32 {
        self.cur += (self.target - self.cur) * self.a;
        self.cur
    }
    pub fn set(&mut self, v: f32) {
        self.target = v;
    }
}

/// Small fast PRNG (xorshift32); deterministic so renders are repeatable.
#[derive(Debug, Clone, Copy)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(seed.max(1))
    }
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in [-1, 1).
    #[inline]
    pub fn bipolar(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}

/// Sine LFO.
#[derive(Debug, Clone, Copy)]
pub struct Lfo {
    phase: f32,
}

impl Lfo {
    pub fn new(phase: f32) -> Self {
        Lfo { phase }
    }
    /// Advance by one sample at `hz` and return sin in [-1, 1].
    #[inline]
    pub fn tick(&mut self, hz: f32, rate: f32) -> f32 {
        let v = (self.phase * 2.0 * PI).sin();
        self.phase += hz / rate;
        if self.phase >= 1.0 {
            self.phase -= self.phase.floor();
        }
        v
    }
    /// Advance by `n` samples without producing output.
    pub fn skip(&mut self, hz: f32, rate: f32, n: usize) {
        self.phase += hz / rate * n as f32;
        self.phase -= self.phase.floor();
    }
    pub fn value(&self) -> f32 {
        (self.phase * 2.0 * PI).sin()
    }
}

/// Smoothed random modulation (low-passed noise), roughly in [-1, 1].
#[derive(Debug, Clone, Copy)]
pub struct Drift {
    rng: Rng,
    s1: f32,
    s2: f32,
}

impl Drift {
    pub fn new(seed: u32) -> Self {
        Drift { rng: Rng::new(seed), s1: 0.0, s2: 0.0 }
    }
    /// One step; `a` is the one-pole coefficient (bandwidth), called once per step.
    #[inline]
    pub fn tick(&mut self, a: f32) -> f32 {
        let n = self.rng.bipolar();
        self.s1 += (n - self.s1) * a;
        self.s2 += (self.s1 - self.s2) * a;
        // two poles shrink the variance; rescale to ~unit peak
        (self.s2 * (1.8 / a.sqrt().max(0.05))).clamp(-1.0, 1.0)
    }
}

/// Circular delay line with fractional reads.
#[derive(Debug, Clone)]
pub struct DelayLine {
    buf: Vec<f32>,
    mask: usize,
    w: usize,
}

impl DelayLine {
    /// A line that can delay up to at least `max` samples.
    pub fn new(max: usize) -> Self {
        let n = (max + 4).next_power_of_two();
        DelayLine { buf: vec![0.0; n], mask: n - 1, w: 0 }
    }
    pub fn max_delay(&self) -> usize {
        self.buf.len() - 4
    }
    #[inline]
    pub fn push(&mut self, x: f32) {
        self.w = (self.w + 1) & self.mask;
        self.buf[self.w] = x;
    }
    /// Sample written `d` pushes ago (0 = the last one pushed).
    #[inline]
    pub fn tap(&self, d: usize) -> f32 {
        self.buf[(self.w.wrapping_sub(d)) & self.mask]
    }
    /// Linear-interpolated read `d` samples back (d >= 0).
    #[inline]
    pub fn read(&self, d: f32) -> f32 {
        let d = d.clamp(0.0, (self.buf.len() - 3) as f32);
        let i = d as usize;
        let f = d - i as f32;
        let a = self.tap(i);
        let b = self.tap(i + 1);
        a + (b - a) * f
    }
    /// Cubic (Catmull-Rom) read, for audible modulated delays.
    #[inline]
    pub fn read_cubic(&self, d: f32) -> f32 {
        let d = d.clamp(1.0, (self.buf.len() - 4) as f32);
        let i = d as usize;
        let t = d - i as f32;
        let y0 = self.tap(i - 1);
        let y1 = self.tap(i);
        let y2 = self.tap(i + 1);
        let y3 = self.tap(i + 2);
        hermite(y0, y1, y2, y3, t)
    }
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
    }
}

/// 4-point Catmull-Rom interpolation between y1 and y2 at t in [0,1).
#[inline]
pub fn hermite(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * t + c2) * t + c1) * t + c0
}

/// Envelope follower (peak, separate attack and release).
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    pub value: f32,
    att: f32,
    rel: f32,
}

impl Envelope {
    pub fn new(attack_s: f32, release_s: f32, rate: f32) -> Self {
        Envelope { value: 0.0, att: coef(attack_s, rate), rel: coef(release_s, rate) }
    }
    pub fn set_times(&mut self, attack_s: f32, release_s: f32, rate: f32) {
        self.att = coef(attack_s, rate);
        self.rel = coef(release_s, rate);
    }
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let a = x.abs();
        let c = if a > self.value { self.att } else { self.rel };
        self.value += (a - self.value) * c;
        self.value
    }
}

/// Turn on flush-to-zero / denormals-are-zero for the current thread (x86),
/// so decaying reverb and filter tails never hit slow denormal arithmetic.
#[inline]
pub fn denormals_off() {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[allow(deprecated)]
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{_mm_getcsr, _mm_setcsr};
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        _mm_setcsr(_mm_getcsr() | 0x8040);
    }
}

/// A fixed delay (used to pad chains to a constant latency).
pub struct FixedDelay {
    line: DelayLine,
    d: usize,
}

impl FixedDelay {
    pub fn new(d: usize) -> Self {
        FixedDelay { line: DelayLine::new(d + 1), d }
    }
}

impl Block for FixedDelay {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        if self.d == 0 {
            return;
        }
        for x in buf.iter_mut() {
            self.line.push(*x);
            *x = self.line.tap(self.d);
        }
    }
    fn set_param(&mut self, _index: usize, _value: f32) {}
    fn latency(&self) -> usize {
        self.d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_line_reads() {
        let mut d = DelayLine::new(16);
        for i in 0..10 {
            d.push(i as f32);
        }
        assert_eq!(d.tap(0), 9.0);
        assert_eq!(d.tap(3), 6.0);
        assert!((d.read(2.5) - 6.5).abs() < 1e-6);
        assert!((d.read_cubic(2.5) - 6.5).abs() < 1e-5);
        let mut f = FixedDelay::new(3);
        let mut b = [1.0, 2.0, 3.0, 4.0, 5.0];
        f.process(&mut b, &mut Ctx::default());
        assert_eq!(b, [0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn rng_is_uniform_and_bounded() {
        let mut r = Rng::new(7);
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for _ in 0..100_000 {
            let v = r.bipolar();
            assert!((-1.0..1.0).contains(&v));
            sum += v as f64;
            sq += (v * v) as f64;
        }
        assert!((sum / 1e5).abs() < 0.01);
        assert!(((sq / 1e5) - 1.0 / 3.0).abs() < 0.01);
    }
}
