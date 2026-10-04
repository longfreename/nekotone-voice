//! Algorithmic reverb: a 16-line feedback delay network.
//!
//! pre-delay → early reflections (8 taps) + 4 series all-pass diffusers →
//! 16 modulated delay lines mixed by a normalised Hadamard matrix, each with
//! its own RT60-derived loop gain and a one-pole damping low-pass → output
//! taps with alternating signs → wet low/high cut. The slow, decorrelated
//! delay modulation removes the metallic ringing of static FDNs, so long
//! tails (the Cathedral) stay smooth. The wet level is normalised by the
//! loop's energy gain, so a longer decay makes a longer, not louder, tail.

use super::filters::{Biquad, Kind};
use super::{Block, Ctx, DelayLine, Lfo};
use crate::voice::params::params;

params! {
    /// Room / hall / cathedral reverb.
    pub struct ReverbParams {
        /// Room size (scales all delays).
        size: [0.2, 2.0, 1.0, "x"],
        /// Decay time (RT60) at mid frequencies.
        decay_s: [0.1, 15.0, 2.0, "s"],
        /// Gap before the reverb starts.
        predelay_ms: [0.0, 250.0, 20.0, "ms"],
        /// High-frequency damping in the tail (0 = bright stone, 1 = dark velvet).
        damping: [0.0, 1.0, 0.4, ""],
        /// Density of the onset (all-pass diffusion).
        diffusion: [0.0, 0.85, 0.7, ""],
        /// Early reflections level.
        early: [0.0, 1.0, 0.4, ""],
        /// Chorus-like modulation of the tail.
        modulation: [0.0, 1.0, 0.4, ""],
        /// Wet low cut.
        lowcut_hz: [20.0, 1000.0, 100.0, "Hz"],
        /// Wet high cut.
        highcut_hz: [1000.0, 20000.0, 9000.0, "Hz"],
        /// Wet level (the dry level falls as the wet rises).
        mix: [0.0, 1.0, 0.3, ""],
    }
}

const N: usize = 16;
const BASE_MS: [f32; N] =
    [31.3, 37.1, 41.9, 46.7, 53.3, 58.1, 63.7, 69.1, 73.9, 79.3, 84.7, 89.9, 95.3, 101.9, 107.1, 113.3];
const AP_MS: [f32; 4] = [4.7, 3.6, 12.7, 9.3];
const ER_MS: [f32; 8] = [7.0, 11.3, 17.9, 23.1, 31.7, 41.3, 53.9, 67.1];
const ER_G: [f32; 8] = [0.72, -0.61, 0.55, -0.47, 0.41, -0.36, 0.3, -0.25];

struct AllPass {
    line: DelayLine,
    d: usize,
}

pub struct Reverb {
    p: ReverbParams,
    rate: f32,
    pre: DelayLine,
    aps: [AllPass; 4],
    lines: Vec<DelayLine>,
    len: [f32; N],
    gain: [f32; N],
    lp: [f32; N],
    damp: f32,
    lfos: [Lfo; N],
    lfo_hz: [f32; N],
    out: [f32; N],
    lowcut: Biquad,
    highcut: Biquad,
    wet_norm: f32,
}

impl Reverb {
    pub fn new(p: ReverbParams, rate: f32) -> Self {
        let max_line = (BASE_MS[N - 1] * 2.0 * 0.001 * rate) as usize + (rate * 0.004) as usize;
        let aps = std::array::from_fn(|i| AllPass { line: DelayLine::new((AP_MS[i] * 2.0 * 0.001 * rate) as usize + 2), d: 1 });
        let lines = (0..N).map(|_| DelayLine::new(max_line)).collect();
        let lfos = std::array::from_fn(|i| Lfo::new(i as f32 * 0.137));
        let lfo_hz = std::array::from_fn(|i| 0.11 + 0.057 * i as f32);
        let mut r = Reverb {
            p,
            rate,
            pre: DelayLine::new((0.26 * rate) as usize + (ER_MS[7] * 2.0 * 0.001 * rate) as usize),
            aps,
            lines,
            len: [0.0; N],
            gain: [0.0; N],
            lp: [0.0; N],
            damp: 0.0,
            lfos,
            lfo_hz,
            out: [0.0; N],
            lowcut: Biquad::pass(),
            highcut: Biquad::pass(),
            wet_norm: 1.0,
        };
        r.update();
        r
    }

    fn update(&mut self) {
        let p = self.p;
        let mut g2 = 0.0;
        for (i, base) in BASE_MS.iter().enumerate() {
            let l = base * p.size * 0.001 * self.rate;
            self.len[i] = l;
            // loop gain for the requested RT60: -60 dB after decay_s
            let g = 10f32.powf(-3.0 * l / (p.decay_s * self.rate));
            self.gain[i] = g;
            g2 += g * g;
        }
        let g2 = g2 / N as f32;
        self.wet_norm = (1.0 - g2).max(0.0005).sqrt() * 1.6;
        for (i, ap) in self.aps.iter_mut().enumerate() {
            ap.d = ((AP_MS[i] * (0.5 + 0.5 * p.size) * 0.001 * self.rate) as usize).max(1);
        }
        // damping: one-pole low-pass coefficient in the loop (0 = none)
        self.damp = p.damping * 0.85;
        self.lowcut.set(Kind::HighPass, p.lowcut_hz, 0.707, 0.0, self.rate);
        self.highcut.set(Kind::LowPass, p.highcut_hz, 0.707, 0.0, self.rate);
    }

    /// Wet output only, for one input sample.
    #[inline]
    fn wet(&mut self, x: f32) -> f32 {
        let p = &self.p;
        self.pre.push(x);
        let pd = p.predelay_ms * 0.001 * self.rate;
        let xin = self.pre.read(pd);
        // early reflections straight from the pre-delayed input
        let mut er = 0.0;
        if p.early > 0.0 {
            for (ms, g) in ER_MS.iter().zip(ER_G.iter()) {
                er += self.pre.read(pd + ms * p.size * 0.001 * self.rate) * g;
            }
            er *= p.early * 0.5;
        }
        // diffusion
        let mut d = xin;
        let gd = p.diffusion;
        for ap in self.aps.iter_mut() {
            let z = ap.line.tap(ap.d - 1);
            let v = d + gd * z;
            ap.line.push(v);
            d = z - gd * v;
        }
        // read the lines (modulated)
        let depth = p.modulation * 0.0006 * self.rate;
        for i in 0..N {
            let m = if depth > 0.0 { depth * (1.0 + self.lfos[i].tick(self.lfo_hz[i], self.rate)) } else { 0.0 };
            let v = self.lines[i].read(self.len[i] + m);
            // damping low-pass then decay gain
            self.lp[i] = v * (1.0 - self.damp) + self.lp[i] * self.damp;
            self.out[i] = self.lp[i] * self.gain[i];
        }
        let mut y = 0.0;
        for (i, o) in self.out.iter().enumerate() {
            y += if i & 1 == 0 { *o } else { -*o };
        }
        // Hadamard mixing (in place, normalised by 1/4 for N = 16)
        let mut h = self.out;
        let mut len = 1;
        while len < N {
            let mut i = 0;
            while i < N {
                for j in i..i + len {
                    let (a, b) = (h[j], h[j + len]);
                    h[j] = a + b;
                    h[j + len] = a - b;
                }
                i += 2 * len;
            }
            len *= 2;
        }
        for (i, (line, hv)) in self.lines.iter_mut().zip(h.iter()).enumerate() {
            let inj = if i % 3 == 0 { d } else if i % 3 == 1 { -d } else { d * 0.8 };
            line.push(hv * 0.25 + inj * 0.35);
        }
        y * 0.25 * self.wet_norm + er
    }
}

impl Block for Reverb {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let mix = self.p.mix;
        let dry = (1.0 - mix * mix).sqrt();
        for x in buf.iter_mut() {
            let w = self.wet(*x);
            let w = self.highcut.tick(self.lowcut.tick(w));
            *x = *x * dry + w * mix;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decay_time(p: ReverbParams) -> f32 {
        let rate = 48000.0;
        let mut r = Reverb::new(ReverbParams { mix: 1.0, predelay_ms: 0.0, early: 0.0, ..p }, rate);
        let n = (rate * (p.decay_s * 1.5 + 0.5)) as usize;
        let mut b = vec![0.0f32; n];
        b[0] = 1.0;
        r.process(&mut b, &mut Ctx::default());
        // energy in 50 ms windows; find -30 dB relative to the 0.1–0.15 s window, double it (T30)
        let w = (rate * 0.05) as usize;
        let e: Vec<f32> = b.chunks(w).map(|c| c.iter().map(|v| v * v).sum::<f32>()).collect();
        let e0 = e[2];
        let i = e.iter().skip(2).position(|v| *v < e0 * 1e-3).unwrap_or(e.len()) + 2;
        (i - 2) as f32 * 0.05 * 2.0
    }

    #[test]
    fn rt60_follows_decay_param() {
        for d in [1.0f32, 3.0] {
            let t = decay_time(ReverbParams { decay_s: d, damping: 0.0, lowcut_hz: 20.0, highcut_hz: 20000.0, ..Default::default() });
            assert!((t / d - 1.0).abs() < 0.35, "decay {d}: measured {t}");
        }
    }

    #[test]
    fn silence_in_silence_out_and_tail_is_dense() {
        let rate = 48000.0;
        let mut r = Reverb::new(ReverbParams { size: 2.0, decay_s: 8.0, mix: 0.6, ..Default::default() }, rate);
        let mut z = vec![0.0f32; 48000];
        r.process(&mut z, &mut Ctx::default());
        assert!(z.iter().all(|v| *v == 0.0));
        let mut b = vec![0.0f32; 96000];
        b[0] = 1.0;
        r.process(&mut b, &mut Ctx::default());
        assert!(b.iter().all(|v| v.is_finite()));
        // echo density: after 300 ms nearly every 1 ms window has energy
        let w = 48;
        let win: Vec<f32> = b[14400..62400].chunks(w).map(|c| c.iter().map(|v| v.abs()).sum::<f32>()).collect();
        let mean = win.iter().sum::<f32>() / win.len() as f32;
        let sparse = win.iter().filter(|v| **v < mean * 0.1).count();
        assert!(sparse < win.len() / 50, "{sparse} sparse windows of {}", win.len());
    }
}
