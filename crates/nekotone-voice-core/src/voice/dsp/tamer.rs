//! Resonance tamer (the realism guard, part 2): finds narrow, *steady*
//! spectral peaks in the processed voice (a "metallic" ringing that a large
//! formant move, a strong EQ or a resonant effect can leave) and cuts them
//! with narrow dynamic notches. No latency: the analysis runs on the side
//! and steers peaking filters in the signal path.
//!
//! What counts as a resonance, 1.5–10 kHz (where ringing sounds metallic):
//! * a local spectral maximum at least 10 dB above the spectrum smoothed over
//!   half an octave, and
//! * **steady**: on the same frequency (±1 bin) for 250 ms. Natural voice
//!   harmonics up there move with the pitch (a 1 % pitch change moves the
//!   25th harmonic by a whole bin), so they are not steady; and
//! * **not a harmonic of the voice**: peaks within 1.5 % of a multiple of the
//!   current output pitch are left alone (a flat robot voice has perfectly
//!   steady harmonics).
//!
//! Up to four notches, each cutting (excess − 6 dB) × 0.7, at most 8 dB,
//! about 70 Hz wide; 50 ms attack, 300 ms release. Allocation-free after
//! construction.

use super::filters::{Biquad, Kind};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

const SLOTS: usize = 4;
const LO_HZ: f32 = 1500.0;
const HI_HZ: f32 = 10000.0;
const EXCESS_DB: f32 = 10.0;
const STEADY_S: f32 = 0.25;
const MAX_CUT_DB: f32 = 8.0;

#[derive(Clone, Copy)]
struct Slot {
    hz: f32,
    /// Current and target cut (dB, ≤ 0).
    gain: f32,
    target: f32,
    filt: Biquad,
    used: bool,
}

pub struct Tamer {
    rate: f32,
    n: usize,
    hop: usize,
    fill: usize,
    hist: Vec<f32>,
    pos: usize,
    win: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    level: Vec<f32>,
    prefix: Vec<f64>,
    age: Vec<u16>,
    age2: Vec<u16>,
    steady: u16,
    slots: [Slot; SLOTS],
    attack: f32,
    release: f32,
    /// Cuts applied so far (for tests and meters).
    pub cuts: u32,
}

impl Tamer {
    pub fn new(rate: f32) -> Self {
        let n = ((rate * 0.043) as usize).next_power_of_two().max(256);
        let hop = n / 8;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let bins = n / 2 + 1;
        let hop_s = hop as f32 / rate;
        Tamer {
            rate,
            n,
            hop,
            fill: 0,
            hist: vec![0.0; n],
            pos: 0,
            win: (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()).collect(),
            time: vec![0.0; n],
            spec: vec![Complex::new(0.0, 0.0); bins],
            scratch: fft.make_scratch_vec(),
            fft,
            level: vec![0.0; bins],
            prefix: vec![0.0; bins + 1],
            age: vec![0; bins],
            age2: vec![0; bins],
            steady: (STEADY_S / hop_s).ceil() as u16,
            slots: [Slot { hz: 0.0, gain: 0.0, target: 0.0, filt: Biquad::default(), used: false }; SLOTS],
            attack: 1.0 - (-hop_s / 0.05).exp(),
            release: 1.0 - (-hop_s / 0.3).exp(),
            cuts: 0,
        }
    }

    /// Process `buf` in place; `out_pitch_hz` is the voice's current output
    /// pitch (0 = unvoiced or unknown), whose harmonics are never cut.
    pub fn process(&mut self, buf: &mut [f32], out_pitch_hz: f32) {
        for x in buf.iter_mut() {
            self.hist[self.pos] = *x;
            self.pos = (self.pos + 1) % self.n;
            self.fill += 1;
            if self.fill >= self.hop {
                self.fill = 0;
                self.analyse(out_pitch_hz);
            }
            let mut y = *x;
            for s in self.slots.iter_mut() {
                if s.used && s.gain < -0.05 {
                    y = s.filt.tick(y);
                }
            }
            *x = y;
        }
    }

    fn analyse(&mut self, f0: f32) {
        let n = self.n;
        for i in 0..n {
            self.time[i] = self.hist[(self.pos + i) % n] * self.win[i];
        }
        if self.fft.process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch).is_err() {
            return;
        }
        let bins = self.spec.len();
        let hz = self.rate / n as f32;
        let mut top = f32::MIN;
        for (l, c) in self.level.iter_mut().zip(self.spec.iter()) {
            *l = 10.0 * (c.norm_sqr() + 1e-20).log10();
            top = top.max(*l);
        }
        self.prefix[0] = 0.0;
        for k in 0..bins {
            self.prefix[k + 1] = self.prefix[k] + self.level[k] as f64;
        }
        let (k_lo, k_hi) = (((LO_HZ / hz) as usize).max(2), ((HI_HZ / hz) as usize).min(bins - 2));
        // candidates: steady local maxima well above their half-octave neighbourhood
        let mut best: [(f32, usize); SLOTS] = [(0.0, 0); SLOTS];
        for k in 0..bins {
            self.age2[k] = 0;
        }
        for k in k_lo..=k_hi {
            let l = self.level[k];
            if !(l > self.level[k - 1] && l >= self.level[k + 1]) || l < top - 60.0 {
                continue;
            }
            let (a, b) = (((k as f32) * 2f32.powf(-0.25)) as usize, (((k as f32) * 2f32.powf(0.25)) as usize + 1).min(bins));
            let env = ((self.prefix[b] - self.prefix[a]) / (b - a).max(1) as f64) as f32;
            let excess = l - env;
            if excess < EXCESS_DB {
                continue;
            }
            let f = k as f32 * hz;
            if f0 > 30.0 {
                let h = (f / f0).round().max(1.0);
                if (f - h * f0).abs() < (0.015 * f).max(1.5 * hz) {
                    continue;
                }
            }
            let age = self.age[k - 1].max(self.age[k]).max(self.age[k + 1]).saturating_add(1);
            self.age2[k] = age;
            if age >= self.steady {
                // keep the largest excesses
                let (i, m) = best.iter().enumerate().min_by(|x, y| x.1 .0.total_cmp(&y.1 .0)).map(|(i, v)| (i, v.0)).unwrap();
                if excess > m {
                    best[i] = (excess, k);
                }
            }
        }
        std::mem::swap(&mut self.age, &mut self.age2);
        // steer the notches
        for s in self.slots.iter_mut() {
            s.target = 0.0;
        }
        for &(excess, k) in best.iter().filter(|b| b.0 > 0.0) {
            let f = k as f32 * hz;
            let cut = -((excess - 6.0) * 0.7).clamp(0.0, MAX_CUT_DB);
            let slot = self
                .slots
                .iter()
                .position(|s| s.used && (s.hz - f).abs() <= 2.0 * hz)
                .or_else(|| self.slots.iter().position(|s| !s.used));
            if let Some(i) = slot {
                let s = &mut self.slots[i];
                if !s.used {
                    s.used = true;
                    s.gain = 0.0;
                    self.cuts = self.cuts.saturating_add(1);
                }
                s.hz = f;
                s.target = cut;
            }
        }
        let (att, rel) = (self.attack, self.release);
        for s in self.slots.iter_mut().filter(|s| s.used) {
            let a = if s.target < s.gain { att } else { rel };
            s.gain += (s.target - s.gain) * a;
            if s.target == 0.0 && s.gain > -0.05 {
                s.used = false;
                s.gain = 0.0;
                continue;
            }
            // about 70 Hz wide (three bins), never wider than a third of the frequency
            let q = (s.hz / (3.0 * hz)).clamp(4.0, 30.0);
            s.filt.set(Kind::Peak, s.hz, q, s.gain, self.rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::dsp::profile::tests::talker_truth;

    const RATE: f32 = 48000.0;

    /// Band level (dB) of `y` between `lo` and `hi` Hz over the settled part.
    fn band(y: &[f32], lo: f32, hi: f32) -> f32 {
        let n = 4096;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
        let mut acc = vec![0.0f64; n / 2 + 1];
        let from = (1.0 * RATE) as usize;
        let mut t = vec![0.0f32; n];
        let mut s = vec![Complex::new(0.0, 0.0); n / 2 + 1];
        for c in y[from..].chunks_exact(n) {
            for (i, v) in c.iter().enumerate() {
                t[i] = v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos());
            }
            fft.process(&mut t, &mut s).unwrap();
            for (a, v) in acc.iter_mut().zip(s.iter()) {
                *a += v.norm_sqr() as f64;
            }
        }
        let hz = RATE / n as f32;
        let e: f64 = acc.iter().enumerate().filter(|(k, _)| (*k as f32 * hz) >= lo && (*k as f32 * hz) < hi).map(|(_, v)| *v).sum();
        10.0 * e.max(1e-30).log10() as f32
    }

    fn run(x: &[f32], pitch: f32) -> (Vec<f32>, u32) {
        let mut t = Tamer::new(RATE);
        let mut y = x.to_vec();
        for c in y.chunks_mut(256) {
            t.process(c, pitch);
        }
        (y, t.cuts)
    }

    #[test]
    fn a_steady_ringing_is_cut_and_the_voice_is_left_alone() {
        let (talk, _) = talker_truth(RATE, 110.0, 1.0, 6.0, 1);
        // a metallic ring at 3.2 kHz, well above the voice there
        let ring: Vec<f32> = talk.iter().enumerate().map(|(i, v)| v + 0.02 * (2.0 * std::f32::consts::PI * 3200.0 * i as f32 / RATE).sin()).collect();
        let (y, cuts) = run(&ring, 0.0);
        let (b0, b1) = (band(&ring, 3150.0, 3250.0), band(&y, 3150.0, 3250.0));
        println!("ring band {b0:.1} -> {b1:.1} dB ({cuts} notches)");
        assert!(b1 < b0 - 5.0, "the ring is cut: {b0:.1} -> {b1:.1} dB");
        // the plain voice: no band moves by more than half a dB
        let (y, cuts) = run(&talk, 0.0);
        for (lo, hi) in [(1500.0, 2000.0), (2000.0, 2500.0), (2500.0, 3150.0), (3150.0, 4000.0), (4000.0, 5000.0), (5000.0, 6300.0), (6300.0, 8000.0)] {
            let (a, b) = (band(&talk, lo, hi), band(&y, lo, hi));
            assert!((a - b).abs() < 0.5, "{lo}-{hi} Hz moved {:.2} dB ({cuts} notches)", b - a);
        }
    }

    #[test]
    fn steady_harmonics_of_the_voice_are_never_cut() {
        // a perfectly flat (robot) voice: its harmonics are steady by design
        let x: Vec<f32> = {
            let mut ph = 0.0f32;
            let (mut y1, mut y2) = (0.0f32, 0.0f32);
            let r = (-std::f32::consts::PI * 300.0 / RATE).exp();
            let c = 2.0 * r * (2.0 * std::f32::consts::PI * 2500.0 / RATE).cos();
            (0..(4.0 * RATE) as usize)
                .map(|_| {
                    ph += 150.0 / RATE;
                    let e = if ph >= 1.0 {
                        ph -= 1.0;
                        1.0
                    } else {
                        0.0
                    };
                    let y = e + c * y1 - r * r * y2;
                    y2 = y1;
                    y1 = y;
                    0.05 * y
                })
                .collect()
        };
        let (y, cuts) = run(&x, 150.0);
        assert_eq!(cuts, 0, "no notch on the voice's own harmonics");
        let (a, b) = (band(&x, 1500.0, 8000.0), band(&y, 1500.0, 8000.0));
        assert!((a - b).abs() < 0.1, "unchanged: {a:.2} vs {b:.2}");
    }
}
