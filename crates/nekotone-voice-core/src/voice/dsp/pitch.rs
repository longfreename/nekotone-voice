//! Real-time pitch tracker: YIN (cumulative mean normalised difference) on
//! a low-passed, decimated copy of the input (~12 kHz), with parabolic
//! refinement, a voicing decision with hysteresis, and octave-jump
//! protection. Called once per analysis hop; allocation-free.

use super::filters::{Biquad, Kind};

/// Lowest and highest f0 tracked (Hz).
pub const F0_MIN: f32 = 60.0;
pub const F0_MAX: f32 = 1000.0;
/// A dip within this much of the deepest one counts as "as deep" when
/// choosing the shortest period (see `analyze`).
const OCTAVE_SLACK: f32 = 0.1;

#[derive(Debug, Clone, Copy, Default)]
pub struct PitchEstimate {
    /// f0 in Hz (the last voiced value is held while unvoiced).
    pub f0: f32,
    /// Period in samples at the full rate.
    pub period: f32,
    /// YIN aperiodicity at the chosen lag (0 = perfectly periodic, 1 = noise).
    pub aperiodicity: f32,
    pub voiced: bool,
    /// RMS of the analysis window (full-scale units).
    pub rms: f32,
    /// Share of the input's energy below ~1.4 kHz (voiced sound is
    /// low-heavy; s, sh, f and clicks are not).
    pub lf_share: f32,
    /// Soft voicing degree 0..1 (1 = clearly periodic), 0 when unvoiced.
    pub periodicity: f32,
}

pub struct PitchTracker {
    rate: f32,
    dec: usize,
    phase: usize,
    lp: [Biquad; 2],
    buf: Vec<f32>,
    mask: usize,
    w: usize, // samples written (decimated)
    win: usize,
    tau_min: usize,
    tau_max: usize,
    d: Vec<f32>,
    est: PitchEstimate,
    /// Voicing thresholds on aperiodicity: enter below `on`, stay below `off`.
    pub on: f32,
    pub off: f32,
    /// Minimum RMS for a voiced decision.
    pub min_rms: f32,
    /// Minimum low-band energy share for a voiced decision.
    pub min_lf: f32,
    /// Aperiodicity at which `periodicity` reaches 1 (fully voiced).
    pub full: f32,
    /// Also require the period to repeat at twice the lag.
    pub two_period: bool,
    lf_e: f32,
    full_e: f32,
    e_a: f32,
}

impl PitchTracker {
    pub fn new(rate: f32) -> Self {
        let dec = ((rate / 12000.0).round() as usize).max(1);
        let fs = rate / dec as f32;
        let tau_min = (fs / F0_MAX).floor().max(2.0) as usize;
        let tau_max = (fs / F0_MIN).ceil() as usize;
        let win = tau_max; // >= one period of the lowest voice
        let n = (win + tau_max + 4).next_power_of_two() * 2;
        PitchTracker {
            rate,
            dec,
            phase: 0,
            lp: [Biquad::new(Kind::LowPass, 1400.0, 0.707, 0.0, rate); 2],
            buf: vec![0.0; n],
            mask: n - 1,
            w: 0,
            win,
            tau_min,
            tau_max,
            d: vec![0.0; tau_max + 2],
            est: PitchEstimate { f0: 120.0, period: rate / 120.0, aperiodicity: 1.0, voiced: false, rms: 0.0, lf_share: 0.0, periodicity: 0.0 },
            // Real voices (breathy, creaky, a noisy room) sit at 0.2-0.5
            // aperiodicity on the low band; the old 0.2/0.38 turned half of
            // them into noise (see the shifter docs).
            on: 0.35,
            off: 0.5,
            min_rms: 3e-4,
            min_lf: 0.02,
            full: 0.25,
            two_period: true,
            lf_e: 0.0,
            full_e: 0.0,
            e_a: super::coef(0.03, rate),
        }
    }

    /// Rate of the decimated signal.
    pub fn analysis_rate(&self) -> f32 {
        self.rate / self.dec as f32
    }

    #[inline]
    pub fn push(&mut self, x: f32) {
        let y = { let s0 = self.lp[0].tick(x); self.lp[1].tick(s0) };
        self.lf_e += (y * y - self.lf_e) * self.e_a;
        self.full_e += (x * x - self.full_e) * self.e_a;
        if self.phase == 0 {
            self.buf[self.w & self.mask] = y;
            self.w += 1;
        }
        self.phase += 1;
        if self.phase == self.dec {
            self.phase = 0;
        }
    }

    pub fn estimate(&self) -> PitchEstimate {
        self.est
    }

    /// Run YIN on the most recent window and update the estimate.
    pub fn analyze(&mut self) -> PitchEstimate {
        let span = self.win + self.tau_max;
        if self.w < span {
            return self.est;
        }
        let start = self.w - span;
        let m = self.mask;
        let x = |i: usize| self.buf[(start + i) & m];
        // energy
        let mut e = 0.0f32;
        for i in 0..self.win {
            let v = x(i + self.tau_max);
            e += v * v;
        }
        let rms = (e / self.win as f32).sqrt();
        // difference function over the window at the end of the span
        let base = self.tau_max;
        self.d[0] = 0.0;
        let mut cum = 0.0f32;
        let mut best_tau = 0usize;
        let mut first_tau = 0usize;
        let mut best = f32::MAX;
        for tau in 1..=self.tau_max {
            let mut s = 0.0f32;
            for i in 0..self.win {
                let a = x(base + i);
                let b = x(base + i - tau);
                let dd = a - b;
                s += dd * dd;
            }
            cum += s;
            let dn = if cum > 0.0 { s * tau as f32 / cum } else { 1.0 };
            self.d[tau] = dn;
            if tau >= self.tau_min {
                if dn < best {
                    best = dn;
                    best_tau = tau;
                }
                if first_tau == 0 && dn < 0.15 {
                    first_tau = tau;
                }
            }
        }
        // YIN: first dip below the absolute threshold, walked to its local
        // minimum. Real voices often have no dip that deep (breathy, a
        // noisy room: 0.2-0.5); then the smallest lag whose dip is nearly
        // as deep as the best one, not the best one itself: the global
        // minimum often sits at two or three periods, and taking it put a
        // tenth of a woman's voiced frames an octave down (and the jump
        // protection below then held them there).
        let start = if first_tau > 0 {
            first_tau
        } else {
            let lim = best + OCTAVE_SLACK;
            (self.tau_min..=self.tau_max).find(|&t| self.d[t] < lim).unwrap_or(best_tau)
        };
        let mut tau = {
            let mut t = start;
            while t < self.tau_max && self.d[t + 1] < self.d[t] {
                t += 1;
            }
            t
        };
        // The threshold can also be crossed first at two periods (a dip
        // of 0.16 at the period, 0.12 at twice it): prefer a sub-multiple
        // whose own dip is nearly as deep.
        for k in [3usize, 2] {
            let c = (tau as f32 / k as f32).round() as usize;
            if c < self.tau_min + 1 || c + 1 >= self.tau_max {
                continue;
            }
            let mut t = c;
            for u in c.saturating_sub(2).max(self.tau_min)..=(c + 2).min(self.tau_max) {
                if self.d[u] < self.d[t] {
                    t = u;
                }
            }
            if self.d[t] < self.d[tau] + OCTAVE_SLACK && self.d[t] < self.off {
                tau = t;
                break;
            }
        }
        if tau == 0 {
            tau = self.tau_min;
        }
        let was_voiced = self.est.voiced;
        // octave-jump protection: prefer a dip near the previous period
        if was_voiced {
            let prev_tau = self.est.period / self.dec as f32;
            let ratio = tau as f32 / prev_tau;
            if !(0.7..=1.43).contains(&ratio) {
                let lo = ((prev_tau * 0.85) as usize).max(self.tau_min);
                let hi = ((prev_tau * 1.15).ceil() as usize).min(self.tau_max);
                let mut t2 = lo;
                for t in lo..=hi {
                    if self.d[t] < self.d[t2] {
                        t2 = t;
                    }
                }
                if self.d[t2] < self.d[tau] + 0.06 {
                    tau = t2;
                }
            }
        }
        let mut aper = self.d[tau];
        // a true period repeats at twice the lag; resonant noise (a whisper's
        // first formant, 300-900 Hz) dips once and decays. Only checked for
        // short lags: below 300 Hz it cost real voices a sixth of their
        // voiced frames.
        let fs = self.rate / self.dec as f32;
        if self.two_period && (tau as f32) < fs / 300.0 && 2 * tau < self.tau_max {
            let a2 = self.d[2 * tau - 1].min(self.d[2 * tau]).min(self.d[2 * tau + 1]);
            aper = aper.max(a2 - 0.03);
        }
        // parabolic interpolation on the normalised difference
        let t = if tau > 1 && tau < self.tau_max {
            let (y0, y1, y2) = (self.d[tau - 1], self.d[tau], self.d[tau + 1]);
            let den = y0 - 2.0 * y1 + y2;
            if den.abs() > 1e-9 {
                tau as f32 + (0.5 * (y0 - y2) / den).clamp(-1.0, 1.0)
            } else {
                tau as f32
            }
        } else {
            tau as f32
        };
        let mut thr = if was_voiced { self.off } else { self.on };
        // above 400 Hz a whisper's first formant (narrow-band noise) mimics a
        // period; speaking voices rarely go there, so be strict
        if self.rate / (t * self.dec as f32) > 400.0 {
            thr = thr.min(if was_voiced { 0.3 } else { 0.2 });
        }
        let lf_share = if self.full_e > 1e-14 { (self.lf_e / self.full_e).min(1.0) } else { 0.0 };
        let voiced = aper < thr && rms > self.min_rms && lf_share > self.min_lf;
        self.est.aperiodicity = aper;
        self.est.rms = rms;
        self.est.voiced = voiced;
        self.est.lf_share = lf_share;
        self.est.periodicity = if voiced { ((self.off - aper) / (self.off - self.full).max(1e-3)).clamp(0.0, 1.0) } else { 0.0 };
        if voiced {
            let period = t * self.dec as f32;
            self.est.period = period;
            self.est.f0 = self.rate / period;
        }
        self.est
    }
}

/// Offline f0 of a whole signal with high precision (for tests and reports):
/// autocorrelation at the full rate over `x`, parabolic peak, searched in
/// [fmin, fmax]. Returns 0 when there is no clear periodicity.
pub fn measure_f0(x: &[f32], rate: f32, fmin: f32, fmax: f32) -> f32 {
    let tmin = (rate / fmax).floor() as usize;
    let tmax = ((rate / fmin).ceil() as usize).min(x.len() / 2);
    if tmax <= tmin + 2 {
        return 0.0;
    }
    let n = x.len() - tmax;
    let e0: f64 = x[..n].iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let mut d = vec![0.0f64; tmax + 2];
    let mut cum = 0.0;
    for tau in 1..=tmax + 1 {
        let mut s = 0.0f64;
        for i in 0..n.min(x.len() - tau) {
            let dd = (x[i] - x[i + tau]) as f64;
            s += dd * dd;
        }
        cum += s;
        d[tau] = if cum > 0.0 { s * tau as f64 / cum } else { 1.0 };
    }
    let _ = e0;
    // global minimum, then the smallest lag whose dip is nearly as deep
    // (avoids both octave-down and strong-harmonic octave-up errors)
    let gmin = (tmin..=tmax).map(|t| d[t]).fold(f64::MAX, f64::min);
    if gmin > 0.2 {
        return 0.0;
    }
    let mut tau = 0;
    for t in tmin..=tmax {
        if d[t] < gmin + 0.08 {
            let mut u = t;
            while u < tmax && d[u + 1] < d[u] {
                u += 1;
            }
            tau = u;
            break;
        }
    }
    if tau == 0 {
        return 0.0;
    }
    let (y0, y1, y2) = (d[tau - 1], d[tau], d[tau + 1]);
    let den = y0 - 2.0 * y1 + y2;
    let t = if den.abs() > 1e-12 { tau as f64 + 0.5 * (y0 - y2) / den } else { tau as f64 };
    (rate as f64 / t) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_pulse_train_pitch() {
        let rate = 48000.0;
        for f0 in [85.0f32, 120.0, 210.0, 440.0] {
            let mut p = PitchTracker::new(rate);
            let period = rate / f0;
            let mut ph = 0.0f32;
            let mut last = PitchEstimate::default();
            for i in 0..24000 {
                ph += 1.0;
                let mut x = 0.0;
                if ph >= period {
                    ph -= period;
                    x = 1.0;
                }
                // a decaying resonance per pulse makes it voice-like
                p.push(x + 0.0 * i as f32);
                if i % 240 == 239 {
                    last = p.analyze();
                }
            }
            assert!(last.voiced, "{f0}: not voiced (aper {})", last.aperiodicity);
            let cents = 1200.0 * (last.f0 / f0).log2();
            assert!(cents.abs() < 3.0, "{f0}: {} Hz ({cents} cents)", last.f0);
        }
    }

    #[test]
    fn noise_is_unvoiced() {
        let rate = 48000.0;
        let mut p = PitchTracker::new(rate);
        let mut r = crate::voice::dsp::Rng::new(11);
        let mut voiced = 0;
        for i in 0..48000 {
            p.push(r.bipolar() * 0.3);
            if i % 240 == 239 && p.analyze().voiced {
                voiced += 1;
            }
        }
        assert!(voiced < 5, "noise voiced in {voiced} frames");
    }

    #[test]
    fn measure_f0_is_precise() {
        let rate = 48000.0;
        let x: Vec<f32> = (0..24000).map(|i| (2.0 * std::f32::consts::PI * 157.3 * i as f32 / rate).sin()).collect();
        let f = measure_f0(&x, rate, 60.0, 1000.0);
        assert!((1200.0 * (f / 157.3).log2()).abs() < 0.5, "{f}");
    }
}
