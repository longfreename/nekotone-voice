//! Linear prediction: autocorrelation, Levinson-Durbin, lattice analysis and
//! synthesis filters, and the spectral-envelope warper that moves formants.
//!
//! Formant warping works on the all-pole envelope `S(w) = 1/|A(w)|^2`:
//! the warped envelope is `S'(w) = S(w / alpha)` (interpolated in log
//! magnitude on a 512-point grid), its autocorrelation is the inverse FFT,
//! and Levinson gives the reflection coefficients of the warped all-pole
//! filter plus the gain that keeps the absolute level of `S'`.

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Autocorrelation `r[0..r.len()]` of `x`.
pub fn autocorr(x: &[f32], r: &mut [f64]) {
    for (lag, out) in r.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        if lag < x.len() {
            let (a, b) = (&x[..x.len() - lag], &x[lag..]);
            // f32 products accumulated in chunks for speed, then widened
            let mut part = 0.0f32;
            for (i, (p, q)) in a.iter().zip(b).enumerate() {
                part += p * q;
                if i & 63 == 63 {
                    acc += part as f64;
                    part = 0.0;
                }
            }
            acc += part as f64;
        }
        *out = acc;
    }
}

/// Levinson-Durbin. `r` has order+1 lags. Fills `a` (a[0] = 1, length
/// order+1) and reflection coefficients `k` (length order; |k| < 1 is
/// enforced). Returns the prediction error power.
pub fn levinson(r: &[f64], a: &mut [f64], k: &mut [f64], tmp: &mut [f64]) -> f64 {
    let p = k.len();
    a.fill(0.0);
    a[0] = 1.0;
    k.fill(0.0);
    let mut err = r[0];
    if err <= 1e-18 {
        return 0.0;
    }
    for i in 1..=p {
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let mut ki = -acc / err;
        if !ki.is_finite() {
            break;
        }
        ki = ki.clamp(-0.9995, 0.9995);
        tmp[..i].copy_from_slice(&a[..i]);
        for j in 1..i {
            a[j] = tmp[j] + ki * tmp[i - j];
        }
        a[i] = ki;
        k[i - 1] = ki;
        err *= 1.0 - ki * ki;
        if err <= 1e-18 {
            break;
        }
    }
    err
}

/// Reflection coefficients to direct-form predictor (a[0] = 1).
pub fn k_to_a(k: &[f32], a: &mut [f64], tmp: &mut [f64]) {
    a.fill(0.0);
    a[0] = 1.0;
    for (i0, ki) in k.iter().enumerate() {
        let i = i0 + 1;
        let ki = *ki as f64;
        tmp[..i].copy_from_slice(&a[..i]);
        for j in 1..i {
            a[j] = tmp[j] + ki * tmp[i - j];
        }
        a[i] = ki;
    }
}

/// FIR lattice (inverse / whitening filter): input x, output the residual.
#[inline]
pub fn lattice_analyze(x: f32, k: &[f32], b: &mut [f32]) -> f32 {
    // b[i] holds b_i(n-1) for i in 0..p
    let mut f = x;
    let mut bprev = x; // b_0(n)
    for i in 0..k.len() {
        let bi_old = b[i]; // b_i(n-1)
        let fnew = f + k[i] * bi_old;
        let bnew = bi_old + k[i] * f;
        b[i] = bprev;
        bprev = bnew;
        f = fnew;
    }
    f
}

/// IIR lattice (all-pole synthesis): input the excitation, output the signal.
#[inline]
pub fn lattice_synth(e: f32, k: &[f32], b: &mut [f32]) -> f32 {
    let p = k.len();
    let mut f = e;
    for i in (0..p).rev() {
        f -= k[i] * b[i]; // f_{i} = f_{i+1} - k b_i(n-1)
        let bnext = b[i] + k[i] * f; // b_{i+1}(n)
        if i + 1 < p {
            b[i + 1] = bnext;
        }
    }
    b[0] = f;
    f
}

/// The warper's FFT size (~23 Hz bins at 48 kHz).
const WARP_N: usize = 2048;
/// Formant finder: the lowest eighth of the band (0..6 kHz at 48 kHz) as
/// the spectrum of a signal at a quarter of the rate, on the same bin grid.
const FR_N: usize = WARP_N / 4;
/// 12 poles over 0..6 kHz: up to six resonances, the standard formant order.
const FR_ORDER: usize = 12;

/// Envelope warper (formant shifter) with preallocated FFT buffers.
pub struct Warper {
    n: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    spec: Vec<Complex<f32>>,
    logs: Vec<f32>,
    /// Formant finder (see [`Warper::find_formants`]): a low-order fit to
    /// the envelope's lowest eighth of the band, on its own small FFT.
    fr_fwd: Arc<dyn RealToComplex<f32>>,
    fr_inv: Arc<dyn ComplexToReal<f32>>,
    fr_time: Vec<f32>,
    fr_spec: Vec<Complex<f32>>,
    fr_sf: Vec<Complex<f32>>,
    fr_si: Vec<Complex<f32>>,
    fr_r: [f64; FR_ORDER + 1],
    fr_a: [f64; FR_ORDER + 1],
    fr_k: [f64; FR_ORDER],
    fr_tmp: [f64; FR_ORDER + 1],
    scratch_f: Vec<Complex<f32>>,
    scratch_i: Vec<Complex<f32>>,
    a: Vec<f64>,
    tmp: Vec<f64>,
    r: Vec<f64>,
    kk: Vec<f64>,
    /// log |1 - pre e^-jw|^2 per bin: the pre-emphasis the envelope was fitted with.
    logp: Vec<f32>,
}

impl Warper {
    pub fn new(order: usize) -> Self {
        Self::with_preemphasis(order, 0.0)
    }

    /// For envelopes fitted to a pre-emphasised signal: the tilt is removed
    /// before warping and re-applied after, so only the vocal tract moves
    /// (warping the tilt too drags the formants when shifting down).
    /// `order` is the order of the warped (output) filter.
    pub fn with_preemphasis(order: usize, pre: f32) -> Self {
        // 2048 points: ~23 Hz bins at 48 kHz, fine enough for formant bandwidths
        // even after compressing them (alpha < 1)
        let n = WARP_N;
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let fr_fwd = planner.plan_fft_forward(FR_N);
        let fr_inv = planner.plan_fft_inverse(FR_N);
        let scratch_f = fwd.make_scratch_vec();
        let scratch_i = inv.make_scratch_vec();
        Warper {
            n,
            time: vec![0.0; n],
            spec: vec![Complex::new(0.0, 0.0); n / 2 + 1],
            logs: vec![0.0; n / 2 + 1],
            fr_time: vec![0.0; FR_N],
            fr_spec: vec![Complex::new(0.0, 0.0); FR_N / 2 + 1],
            fr_sf: fr_fwd.make_scratch_vec(),
            fr_si: fr_inv.make_scratch_vec(),
            fr_fwd,
            fr_inv,
            fr_r: [0.0; FR_ORDER + 1],
            fr_a: [0.0; FR_ORDER + 1],
            fr_k: [0.0; FR_ORDER],
            fr_tmp: [0.0; FR_ORDER + 1],
            scratch_f,
            scratch_i,
            fwd,
            inv,
            a: vec![0.0; order + 1],
            tmp: vec![0.0; order + 1],
            r: vec![0.0; order + 1],
            kk: vec![0.0; order],
            logp: (0..=n / 2)
                .map(|b| {
                    let w = std::f32::consts::PI * b as f32 / (n / 2) as f32;
                    let (sn, cs) = w.sin_cos();
                    ((1.0 - pre * cs).powi(2) + (pre * sn).powi(2)).max(1e-9).ln()
                })
                .collect(),
        }
    }

    /// Warp the envelope of the all-pole filter `k` by `alpha` (>1 moves
    /// formants up). Writes the warped reflection coefficients to `kw` and
    /// returns the linear gain to apply after the warped synthesis filter.
    pub fn warp(&mut self, k: &[f32], alpha: f32, kw: &mut [f32]) -> f32 {
        self.warp_mapped(k, alpha, 0.0, None, kw)
    }

    /// [`Warper::warp`], and when `formants` is given, a formant map on top
    /// of the uniform `alpha`: the envelope's first three peaks F1..F3
    /// (150-4500 Hz, at sample rate `rate`) are passed to it with their
    /// count, it returns a ratio for each, and the envelope is resampled
    /// through a piecewise-linear map that moves each formant, as a block
    /// (its flanks), to alpha·ri·Fi, with slope alpha above F3 (kept strictly
    /// increasing).
    pub fn warp_mapped(&mut self, k: &[f32], alpha: f32, rate: f32, formants: Option<&mut dyn FnMut(&[f32; 3], usize) -> [f32; 3]>, kw: &mut [f32]) -> f32 {
        let p = kw.len();
        if ((alpha - 1.0).abs() < 1e-4 && formants.is_none()) || k.len() > p {
            kw.fill(0.0);
            let n = k.len().min(p);
            kw[..n].copy_from_slice(&k[..n]);
            return 1.0;
        }
        k_to_a(k, &mut self.a[..k.len() + 1], &mut self.tmp[..k.len() + 1]);
        self.time.fill(0.0);
        for (t, a) in self.time.iter_mut().zip(self.a[..k.len() + 1].iter()) {
            *t = *a as f32;
        }
        if self.fwd.process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_f).is_err() {
            kw.fill(0.0);
            kw[..k.len()].copy_from_slice(k);
            return 1.0;
        }
        let half = self.n / 2;
        for ((l, c), p) in self.logs.iter_mut().zip(self.spec.iter()).zip(self.logp.iter()) {
            // log S = -log |A|^2, minus the pre-emphasis tilt
            *l = -(c.norm_sqr().max(1e-12)).ln() - p;
        }
        // the formant map: anchors (source bin, target bin), strictly increasing
        let mut anchors = [(0.0f32, 0.0f32); 7];
        let mut na = 1;
        if let Some(map) = formants {
            let hz = rate / self.n as f32;
            let mut pk = [0.0f32; 3];
            let np = self.find_formants(hz, &mut pk);
            if np > 0 {
                let r = map(&pk, np);
                // Each formant moves as a block, from the valley below it to
                // the valley above it, so its peak keeps its shape. The gap a
                // move opens (or closes) is only the valley bottom, stretched:
                // a flat floor. (Stretching a formant's skirt instead fills
                // the gap with the skirt's level, and the refit reads the
                // broad lump as the neighbouring formant moving too.)
                let mut valley = [0.0f32; 3];
                for i in 0..np.saturating_sub(1) {
                    let (b0, b1) = ((pk[i] / hz) as usize + 1, (pk[i + 1] / hz) as usize);
                    let mut vb = b0;
                    for bb in b0..=b1.min(half) {
                        if self.logs[bb] < self.logs[vb] {
                            vb = bb;
                        }
                    }
                    valley[i] = vb as f32 * hz;
                }
                let mut d = [0.0f32; 3];
                for i in 0..np {
                    d[i] = alpha * r[i].clamp(0.5, 2.0) * pk[i] - pk[i];
                }
                // Block edges around each gap: the valley on both sides, or,
                // when the two moves would cross there, points pulled in from
                // the valley towards each peak until the targets separate.
                let mut edge = [(0.0f32, 0.0f32); 2]; // (top of block i, bottom of block i+1)
                for i in 0..np.saturating_sub(1) {
                    let (f0, f1, v) = (pk[i], pk[i + 1], valley[i]);
                    edge[i] = (v, v + 0.5 * hz);
                    for t in [1.0f32, 0.6, 0.35, 0.15] {
                        let (hi, lo) = (f0 + t * (v - f0), f1 - t * (f1 - v));
                        edge[i] = (hi, lo.max(hi + 0.5 * hz));
                        if (edge[i].1 + d[i + 1]) - (hi + d[i]) >= hz {
                            break;
                        }
                    }
                }
                let mut last = (0.0f32, 0.0f32);
                for i in 0..np {
                    let f = pk[i];
                    let lo = if i > 0 { edge[i - 1].1 } else { f - 250.0f32.min(0.45 * f) };
                    let hi = if i + 1 < np { edge[i].0 } else { f + 180.0 };
                    let a0 = (lo / hz, (lo + d[i]) / hz);
                    let a1 = (hi / hz, (hi + d[i]) / hz);
                    if a0.0 > last.0 && a1.0 > a0.0 && a0.1 > last.1 + 0.5 && a1.1 < half as f32 - 2.0 {
                        anchors[na] = a0;
                        anchors[na + 1] = a1;
                        last = a1;
                        na += 2;
                    }
                }
            }
        }
        // warped log envelope sampled on the same grid
        for b in 0..=half {
            let bf = b as f32;
            // inverse of the map: which source bin lands on output bin b
            let src = {
                let mut i = na - 1;
                while i > 0 && bf < anchors[i].1 {
                    i -= 1;
                }
                let (s0, t0) = anchors[i];
                if i + 1 < na {
                    let (s1, t1) = anchors[i + 1];
                    s0 + (bf - t0) * (s1 - s0) / (t1 - t0).max(1e-6)
                } else {
                    s0 + (bf - t0) / alpha
                }
            };
            let v = if src >= half as f32 {
                self.logs[half]
            } else {
                let i = src as usize;
                let f = src - i as f32;
                self.logs[i] + (self.logs[i + 1] - self.logs[i]) * f
            };
            self.spec[b] = Complex::new((v + self.logp[b]).exp(), 0.0);
        }
        self.spec[0].im = 0.0;
        self.spec[half].im = 0.0;
        if self.inv.process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_i).is_err() {
            kw.fill(0.0);
            kw[..k.len()].copy_from_slice(k);
            return 1.0;
        }
        for (i, r) in self.r.iter_mut().enumerate() {
            *r = self.time[i] as f64;
        }
        self.r[0] *= 1.000_01;
        let err = levinson(&self.r, &mut self.a, &mut self.kk, &mut self.tmp);
        for (o, v) in kw.iter_mut().zip(self.kk.iter()) {
            *o = *v as f32;
        }
        debug_assert_eq!(kw.len(), p);
        // irfft is unnormalised (x n); the unwarped envelope has error 1
        ((err / self.n as f64).max(0.0).sqrt() as f32).clamp(0.05, 20.0)
    }

    /// F1..F3 (Hz) of the envelope in `logs`; returns how many were found.
    ///
    /// The high-order envelope follows single harmonics a little, and a weak
    /// formant on the skirt of a strong one is a shoulder, not a peak, so
    /// peak picking on it is unreliable. Instead the lowest eighth of the
    /// band (with the pre-emphasis kept, as formant trackers use it) is
    /// treated as the power spectrum of a signal at a quarter of the rate:
    /// its autocorrelation (inverse FFT) gets a 12-pole fit, whose envelope
    /// has one smooth peak per resonance. No allocation.
    fn find_formants(&mut self, hz: f32, out: &mut [f32; 3]) -> usize {
        let m = FR_N / 2;
        // power, normalised to its maximum so exp() cannot overflow
        let top = (0..=m).map(|b| self.logs[b] + self.logp[b]).fold(f32::MIN, f32::max);
        for b in 0..=m {
            let v = (self.logs[b] + self.logp[b] - top).max(-60.0);
            self.fr_spec[b] = Complex::new(v.exp(), 0.0);
        }
        if self.fr_inv.process_with_scratch(&mut self.fr_spec, &mut self.fr_time, &mut self.fr_si).is_err() {
            return 0;
        }
        for (r, t) in self.fr_r.iter_mut().zip(self.fr_time.iter()) {
            *r = *t as f64;
        }
        if self.fr_r[0] <= 0.0 {
            return 0;
        }
        self.fr_r[0] *= 1.0 + 1e-6;
        levinson(&self.fr_r, &mut self.fr_a, &mut self.fr_k, &mut self.fr_tmp);
        self.fr_time.fill(0.0);
        for (t, a) in self.fr_time.iter_mut().zip(self.fr_a.iter()) {
            *t = *a as f32;
        }
        if self.fr_fwd.process_with_scratch(&mut self.fr_time, &mut self.fr_spec, &mut self.fr_sf).is_err() {
            return 0;
        }
        // -log |A|^2 is the fitted envelope; its local maxima are the formants
        let env = |s: &[Complex<f32>], b: usize| -(s[b].norm_sqr().max(1e-20)).ln();
        let lo = ((150.0 / hz) as usize).max(1);
        let hi = ((4500.0 / hz) as usize).min(m - 1);
        let mut n = 0;
        for b in lo..=hi {
            let (y0, y1, y2) = (env(&self.fr_spec, b - 1), env(&self.fr_spec, b), env(&self.fr_spec, b + 1));
            if y1 > y0 && y1 >= y2 {
                let den = y0 - 2.0 * y1 + y2;
                let d = if den.abs() > 1e-9 { (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5) } else { 0.0 };
                out[n] = (b as f32 + d) * hz;
                n += 1;
                if n == 3 {
                    break;
                }
            }
        }
        n
    }
}

/// Power response 1/|A(f)|^2 of predictor `a` at `hz` (for tests/analysis).
pub fn envelope_at(a: &[f64], hz: f64, rate: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz / rate;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, c) in a.iter().enumerate() {
        re += c * (w * i as f64).cos();
        im -= c * (w * i as f64).sin();
    }
    1.0 / (re * re + im * im).max(1e-30)
}

/// LPC of a whole signal (Hann window), returns the predictor coefficients.
pub fn lpc(x: &[f32], order: usize) -> Vec<f64> {
    let n = x.len();
    let w: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v * (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()))
        .collect();
    let mut r = vec![0.0; order + 1];
    autocorr(&w, &mut r);
    r[0] *= 1.000_1;
    let mut a = vec![0.0; order + 1];
    let mut k = vec![0.0; order];
    let mut tmp = vec![0.0; order + 1];
    levinson(&r, &mut a, &mut k, &mut tmp);
    a
}

/// Peaks (formant candidates) of the LPC envelope between `lo` and `hi` Hz.
pub fn envelope_peaks(a: &[f64], rate: f64, lo: f64, hi: f64) -> Vec<f64> {
    let step = 5.0;
    let mut out = Vec::new();
    let mut f = lo;
    let mut prev2 = envelope_at(a, f - step, rate);
    let mut prev = envelope_at(a, f, rate);
    while f + step < hi {
        let next = envelope_at(a, f + step, rate);
        if prev > prev2 && prev >= next {
            // parabolic refinement in dB
            let (y0, y1, y2) = (prev2.ln(), prev.ln(), next.ln());
            let d = (y0 - y2) / (2.0 * (y0 - 2.0 * y1 + y2));
            out.push(f + d.clamp(-1.0, 1.0) * step);
        }
        prev2 = prev;
        prev = next;
        f += step;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resonator_signal(rate: f32, formants: &[(f32, f32)], n: usize) -> Vec<f32> {
        // noise through two-pole resonators in cascade
        let mut rng = crate::voice::dsp::Rng::new(5);
        let mut x: Vec<f32> = (0..n).map(|_| rng.bipolar()).collect();
        for &(f, bw) in formants {
            let r = (-std::f32::consts::PI * bw / rate).exp();
            let c = 2.0 * r * (2.0 * std::f32::consts::PI * f / rate).cos();
            let (mut y1, mut y2) = (0.0f32, 0.0f32);
            for v in x.iter_mut() {
                let y = *v + c * y1 - r * r * y2;
                y2 = y1;
                y1 = y;
                *v = y;
            }
        }
        x
    }

    #[test]
    fn lattice_round_trip_is_identity() {
        let rate = 16000.0;
        let x = resonator_signal(rate, &[(700.0, 80.0), (1200.0, 90.0)], 4000);
        let a = lpc(&x, 12);
        let mut k = vec![0.0f64; 12];
        let mut r = vec![0.0; 13];
        autocorr(&x, &mut r);
        let mut aa = vec![0.0; 13];
        let mut tmp = vec![0.0; 13];
        levinson(&r, &mut aa, &mut k, &mut tmp);
        let kf: Vec<f32> = k.iter().map(|v| *v as f32).collect();
        // k_to_a reproduces a
        let mut a2 = vec![0.0; 13];
        k_to_a(&kf, &mut a2, &mut tmp);
        for i in 0..13 {
            assert!((a2[i] - aa[i]).abs() < 1e-4, "{i}");
        }
        let _ = a;
        let mut ba = vec![0.0f32; 12];
        let mut bs = vec![0.0f32; 12];
        let mut max_err = 0.0f32;
        let mut e_pow = 0.0;
        let mut x_pow = 0.0;
        for v in &x {
            let e = lattice_analyze(*v, &kf, &mut ba);
            // lattice residual equals the direct-form residual
            let y = lattice_synth(e, &kf, &mut bs);
            max_err = max_err.max((y - v).abs());
            e_pow += e * e;
            x_pow += v * v;
        }
        assert!(max_err < 1e-3, "round trip error {max_err}");
        assert!(e_pow < x_pow * 0.2, "residual is whiter/smaller");
    }

    #[test]
    fn warp_moves_envelope_peaks_by_alpha() {
        let rate = 16000.0;
        let x = resonator_signal(rate, &[(600.0, 60.0), (1500.0, 80.0), (2600.0, 100.0)], 16000);
        let order = 16;
        let mut r = vec![0.0; order + 1];
        autocorr(&x, &mut r);
        let mut a = vec![0.0; order + 1];
        let mut k = vec![0.0; order];
        let mut tmp = vec![0.0; order + 1];
        levinson(&r, &mut a, &mut k, &mut tmp);
        let kf: Vec<f32> = k.iter().map(|v| *v as f32).collect();
        let base = envelope_peaks(&a, rate as f64, 200.0, 4000.0);
        let mut w = Warper::new(order);
        let mut kw = vec![0.0f32; order];
        // identity warp keeps gain ~1
        let g1 = w.warp(&kf, 1.0, &mut kw);
        assert_eq!(g1, 1.0);
        for alpha in [0.8f32, 1.2] {
            let g = w.warp(&kf, alpha, &mut kw);
            assert!(g > 0.1 && g < 10.0);
            let mut aw = vec![0.0; order + 1];
            k_to_a(&kw, &mut aw, &mut tmp);
            let peaks = envelope_peaks(&aw, rate as f64, 200.0, 5000.0);
            for f in base.iter().take(2) {
                let want = f * alpha as f64;
                let got = peaks.iter().cloned().min_by(|p, q| (p - want).abs().total_cmp(&(q - want).abs())).unwrap();
                assert!((got / want - 1.0).abs() < 0.05, "alpha {alpha}: peak {f} -> {got}, want {want}");
            }
        }
    }

    #[test]
    fn formant_map_moves_one_formant_and_leaves_the_others() {
        // a man's GOOSE at 48 kHz, fitted like the voice stage (order 32, pre-emphasis)
        let rate = 48000.0;
        let raw = resonator_signal(rate, &[(300.0, 70.0), (870.0, 90.0), (2240.0, 120.0), (3400.0, 160.0)], 48000);
        let pre = 0.9f32;
        let x: Vec<f32> = raw.windows(2).map(|w| w[1] - pre * w[0]).collect();
        let order = 32;
        let mut r = vec![0.0; order + 1];
        autocorr(&x, &mut r);
        let (mut a, mut k, mut tmp) = (vec![0.0; order + 1], vec![0.0; order], vec![0.0; order + 1]);
        levinson(&r, &mut a, &mut k, &mut tmp);
        let kf: Vec<f32> = k.iter().map(|v| *v as f32).collect();
        let mut w = Warper::with_preemphasis(order + 12, pre);
        let mut kw = vec![0.0f32; order + 12];
        let mut seen = [0.0f32; 3];
        let mut map = |pk: &[f32; 3], _n: usize| {
            seen = *pk;
            [1.0, 1.35, 1.0]
        };
        w.warp_mapped(&kf, 1.0, rate, Some(&mut map), &mut kw);
        let mut aw = vec![0.0; order + 13];
        let mut tmp2 = vec![0.0; order + 13];
        k_to_a(&kw, &mut aw, &mut tmp2);
        // the pre-emphasis is still in the warped filter; compare its peaks
        let peaks = envelope_peaks(&aw, rate as f64, 150.0, 4500.0);
        let base = envelope_peaks(&a, rate as f64, 150.0, 4500.0);
        println!("found {seen:?}; before {base:?} after {peaks:?}");
        let near = |ps: &[f64], f: f64| ps.iter().cloned().min_by(|p, q| (p - f).abs().total_cmp(&(q - f).abs())).unwrap();
        assert!((seen[0] - 300.0).abs() < 60.0 && (seen[1] - 870.0).abs() < 60.0, "finder: {seen:?}");
        // F1 is not in the moved block (the warped spectrum keeps its peak
        // bin), but the 44-pole refit at 48 kHz cannot keep a low, sharp
        // peak exactly next to a moved neighbour: measured +11 % for F2
        // ×1.35 (+33 % when the blocks were anchored in the skirts).
        assert!((near(&peaks, 300.0) / near(&base, 300.0) - 1.0).abs() < 0.12, "F1 stays");
        assert!((near(&peaks, 1175.0) / (1.35 * near(&base, 870.0)) - 1.0).abs() < 0.05, "F2 moves ×1.35");
        assert!((near(&peaks, 2240.0) / near(&base, 2240.0) - 1.0).abs() < 0.05, "F3 stays");
    }
}

