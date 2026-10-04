//! Speaker profile: who is talking, measured continuously from the voice
//! itself, so that voices with a *target* (see [`super::shifter`]) can move
//! any speaker to the same place instead of applying a fixed shift.
//!
//! ```text
//!  x ─┬─ YIN (every 10 ms) ─▶ voiced f0 ──────────────▶ log-f0 histogram ──▶ median f0, spread
//!     └─ ↓ ~12 kHz ─ LPC (every 20 ms, voiced frames) ─▶ formants F1..F4 ─▶ dispersion ΔF histogram ─▶ tract
//! ```
//!
//! * **f0** is the median of voiced f0 on a log scale; **spread** the robust
//!   standard deviation (IQR / 1.349) in semitones: how much you move.
//! * **tract** is the vocal-tract scale, half from the formant dispersion ΔF (a
//!   uniform tube of length L resonates at (2i − 1)·c / 4L, so the formants
//!   are spaced ΔF = c / 2L apart): each voiced frame's formants are fitted
//!   by least squares to Fᵢ = (2i − 1)·ΔF / 2 and `tract` = median ΔF /
//!   1000 Hz, which is about 1.0 for an average adult man (17.5 cm) and
//!   about 1.17 for an average woman. Vowels move single formants a lot;
//!   the dispersion over many frames is what follows the speaker. The other
//!   half is the tract that usually goes with the speaking pitch, because
//!   formant picking on real speech is easily thrown (see `update`).
//! * Evidence decays (τ = 20 s), so the profile follows a new speaker
//!   within seconds but does not jump on one odd word. Until there is
//!   enough speech it leans on a prior: neutral, or a saved calibration.
//!
//! Allocation-free after construction; cost ≈ 1 % of a core at 48 kHz.

use super::filters::{Biquad, Kind};
use super::lpc::{autocorr, levinson};
use super::pitch::PitchTracker;
use serde::{Deserialize, Serialize};

/// What the profiler knows about the speaker.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Profile {
    /// Median speaking pitch (Hz).
    pub f0_hz: f32,
    /// Pitch movement: robust standard deviation in semitones.
    pub spread_st: f32,
    /// Vocal-tract scale: formant dispersion / 1000 Hz (≈1.0 man, ≈1.17 woman, ≈1.3 child).
    pub tract: f32,
    /// Seconds of voiced speech behind the numbers (decaying; 0 = prior only).
    pub voiced_secs: f32,
    /// The style layer (level, body, brightness, breathiness, speaking
    /// rate), measured from a calibration sample; carried unchanged by the
    /// live profiler (see [`super::style`]).
    pub style: Option<super::style::VoiceStyle>,
}

impl Default for Profile {
    fn default() -> Self {
        Profile::NEUTRAL
    }
}

impl Profile {
    /// Between an average man and an average woman: the prior before any speech.
    pub const NEUTRAL: Profile = Profile { f0_hz: 150.0, spread_st: 2.5, tract: 1.08, voiced_secs: 0.0, style: None };

    /// Enough speech that targets can trust it.
    pub fn is_measured(&self) -> bool {
        self.voiced_secs >= 1.0
    }

    /// Clamp into plausible human ranges (a hand-edited or corrupt saved profile).
    pub fn sanitized(mut self) -> Profile {
        let ok = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        self.f0_hz = ok(self.f0_hz, F0_LO, F0_HI, Profile::NEUTRAL.f0_hz);
        self.spread_st = ok(self.spread_st, 0.3, 12.0, Profile::NEUTRAL.spread_st);
        self.tract = ok(self.tract, TRACT_LO, TRACT_HI, Profile::NEUTRAL.tract);
        self.voiced_secs = ok(self.voiced_secs, 0.0, 3600.0, 0.0);
        self.style = self.style.map(|s| s.sanitized());
        self
    }
}

const F0_LO: f32 = 50.0;
const F0_HI: f32 = 800.0;
/// f0 histogram resolution: bins per octave (a quarter semitone).
const F0_BPO: f32 = 48.0;
const F0_BINS: usize = 192; // log2(800/50) = 4 octaves
const TRACT_LO: f32 = 0.6;
const TRACT_HI: f32 = 2.0;
/// ΔF histogram resolution: bins per octave (≈0.7 %).
const TR_BPO: f32 = 96.0;
const TR_BINS: usize = 168; // log2(2.0/0.6) = 1.74 octaves
/// Formant search band and grid step.
const FMT_LO: f32 = 180.0;
const FMT_HI: f32 = 4800.0;
const FMT_STEP: f32 = 20.0;
const FMT_GRID: usize = ((FMT_HI - FMT_LO) / FMT_STEP) as usize + 1;
const LPC_ORDER: usize = 14;
/// Evidence decay time constant (s).
const FORGET_S: f32 = 20.0;
/// Weight (seconds of speech) of the neutral prior.
pub const NEUTRAL_PRIOR_S: f32 = 0.5;
/// Weight of a saved calibration.
pub const CALIBRATED_PRIOR_S: f32 = 4.0;
/// Seconds of voiced speech over which a prior's weight falls by e.
const PRIOR_FADE_S: f32 = 1.5;

/// The vocal-tract scale that usually goes with a speaking pitch: 1.0 at
/// 110 Hz (an average man), 1.17 at 210 Hz (an average woman), a power law
/// between and beyond, kept within adult-to-child sizes.
pub fn tract_for_pitch(f0_hz: f32) -> f32 {
    const BETA: f32 = 0.2428; // ln 1.17 / ln (210 / 110)
    (f0_hz.max(1.0) / 110.0).powf(BETA).clamp(0.9, 1.35)
}

/// Decaying histogram on a log axis with quantiles.
struct LogHist {
    bins: Vec<f32>,
    lo: f32,
    bpo: f32,
    total: f32,
}

impl LogHist {
    fn new(n: usize, lo: f32, bpo: f32) -> Self {
        LogHist { bins: vec![0.0; n], lo, bpo, total: 0.0 }
    }
    fn add(&mut self, v: f32, w: f32) {
        let x = (v / self.lo).log2() * self.bpo;
        // below the axis, or NaN (a zero or non-finite value)
        if x.is_nan() || x < 0.0 {
            return;
        }
        let i = x as usize;
        if i < self.bins.len() {
            self.bins[i] += w;
            self.total += w;
        }
    }
    fn clear(&mut self) {
        self.bins.fill(0.0);
        self.total = 0.0;
    }
    fn decay(&mut self, d: f32) {
        for b in self.bins.iter_mut() {
            *b *= d;
        }
        self.total *= d;
    }
    /// Value at quantile `q` (linear inside the bin).
    fn quantile(&self, q: f32) -> Option<f32> {
        if self.total <= 1e-9 {
            return None;
        }
        let want = q * self.total;
        let mut acc = 0.0;
        for (i, b) in self.bins.iter().enumerate() {
            if acc + b >= want && *b > 0.0 {
                let frac = ((want - acc) / b).clamp(0.0, 1.0);
                return Some(self.lo * 2f32.powf((i as f32 + frac) / self.bpo));
            }
            acc += b;
        }
        None
    }
}

/// Measures a [`Profile`] from a live voice.
pub struct Profiler {
    rate: f32,
    hop: usize,
    phase: usize,
    pitch: PitchTracker,
    // formant analysis on a decimated copy
    dec: usize,
    dec_phase: usize,
    aa: [Biquad; 2],
    dbuf: Vec<f32>,
    dmask: usize,
    dw: usize,
    pre_prev: f32,
    win: Vec<f32>,
    wbuf: Vec<f32>,
    r: Vec<f64>,
    a: Vec<f64>,
    k: Vec<f64>,
    tmp: Vec<f64>,
    cos_t: Vec<f32>,
    sin_t: Vec<f32>,
    env: Vec<f32>,
    fmt_every: usize,
    frames: usize,
    f0_hist: LogHist,
    tr_hist: LogHist,
    /// f0 of every voiced frame over the last ~2 s (events included): the
    /// reference the event guard compares a frame's pitch with.
    recent: LogHist,
    recent_decay: f32,
    decay_levels: f32,
    decay: f32,
    prior: Profile,
    prior_w: f32,
    out: Profile,
    /// Levels (RMS) of normal voiced frames over ~8 s, for the shout guard.
    levels: LogHist,
    /// Voiced frames left out as expressive events (shouts, laughs, fry).
    events: u32,
    /// Running share of voiced frames that look like events (τ 1.5 s): a
    /// burst is an event, but what persists is the new normal (a new
    /// speaker, or you in a different register).
    event_run: f32,
    event_a: f32,
}

impl Profiler {
    /// A profiler that starts from `prior` (a saved calibration, or `None` for neutral).
    pub fn new(rate: f32, prior: Option<Profile>) -> Self {
        let hop = (rate * 0.010).round().max(1.0) as usize;
        let dec = ((rate / 12000.0).round() as usize).max(1);
        let fs = rate / dec as f32;
        let wlen = (fs * 0.025).round() as usize;
        let n = (wlen * 2).next_power_of_two();
        let win: Vec<f32> = (0..wlen).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / wlen as f32).cos()).collect();
        let mut cos_t = vec![0.0; FMT_GRID * (LPC_ORDER + 1)];
        let mut sin_t = vec![0.0; FMT_GRID * (LPC_ORDER + 1)];
        for g in 0..FMT_GRID {
            let w = 2.0 * std::f64::consts::PI * (FMT_LO + g as f32 * FMT_STEP) as f64 / fs as f64;
            for i in 0..=LPC_ORDER {
                cos_t[g * (LPC_ORDER + 1) + i] = (w * i as f64).cos() as f32;
                sin_t[g * (LPC_ORDER + 1) + i] = (w * i as f64).sin() as f32;
            }
        }
        let (prior_p, prior_w) = match prior {
            Some(p) => (p.sanitized(), CALIBRATED_PRIOR_S),
            None => (Profile::NEUTRAL, NEUTRAL_PRIOR_S),
        };
        // anti-alias well below the decimated Nyquist (fs / 2 ≥ 5.5 kHz)
        let aa_hz = (fs * 0.42).min(rate * 0.45);
        Profiler {
            rate,
            hop,
            phase: 0,
            pitch: PitchTracker::new(rate),
            dec,
            dec_phase: 0,
            aa: [Biquad::new(Kind::LowPass, aa_hz, 0.54, 0.0, rate), Biquad::new(Kind::LowPass, aa_hz, 1.31, 0.0, rate)],
            dbuf: vec![0.0; n],
            dmask: n - 1,
            dw: 0,
            pre_prev: 0.0,
            win,
            wbuf: vec![0.0; wlen],
            r: vec![0.0; LPC_ORDER + 1],
            a: vec![0.0; LPC_ORDER + 1],
            k: vec![0.0; LPC_ORDER],
            tmp: vec![0.0; LPC_ORDER + 1],
            cos_t,
            sin_t,
            env: vec![0.0; FMT_GRID],
            fmt_every: 2,
            frames: 0,
            f0_hist: LogHist::new(F0_BINS, F0_LO, F0_BPO),
            tr_hist: LogHist::new(TR_BINS, TRACT_LO * 1000.0, TR_BPO),
            recent: LogHist::new(F0_BINS, F0_LO, F0_BPO),
            recent_decay: (-(hop as f32 / rate) / 2.0).exp(),
            decay_levels: (-(hop as f32 / rate) / 8.0).exp(),
            decay: (-(hop as f32 / rate) / FORGET_S).exp(),
            prior: prior_p,
            prior_w,
            out: Profile { voiced_secs: 0.0, ..prior_p },
            // RMS 1e-4..1 (-80..0 dBFS) at 1 dB per bin
            levels: LogHist::new(80, 1e-4, 6.02),
            events: 0,
            event_run: 0.0,
            event_a: hop as f32 / (1.5 * rate),
        }
    }

    /// The current profile.
    pub fn profile(&self) -> Profile {
        self.out
    }

    /// Forget what was measured and start from `prior` (a new calibration,
    /// or `None` for neutral). Allocation-free (audio thread).
    pub fn restart(&mut self, prior: Option<Profile>) {
        let (p, w) = match prior {
            Some(p) => (p.sanitized(), CALIBRATED_PRIOR_S),
            None => (Profile::NEUTRAL, NEUTRAL_PRIOR_S),
        };
        self.prior = p;
        self.prior_w = w;
        self.f0_hist.clear();
        self.tr_hist.clear();
        self.recent.clear();
        self.out = Profile { voiced_secs: 0.0, ..p };
        self.levels.clear();
        self.events = 0;
        self.event_run = 0.0;
    }

    /// Feed samples (after the gate: silence adds no evidence).
    pub fn process(&mut self, buf: &[f32]) {
        for &x in buf {
            self.pitch.push(x);
            let y = { let s0 = self.aa[0].tick(x); self.aa[1].tick(s0) };
            if self.dec_phase == 0 {
                let yp = y - 0.9 * self.pre_prev;
                self.pre_prev = y;
                self.dbuf[self.dw & self.dmask] = yp;
                self.dw += 1;
            }
            self.dec_phase += 1;
            if self.dec_phase == self.dec {
                self.dec_phase = 0;
            }
            self.phase += 1;
            if self.phase == self.hop {
                self.phase = 0;
                self.frame();
            }
        }
    }

    fn frame(&mut self) {
        let est = self.pitch.analyze();
        let dt = self.hop as f32 / self.rate;
        self.f0_hist.decay(self.decay);
        self.tr_hist.decay(self.decay);
        self.recent.decay(self.recent_decay);
        self.levels.decay(self.decay_levels);
        self.frames += 1;
        // confident, clearly periodic frames only
        let good = est.voiced && est.periodicity > 0.35 && est.rms > 2e-3;
        if good && self.expressive_event(est.f0, est.rms) {
            // a shout, a laugh or a squeal, or fry: real, but not how you
            // normally speak, so it does not move your identity
            self.events = self.events.saturating_add(1);
        } else if good {
            self.f0_hist.add(est.f0, dt);
            if self.frames % self.fmt_every == 0 {
                if let Some(df) = self.dispersion() {
                    self.tr_hist.add(df, dt * self.fmt_every as f32);
                }
            }
        }
        if self.frames % 5 == 0 {
            self.update();
        }
    }

    /// Voiced frames left out so far as expressive events.
    pub fn events(&self) -> u32 {
        self.events
    }

    /// Is this voiced frame an expressive event rather than normal speech?
    /// A shout (8 dB over the loud end of your normal voiced frames), or a pitch
    /// more than an octave above or 1.5 octaves below the median of the last
    /// ~2 s
    /// (a laugh, a squeal, vocal fry). Updates the running level with
    /// normal frames only. The first seconds are always normal: there is
    /// no reference yet. What lasts over ~1.5 s is not an event.
    fn expressive_event(&mut self, f0: f32, rms: f32) -> bool {
        let db = 20.0 * rms.max(1e-9).log10();
        // the pitch reference: the last ~2 s of voiced frames, whatever they
        // were (a 1 s laugh stays a minority; a new speaker takes over)
        let recent_med = self.recent.quantile(0.5);
        let recent_s = self.recent.total;
        self.recent.add(f0, self.hop as f32 / self.rate);
        // a shout: 8 dB over the loud end (90th percentile) of your normal
        // voiced frames of the last ~8 s (syllables vary by 10 dB or more, so
        // an average would call loud vowels shouts)
        let loud = self.levels.quantile(0.9);
        let shout = self.f0_hist.total > 2.0 && loud.is_some_and(|l| db > 20.0 * l.log10() + 8.0);
        let off_pitch = match recent_med {
            Some(med) if recent_s > 0.8 => f0 > med * 2.0 || f0 < med * 0.35,
            _ => false,
        };
        let candidate = shout || off_pitch;
        self.event_run += (if candidate { 1.0 } else { 0.0 } - self.event_run) * self.event_a.min(1.0);
        // a burst (a laugh, a shout) is an event; what lasts is the new normal
        if candidate && self.event_run < 0.6 {
            return true;
        }
        self.levels.add(rms, self.hop as f32 / self.rate);
        false
    }

    /// Formant dispersion (Hz) of the latest window, or None.
    fn dispersion(&mut self) -> Option<f32> {
        let wlen = self.win.len();
        if self.dw < wlen {
            return None;
        }
        let start = self.dw - wlen;
        for i in 0..wlen {
            self.wbuf[i] = self.dbuf[(start + i) & self.dmask] * self.win[i];
        }
        autocorr(&self.wbuf, &mut self.r);
        if self.r[0] < 1e-9 {
            return None;
        }
        self.r[0] *= 1.0 + 1e-4;
        // light lag window: formants, not harmonics
        let fs = (self.rate / self.dec as f32) as f64;
        for (i, r) in self.r.iter_mut().enumerate().skip(1) {
            let x = 2.0 * std::f64::consts::PI * 40.0 * i as f64 / fs;
            *r *= (-0.5 * x * x).exp();
        }
        levinson(&self.r, &mut self.a, &mut self.k, &mut self.tmp);
        // log |1/A|^2 on the grid
        let m = LPC_ORDER + 1;
        for g in 0..FMT_GRID {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for i in 0..m {
                let c = self.a[i] as f32;
                re += c * self.cos_t[g * m + i];
                im -= c * self.sin_t[g * m + i];
            }
            self.env[g] = -(re * re + im * im).max(1e-20).ln();
        }
        // the first four peaks with at least 1.5 dB of prominence
        let mut f = [0.0f32; 4];
        let mut nf = 0;
        let prom = 1.5 / 4.343; // dB → natural log of power
        for g in 1..FMT_GRID - 1 {
            let (y0, y1, y2) = (self.env[g - 1], self.env[g], self.env[g + 1]);
            if y1 > y0 && y1 >= y2 {
                // prominence against the lowest point within ±300 Hz
                let span = (300.0 / FMT_STEP) as usize;
                let lo_l = self.env[g.saturating_sub(span)..g].iter().cloned().fold(f32::MAX, f32::min);
                let lo_r = self.env[g + 1..(g + 1 + span).min(FMT_GRID)].iter().cloned().fold(f32::MAX, f32::min);
                if y1 - lo_l.max(lo_r) < prom {
                    continue;
                }
                let den = y0 - 2.0 * y1 + y2;
                let d = if den.abs() > 1e-9 { (0.5 * (y0 - y2) / den).clamp(-1.0, 1.0) } else { 0.0 };
                f[nf] = FMT_LO + (g as f32 + d) * FMT_STEP;
                nf += 1;
                if nf == 4 {
                    break;
                }
            }
        }
        if nf < 3 {
            return None;
        }
        // least squares Fi = c_i ΔF with c_i = (2i - 1) / 2
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (i, fi) in f[..nf].iter().enumerate() {
            let c = (2 * i + 1) as f32 * 0.5;
            num += fi * c;
            den += c * c;
        }
        let df = num / den;
        (df > TRACT_LO * 1000.0 && df < TRACT_HI * 1000.0).then_some(df)
    }

    fn update(&mut self) {
        let w_f0 = self.f0_hist.total;
        let w_tr = self.tr_hist.total;
        let blend = |prior: f32, meas: Option<f32>, w: f32, pw: f32| match meas {
            Some(m) => ((prior.ln() * pw + m.ln() * w) / (pw + w)).exp(),
            None => prior,
        };
        // The prior gives way as speech comes in (after ~3 s of voiced
        // speech it has 15 % or less of its weight left): a calibration is
        // an instant start, never an anchor against the person talking now.
        let pw = self.prior_w * (-w_f0 / PRIOR_FADE_S).exp();
        let med = self.f0_hist.quantile(0.5);
        let f0 = blend(self.prior.f0_hz, med, w_f0, pw);
        let spread = match (self.f0_hist.quantile(0.25), self.f0_hist.quantile(0.75)) {
            (Some(a), Some(b)) => {
                let s = (12.0 * (b / a).log2() / 1.349).max(0.3);
                (self.prior.spread_st * pw + s * w_f0) / (pw + w_f0)
            }
            _ => self.prior.spread_st,
        };
        let formants = blend(self.prior.tract, self.tr_hist.quantile(0.5).map(|d| d / 1000.0), w_tr, pw);
        // Formant picking is easily thrown on real speech (consonant
        // transitions, a missing top band): steady it with the tract that
        // goes with this speaking pitch (in adults they go together), half
        // and half on a log scale, so one wrong reading moves it at most half.
        let tract = (0.5 * formants.ln() + 0.5 * tract_for_pitch(f0).ln()).exp();
        self.out = Profile { f0_hz: f0, spread_st: spread, tract, voiced_secs: w_f0, style: self.prior.style }.sanitized();
    }
}

/// Profile of a recording (the calibration sentence): the same analysis
/// as live, over the whole clip, with no prior weight left once it has
/// enough speech.
pub fn measure(samples: &[f32], rate: f32) -> Profile {
    let mut p = Profiler::new(rate, None);
    // the prior should not pull a calibration: shrink it to nothing
    p.prior_w = 1e-3;
    // one long memory: the whole clip counts equally
    p.decay = 1.0;
    p.process(samples);
    p.update();
    p.profile()
}

/// Most effective seconds of speech a saved identity keeps (older sessions
/// fade as new ones are merged in).
pub const LEARNED_CAP_S: f32 = 120.0;
/// Least voiced speech (effective seconds) a session needs to be learnt from.
pub const LEARN_MIN_S: f32 = 5.0;

/// Continuous learning: `saved` (a calibration, or what earlier sessions
/// learnt) refined by what the live profiler measured in a session. None
/// when the session should not be learnt from: too little speech, or not
/// the same person (pitch more than 5 semitones or tract more than 12 %
/// away from the saved identity: someone else at the microphone, or you
/// doing a character voice). Weighted on log scales by seconds of speech,
/// the saved side capped at [`LEARNED_CAP_S`]; the style layer is kept.
pub fn learn(saved: Option<Profile>, live: Profile) -> Option<Profile> {
    let live = live.sanitized();
    if live.voiced_secs < LEARN_MIN_S {
        return None;
    }
    let Some(saved) = saved.map(|p| p.sanitized()) else {
        return Some(Profile { voiced_secs: live.voiced_secs.min(LEARNED_CAP_S), ..live });
    };
    let st = 12.0 * (live.f0_hz / saved.f0_hz).log2();
    let tr = (live.tract / saved.tract).ln().abs();
    if st.abs() > 5.0 || tr > 0.12f32.ln_1p() {
        return None;
    }
    let (ws, wl) = (saved.voiced_secs.clamp(1.0, LEARNED_CAP_S), live.voiced_secs);
    let mix = |a: f32, b: f32| ((a.ln() * ws + b.ln() * wl) / (ws + wl)).exp();
    Some(
        Profile {
            f0_hz: mix(saved.f0_hz, live.f0_hz),
            spread_st: (saved.spread_st * ws + live.spread_st * wl) / (ws + wl),
            tract: mix(saved.tract, live.tract),
            voiced_secs: (ws + wl).min(LEARNED_CAP_S),
            style: saved.style,
        }
        .sanitized(),
    )
}

/// [`measure`] and the style layer: the full identity from a calibration sample.
pub fn measure_identity(samples: &[f32], rate: f32) -> Profile {
    Profile { style: super::style::measure_style(samples, rate), ..measure(samples, rate) }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A glottal-pulse vowel sequence through formant resonators (the
    /// uniform-tube pattern scaled by `tract`), with intonation around `f0`.
    pub(crate) fn talker(rate: f32, f0: f32, tract: f32, secs: f32, seed: u32) -> Vec<f32> {
        talker_truth(rate, f0, tract, secs, seed).0
    }

    /// [`talker`] and the true median f0 of its voiced samples.
    pub(crate) fn talker_truth(rate: f32, f0: f32, tract: f32, secs: f32, seed: u32) -> (Vec<f32>, f32) {
        talker_with(rate, f0, tract, secs, seed, 1.0)
    }

    /// The talker with its melody scaled by `melody` (0 = a steady pitch,
    /// for measurements that need one, such as harmonics-to-noise).
    pub(crate) fn talker_with(rate: f32, f0: f32, tract: f32, secs: f32, seed: u32, melody: f32) -> (Vec<f32>, f32) {
        // male Peterson-Barney-like vowels (Hz), scaled by tract
        const V: [[f32; 3]; 5] = [[730.0, 1090.0, 2440.0], [270.0, 2290.0, 3010.0], [300.0, 870.0, 2240.0], [530.0, 1840.0, 2480.0], [570.0, 840.0, 2410.0]];
        let n = (secs * rate) as usize;
        let mut out = vec![0.0f32; n];
        let mut ph = 0.0f32;
        let mut rng = crate::voice::dsp::Rng::new(seed);
        let syl = (0.22 * rate) as usize;
        let mut state = [[0.0f32; 2]; 4];
        let mut truth: Vec<f32> = Vec::with_capacity(n / 64);
        for i in 0..n {
            let s = i / syl;
            let t = i as f32 / rate;
            // a lively but stationary melody (±3.5 st, no slow drift), so
            // any few seconds have `f0` as their median
            let st = melody * (2.0 * (2.0 * std::f32::consts::PI * 0.5 * t).sin() + 1.5 * (2.0 * std::f32::consts::PI * 1.3 * t + 0.7).sin());
            let f = f0 * 2f32.powf(st / 12.0);
            ph += f / rate;
            let mut e = 0.0;
            if ph >= 1.0 {
                ph -= 1.0;
                e = 1.0;
            }
            // pauses between words
            let gap = (s % 4 == 3) && (i % syl) > syl / 2;
            let x = if gap { 0.0 } else { e + 0.02 * rng.bipolar() };
            if !gap && i % 64 == 0 {
                truth.push(f);
            }
            let v = &V[(s * 7 + seed as usize) % 5];
            let mut y = x;
            let fmts = [v[0] * tract, v[1] * tract, v[2] * tract, 3500.0 * tract];
            for (k, fr) in fmts.iter().enumerate() {
                let bw = 60.0 + 0.05 * fr;
                let r = (-std::f32::consts::PI * bw / rate).exp();
                let c = 2.0 * r * (2.0 * std::f32::consts::PI * fr / rate).cos();
                let yn = y + c * state[k][0] - r * r * state[k][1];
                state[k][1] = state[k][0];
                state[k][0] = yn;
                y = yn * (1.0 - r);
            }
            out[i] = y;
        }
        let peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
        out.iter_mut().for_each(|v| *v *= 0.5 / peak);
        truth.sort_by(|a, b| a.total_cmp(b));
        let med = truth[truth.len() / 2];
        (out, med)
    }

    #[test]
    fn measures_pitch_and_tract_of_a_man_and_a_woman() {
        let rate = 48000.0;
        let (m, m_f0) = talker_truth(rate, 110.0, 1.0, 8.0, 1);
        let (w, w_f0) = talker_truth(rate, 210.0, 1.17, 8.0, 2);
        let (man, woman) = (measure(&m, rate), measure(&w, rate));
        let cents = |a: f32, b: f32| 1200.0 * (a / b).log2();
        // within 50 cents: the targets need a semitone
        assert!(cents(man.f0_hz, m_f0).abs() < 50.0, "{man:?} vs {m_f0}");
        assert!(cents(woman.f0_hz, w_f0).abs() < 50.0, "{woman:?} vs {w_f0}");
        // the ratio is what the targets use; absolute scale depends on vowels
        let ratio = woman.tract / man.tract;
        assert!((ratio - 1.17).abs() < 0.05, "tract ratio {ratio} ({man:?} / {woman:?})");
        assert!(man.voiced_secs > 3.0, "{man:?}");
    }

    #[test]
    fn follows_a_new_speaker_within_seconds_and_starts_from_the_prior() {
        let rate = 48000.0;
        let cal = Profile { f0_hz: 115.0, spread_st: 2.0, tract: 1.0, voiced_secs: 30.0, style: None };
        let mut p = Profiler::new(rate, Some(cal));
        assert!((p.profile().f0_hz - 115.0).abs() < 0.5);
        let (x, truth) = talker_truth(rate, 220.0, 1.17, 6.0, 3);
        p.process(&x);
        let got = p.profile();
        // what is left of the prior after ~4.5 s of speech is worth ~50 cents here
        assert!((1200.0 * (got.f0_hz / truth).log2()).abs() < 100.0, "{got:?} vs {truth}");
        assert!(got.tract > 1.08, "{got:?}");
    }

    #[test]
    fn silence_and_noise_leave_the_prior_alone() {
        let rate = 48000.0;
        let mut p = Profiler::new(rate, None);
        p.process(&vec![0.0; 48000]);
        let mut rng = crate::voice::dsp::Rng::new(9);
        let noise: Vec<f32> = (0..96000).map(|_| 0.2 * rng.bipolar()).collect();
        p.process(&noise);
        let got = p.profile();
        assert!((got.f0_hz - Profile::NEUTRAL.f0_hz).abs() < 5.0, "{got:?}");
        assert!(got.voiced_secs < 0.2, "{got:?}");
    }

    #[test]
    fn shouts_and_laughs_do_not_move_the_identity() {
        let rate = 48000.0;
        let (talk, truth) = talker_truth(rate, 110.0, 1.0, 20.0, 4);
        // every 4 s: a 1 s shout (+18 dB) or a 1 s laugh (an octave and a half up)
        let (shout, _) = talker_truth(rate, 110.0, 1.0, 1.0, 5);
        let (laugh, _) = talker_truth(rate, 110.0 * 2.83, 1.0, 1.0, 6);
        let mut x = Vec::new();
        for (k, c) in talk.chunks((4.0 * rate) as usize).enumerate() {
            x.extend_from_slice(c);
            if k % 2 == 0 {
                x.extend(shout.iter().map(|v| v * 8.0));
            } else {
                x.extend_from_slice(&laugh);
            }
        }
        let mut guarded = Profiler::new(rate, None);
        guarded.decay = 1.0;
        guarded.process(&x);
        guarded.update();
        let g = guarded.profile();
        println!("truth {truth:.1} Hz; with events {:.1} Hz, spread {:.2} st, {} event frames", g.f0_hz, g.spread_st, guarded.events());
        assert!(guarded.events() > 150, "the events are recognised: {}", guarded.events());
        assert!((12.0 * (g.f0_hz / truth).log2()).abs() < 0.5, "median stays on the talker: {:.1} vs {truth:.1}", g.f0_hz);
        // and plain speech is never an event
        let mut plain = Profiler::new(rate, None);
        plain.process(&talk);
        assert!(plain.events() < 20, "normal speech flagged {} times", plain.events());
    }

    #[test]
    fn learning_refines_the_saved_identity_but_not_from_someone_else() {
        let saved = Profile { f0_hz: 110.0, spread_st: 3.0, tract: 1.04, voiced_secs: 30.0, style: None };
        let session = Profile { f0_hz: 116.0, spread_st: 4.0, tract: 1.02, voiced_secs: 15.0, style: None };
        let l = learn(Some(saved), session).expect("same person: learnt");
        assert!(l.f0_hz > 110.0 && l.f0_hz < 116.0 && (l.voiced_secs - 45.0).abs() < 0.01);
        assert!((l.f0_hz - (110f32.ln() * 30.0 / 45.0 + 116f32.ln() * 15.0 / 45.0).exp()).abs() < 0.01, "weighted by speech");
        // someone else (a woman at the mic), or a character voice: not learnt
        assert!(learn(Some(saved), Profile { f0_hz: 210.0, tract: 1.17, ..session }).is_none());
        assert!(learn(Some(saved), Profile { tract: 1.25, ..session }).is_none());
        // too little speech: not learnt
        assert!(learn(Some(saved), Profile { voiced_secs: 2.0, ..session }).is_none());
        // the first session starts an identity; the cap keeps it learning
        assert_eq!(learn(None, session).unwrap().f0_hz, 116.0);
        let mut p = saved;
        for _ in 0..40 {
            p = learn(Some(p), session).unwrap();
        }
        assert!(p.voiced_secs <= LEARNED_CAP_S && (p.f0_hz - 116.0).abs() < 0.5, "new sessions keep counting: {p:?}");
        // the style layer is kept
        let styled = Profile { style: Some(crate::voice::dsp::style::VoiceStyle::default()), ..saved };
        assert!(learn(Some(styled), session).unwrap().style.is_some());
    }
}
