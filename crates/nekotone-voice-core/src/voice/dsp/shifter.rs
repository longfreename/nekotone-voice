//! The voice stage: independent pitch and formant shifting by LP-PSOLA
//! (linear-prediction pitch-synchronous overlap-add), plus harmoniser
//! voices, pitch/formant LFOs, monotone ("robot") pitch, whispering, growl
//! and a synthetic-pulse (vocoder) excitation.
//!
//! ```text
//!  x ─ pre-emphasis ─┬─ LPC every 5 ms (asymmetric 20 ms window, lattice k) ─┬─ warp envelope by formant ratio
//!                    │                                                        │        (k', gain)
//!                    └─ inverse lattice ─▶ residual e ─┬─ pitch marks (GCIs)   │
//!  x ─ decimate ─ YIN ─▶ f0, periodicity ──────────────┤                      │
//!                                                      ├─ PSOLA grains at new spacing (1..5 voices) ─┐
//!                                                      └─ noise × envelope(e)  (unvoiced, whisper) ──┤
//!                                         excitation = v·grains + √(1-v²)·noise   (v = soft voicing)
//!                                                        ─▶ synthesis lattice(k') × gain ─▶ de-emphasis ─▶ y
//! ```
//!
//! Why this design: the residual of LPC inverse filtering is spectrally flat
//! and pulse-like in voiced speech, so PSOLA on the residual moves the pitch
//! without dragging formants along or smearing them, and the formants are
//! then re-imposed from a separately warped envelope — pitch and formant are
//! truly independent. Voiced output is rebuilt from grains placed at the new
//! period, and unvoiced output from energy-matched noise shaped by the
//! (warped) envelope, so no component of the output carries the original
//! f0. Latency is fixed and small (25 ms at any rate: 12 ms grain half-width
//! twice plus 1 ms guard; it was 11 ms with a 5 ms cap, which made low
//! voices buzz, see `Tuning::cap_ms`) because PSOLA only needs one grain of
//! lookahead, unlike a phase vocoder that needs a 40–80 ms FFT frame for low
//! male voices.
//!
//! Voicing (0.4 listening pass): the voiced/unvoiced split is soft. The
//! tracker's periodicity (from YIN aperiodicity, with a low-band energy
//! share so s/sh/clicks never count as voiced) crossfades grains and noise
//! at equal power. The first version switched hard at aperiodicity 0.2/0.38,
//! which on real voices (breathy, creaky, a noisy room: 0.2–0.5) turned
//! half of every sentence — and all of a quiet voice — into hoarse noise;
//! that was the main reason every shifted preset sounded rough.
//! Pitch marks are picked near where the previous period predicts
//! (proximity-weighted residual peak), which removes the mark jitter that
//! made low voices warble.

use super::lpc::{autocorr, lattice_analyze, lattice_synth, levinson, Warper};
use super::pitch::PitchTracker;
use super::{coef, db_to_lin, Block, Ctx, Drift, Rng};
use crate::voice::params::params;
use std::f32::consts::PI;

params! {
    /// Pitch and formant shifter with up to four extra harmony voices.
    pub struct VoiceParams {
        /// Pitch shift of the main voice.
        pitch_st: [-24.0, 24.0, 0.0, "st"],
        /// Formant (vocal-tract size) ratio; >1 sounds smaller/brighter, <1 bigger/darker.
        formant: [0.5, 2.0, 1.0, "x"],
        /// Pitch vibrato rate.
        vibrato_hz: [0.0, 12.0, 5.0, "Hz"],
        /// Pitch vibrato depth.
        vibrato_st: [0.0, 12.0, 0.0, "st"],
        /// Formant wobble rate.
        formant_lfo_hz: [0.0, 12.0, 2.0, "Hz"],
        /// Formant wobble depth (fraction of the formant ratio).
        formant_lfo_depth: [0.0, 0.6, 0.0, ""],
        /// Replace the voiced excitation with noise (1 = full whisper).
        whisper: [0.0, 1.0, 0.0, ""],
        /// 1 = monotone at `fixed_hz` (robot), 0 = follow your intonation.
        flatten: [0.0, 1.0, 0.0, ""],
        /// Monotone pitch used by `flatten`.
        fixed_hz: [40.0, 600.0, 110.0, "Hz"],
        /// Random pitch drift per voice (natural roughness, choir spread).
        jitter_cents: [0.0, 60.0, 0.0, "cents"],
        /// Level of the main voice.
        main_db: [-60.0, 12.0, 0.0, "dB"],
        /// Harmony voice 1 interval relative to the main voice.
        v1_st: [-24.0, 24.0, -12.0, "st"],
        /// Harmony voice 1 level (-60 = off).
        v1_db: [-60.0, 12.0, -60.0, "dB"],
        /// Harmony voice 2 interval.
        v2_st: [-24.0, 24.0, 7.0, "st"],
        /// Harmony voice 2 level (-60 = off).
        v2_db: [-60.0, 12.0, -60.0, "dB"],
        /// Harmony voice 3 interval.
        v3_st: [-24.0, 24.0, 4.0, "st"],
        /// Harmony voice 3 level (-60 = off).
        v3_db: [-60.0, 12.0, -60.0, "dB"],
        /// Harmony voice 4 interval.
        v4_st: [-24.0, 24.0, 12.0, "st"],
        /// Harmony voice 4 level (-60 = off).
        v4_db: [-60.0, 12.0, -60.0, "dB"],
        /// Output gain.
        gain_db: [-24.0, 24.0, 0.0, "dB"],
        /// Growl: irregular sub-harmonic roughness on voiced sound only (monsters, vocal fry).
        growl: [0.0, 1.0, 0.0, ""],
        /// Breath: a breathy voice (softer upper harmonics and aspiration noise riding on voiced sound).
        breath: [0.0, 1.0, 0.0, ""],
        /// Excitation: 0 = your own voice's source (natural), 1 = clean synthetic pulses (vocoder, robot).
        pulse: [0.0, 1.0, 0.0, ""],
        /// Level of the unvoiced sounds (s, sh, f, breaths); -60 = voiced sound only (for layers).
        noise_db: [-60.0, 12.0, 0.0, "dB"],
        /// Target speaking pitch (median). 0 = a fixed shift of `pitch_st`. With a target the
        /// shift adapts to whoever is talking (see `profile`) and `pitch_st` fine-tunes it.
        target_hz: [0.0, 600.0, 0.0, "Hz"],
        /// Target vocal-tract scale (1 ≈ an average man, 1.17 an average woman, 1.3 a child).
        /// 0 = a fixed `formant` ratio. With a target, `formant` fine-tunes it.
        target_tract: [0.0, 2.0, 0.0, "x"],
        /// Intonation: 1 = your own melody, below 1 steadier, above 1 livelier (scaled around your median pitch).
        expr: [0.0, 2.0, 1.0, ""],
        /// Accent colour: moves your vowels, r-sounds and melody towards an accent (see `accent`).
        accent: [0.0, 7.0, 0.0, "choice:None|British RP|Australian|Irish|Scottish|Southern US|Indian|Welsh"],
        /// How strongly the accent colours your voice.
        accent_amount: [0.0, 1.0, 1.0, ""],
    }
}

/// How fast a voice with targets follows the speaker profile (s): quickly
/// while the profile is still learning a new voice, then slowly, so the
/// shift never moves within a phrase.
const ADAPT_TAU_S: f32 = 1.5;
const ADAPT_TAU_LEARNING_S: f32 = 0.3;
/// Seconds of voiced speech after which the profile counts as learnt.
const PROFILE_LEARNT_S: f32 = 4.0;

params! {
    /// Whisper: the voice is re-synthesised from noise through your (optionally shifted) vocal tract.
    pub struct WhisperParams {
        /// 1 = pure whisper, lower keeps some voiced tone.
        amount: [0.0, 1.0, 1.0, ""],
        /// Formant ratio.
        formant: [0.5, 2.0, 1.0, "x"],
        /// Output gain.
        gain_db: [-24.0, 24.0, 0.0, "dB"],
    }
}

impl WhisperParams {
    pub fn to_voice(self) -> VoiceParams {
        VoiceParams { whisper: self.amount, formant: self.formant, gain_db: self.gain_db, ..Default::default() }
    }
}

/// Engine tuning (experiments only; the defaults are what ships).
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(default)]
pub(crate) struct Tuning {
    pub cap_ms: f32,
    pub on: f32,
    pub off: f32,
    pub full: f32,
    pub min_lf: f32,
    pub v_tau_ms: f32,
    pub prox: f32,
    pub mag_tau_ms: f32,
    pub hard: bool,
    /// Track marks by waveform similarity with the previous period.
    pub xcorr: bool,
    /// Search radius (fraction of a period) for the similarity search.
    pub radius: f32,
    /// Use the tracked mark spacing as the input period.
    pub spacing: bool,
    /// Lag-window bandwidth as a multiple of f0 (0 = fixed `lag_min_hz`).
    pub lag_k: f32,
    pub lag_min_hz: f32,
    pub lag_max_hz: f32,
    /// Length of the binomial smoothing kernel of synthetic pulses (1, 3, 5, 7).
    pub pulse_taps: usize,
    /// Use the forward mark spacing and start voiced runs on a mark.
    pub forward: bool,
    pub two_period: bool,
    /// Snap a mark found by waveform similarity onto the glottal pulse (the
    /// residual peak) within this fraction of a period (0 = off). Off: it
    /// helps clean synthetic pulses (+4.7 dB of 1.5-5 kHz periodicity) but
    /// doubled the short-term pitch wobble of recorded voices, whose
    /// residual has several peaks per period (adaptive_tests::warble_of_recordings).
    pub snap: f32,
    /// Realism guard: a pitch mark more than this many semitones from the
    /// recent marks (a tracker error, often an octave, on noisy or creaky
    /// voice) takes its output pitch from them instead, for up to
    /// `guard_marks` marks; what persists is a real jump (0 = off).
    pub guard_st: f32,
    pub guard_marks: u8,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            // Grain half-width cap. 5 ms (the old 11 ms latency) cut every
            // grain of a voice under 200 Hz short of its period: the grains
            // no longer overlapped and the gaps pulsed at the pitch rate,
            // which was the "buzz" of the voice changer (a pass-through of
            // a 93 Hz voice: 2.6x the dry voice's pitch-rate modulation in
            // 2-6 kHz, +2.5 dB of 3-6 kHz). 12 ms covers voices down to
            // ~83 Hz: pass-through within 0.003 of the dry voice; relaxed-me
            // 0.172 -> 0.060 and James 0.152 -> 0.055 (dry 0.047). Latency
            // 25 ms instead of 11 (adaptive_tests::render_presets_of_recordings).
            cap_ms: 12.0,
            on: 0.35,
            off: 0.5,
            full: 0.25,
            min_lf: 0.02,
            v_tau_ms: 5.0,
            prox: 0.5,
            mag_tau_ms: 0.25,
            hard: true,
            xcorr: true,
            radius: 0.2,
            spacing: true,
            // 0.7 f0: formants move by the ratio asked for up to ~220 Hz voices
            // (1.3 left most of a woman's formants in the residual, where the
            // warp cannot move them: x1.24 moved them by 0 %), with the same
            // -35 dB of original f0 in the output (measured in adaptive_tests)
            lag_k: 0.7,
            lag_min_hz: 60.0,
            lag_max_hz: 350.0,
            pulse_taps: 5,
            forward: true,
            two_period: true,
            snap: 0.0,
            guard_st: 7.0,
            guard_marks: 6,
        }
    }
}

#[cfg(test)]
pub(crate) static TUNE: std::sync::Mutex<Option<Tuning>> = std::sync::Mutex::new(None);

fn tuning() -> Tuning {
    #[cfg(test)]
    if let Some(t) = *TUNE.lock().unwrap() {
        return t;
    }
    Tuning::default()
}

const PRE: f32 = 0.97;
const VOICES: usize = 5;
const FRAMES: usize = 64;
const MARKS: usize = 512;
const HANN_N: usize = 2048;

#[inline]
fn fidx(j: i64) -> usize {
    (j.max(0) as usize) % FRAMES
}

#[derive(Clone)]
struct Frame {
    k: Vec<f32>,
    kw: Vec<f32>,
    gain: f32,
    /// Soft voicing degree (0 = unvoiced).
    deg: f32,
}

#[derive(Clone, Copy, Default)]
struct Mark {
    pos: i64,
    period: f32,
    voiced: bool,
}

#[derive(Clone, Copy)]
struct VoiceState {
    /// Realism guard: log f0 of the recent marks, outliers in a row, primed.
    guard_ln: f32,
    guard_run: u8,
    guard_ready: bool,
    next_out: f64,
    /// The last grain was voiced.
    voiced: bool,
    drift: Drift,
    rng: Rng,
    alt: bool,
}

/// Grain half-width cap (seconds). Latency is twice this plus the guard.
const GUARD_S: f32 = 0.001;

pub struct Shifter {
    p: VoiceParams,
    rate: f32,
    order: usize,
    /// Order of the (warped) synthesis filter: higher than the analysis,
    /// because a warped all-pole envelope is no longer exactly all-pole.
    order_w: usize,
    hop: usize,
    cap: usize,
    guard: usize,
    delay: usize,
    mask: usize,
    n: i64,
    e: i64,
    xin: Vec<f32>,
    res: Vec<f32>,
    /// Smoothed |residual| for picking glottal closures.
    mag: Vec<f32>,
    /// Low-passed input for pitch-synchronous mark tracking.
    xl: Vec<f32>,
    xl_f: [super::filters::Biquad; 2],
    mag_s: f32,
    mag_a: f32,
    envr: Vec<f32>,
    acc: Vec<f32>,
    pre_prev: f32,
    env1: f32,
    env2: f32,
    env_a: f32,
    inv_b: Vec<f32>,
    syn_b: Vec<f32>,
    k_cur: Vec<f32>,
    k_syn: Vec<f32>,
    deemph: f32,
    frames: Vec<Frame>,
    // LPC analysis workspace
    window: Vec<f32>,
    wbuf: Vec<f32>,
    lagw: Vec<f64>,
    lag_fb: f32,
    r: Vec<f64>,
    a: Vec<f64>,
    kk: Vec<f64>,
    tmp: Vec<f64>,
    warper: Warper,
    pitch: PitchTracker,
    marks: Vec<Mark>,
    m_head: usize,
    m_len: usize,
    voices: [VoiceState; VOICES],
    hann: Vec<f32>,
    rng: Rng,
    v: f32,
    v_a: f32,
    flfo_phase: f32,
    out_gain: f32,
    noise_gain: f32,
    voice_gain: [f32; VOICES],
    voice_st: [f32; VOICES],
    last_out_f0: f32,
    /// Marks the realism guard corrected (for tests and meters).
    guarded: u32,
    pulse_k: Vec<f32>,
    pulse_norm: f32,
    tune: Tuning,
    // targets: the adaptive part of the shift, glided towards the profile
    adapt_st: f32,
    adapt_ln_alpha: f32,
    adapt_ln_f0: f32,
    adapt_ready: bool,
    // breath: one-pole splits of the voiced source (2 kHz) and the aspiration noise (1 kHz)
    br_lp: f32,
    br_c: f32,
    asp_lp: f32,
    asp_c: f32,
    // accent colour: smoothed formant ratios, the speaker's tract, phrase timing
    acc_r: [f32; 3],
    prof_tract: f32,
    run_start: i64,
    unvoiced_frames: u32,
}

impl Shifter {
    /// Fixed latency in samples at `rate`.
    pub fn latency_for(rate: f32) -> usize {
        let t = tuning();
        2 * (rate * t.cap_ms * 0.001).round() as usize + (rate * GUARD_S).round() as usize
    }

    pub fn new(p: VoiceParams, rate: f32) -> Self {
        let tune = tuning();
        let order = ((rate / 1500.0).round() as usize).clamp(10, 36);
        let hop = (rate * 0.005).round() as usize;
        let cap = (rate * tune.cap_ms * 0.001).round() as usize;
        let guard = (rate * GUARD_S).round() as usize;
        let size = ((rate * 0.35) as usize).next_power_of_two();
        let wlen = (rate * 0.020).round() as usize;
        // asymmetric window: long sine rise, short cosine fall ending at "now"
        let rise = wlen * 3 / 4;
        let fall = wlen - rise;
        let mut window = vec![0.0f32; wlen];
        for (i, w) in window.iter_mut().enumerate() {
            *w = if i < rise {
                (PI * 0.5 * i as f32 / rise as f32).sin().powi(2)
            } else {
                (PI * 0.5 * (i - rise) as f32 / fall as f32).cos()
            };
        }
        let lagw: Vec<f64> = (0..=order)
            .map(|i| {
                let x = 2.0 * std::f64::consts::PI * 60.0 * i as f64 / rate as f64;
                (-0.5 * x * x).exp()
            })
            .collect();
        let hann: Vec<f32> = (0..HANN_N).map(|i| 0.5 + 0.5 * (PI * i as f32 / (HANN_N - 1) as f32).cos()).collect();
        let order_w = order + 12;
        let frame = Frame { k: vec![0.0; order], kw: vec![0.0; order_w], gain: 1.0, deg: 0.0 };
        let mut pitch = PitchTracker::new(rate);
        pitch.on = tune.on;
        pitch.off = tune.off;
        pitch.full = tune.full;
        pitch.min_lf = tune.min_lf;
        pitch.two_period = tune.two_period;
        let delay = Self::latency_for(rate);
        let mut s = Shifter {
            p,
            rate,
            order,
            order_w,
            hop,
            cap,
            guard,
            delay,
            mask: size - 1,
            n: 0,
            e: -(delay as i64),
            xin: vec![0.0; size],
            res: vec![0.0; size],
            mag: vec![0.0; size],
            xl: vec![0.0; size],
            xl_f: [super::filters::Biquad::new(super::filters::Kind::LowPass, 1000.0, 0.707, 0.0, rate); 2],
            mag_s: 0.0,
            mag_a: coef(tune.mag_tau_ms * 0.001, rate),
            envr: vec![0.0; size],
            acc: vec![0.0; size],
            pre_prev: 0.0,
            env1: 0.0,
            env2: 0.0,
            env_a: coef(0.006, rate),
            inv_b: vec![0.0; order],
            syn_b: vec![0.0; order_w],
            k_cur: vec![0.0; order],
            k_syn: vec![0.0; order_w],
            deemph: 0.0,
            frames: vec![frame; FRAMES],
            window,
            wbuf: vec![0.0; wlen],
            lagw,
            lag_fb: 60.0,
            r: vec![0.0; order + 1],
            a: vec![0.0; order + 1],
            kk: vec![0.0; order],
            tmp: vec![0.0; order + 1],
            warper: Warper::with_preemphasis(order_w, PRE),
            pitch,
            marks: vec![Mark::default(); MARKS],
            m_head: 0,
            m_len: 0,
            voices: [VoiceState { guard_ln: 0.0, guard_run: 0, guard_ready: false, next_out: 0.0, voiced: false, drift: Drift::new(1), rng: Rng::new(1), alt: false }; VOICES],
            hann,
            rng: Rng::new(0x00c0_ffee),
            v: 0.0,
            v_a: coef(tune.v_tau_ms * 0.001, rate),
            flfo_phase: 0.0,
            out_gain: 1.0,
            noise_gain: 1.0,
            voice_gain: [0.0; VOICES],
            voice_st: [0.0; VOICES],
            last_out_f0: 0.0,
            guarded: 0,
            pulse_k: Vec::new(),
            pulse_norm: 1.0,
            tune,
            adapt_st: 0.0,
            adapt_ln_alpha: 0.0,
            adapt_ln_f0: super::profile::Profile::NEUTRAL.f0_hz.ln(),
            adapt_ready: false,
            br_lp: 0.0,
            br_c: 1.0 - (-2.0 * PI * 2000.0 / rate).exp(),
            asp_lp: 0.0,
            asp_c: 1.0 - (-2.0 * PI * 1000.0 / rate).exp(),
            acc_r: [1.0; 3],
            prof_tract: 1.0,
            run_start: 0,
            unvoiced_frames: u32::MAX / 2,
        };
        // binomial pulse kernel (row of Pascal's triangle), unit energy
        let taps = (tune.pulse_taps.clamp(1, 9) - 1) | 1;
        let mut row = vec![1.0f32];
        while row.len() < taps {
            let mut next = vec![1.0f32; row.len() + 1];
            for i in 1..row.len() {
                next[i] = row[i - 1] + row[i];
            }
            row = next;
        }
        let e: f32 = row.iter().map(|v| v * v).sum();
        s.pulse_norm = 1.0 / e.sqrt();
        s.pulse_k = row;
        for (i, v) in s.voices.iter_mut().enumerate() {
            v.drift = Drift::new(0x9e37_79b9u32.wrapping_mul(i as u32 + 1));
            v.rng = Rng::new(0x51ed_270bu32.wrapping_mul(i as u32 + 3));
        }
        s.update();
        s
    }

    fn update(&mut self) {
        let p = self.p;
        let g = |db: f32| if db <= -59.5 { 0.0 } else { db_to_lin(db) };
        self.voice_gain = [g(p.main_db), g(p.v1_db), g(p.v2_db), g(p.v3_db), g(p.v4_db)];
        self.voice_st = [0.0, p.v1_st, p.v2_st, p.v3_st, p.v4_st];
        self.out_gain = db_to_lin(p.gain_db);
        self.noise_gain = g(p.noise_db);
    }

    pub fn params(&self) -> VoiceParams {
        self.p
    }

    /// Reflection coefficients of the latest analysis frame: the estimated
    /// vocal-tract envelope and the formant-warped one used for synthesis
    /// (for analysis displays and tests).
    pub fn envelopes(&self) -> (Vec<f32>, Vec<f32>) {
        let j = (self.n / self.hop as i64 - 1).max(0);
        let f = &self.frames[fidx(j)];
        (f.k.clone(), f.kw.clone())
    }

    /// Latest input pitch estimate.
    pub fn input_pitch(&self) -> super::pitch::PitchEstimate {
        self.pitch.estimate()
    }

    fn analyze_frame(&mut self) {
        let j = self.n / self.hop as i64;
        // a phrase starts after 150 ms of unvoiced sound (for the accent's melody)
        if self.pitch.analyze().voiced {
            if self.unvoiced_frames as f32 * self.hop as f32 > 0.15 * self.rate {
                self.run_start = self.n;
            }
            self.unvoiced_frames = 0;
        } else {
            self.unvoiced_frames = self.unvoiced_frames.saturating_add(1);
        }
        let wlen = self.window.len();
        for i in 0..wlen {
            let pos = self.n - wlen as i64 + i as i64;
            self.wbuf[i] = if pos < 0 { 0.0 } else { self.xin[(pos as usize) & self.mask] * self.window[i] };
        }
        autocorr(&self.wbuf, &mut self.r);
        let slot = (j as usize) % FRAMES;
        let est = self.pitch.estimate();
        // Gaussian lag window = the power spectrum smoothed over `fb` Hz.
        // Tied to f0 so the all-pole envelope does not resolve single
        // harmonics (a strong first harmonic would become a resonance at
        // the ORIGINAL f0), but narrow enough to capture the formants:
        // what the envelope misses stays in the residual, and the warp
        // cannot move it (see `Tuning::lag_k`).
        let fb = if est.voiced { (self.tune.lag_k * est.f0).clamp(self.tune.lag_min_hz, self.tune.lag_max_hz) } else { self.tune.lag_min_hz };
        if (fb - self.lag_fb).abs() > 0.5 {
            self.lag_fb = fb;
            for (i, w) in self.lagw.iter_mut().enumerate() {
                let x = 2.0 * std::f64::consts::PI * fb as f64 * i as f64 / self.rate as f64;
                *w = (-0.5 * x * x).exp();
            }
        }
        for (r, w) in self.r.iter_mut().zip(self.lagw.iter()) {
            *r *= w;
        }
        // formant ratio for this frame (with wobble)
        let depth = self.p.formant_lfo_depth;
        let mut alpha = self.p.formant * self.adapt_ln_alpha.exp();
        if depth > 0.0 {
            alpha *= 1.0 + depth * (2.0 * PI * self.flfo_phase).sin();
        }
        self.flfo_phase += self.p.formant_lfo_hz * self.hop as f32 / self.rate;
        self.flfo_phase -= self.flfo_phase.floor();
        let alpha = alpha.clamp(0.35, 2.5);
        let deg = if self.tune.hard { if est.voiced { 1.0 } else { 0.0 } } else { est.periodicity };
        if self.r[0] < 1e-10 {
            let f = &mut self.frames[slot];
            f.k.fill(0.0);
            f.kw.fill(0.0);
            f.gain = 1.0;
            f.deg = 0.0;
            return;
        }
        self.r[0] *= 1.0 + 1e-4;
        levinson(&self.r, &mut self.a, &mut self.kk, &mut self.tmp);
        let f = &mut self.frames[slot];
        for (o, v) in f.k.iter_mut().zip(self.kk.iter()) {
            *o = *v as f32;
        }
        let accent = super::accent::accent_index(self.p.accent);
        f.gain = if accent == 0 || !est.voiced {
            // vowels only: noise (s, f, sh) has no formants to classify;
            // the next vowel's colour then fades in from none
            self.acc_r = [1.0; 3];
            self.warper.warp(&f.k, alpha, &mut f.kw)
        } else {
            let (amount, tract, rate) = (self.p.accent_amount, self.prof_tract, self.rate);
            let acc = &mut self.acc_r;
            let mut map = |pk: &[f32; 3], n: usize| {
                // measured in the speaker's own vowel space (before the tract
                // shift), smoothed over ~10 ms so a formant that flickers in
                // and out of a class does not warble
                let want = super::accent::vowel_ratios(accent, pk, n, tract, amount);
                for i in 0..3 {
                    acc[i] += (want[i] - acc[i]) * 0.4;
                }
                *acc
            };
            self.warper.warp_mapped(&f.k, alpha, rate, Some(&mut map), &mut f.kw)
        };
        f.deg = deg;
    }

    #[inline]
    fn ingest(&mut self, x: f32) {
        let xp = x - PRE * self.pre_prev;
        self.pre_prev = x;
        let idx = (self.n as usize) & self.mask;
        self.xin[idx] = xp;
        self.pitch.push(x);
        let l0 = self.xl_f[0].tick(x);
        self.xl[idx] = self.xl_f[1].tick(l0);
        // interpolated inverse filter at time n
        let j = self.n / self.hop as i64;
        let frac = (self.n - j * self.hop as i64) as f32 / self.hop as f32;
        {
            let (fa, fb) = (&self.frames[fidx(j - 1)], &self.frames[fidx(j)]);
            for i in 0..self.order {
                self.k_cur[i] = fa.k[i] + (fb.k[i] - fa.k[i]) * frac;
            }
        }
        let e = lattice_analyze(xp, &self.k_cur, &mut self.inv_b);
        self.res[idx] = e;
        self.mag_s += (e.abs() - self.mag_s) * self.mag_a;
        self.mag[idx] = self.mag_s;
        self.env1 += (e * e - self.env1) * self.env_a;
        self.env2 += (self.env1 - self.env2) * self.env_a;
        self.envr[idx] = self.env2.max(0.0).sqrt();
        self.n += 1;
    }

    #[inline]
    fn mark(&self, i: usize) -> Mark {
        self.marks[(self.m_head + i) % MARKS]
    }

    fn push_mark(&mut self, m: Mark) {
        if self.m_len == MARKS {
            self.m_head = (self.m_head + 1) % MARKS;
            self.m_len -= 1;
        }
        self.marks[(self.m_head + self.m_len) % MARKS] = m;
        self.m_len += 1;
    }

    /// Position in `lo..=hi` whose neighbourhood (±`hw`) best matches the
    /// neighbourhood of `anchor` (normalised cross-correlation of the
    /// low-passed input): coarse search in steps of 2, then refined.
    fn best_match(&self, anchor: i64, lo: i64, hi: i64, hw: i64) -> (i64, f32) {
        let m = self.mask;
        let xl = &self.xl;
        let at = |i: i64| xl[(i.max(0) as usize) & m];
        let mut ea = 0.0f32;
        for i in -hw..=hw {
            ea += at(anchor + i) * at(anchor + i);
        }
        if ea <= 1e-12 {
            return (lo.max(anchor + 1), 0.0);
        }
        let score = |c: i64| {
            let (mut xy, mut ec) = (0.0f32, 0.0f32);
            for i in -hw..=hw {
                let b = at(c + i);
                xy += at(anchor + i) * b;
                ec += b * b;
            }
            if ec <= 1e-12 { -1.0 } else { xy / (ea * ec).sqrt() }
        };
        let (mut best, mut bs) = (lo, f32::MIN);
        let mut c = lo;
        while c <= hi {
            let v = score(c);
            if v > bs {
                bs = v;
                best = c;
            }
            c += 2;
        }
        for c in [best - 1, best + 1] {
            if c >= lo && c <= hi {
                let v = score(c);
                if v > bs {
                    bs = v;
                    best = c;
                }
            }
        }
        (best.max(anchor + 1), bs)
    }

    fn update_marks(&mut self) {
        let est = self.pitch.estimate();
        if self.m_len == 0 {
            self.push_mark(Mark { pos: 0, period: self.cap as f32, voiced: false });
        }
        loop {
            let last = self.mark(self.m_len - 1);
            if est.voiced {
                let t0 = est.period.max(2.0);
                if last.voiced && self.tune.xcorr {
                    // pitch-synchronous tracking: the next mark is where the
                    // waveform best repeats the previous period, so marks keep
                    // one phase of the cycle and the re-spaced grains stay
                    // periodic (peak picking jittered by a sample or ten)
                    let expect = last.pos as f32 + t0;
                    let r = (self.tune.radius * t0).max(2.0) as i64;
                    let hw = (0.5 * t0).min(self.rate * 0.004).max(8.0) as i64;
                    let (lo, hi) = (expect as i64 - r, expect as i64 + r);
                    if hi + hw + 1 >= self.n {
                        break;
                    }
                    let (mut best, score) = self.best_match(last.pos, lo, hi, hw);
                    if score > 0.3 && self.tune.snap > 0.0 {
                        // (experiment, off by default: see `Tuning::snap`)
                        let rs = ((self.tune.snap * t0) as i64).max(2);
                        let (a, b) = ((best - rs).max(last.pos + 1), best + rs);
                        if b + 1 < self.n {
                            let inv = 1.0 / rs as f32;
                            let mut bv = -1.0f32;
                            let mut at = best;
                            for i in a..=b {
                                let d = (i - best) as f32 * inv;
                                let v = self.mag[(i.max(0) as usize) & self.mask] * (1.0 - 0.5 * d * d);
                                if v > bv {
                                    bv = v;
                                    at = i;
                                }
                            }
                            best = at;
                        }
                    }
                    if score > 0.3 {
                        let spacing = (best - last.pos) as f32;
                        let period = if self.tune.spacing { spacing.clamp(0.8 * t0, 1.25 * t0) } else { t0 };
                        self.push_mark(Mark { pos: best, period, voiced: true });
                        continue;
                    }
                    // no repetition found: fall back to the residual peak
                }
                let (lo, hi) = if last.voiced {
                    (last.pos as f32 + 0.7 * t0, last.pos as f32 + 1.3 * t0)
                } else {
                    (last.pos as f32 + 0.5 * t0, last.pos as f32 + 1.5 * t0)
                };
                let lo = lo.floor() as i64;
                let hi = hi.ceil() as i64;
                if hi + 1 >= self.n {
                    break;
                }
                // strongest residual peak, weighted towards where the last
                // period predicts it (continuity: no mark jitter)
                let expect = last.pos as f32 + t0;
                let prox = if last.voiced { self.tune.prox } else { 0.0 };
                let inv = 1.0 / (0.3 * t0);
                let mut best = lo;
                let mut bv = -1.0f32;
                for i in lo..=hi {
                    let d = (i as f32 - expect) * inv;
                    let v = self.mag[(i.max(0) as usize) & self.mask] * (1.0 - prox * d * d).max(0.05);
                    if v > bv {
                        bv = v;
                        best = i;
                    }
                }
                self.push_mark(Mark { pos: best.max(last.pos + 1), period: t0, voiced: true });
            } else {
                let t0 = (self.cap as f32).min(est.period.max(self.rate * 0.0025));
                let pos = last.pos + t0.round() as i64;
                if pos >= self.n {
                    break;
                }
                self.push_mark(Mark { pos, period: t0, voiced: false });
            }
        }
        // forget marks older than any grain can still need
        let keep_from = self.n - (self.delay as i64 + 4 * self.cap as i64 + (self.rate * 0.04) as i64);
        while self.m_len > 4 && self.mark(1).pos < keep_from {
            self.m_head = (self.m_head + 1) % MARKS;
            self.m_len -= 1;
        }
    }

    fn schedule(&mut self) {
        let limit = (self.n - self.cap as i64 - self.guard as i64) as f64;
        let p = self.p;
        for vi in 0..VOICES {
            if self.voice_gain[vi] == 0.0 {
                // keep idle voices in step so they start cleanly when enabled
                if self.voices[vi].next_out < limit {
                    self.voices[vi].next_out = limit;
                }
                continue;
            }
            while self.voices[vi].next_out <= limit {
                let m = self.voices[vi].next_out;
                // nearest input mark whose grain is fully available
                let mut best: Option<(usize, Mark)> = None;
                let mut bd = f64::MAX;
                for i in 0..self.m_len {
                    let mk = self.mark(i);
                    let half = (mk.period.min(self.cap as f32)).round() as i64;
                    if mk.pos + half > self.n {
                        break;
                    }
                    let d = (mk.pos as f64 - m).abs();
                    if d < bd {
                        bd = d;
                        best = Some((i, mk));
                    } else if (mk.pos as f64) > m {
                        break;
                    }
                }
                let Some((bi, mk)) = best else {
                    self.voices[vi].next_out += self.cap as f64;
                    self.voices[vi].voiced = false;
                    continue;
                };
                if !mk.voiced {
                    // unvoiced: the noise path carries the sound; no grain
                    self.voices[vi].next_out += mk.period.max(2.0) as f64;
                    self.voices[vi].voiced = false;
                    continue;
                }
                // the input period is the distance to the NEXT mark, so at
                // ratio 1 the output grid lands on the marks exactly and the
                // grains overlap in phase (the backward spacing let the grid
                // drift against the marks and comb-filter the overlaps)
                let mut t0i = mk.period.max(2.0);
                if self.tune.forward && bi + 1 < self.m_len {
                    let nx = self.mark(bi + 1);
                    if nx.voiced {
                        t0i = ((nx.pos - mk.pos) as f32).clamp(0.7 * t0i, 1.4 * t0i);
                    }
                }
                // a voiced run starts on its first mark (zero offset)
                if self.tune.forward && !self.voices[vi].voiced && (mk.pos as f64) > m {
                    self.voices[vi].next_out = mk.pos as f64;
                }
                if !self.voices[vi].voiced {
                    // a new voiced run: the guard starts from its first mark
                    self.voices[vi].guard_ready = false;
                }
                self.voices[vi].voiced = true;
                let m = self.voices[vi].next_out;
                let t = m as f32 / self.rate;
                let vib = if p.vibrato_st > 0.0 { p.vibrato_st * (2.0 * PI * p.vibrato_hz * t).sin() } else { 0.0 };
                let jit = if p.jitter_cents > 0.0 { p.jitter_cents / 100.0 * self.voices[vi].drift.tick(0.08) } else { 0.0 };
                let f_in = {
                    let raw = self.rate / t0i;
                    let g = self.tune.guard_st;
                    let vs = &mut self.voices[vi];
                    let lf = raw.ln();
                    if g <= 0.0 || !vs.guard_ready {
                        (vs.guard_ln, vs.guard_run, vs.guard_ready) = (lf, 0, true);
                        raw
                    } else if (12.0 * (lf - vs.guard_ln) / std::f32::consts::LN_2).abs() > g && vs.guard_run < self.tune.guard_marks {
                        // an outlier: sing the recent pitch, not the misread one
                        vs.guard_run += 1;
                        self.guarded = self.guarded.wrapping_add(1);
                        vs.guard_ln.exp()
                    } else {
                        if vs.guard_run >= self.tune.guard_marks {
                            // it lasted: a real jump, the new reference
                            vs.guard_ln = lf;
                        }
                        vs.guard_run = 0;
                        vs.guard_ln += (lf - vs.guard_ln) * 0.3;
                        raw
                    }
                };
                let st = self.voice_st[vi] + vib + jit;
                // with a pitch target: the shift that takes this speaker's
                // median to the target, and the melody around it scaled by `expr`
                let accent = super::accent::accent_index(p.accent);
                let adapt = if p.target_hz > 0.0 || accent > 0 || (p.expr - 1.0).abs() > 1e-3 {
                    let mel = super::accent::melody(accent, p.accent_amount);
                    let melody = 12.0 * (f_in.ln() - self.adapt_ln_f0) / std::f32::consts::LN_2;
                    let expr = p.expr * mel.range;
                    // time into the phrase at this grain (output clock)
                    let t_run = ((m - self.run_start as f64) / self.rate as f64).max(0.0) as f32;
                    self.adapt_st + ((expr - 1.0) * melody).clamp(-12.0, 12.0) + mel.offset(t_run)
                } else {
                    0.0
                };
                let f_rel = f_in * 2f32.powf((p.pitch_st + adapt + st) / 12.0);
                let f_out = if p.flatten > 0.0 {
                    let f_fix = p.fixed_hz * 2f32.powf(st / 12.0);
                    (f_rel.ln() * (1.0 - p.flatten) + f_fix.ln() * p.flatten).exp()
                } else {
                    f_rel
                };
                let mut t0o = (self.rate / f_out).clamp(self.rate / 2000.0, self.rate / 25.0);
                if vi == 0 {
                    self.last_out_f0 = self.rate / t0o;
                }
                let mut g = self.voice_gain[vi] * (t0o / t0i).sqrt().clamp(0.3, 2.5);
                if p.growl > 0.0 {
                    // irregular period doubling (alternating strong/weak
                    // pulses that sometimes skip a beat), shimmer and jitter:
                    // a vocal-fry growl on voiced sound only
                    let vs = &mut self.voices[vi];
                    if vs.rng.unit() > 0.3 * p.growl {
                        vs.alt = !vs.alt;
                    }
                    let sub = if vs.alt { 1.0 - 0.8 * p.growl } else { 1.0 };
                    let shimmer = 1.0 + 0.45 * p.growl * vs.rng.bipolar();
                    g *= sub * shimmer / (1.0 - 0.4 * p.growl);
                    t0o *= 1.0 + 0.035 * p.growl * vs.rng.bipolar();
                }
                let center = m.round() as i64;
                // overlap-add the Hann-windowed residual grain centred on the mark
                let gr = g * (1.0 - p.pulse);
                if gr != 0.0 {
                    let half = (t0i.min(self.cap as f32)).round().max(2.0) as i64;
                    let scale = (HANN_N - 1) as f32 / half as f32;
                    for i in (1 - half)..half {
                        let w = self.hann[((i.unsigned_abs() as f32) * scale) as usize];
                        let src = self.res[((mk.pos + i) as usize) & self.mask];
                        let dst = ((center + i) as usize) & self.mask;
                        self.acc[dst] += gr * w * src;
                    }
                }
                if p.pulse > 0.0 {
                    // a flat pulse with the residual's energy per output
                    // period, binomially smoothed (-3 dB near 6 kHz at 48 kHz,
                    // so it buzzes less than a bare impulse) and split between
                    // two samples for exact timing
                    let rms = self.envr[(mk.pos as usize) & self.mask];
                    let amp = g * p.pulse * rms * t0o.sqrt() * self.pulse_norm;
                    let fr = (m - m.floor()) as f32;
                    let c = m.floor() as i64 - (self.pulse_k.len() / 2) as i64;
                    for (j, w) in self.pulse_k.iter().enumerate() {
                        let i0 = ((c + j as i64) as usize) & self.mask;
                        let i1 = ((c + j as i64 + 1) as usize) & self.mask;
                        self.acc[i0] += amp * w * (1.0 - fr);
                        self.acc[i1] += amp * w * fr;
                    }
                }
                self.voices[vi].next_out += t0o as f64;
            }
        }
    }

    /// Move the adaptive shift towards what takes `ctx.profile` to the targets.
    fn adapt(&mut self, ctx: &Ctx, n: usize) {
        let p = &self.p;
        self.prof_tract = ctx.profile.tract;
        if p.target_hz <= 0.0 && p.target_tract <= 0.0 && (super::accent::accent_index(p.accent) > 0 || (p.expr - 1.0).abs() > 1e-3) {
            // an accent or an intonation change without targets (a voice
            // relative to yours) still needs the speaker's median: the melody
            // is scaled around it
            let a = 1.0 - (-(n as f32) / (ADAPT_TAU_S * self.rate)).exp();
            self.adapt_ln_f0 += (ctx.profile.f0_hz.ln() - self.adapt_ln_f0) * a;
            self.adapt_st = 0.0;
            self.adapt_ln_alpha = 0.0;
            return;
        }
        if p.target_hz <= 0.0 && p.target_tract <= 0.0 {
            self.adapt_st = 0.0;
            self.adapt_ln_alpha = 0.0;
            self.adapt_ready = false;
            return;
        }
        let prof = ctx.profile;
        let want_st = if p.target_hz > 0.0 { (12.0 * (p.target_hz / prof.f0_hz).log2()).clamp(-30.0, 30.0) } else { 0.0 };
        let want_ln_alpha = if p.target_tract > 0.0 { (p.target_tract / prof.tract).ln().clamp(-0.8, 0.8) } else { 0.0 };
        let want_ln_f0 = prof.f0_hz.ln();
        if !self.adapt_ready {
            // start where the profile already points (a calibration, or neutral)
            self.adapt_st = want_st;
            self.adapt_ln_alpha = want_ln_alpha;
            self.adapt_ln_f0 = want_ln_f0;
            self.adapt_ready = true;
            return;
        }
        let tau = if prof.voiced_secs < PROFILE_LEARNT_S { ADAPT_TAU_LEARNING_S } else { ADAPT_TAU_S };
        let a = 1.0 - (-(n as f32) / (tau * self.rate)).exp();
        self.adapt_st += (want_st - self.adapt_st) * a;
        self.adapt_ln_alpha += (want_ln_alpha - self.adapt_ln_alpha) * a;
        self.adapt_ln_f0 += (want_ln_f0 - self.adapt_ln_f0) * a;
    }

    /// Pitch marks the realism guard has corrected so far.
    pub fn guarded(&self) -> u32 {
        self.guarded
    }

    /// The adaptive part of the shift now: semitones and formant ratio (for meters and tests).
    pub fn adaptive_shift(&self) -> (f32, f32) {
        (self.adapt_st, self.adapt_ln_alpha.exp())
    }

    #[inline]
    fn emit(&mut self) -> f32 {
        let p = self.e;
        self.e += 1;
        if p < 0 {
            return 0.0;
        }
        let idx = (p as usize) & self.mask;
        let a = self.acc[idx];
        self.acc[idx] = 0.0;
        let j = p / self.hop as i64;
        let frac = (p - j * self.hop as i64) as f32 / self.hop as f32;
        let (deg, gain) = {
            let (fa, fb) = (&self.frames[fidx(j - 1)], &self.frames[fidx(j)]);
            for i in 0..self.order_w {
                self.k_syn[i] = fa.kw[i] + (fb.kw[i] - fa.kw[i]) * frac;
            }
            (fb.deg, fa.gain + (fb.gain - fa.gain) * frac)
        };
        self.v += (deg - self.v) * self.v_a;
        let env = self.envr[idx];
        let noise = self.rng.bipolar() * 1.732_050_8 * env;
        let v = self.v;
        let w = self.p.whisper;
        let bn = self.p.breath;
        // grains and noise cross at equal power (they are uncorrelated);
        // whisper replaces the voiced part with noise
        let nv = (1.0 - v * v).max(0.0).sqrt();
        let voiced = (1.0 - w) * v;
        // Breath: a breathy source closes softly, so it loses its upper
        // harmonics (up to -8 dB above ~2 kHz), and aspiration noise rides
        // on it, mostly above ~1 kHz (full-band noise sounded like hiss and
        // rumble and barely changed the voice: -0.6 dB of 1.5-5 kHz HNR at 0.3).
        let (mut a, mut asp) = (a, 0.0);
        if bn > 0.0 {
            self.br_lp += (a - self.br_lp) * self.br_c;
            a -= bn * 0.6 * (a - self.br_lp);
            self.asp_lp += (noise - self.asp_lp) * self.asp_c;
            asp = (noise - self.asp_lp) * 1.6 * bn * voiced;
        }
        let exc = voiced * a + noise * ((1.0 - w) * nv * self.noise_gain + w) + asp;
        let y = lattice_synth(exc, &self.k_syn, &mut self.syn_b) * gain;
        let out = y + PRE * self.deemph;
        self.deemph = out;
        out * self.out_gain
    }
}

impl Block for Shifter {
    fn process(&mut self, buf: &mut [f32], ctx: &mut Ctx) {
        self.adapt(ctx, buf.len());
        let hop = self.hop as i64;
        let mut i = 0;
        while i < buf.len() {
            if self.n % hop == 0 {
                self.analyze_frame();
            }
            let to_boundary = (hop - self.n % hop) as usize;
            let seg = to_boundary.min(buf.len() - i);
            for &x in &buf[i..i + seg] {
                self.ingest(x);
            }
            i += seg;
            // keep marks and grains current at hop granularity so that large
            // host buffers never outrun the mark/grain rings
            self.update_marks();
            self.schedule();
        }
        for o in buf.iter_mut() {
            *o = self.emit();
        }
        let est = self.pitch.estimate();
        ctx.pitch_hz = if est.voiced { est.f0 } else { 0.0 };
        ctx.out_pitch_hz = if est.voiced { self.last_out_f0 } else { 0.0 };
    }

    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }

    fn latency(&self) -> usize {
        self.delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_is_twenty_five_ms() {
        // 2 × 12 ms grain cap + 1 ms guard
        assert_eq!(Shifter::latency_for(48000.0), 1200);
        assert_eq!(Shifter::latency_for(44100.0), 1102);
        let s = Shifter::new(VoiceParams::default(), 48000.0);
        assert_eq!(s.latency(), 1200);
    }

    #[test]
    fn silence_stays_silent() {
        let mut s = Shifter::new(VoiceParams { pitch_st: 5.0, whisper: 0.5, growl: 0.5, pulse: 0.5, breath: 0.5, ..Default::default() }, 48000.0);
        let mut b = vec![0.0f32; 48000];
        for c in b.chunks_mut(256) {
            s.process(c, &mut Ctx::default());
        }
        assert!(b.iter().all(|v| *v == 0.0));
    }
}
