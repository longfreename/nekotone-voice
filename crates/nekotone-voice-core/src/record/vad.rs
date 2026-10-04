//! Voice activity detection on 16 kHz mono: cuts a live stream into
//! utterances for the transcriber.
//!
//! Each 20 ms frame is speech when all of these hold:
//! - its energy is `snr_db` above an adaptive noise floor (the floor falls
//!   quickly and rises slowly, and only on frames judged not to be speech);
//! - most of its energy lies in the voice band (250–4000 Hz);
//! - its spectrum in that band is *not* flat (speech has harmonics and
//!   formants; fans, hiss and rain are flat).
//!
//! An utterance starts after `onset_frames` speech frames in a row (with
//! `pre_roll` of audio before it, so the first consonant is not clipped),
//! ends after `hangover` of non-speech, is dropped when it holds less than
//! `min_speech` of speech, and is cut at a pause after `soft_max` (hard at
//! `hard_max`, below Whisper's 30 s window).
//!
//! With `seamless_cuts`, a cut (soft or hard) opens the next utterance at
//! once, so the speech after it is kept whole: no onset wait, no 0.3 s
//! minimum for the piece after a cut. A soft cut then needs a gap between
//! words (`soft_gap_frames` of non-speech), not a single quiet frame.

use realfft::RealFftPlanner;
use std::sync::Arc;

pub(crate) const RATE: usize = 16_000;
const FRAME: usize = 320; // 20 ms
const FFT: usize = 512;

/// Tuning of the detector.
#[derive(Debug, Clone)]
pub(crate) struct VadConfig {
    pub snr_db: f32,
    pub min_level_db: f32,
    pub band_ratio: f32,
    pub max_flatness: f32,
    pub onset_frames: usize,
    pub hangover_secs: f32,
    pub pre_roll_secs: f32,
    pub min_speech_secs: f32,
    pub soft_max_secs: f32,
    pub hard_max_secs: f32,
    /// Non-speech frames in a row that make a soft cut (after `soft_max`).
    pub soft_gap_frames: usize,
    /// A cut carries straight on into the next utterance (no audio lost).
    pub seamless_cuts: bool,
    /// Learn the speaker's pauses and set the hangover from them (see
    /// [`PauseLearner`]); `hangover_secs` is the start value.
    pub adapt_hangover: Option<PauseLearner>,
}

impl Default for VadConfig {
    fn default() -> Self {
        VadConfig {
            snr_db: 9.0,
            min_level_db: -62.0,
            band_ratio: 0.55,
            max_flatness: 0.4,
            onset_frames: 3,
            hangover_secs: 0.35,
            pre_roll_secs: 0.2,
            min_speech_secs: 0.3,
            soft_max_secs: 15.0,
            hard_max_secs: 28.0,
            soft_gap_frames: 1,
            seamless_cuts: false,
            adapt_hangover: None,
        }
    }
}

impl VadConfig {
    /// Live captions: shorter pauses end a line (300 ms) and lines are cut
    /// sooner (at a pause after 4 s, always by 8 s), so each phrase reaches
    /// the transcriber ~0.15 s after it ends instead of waiting for a long
    /// sentence to finish.
    pub fn captions() -> VadConfig {
        VadConfig { hangover_secs: 0.3, soft_max_secs: 4.0, hard_max_secs: 8.0, ..VadConfig::default() }
    }
}

/// A stretch of speech: 16 kHz sample positions on the source's timeline.
#[derive(Debug, Clone)]
pub(crate) struct Utterance {
    pub start: u64,
    pub end: u64,
    pub samples: Vec<f32>,
}

impl Utterance {
    pub fn start_secs(&self) -> f64 {
        self.start as f64 / RATE as f64
    }
    pub fn end_secs(&self) -> f64 {
        self.end as f64 / RATE as f64
    }
}

/// Per-frame measurements (exposed for tests and tuning).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrameInfo {
    pub energy_db: f32,
    pub band_ratio: f32,
    pub flatness: f32,
    pub speech: bool,
}

/// Learns how long a speaker pauses and picks the silence that ends what
/// they say: the `quantile` of their recent pauses (from 60 ms to 3 s, inside
/// phrases and between them), within `min_secs..max_secs`. Most pauses are
/// inside a sentence (measured on the owner: 70 % under 250 ms, 11 % over
/// 500 ms), so a high quantile waits through them and ends at the long ones,
/// whether the speaker is quick or slow.
#[derive(Debug, Clone)]
pub(crate) struct PauseLearner {
    pub quantile: f32,
    pub min_secs: f32,
    pub max_secs: f32,
    /// Pauses heard before the learnt value is used.
    pub warm_up: usize,
    /// How many recent pauses count.
    pub window: usize,
    pauses: std::collections::VecDeque<f32>,
}

impl PauseLearner {
    pub fn new(quantile: f32) -> PauseLearner {
        PauseLearner { quantile, min_secs: 0.25, max_secs: 0.9, warm_up: 15, window: 120, pauses: Default::default() }
    }

    pub fn observe(&mut self, secs: f32) {
        if (0.06..=3.0).contains(&secs) {
            self.pauses.push_back(secs);
            while self.pauses.len() > self.window {
                self.pauses.pop_front();
            }
        }
    }

    /// The learnt hangover, or `start` until enough pauses were heard.
    pub fn hangover(&self, start: f32) -> f32 {
        if self.pauses.len() < self.warm_up {
            return start;
        }
        let mut v: Vec<f32> = self.pauses.iter().copied().collect();
        v.sort_by(f32::total_cmp);
        let i = ((v.len() - 1) as f32 * self.quantile.clamp(0.0, 1.0)).round() as usize;
        v[i].clamp(self.min_secs, self.max_secs)
    }

    pub fn heard(&self) -> usize {
        self.pauses.len()
    }
}

pub(crate) struct Vad {
    cfg: VadConfig,
    /// End of the last speech frame (absolute sample position), for pauses.
    last_speech_end: Option<u64>,
    /// The hangover in use (learnt or configured), seconds.
    hangover_now: f32,
    fft: Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<realfft::num_complex::Complex<f32>>,
    /// Pre-emphasised power spectrum of the previous frame (averaged with
    /// the current one for a steadier flatness).
    prev_spec: Vec<f64>,
    last_sample: f32,
    pending: Vec<f32>,
    /// Absolute sample position of `pending[0]`.
    pos: u64,
    floor_db: Option<f32>,
    /// Recent audio for the pre-roll.
    history: std::collections::VecDeque<f32>,
    run: usize,
    active: Option<Active>,
}

struct Active {
    start: u64,
    samples: Vec<f32>,
    speech_frames: usize,
    silence_frames: usize,
    /// Opened by a seamless cut: the rest of a phrase.
    continued: bool,
}

impl Vad {
    pub fn new(cfg: VadConfig) -> Vad {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT);
        let window = (0..FRAME).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / FRAME as f32).cos()).collect();
        let scratch_in = fft.make_input_vec();
        let scratch_out = fft.make_output_vec();
        Vad {
            fft,
            window,
            scratch_in,
            scratch_out,
            prev_spec: vec![0.0; FFT / 2 + 1],
            last_sample: 0.0,
            pending: vec![],
            pos: 0,
            floor_db: None,
            history: Default::default(),
            run: 0,
            active: None,
            last_speech_end: None,
            hangover_now: cfg.hangover_secs,
            cfg,
        }
    }

    /// The silence that ends an utterance now (seconds): learnt or configured.
    pub fn hangover_secs(&self) -> f32 {
        self.hangover_now
    }

    /// Pauses learnt from so far (0 without adaptation).
    pub fn pauses_heard(&self) -> usize {
        self.cfg.adapt_hangover.as_ref().map(|l| l.heard()).unwrap_or(0)
    }

    /// Where the next sample fed will sit (16 kHz samples).
    #[allow(dead_code)]
    pub fn position(&self) -> u64 {
        self.pos + self.pending.len() as u64
    }

    /// True while an utterance is open (speech started, end not yet found).
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Feed 16 kHz mono audio; finished utterances are appended to `out`.
    pub fn feed(&mut self, samples: &[f32], out: &mut Vec<Utterance>) {
        self.pending.extend_from_slice(samples);
        let mut off = 0;
        while self.pending.len() - off >= FRAME {
            let frame: Vec<f32> = self.pending[off..off + FRAME].to_vec();
            let info = self.analyse(&frame);
            self.step(&frame, info, out);
            off += FRAME;
            self.pos += FRAME as u64;
        }
        self.pending.drain(..off);
    }

    /// End of input (stop or pause): close the current utterance.
    pub fn flush(&mut self, out: &mut Vec<Utterance>) {
        let rest = std::mem::take(&mut self.pending);
        if let Some(a) = self.active.as_mut() {
            a.samples.extend_from_slice(&rest);
        }
        self.pos += rest.len() as u64;
        self.close(out);
        self.run = 0;
        self.history.clear();
    }

    /// Measure one frame (also used by the tests).
    pub fn analyse(&mut self, frame: &[f32]) -> FrameInfo {
        let energy = frame.iter().map(|x| x * x).sum::<f32>() / frame.len().max(1) as f32;
        let energy_db = 10.0 * (energy + 1e-12).log10();
        let hz_per_bin = RATE as f32 / FFT as f32;
        // Raw spectrum: how much energy sits in the voice band.
        self.scratch_in.iter_mut().for_each(|v| *v = 0.0);
        for (i, (&x, &w)) in frame.iter().zip(&self.window).enumerate() {
            self.scratch_in[i] = x * w;
        }
        let _ = self.fft.process(&mut self.scratch_in, &mut self.scratch_out);
        let (mut band, mut total) = (0f64, 0f64);
        for (b, c) in self.scratch_out.iter().enumerate() {
            let hz = b as f32 * hz_per_bin;
            let p = (c.re * c.re + c.im * c.im) as f64;
            if (60.0..=7800.0).contains(&hz) {
                total += p;
            }
            if (250.0..=4000.0).contains(&hz) {
                band += p;
            }
        }
        let band_ratio = if total > 0.0 { (band / total) as f32 } else { 0.0 };
        // Pre-emphasised spectrum (tilt removed), averaged over two frames:
        // noise of any colour comes out flat, voiced speech keeps its
        // harmonic peaks.
        self.scratch_in.iter_mut().for_each(|v| *v = 0.0);
        let mut prev = self.last_sample;
        for (i, (&x, &w)) in frame.iter().zip(&self.window).enumerate() {
            self.scratch_in[i] = (x - 0.97 * prev) * w;
            prev = x;
        }
        self.last_sample = prev;
        let _ = self.fft.process(&mut self.scratch_in, &mut self.scratch_out);
        let (mut log_sum, mut lin_sum, mut n) = (0f64, 0f64, 0usize);
        for (b, c) in self.scratch_out.iter().enumerate() {
            let hz = b as f32 * hz_per_bin;
            let p = (c.re * c.re + c.im * c.im) as f64 + 1e-14;
            let avg = 0.5 * (p + self.prev_spec[b]);
            self.prev_spec[b] = p;
            if (250.0..=4000.0).contains(&hz) {
                log_sum += (avg + 1e-14).ln();
                lin_sum += avg;
                n += 1;
            }
        }
        let flatness = if n > 0 && lin_sum > 0.0 { ((log_sum / n as f64).exp() / (lin_sum / n as f64)) as f32 } else { 1.0 };
        let floor = *self.floor_db.get_or_insert(energy_db.max(-90.0));
        let speech = energy_db > floor + self.cfg.snr_db
            && energy_db > self.cfg.min_level_db
            && band_ratio > self.cfg.band_ratio
            && flatness < self.cfg.max_flatness;
        // Noise floor: fast down, slow up, only on non-speech frames.
        let f = if energy_db < floor {
            0.7 * floor + 0.3 * energy_db
        } else if !speech {
            floor + ((energy_db - floor) * 0.02).min(0.1)
        } else {
            floor
        };
        self.floor_db = Some(f.max(-100.0));
        FrameInfo { energy_db, band_ratio, flatness, speech }
    }

    fn step(&mut self, frame: &[f32], info: FrameInfo, out: &mut Vec<Utterance>) {
        let frame_secs = FRAME as f32 / RATE as f32;
        if info.speech {
            // a pause just ended: learn it (inside a phrase or between phrases)
            if let (Some(end), Some(l)) = (self.last_speech_end, self.cfg.adapt_hangover.as_mut()) {
                if self.pos > end {
                    l.observe((self.pos - end) as f32 / RATE as f32);
                    self.hangover_now = l.hangover(self.cfg.hangover_secs);
                }
            }
            self.last_speech_end = Some(self.pos + FRAME as u64);
        }
        let hang = (self.hangover_now / frame_secs).ceil() as usize;
        match self.active.as_mut() {
            Some(a) => {
                a.samples.extend_from_slice(frame);
                if info.speech {
                    a.speech_frames += 1;
                    a.silence_frames = 0;
                } else {
                    a.silence_frames += 1;
                }
                let len_secs = a.samples.len() as f32 / RATE as f32;
                let pause = a.silence_frames >= hang;
                let cut = (len_secs >= self.cfg.soft_max_secs && a.silence_frames >= self.cfg.soft_gap_frames.max(1))
                    || len_secs >= self.cfg.hard_max_secs;
                if pause {
                    self.close(out);
                } else if cut {
                    self.close(out);
                    if self.cfg.seamless_cuts {
                        let start = self.pos + FRAME as u64;
                        self.active = Some(Active { start, samples: Vec::new(), speech_frames: 0, silence_frames: 0, continued: true });
                        return;
                    }
                }
            }
            None => {
                if info.speech {
                    self.run += 1;
                } else {
                    self.run = 0;
                }
                if self.run >= self.cfg.onset_frames {
                    // Start: pre-roll + the onset frames (already in history) + this frame.
                    let pre = (self.cfg.pre_roll_secs * RATE as f32) as usize + (self.run - 1) * FRAME;
                    let take = pre.min(self.history.len());
                    let mut samples: Vec<f32> = self.history.iter().skip(self.history.len() - take).copied().collect();
                    samples.extend_from_slice(frame);
                    let start = self.pos - take as u64;
                    self.active = Some(Active { start, samples, speech_frames: self.run, silence_frames: 0, continued: false });
                    self.run = 0;
                    self.history.clear();
                    return;
                }
            }
        }
        if self.active.is_none() {
            self.history.extend(frame.iter().copied());
            let cap = (self.cfg.pre_roll_secs * RATE as f32) as usize + self.cfg.onset_frames * FRAME + FRAME;
            while self.history.len() > cap {
                self.history.pop_front();
            }
        }
    }

    fn close(&mut self, out: &mut Vec<Utterance>) {
        let Some(mut a) = self.active.take() else { return };
        let frame_secs = FRAME as f32 / RATE as f32;
        // Trim trailing silence beyond a short tail.
        let keep_tail = (0.15 / frame_secs) as usize;
        if a.silence_frames > keep_tail {
            let cut = (a.silence_frames - keep_tail) * FRAME;
            let n = a.samples.len().saturating_sub(cut);
            a.samples.truncate(n);
        }
        // the rest of a cut phrase is kept however short (a last word)
        if a.speech_frames as f32 * frame_secs >= self.cfg.min_speech_secs || (a.continued && a.speech_frames >= 2) {
            let end = a.start + a.samples.len() as u64;
            out.push(Utterance { start: a.start, end, samples: a.samples });
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic noise in -1..1.
    pub fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Pink-ish noise (a leaky integrator on white noise).
    pub fn pink(n: usize, seed: u64) -> Vec<f32> {
        let w = noise(n, seed);
        let mut y = 0f32;
        let mut out: Vec<f32> = w
            .iter()
            .map(|x| {
                y = 0.97 * y + 0.03 * x * 4.0;
                y
            })
            .collect();
        let peak = out.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        out.iter_mut().for_each(|v| *v /= peak);
        out
    }

    /// A speech-like burst: a glottal buzz (harmonics of a gliding f0)
    /// shaped by three formants, amplitude-modulated at a syllable rate.
    pub fn speechish(secs: f32, rate: usize, amp: f32, seed: u32) -> Vec<f32> {
        let n = (secs * rate as f32) as usize;
        let formants = [(500.0f32, 90.0f32), (1500.0, 120.0), (2500.0, 160.0)];
        let mut phase = 0f32;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / rate as f32;
            let f0 = 130.0 + 25.0 * (2.0 * std::f32::consts::PI * (0.7 + seed as f32 * 0.1) * t).sin();
            phase += f0 / rate as f32;
            let mut v = 0.0;
            let mut h = 1;
            while (h as f32) * f0 < 4000.0 {
                let hz = h as f32 * f0;
                let env: f32 = formants.iter().map(|(fc, bw)| 1.0 / (1.0 + ((hz - fc) / bw).powi(2))).sum::<f32>() + 0.02;
                v += env * (2.0 * std::f32::consts::PI * phase * h as f32).sin();
                h += 1;
            }
            let syll = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
            out.push(v * syll);
        }
        let peak = out.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        out.iter_mut().for_each(|v| *v *= amp / peak);
        out
    }

    #[test]
    fn finds_speech_bursts_in_noise_with_tight_boundaries() {
        // 1 s room noise, then 3 × (1.5 s speech + 1.2 s pause).
        let mut x: Vec<f32> = noise(RATE * 10, 7).iter().map(|v| v * 0.003).collect();
        let bursts = [(1.0f32, 2.5f32), (3.7, 5.2), (6.4, 7.9)];
        for (k, (a, b)) in bursts.iter().enumerate() {
            let s = speechish(b - a, RATE, 0.3, k as u32);
            let off = (a * RATE as f32) as usize;
            for (i, v) in s.iter().enumerate() {
                x[off + i] += v;
            }
        }
        let mut vad = Vad::new(VadConfig::default());
        let mut utts = vec![];
        for chunk in x.chunks(777) {
            vad.feed(chunk, &mut utts);
        }
        vad.flush(&mut utts);
        assert_eq!(utts.len(), 3, "{:?}", utts.iter().map(|u| (u.start_secs(), u.end_secs())).collect::<Vec<_>>());
        for (u, (a, b)) in utts.iter().zip(bursts) {
            assert!((u.start_secs() - a as f64).abs() < 0.25, "start {} vs {a}", u.start_secs());
            assert!(u.end_secs() >= b as f64 - 0.15 && u.end_secs() < b as f64 + 0.45, "end {} vs {b}", u.end_secs());
            assert_eq!(u.samples.len() as u64, u.end - u.start);
        }
    }

    #[test]
    fn ignores_steady_noise() {
        for (name, sig) in [("white", noise(RATE * 8, 3)), ("pink", pink(RATE * 8, 5))] {
            // Quiet room first, then the noise comes on loudly (a fan, rain).
            let mut x: Vec<f32> = noise(RATE, 11).iter().map(|v| v * 0.002).collect();
            x.extend(sig.iter().map(|v| v * 0.2));
            let mut vad = Vad::new(VadConfig::default());
            let mut utts = vec![];
            vad.feed(&x, &mut utts);
            vad.flush(&mut utts);
            let secs: f64 = utts.iter().map(|u| u.end_secs() - u.start_secs()).sum();
            assert!(secs < 0.6, "{name} noise gave {} utterances, {secs:.2} s", utts.len());
        }
    }

    #[test]
    fn long_speech_is_cut_below_whisper_window() {
        let x = speechish(40.0, RATE, 0.3, 1);
        let mut vad = Vad::new(VadConfig::default());
        let mut utts = vec![];
        vad.feed(&x, &mut utts);
        vad.flush(&mut utts);
        assert!(utts.len() >= 2);
        assert!(utts.iter().all(|u| u.end_secs() - u.start_secs() <= 28.05));
    }

    #[test]
    fn seamless_cuts_lose_nothing_between_pieces() {
        let x = speechish(30.0, RATE, 0.3, 1);
        let cfg = VadConfig { hangover_secs: 0.5, soft_max_secs: 2.5, hard_max_secs: 6.0, soft_gap_frames: 3, seamless_cuts: true, ..VadConfig::default() };
        let mut vad = Vad::new(cfg);
        let mut utts = vec![];
        vad.feed(&x, &mut utts);
        vad.flush(&mut utts);
        assert!(utts.len() >= 5, "{} pieces", utts.len());
        assert!(utts.iter().all(|u| u.end_secs() - u.start_secs() <= 6.05));
        let cut = |u: &Utterance| u.end_secs() - u.start_secs() >= 2.5;
        let joined = utts.windows(2).filter(|w| cut(&w[0])).filter(|w| w[0].end == w[1].start).count();
        let cuts = utts.windows(2).filter(|w| cut(&w[0])).count();
        assert!(cuts >= 4 && joined == cuts, "{joined} of {cuts} cuts carry straight on");
    }

    /// Speech with pauses drawn from `pauses` (seconds), cycling.
    fn talk_with_pauses(pauses: &[f32], n: usize) -> Vec<f32> {
        let mut x = vec![0.0f32; RATE];
        for k in 0..n {
            x.extend(speechish(0.7 + 0.1 * (k % 4) as f32, RATE, 0.3, 100 + k as u32));
            x.extend(std::iter::repeat_n(0.0, (pauses[k % pauses.len()] * RATE as f32) as usize));
        }
        x.extend(std::iter::repeat_n(0.0, RATE * 2));
        for (i, v) in x.iter_mut().enumerate() {
            *v += 0.001 * ((i as f32 * 12.9898).sin() * 43758.545).fract();
        }
        x
    }

    #[test]
    fn the_hangover_follows_a_quick_and_a_slow_speaker() {
        // a quick speaker: pauses 0.1-0.3 s inside sentences, 0.5 s between;
        // a slow one: 0.3-0.7 s inside, 1.2 s between
        let quick = talk_with_pauses(&[0.12, 0.2, 0.15, 0.3, 0.1, 0.25, 0.18, 0.5], 60);
        let slow = talk_with_pauses(&[0.35, 0.6, 0.45, 0.7, 0.3, 0.55, 0.5, 1.2], 60);
        let learnt = |x: &[f32]| {
            let cfg = VadConfig { hangover_secs: 0.4, adapt_hangover: Some(PauseLearner::new(0.85)), ..VadConfig::default() };
            let mut v = Vad::new(cfg);
            let mut u = vec![];
            v.feed(x, &mut u);
            (v.hangover_secs(), v.pauses_heard())
        };
        let (q, nq) = learnt(&quick);
        let (s, ns) = learnt(&slow);
        assert!(nq >= 40 && ns >= 40, "pauses heard: {nq}, {ns}");
        assert!((0.25..=0.35).contains(&q), "quick speaker: {q} s (waits through 0.3 s, ends at 0.5 s)");
        assert!((0.6..=0.8).contains(&s), "slow speaker: {s} s (waits through 0.7 s pauses)");
        let mut l = PauseLearner::new(0.85);
        assert_eq!(l.hangover(0.4), 0.4, "the start value until enough pauses");
        for _ in 0..20 {
            l.observe(0.05);
            l.observe(5.0);
        }
        assert_eq!(l.heard(), 0, "dips under 60 ms and gaps over 3 s are not pauses");
    }

    /// Fixed vs learnt hangovers on real speech (ignored: needs recordings):
    /// NEKOTONE_PAUSE_WAVS=a.wav;b.wav. Each recording is tried as it is,
    /// with its pauses 1.8x longer (a slower speaker) and 0.6x (a quicker
    /// one). The truth is the speaker's own timing: the owner's pauses split
    /// into short ones inside a sentence (70 % under 250 ms) and long ones at
    /// the end of a thought (11 % over 500 ms) with few between, so a pause
    /// of 450 ms or more (times the speed) ends a thought. An early cut is a
    /// cut in a shorter pause (it speaks while you are mid-sentence); a missed
    /// end is a long pause with no cut (you wait for nothing to happen).
    /// (Whisper's sentence ends were tried first: its word times are too
    /// rough, every setting scored ~87 % "mid-sentence".)
    #[test]
    #[ignore]
    fn learnt_hangover_on_real_speech() {
        let list = std::env::var("NEKOTONE_PAUSE_WAVS").expect("NEKOTONE_PAUSE_WAVS=a.wav;b.wav");
        let fs = FRAME as f32 / RATE as f32;
        let mut sets: Vec<(String, Vec<f32>, f32)> = Vec::new();
        for file in list.split(';').filter(|s| !s.is_empty()) {
            let c = crate::audio::decode(std::path::Path::new(file)).unwrap();
            let x = crate::audio::resample(&c, RATE as u32).unwrap().samples;
            let mut lab = Vad::new(VadConfig::default());
            let speech: Vec<bool> = x.chunks_exact(FRAME).map(|f| lab.analyse(f).speech).collect();
            let name = std::path::Path::new(file).file_stem().unwrap().to_string_lossy().to_string();
            for (tag, k) in [("as recorded", 1.0f32), ("slower x1.8", 1.8), ("quicker x0.6", 0.6)] {
                let mut y = Vec::with_capacity(x.len());
                let mut i = 0;
                while i < speech.len() {
                    if speech[i] {
                        y.extend_from_slice(&x[i * FRAME..(i + 1) * FRAME]);
                        i += 1;
                        continue;
                    }
                    let j = (i..speech.len()).find(|&j| speech[j]).unwrap_or(speech.len());
                    let n = j - i;
                    let m = ((n as f32 * k).round() as usize).max(1);
                    for q in 0..m {
                        let src = i + (q * n / m).min(n - 1);
                        y.extend_from_slice(&x[src * FRAME..(src + 1) * FRAME]);
                    }
                    i = j;
                }
                sets.push((format!("{name} ({tag})"), y, 0.45 * k));
            }
        }
        let configs: Vec<(String, f32, Option<f32>)> = vec![
            ("fixed 275 ms".into(), 0.275, None),
            ("fixed 400 ms".into(), 0.4, None),
            ("fixed 525 ms".into(), 0.525, None),
            ("auto q0.80".into(), 0.4, Some(0.80)),
            ("auto q0.85".into(), 0.4, Some(0.85)),
            ("auto q0.90".into(), 0.4, Some(0.90)),
        ];
        println!("{:<30} {:<13} {:>5} {:>6} {:>7} {:>8}", "recording", "hangover", "cuts", "early", "missed", "wait ms");
        let mut tot = vec![(0usize, 0usize, 0usize, 0usize, 0f32); configs.len()];
        for (name, y, end_pause) in &sets {
            // the pauses (runs of non-speech between speech, >= 60 ms), as labelled
            let mut lab = Vad::new(VadConfig::default());
            let sp: Vec<bool> = y.chunks_exact(FRAME).map(|f| lab.analyse(f).speech).collect();
            let mut pauses: Vec<(f32, f32)> = Vec::new(); // (start, length) seconds
            let mut i = sp.iter().position(|&s| s).unwrap_or(sp.len());
            while i < sp.len() {
                if sp[i] {
                    i += 1;
                    continue;
                }
                let j = (i..sp.len()).find(|&j| sp[j]).unwrap_or(sp.len());
                if j < sp.len() && (j - i) >= 3 {
                    pauses.push((i as f32 * fs, (j - i) as f32 * fs));
                }
                i = j;
            }
            let ends = pauses.iter().filter(|p| p.1 >= *end_pause).count();
            for (ci, (label, h, q)) in configs.iter().enumerate() {
                let cfg = VadConfig { hangover_secs: *h, adapt_hangover: q.map(PauseLearner::new), ..VadConfig::default() };
                let mut v = Vad::new(cfg);
                let mut u = vec![];
                let mut waits = Vec::new();
                for chunk in y.chunks(FRAME) {
                    let before = u.len();
                    v.feed(chunk, &mut u);
                    for _ in before..u.len() {
                        waits.push(v.hangover_secs());
                    }
                }
                // each cut sits in a pause: the one that holds the utterance's end
                let mut early = 0;
                let mut cut_pauses = std::collections::HashSet::new();
                for x in &u {
                    let at = x.end_secs() as f32;
                    if let Some((pi, p)) = pauses.iter().enumerate().find(|(_, p)| at >= p.0 && at <= p.0 + p.1 + fs) {
                        cut_pauses.insert(pi);
                        if p.1 < *end_pause {
                            early += 1;
                        }
                    }
                }
                let missed = pauses.iter().enumerate().filter(|(pi, p)| p.1 >= *end_pause && !cut_pauses.contains(pi)).count();
                let wait = if waits.is_empty() { 0.0 } else { waits.iter().sum::<f32>() / waits.len() as f32 };
                println!("{:<30} {:<13} {:>5} {:>6} {:>4}/{:<3} {:>7.0}", name, label, u.len(), early, missed, ends, wait * 1000.0);
                let t = &mut tot[ci];
                t.0 += u.len();
                t.1 += early;
                t.2 += missed;
                t.3 += ends;
                t.4 += wait * u.len() as f32;
            }
        }
        println!("\nall recordings, at all three speeds:");
        for (ci, (label, _, _)) in configs.iter().enumerate() {
            let t = tot[ci];
            println!("  {:<13} cuts {:>4}   early cuts {:>4} ({:>3.0}% of cuts)   missed ends {:>4} of {} ({:>3.0}%)   mean wait {:>4.0} ms", label, t.0, t.1, 100.0 * t.1 as f32 / t.0.max(1) as f32, t.2, t.3, 100.0 * t.2 as f32 / t.3.max(1) as f32, 1000.0 * t.4 / t.0.max(1) as f32);
        }
    }
}

