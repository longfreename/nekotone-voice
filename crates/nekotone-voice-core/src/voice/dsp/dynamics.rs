//! Dynamics: the front-end noise gate/expander (hysteresis + hold + optional
//! automatic noise-floor tracking), a split-band de-esser, a compressor and
//! the lookahead brick-wall limiter that ends every signal path.

use super::filters::{Biquad, Kind};
use super::{coef, db_to_lin, lin_to_db, Block, Ctx};
use crate::voice::params::params;

params! {
    /// Noise gate / downward expander. Opens above `threshold_db`, closes
    /// `hysteresis_db` lower after `hold_ms`; closed gain is `floor_db`.
    pub struct GateParams {
        /// Opening threshold (dBFS, measured after a 150 Hz side-chain high-pass).
        threshold_db: [-90.0, 0.0, -48.0, "dB"],
        /// The gate closes this far below the opening threshold.
        hysteresis_db: [0.0, 24.0, 6.0, "dB"],
        /// Opening time.
        attack_ms: [0.1, 50.0, 1.0, "ms"],
        /// Stay open this long after the level drops.
        hold_ms: [0.0, 1000.0, 120.0, "ms"],
        /// Closing time.
        release_ms: [5.0, 2000.0, 90.0, "ms"],
        /// Gain when closed (-100 = silence; -20 = gentle expander).
        floor_db: [-100.0, 0.0, -100.0, "dB"],
        /// 1 = raise the threshold to 16 dB above the measured noise floor when that is higher.
        auto: [0.0, 1.0, 1.0, ""],
    }
}

pub struct Gate {
    p: GateParams,
    rate: f32,
    sc: Biquad,
    env: f32,
    env_att: f32,
    env_rel: f32,
    open: bool,
    hold_left: usize,
    gain: f32,
    g_att: f32,
    g_rel: f32,
    floor: f32,
    // noise floor tracking: minimum of 20 ms RMS frames over ~3 s
    frame_acc: f32,
    frame_n: usize,
    frame_len: usize,
    mins: [f32; 16],
    min_i: usize,
    min_frames: usize,
    noise_floor_db: f32,
}

impl Gate {
    pub fn new(p: GateParams, rate: f32) -> Self {
        let mut g = Gate {
            p,
            rate,
            sc: Biquad::new(Kind::HighPass, 150.0, 0.707, 0.0, rate),
            env: 0.0,
            env_att: coef(0.0005, rate),
            env_rel: coef(0.030, rate),
            open: false,
            hold_left: 0,
            gain: 0.0,
            g_att: 0.0,
            g_rel: 0.0,
            floor: 0.0,
            frame_acc: 0.0,
            frame_n: 0,
            frame_len: (rate * 0.02) as usize,
            mins: [1.0; 16],
            min_i: 0,
            min_frames: 0,
            noise_floor_db: -120.0,
        };
        g.update();
        g.gain = g.floor;
        g
    }
    fn update(&mut self) {
        self.g_att = coef(self.p.attack_ms / 1000.0, self.rate);
        self.g_rel = coef(self.p.release_ms / 1000.0, self.rate);
        self.floor = if self.p.floor_db <= -99.0 { 0.0 } else { db_to_lin(self.p.floor_db) };
    }
    pub fn is_open(&self) -> bool {
        self.open
    }
    /// Current effective opening threshold in dBFS.
    pub fn threshold_db(&self) -> f32 {
        if self.p.auto > 0.5 {
            self.p.threshold_db.max(self.noise_floor_db + 16.0)
        } else {
            self.p.threshold_db
        }
    }
    pub fn noise_floor_db(&self) -> f32 {
        self.noise_floor_db
    }
    /// Current gain (0..1).
    pub fn gain(&self) -> f32 {
        self.gain
    }
    pub fn set_params(&mut self, p: GateParams) {
        self.p = p;
        self.update();
    }

    fn track_floor(&mut self, s: f32) {
        self.frame_acc += s * s;
        self.frame_n += 1;
        if self.frame_n >= self.frame_len {
            let rms = (self.frame_acc / self.frame_n as f32).sqrt();
            self.frame_acc = 0.0;
            self.frame_n = 0;
            // 16 slots, each the minimum over 10 frames (200 ms): ~3.2 s memory
            let slot = self.min_i / 10 % 16;
            if self.min_i % 10 == 0 {
                self.mins[slot] = rms;
            } else {
                self.mins[slot] = self.mins[slot].min(rms);
            }
            self.min_i = self.min_i.wrapping_add(1);
            self.min_frames = self.min_frames.saturating_add(1);
            if self.min_frames >= 30 {
                let m = self.mins.iter().take((self.min_frames / 10).min(16)).cloned().fold(f32::MAX, f32::min);
                self.noise_floor_db = lin_to_db(m);
            }
        }
    }
}

impl Block for Gate {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let thr_open = db_to_lin(self.threshold_db());
        let thr_close = thr_open * db_to_lin(-self.p.hysteresis_db);
        let hold = (self.p.hold_ms / 1000.0 * self.rate) as usize;
        for x in buf.iter_mut() {
            let s = self.sc.tick(*x);
            self.track_floor(s);
            let a = s.abs();
            let c = if a > self.env { self.env_att } else { self.env_rel };
            self.env += (a - self.env) * c;
            if self.env >= thr_open {
                self.open = true;
                self.hold_left = hold;
            } else if self.open {
                if self.env >= thr_close {
                    self.hold_left = hold;
                } else if self.hold_left > 0 {
                    self.hold_left -= 1;
                } else {
                    self.open = false;
                }
            }
            let target = if self.open { 1.0 } else { self.floor };
            let k = if target > self.gain { self.g_att } else { self.g_rel };
            self.gain += (target - self.gain) * k;
            if self.gain < 1e-6 && !self.open && self.floor == 0.0 {
                self.gain = 0.0;
            }
            *x *= self.gain;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

params! {
    /// Split-band de-esser: turns down the band above `hz` when it exceeds the threshold.
    pub struct DeEsserParams {
        /// Split frequency.
        hz: [2000.0, 12000.0, 5500.0, "Hz"],
        /// Threshold for the sibilant band (dBFS).
        threshold_db: [-60.0, 0.0, -30.0, "dB"],
        /// Compression ratio above the threshold.
        ratio: [1.0, 20.0, 4.0, ""],
        /// Maximum reduction.
        range_db: [0.0, 24.0, 10.0, "dB"],
    }
}

pub struct DeEsser {
    p: DeEsserParams,
    rate: f32,
    sc: Biquad,
    shelf: Biquad,
    env: f32,
    att: f32,
    rel: f32,
    g_db: f32,
    applied_db: f32,
    count: usize,
}

impl DeEsser {
    pub fn new(p: DeEsserParams, rate: f32) -> Self {
        DeEsser {
            p,
            rate,
            sc: Biquad::new(Kind::HighPass, p.hz, 0.707, 0.0, rate),
            shelf: Biquad::pass(),
            env: 0.0,
            att: coef(0.001, rate),
            rel: coef(0.060, rate),
            g_db: 0.0,
            applied_db: 0.0,
            count: 0,
        }
    }
    pub fn set_params(&mut self, p: DeEsserParams) {
        self.p = p;
        self.sc.set(Kind::HighPass, p.hz, 0.707, 0.0, self.rate);
        self.applied_db = 1.0; // force a coefficient update
    }
}

impl Block for DeEsser {
    /// Dynamic high shelf: the side-chain (high-passed) level above the
    /// threshold sets a shelf cut at `hz`, updated every 16 samples.
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        let thr = self.p.threshold_db;
        let slope = 1.0 - 1.0 / self.p.ratio.max(1.0);
        let gs = coef(0.002, self.rate);
        for x in buf.iter_mut() {
            let a = self.sc.tick(*x).abs();
            let c = if a > self.env { self.att } else { self.rel };
            self.env += (a - self.env) * c;
            let over = lin_to_db(self.env) - thr;
            let target = if over > 0.0 { (-over * slope).max(-self.p.range_db) } else { 0.0 };
            self.g_db += (target - self.g_db) * gs;
            self.count += 1;
            if self.count >= 16 {
                self.count = 0;
                let q = (self.g_db * 4.0).round() / 4.0;
                if q != self.applied_db {
                    self.applied_db = q;
                    if q == 0.0 {
                        self.shelf.set(Kind::HighShelf, self.p.hz, 0.707, 0.0, self.rate);
                    } else {
                        self.shelf.set(Kind::HighShelf, self.p.hz * 0.8, 0.707, q, self.rate);
                    }
                }
            }
            *x = self.shelf.tick(*x);
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            let p = self.p;
            self.set_params(p);
        }
    }
}

params! {
    /// Feed-forward compressor with soft knee.
    pub struct CompressorParams {
        /// Threshold (dBFS).
        threshold_db: [-60.0, 0.0, -20.0, "dB"],
        /// Ratio (1 = off, 20 = limiting).
        ratio: [1.0, 20.0, 3.0, ""],
        /// Attack time.
        attack_ms: [0.1, 200.0, 5.0, "ms"],
        /// Release time.
        release_ms: [5.0, 2000.0, 120.0, "ms"],
        /// Soft-knee width.
        knee_db: [0.0, 24.0, 6.0, "dB"],
        /// Make-up gain.
        makeup_db: [-12.0, 36.0, 0.0, "dB"],
    }
}

/// The compressor's level detector: RMS over ~10 ms, steady across a
/// pitch period (a low voice's is 5-12 ms). It once read each sample's own
/// size, so with a 4 ms attack the gain dipped on every glottal pulse and
/// recovered in between: pitch-rate pulsing and waveform distortion (the
/// "buzz" of the compressed voices; measured on the owner's recordings as
/// 3-5x the dry voice's pitch-rate modulation in 2-6 kHz).
const DETECT_SECS: f32 = 0.010;
/// Shortest gain attack allowed, for the same reason.
const MIN_ATTACK_MS: f32 = 5.0;

pub struct Compressor {
    p: CompressorParams,
    rate: f32,
    env_db: f32,
    att: f32,
    rel: f32,
    makeup: f32,
    /// Running mean square (the detector).
    power: f32,
    det: f32,
}

impl Compressor {
    pub fn new(p: CompressorParams, rate: f32) -> Self {
        let mut c = Compressor { p, rate, env_db: 0.0, att: 0.0, rel: 0.0, makeup: 1.0, power: 0.0, det: 0.0 };
        c.update();
        c
    }
    fn update(&mut self) {
        self.att = coef(self.p.attack_ms.max(MIN_ATTACK_MS) / 1000.0, self.rate);
        self.rel = coef(self.p.release_ms / 1000.0, self.rate);
        self.makeup = db_to_lin(self.p.makeup_db);
        self.det = coef(DETECT_SECS, self.rate);
    }
    /// Static gain reduction in dB (<= 0) for an input level in dBFS.
    pub fn curve(&self, lvl: f32) -> f32 {
        let t = self.p.threshold_db;
        let k = self.p.knee_db;
        let s = 1.0 / self.p.ratio.max(1.0) - 1.0;
        let over = lvl - t;
        if k > 0.0 && over.abs() <= k / 2.0 {
            s * (over + k / 2.0).powi(2) / (2.0 * k)
        } else if over > 0.0 {
            s * over
        } else {
            0.0
        }
    }
}

impl Block for Compressor {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            self.power += (*x * *x - self.power) * self.det;
            // RMS + 3 dB: a sine's peak level, so thresholds keep their meaning
            let lvl = 10.0 * (self.power + 1e-12).log10() + 3.0;
            let gr = self.curve(lvl);
            // smooth the gain reduction: attack when reducing more
            let c = if gr < self.env_db { self.att } else { self.rel };
            self.env_db += (gr - self.env_db) * c;
            *x *= db_to_lin(self.env_db) * self.makeup;
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
}

params! {
    /// Lookahead brick-wall limiter: the output never exceeds the ceiling.
    pub struct LimiterParams {
        /// Ceiling (dBFS).
        ceiling_db: [-24.0, 0.0, -1.0, "dB"],
        /// Release time.
        release_ms: [5.0, 1000.0, 80.0, "ms"],
        /// Input gain before limiting.
        drive_db: [-24.0, 24.0, 0.0, "dB"],
    }
}

/// Lookahead limiter. The gain for each sample is the minimum required over
/// the next `L` samples, smoothed by a box average of length `L` so it is in
/// place before the peak arrives; a final clamp guarantees the ceiling.
pub struct Limiter {
    p: LimiterParams,
    rate: f32,
    la: usize,
    delay: Vec<f32>,
    // running minimum over the last `la` targets (monotonic deque in a ring)
    dq_val: Vec<f32>,
    dq_idx: Vec<u64>,
    dq_head: usize,
    dq_len: usize,
    t: u64,
    rel_g: f32,
    rel: f32,
    box_buf: Vec<f32>,
    box_sum: f64,
    pos: usize,
    ceil: f32,
    drive: f32,
    /// Smallest gain applied since the last `take_reduction` (for meters).
    pub min_gain: f32,
}

impl Limiter {
    pub fn new(p: LimiterParams, rate: f32) -> Self {
        let la = ((rate * 0.0015) as usize).max(4);
        let mut l = Limiter {
            p,
            rate,
            la,
            delay: vec![0.0; la],
            dq_val: vec![0.0; la + 2],
            dq_idx: vec![0; la + 2],
            dq_head: 0,
            dq_len: 0,
            t: 0,
            rel_g: 1.0,
            rel: 0.0,
            box_buf: vec![1.0; la],
            box_sum: la as f64,
            pos: 0,
            ceil: 1.0,
            drive: 1.0,
            min_gain: 1.0,
        };
        l.update();
        l
    }
    fn update(&mut self) {
        self.rel = coef(self.p.release_ms / 1000.0, self.rate);
        self.ceil = db_to_lin(self.p.ceiling_db);
        self.drive = db_to_lin(self.p.drive_db);
    }
    pub fn ceiling(&self) -> f32 {
        self.ceil
    }
    pub fn set_ceiling_db(&mut self, db: f32) {
        self.p.ceiling_db = db.clamp(-24.0, 0.0);
        self.update();
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let x = x * self.drive;
        let a = x.abs();
        let target = if a > self.ceil { self.ceil / a } else { 1.0 };
        // push into the running-min deque
        let cap = self.dq_val.len();
        while self.dq_len > 0 {
            let back = (self.dq_head + self.dq_len - 1) % cap;
            if self.dq_val[back] >= target {
                self.dq_len -= 1;
            } else {
                break;
            }
        }
        let slot = (self.dq_head + self.dq_len) % cap;
        self.dq_val[slot] = target;
        self.dq_idx[slot] = self.t;
        self.dq_len += 1;
        // window of la+1 targets: every gain averaged for the delayed sample saw its peak
        while self.dq_idx[self.dq_head] + (self.la as u64) < self.t {
            self.dq_head = (self.dq_head + 1) % cap;
            self.dq_len -= 1;
        }
        let held = self.dq_val[self.dq_head];
        self.t += 1;
        // release smoothing (instant down, exponential up)
        if held < self.rel_g {
            self.rel_g = held;
        } else {
            self.rel_g += (held - self.rel_g) * self.rel;
        }
        // box average so the gain ramps down over the lookahead
        self.box_sum += (self.rel_g - self.box_buf[self.pos]) as f64;
        self.box_buf[self.pos] = self.rel_g;
        let delayed = self.delay[self.pos];
        self.delay[self.pos] = x;
        self.pos += 1;
        if self.pos == self.la {
            self.pos = 0;
            // re-sum occasionally to stop floating-point drift
            if self.t % (self.la as u64 * 64) == 0 {
                self.box_sum = self.box_buf.iter().map(|v| *v as f64).sum();
            }
        }
        let g = ((self.box_sum / self.la as f64) as f32).min(1.0);
        self.min_gain = self.min_gain.min(g);
        (delayed * g).clamp(-self.ceil, self.ceil)
    }
}

impl Block for Limiter {
    fn process(&mut self, buf: &mut [f32], _ctx: &mut Ctx) {
        for x in buf.iter_mut() {
            *x = self.tick(*x);
        }
    }
    fn set_param(&mut self, i: usize, v: f32) {
        if self.p.set(i, v) {
            self.update();
        }
    }
    fn latency(&self) -> usize {
        self.la
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn rms(b: &[f32]) -> f32 {
        (b.iter().map(|v| v * v).sum::<f32>() / b.len().max(1) as f32).sqrt()
    }

    #[test]
    fn gate_closes_on_quiet_noise_and_opens_on_voice() {
        let rate = 48000.0;
        let mut g = Gate::new(GateParams { auto: 0.0, threshold_db: -40.0, ..Default::default() }, rate);
        let mut rng = crate::voice::dsp::Rng::new(3);
        // -60 dBFS noise: must be gated to silence
        let mut noise: Vec<f32> = (0..48000).map(|_| rng.bipolar() * 0.001 * 1.7).collect();
        g.process(&mut noise, &mut Ctx::default());
        assert!(rms(&noise[24000..]) < 1e-6, "gate leaked {}", rms(&noise[24000..]));
        assert!(!g.is_open());
        // -12 dBFS tone opens it
        let mut tone: Vec<f32> = (0..9600).map(|i| 0.25 * (2.0 * PI * 300.0 * i as f32 / rate).sin()).collect();
        g.process(&mut tone, &mut Ctx::default());
        assert!(g.is_open());
        assert!(rms(&tone[4800..]) > 0.17);
        // hysteresis + hold: a level between the two thresholds keeps it open
        let mut mid: Vec<f32> = (0..4800).map(|i| 0.0125 * (2.0 * PI * 300.0 * i as f32 / rate).sin()).collect();
        g.process(&mut mid, &mut Ctx::default());
        assert!(g.is_open(), "-41 dB after -12 dB should stay open (hysteresis)");
    }

    #[test]
    fn gate_auto_threshold_tracks_noise_floor() {
        let rate = 48000.0;
        let mut g = Gate::new(GateParams { threshold_db: -80.0, auto: 1.0, ..Default::default() }, rate);
        let mut rng = crate::voice::dsp::Rng::new(9);
        // -40 dBFS noise (a noisy room)
        let mut noise: Vec<f32> = (0..48000 * 4).map(|_| rng.bipolar() * 0.01 * 1.7).collect();
        for c in noise.chunks_mut(256) {
            g.process(c, &mut Ctx::default());
        }
        assert!((g.noise_floor_db() + 42.0).abs() < 4.0, "floor {}", g.noise_floor_db());
        assert!(g.threshold_db() > -30.0);
        assert!(rms(&noise[48000 * 3..]) < 1e-5);
    }

    #[test]
    fn deesser_reduces_sibilants_only() {
        let rate = 48000.0;
        let run = |hz: f32| {
            let mut d = DeEsser::new(DeEsserParams { threshold_db: -30.0, ..Default::default() }, rate);
            let mut b: Vec<f32> = (0..9600).map(|i| 0.3 * (2.0 * PI * hz * i as f32 / rate).sin()).collect();
            d.process(&mut b, &mut Ctx::default());
            rms(&b[4800..]) / (0.3 / 2f32.sqrt())
        };
        assert!(run(300.0) > 0.97);
        assert!(run(8000.0) < 0.5);
    }

    #[test]
    fn compressor_curve_and_gain() {
        let c = Compressor::new(CompressorParams { threshold_db: -20.0, ratio: 4.0, knee_db: 0.0, ..Default::default() }, 48000.0);
        assert_eq!(c.curve(-30.0), 0.0);
        assert!((c.curve(-8.0) + 9.0).abs() < 1e-4);
    }

    /// A low, pulsy voice-like wave (100 Hz, sharp glottal pulses) well over
    /// the threshold: the gain may settle, but must not move within a pitch
    /// period (that is audible as buzz/rasp).
    #[test]
    fn compressor_does_not_modulate_within_a_pitch_period() {
        let rate = 48000.0;
        let mut c = Compressor::new(CompressorParams { threshold_db: -24.0, ratio: 3.0, attack_ms: 4.0, release_ms: 120.0, knee_db: 6.0, makeup_db: 0.0 }, rate);
        let period = 480; // 100 Hz
        let x: Vec<f32> = (0..48000).map(|i| {
            let ph = (i % period) as f32 / period as f32;
            0.5 * (-ph * 18.0).exp() * (2.0 * std::f32::consts::PI * ph * 7.0).cos()
        }).collect();
        let mut y = x.clone();
        c.process(&mut y, &mut Ctx::default());
        // gain per sample where the input is not tiny, over the last 0.2 s
        let gains: Vec<f32> = (38400..48000).filter(|&i| x[i].abs() > 0.02).map(|i| y[i] / x[i]).collect();
        let (lo, hi) = gains.iter().fold((f32::MAX, f32::MIN), |(a, b), &g| (a.min(g), b.max(g)));
        let swing_db = 20.0 * (hi / lo).log10();
        assert!(swing_db < 0.5, "the gain swings {swing_db:.2} dB within pitch periods");
        assert!(hi < 0.95, "it should still compress (gain {hi:.2})");
    }

    #[test]
    fn limiter_never_exceeds_ceiling() {
        let rate = 48000.0;
        let mut l = Limiter::new(LimiterParams { ceiling_db: -1.0, ..Default::default() }, rate);
        let ceil = db_to_lin(-1.0);
        let mut rng = crate::voice::dsp::Rng::new(1);
        let mut b: Vec<f32> = (0..48000).map(|i| 4.0 * rng.bipolar() * if i % 5000 < 100 { 3.0 } else { 0.3 }).collect();
        l.process(&mut b, &mut Ctx::default());
        assert!(b.iter().all(|v| v.abs() <= ceil + 1e-7));
        // a quiet signal passes unchanged (delayed by the lookahead)
        let mut l = Limiter::new(LimiterParams::default(), rate);
        let src: Vec<f32> = (0..2000).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
        let mut b = src.clone();
        l.process(&mut b, &mut Ctx::default());
        let la = l.latency();
        for i in la..2000 {
            assert!((b[i] - src[i - la]).abs() < 1e-5);
        }
    }
}
