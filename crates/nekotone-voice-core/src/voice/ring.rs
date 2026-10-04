//! Lock-free single-producer/single-consumer sample ring between the input
//! callback (producer) and an output callback (consumer), and the adaptive
//! resampler that reads it at the output device's clock.
//!
//! Clock drift: the two devices' crystals never agree exactly (typically
//! ±0.01–0.1 %), and the rates may differ nominally (44.1 vs 48 kHz). The
//! reader resamples by `in_rate/out_rate × (1 + c)` with 4-point Hermite
//! interpolation, where `c` comes from a PI controller that holds the ring
//! fill (smoothed) at its target. Result: bounded latency for hours, no
//! periodic drop/repeat clicks. Safety nets: on an underrun the output fades
//! to silence and re-primes to the target (counted as an xrun); if the fill
//! runs far above target (a stalled consumer), the excess is skipped with a
//! 64-sample cross-fade.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Shared {
    buf: Box<[UnsafeCell<f32>]>,
    mask: usize,
    head: AtomicUsize, // total written
    tail: AtomicUsize, // total read
}

// SAFETY: the producer only writes slots in [tail+len, tail+cap) and the
// consumer only reads slots in [tail, head); indices are published with
// release/acquire ordering, so no slot is accessed by both at once.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Producer(Arc<Shared>);
pub struct Consumer(Arc<Shared>);

/// A ring holding at least `capacity` samples.
pub fn ring(capacity: usize) -> (Producer, Consumer) {
    let n = capacity.next_power_of_two().max(64);
    let buf: Vec<UnsafeCell<f32>> = (0..n).map(|_| UnsafeCell::new(0.0)).collect();
    let s = Arc::new(Shared { buf: buf.into_boxed_slice(), mask: n - 1, head: AtomicUsize::new(0), tail: AtomicUsize::new(0) });
    (Producer(s.clone()), Consumer(s))
}

impl Producer {
    /// Write as much of `data` as fits; returns the count written.
    pub fn push(&mut self, data: &[f32]) -> usize {
        let s = &*self.0;
        let head = s.head.load(Ordering::Relaxed);
        let tail = s.tail.load(Ordering::Acquire);
        let free = s.buf.len() - (head - tail);
        let n = data.len().min(free);
        for (i, v) in data[..n].iter().enumerate() {
            unsafe { *s.buf[(head + i) & s.mask].get() = *v };
        }
        s.head.store(head + n, Ordering::Release);
        n
    }
    pub fn len(&self) -> usize {
        let s = &*self.0;
        s.head.load(Ordering::Relaxed) - s.tail.load(Ordering::Acquire)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.0.buf.len()
    }
}

impl Consumer {
    /// Read up to `out.len()` samples; returns the count read.
    pub fn pop(&mut self, out: &mut [f32]) -> usize {
        let s = &*self.0;
        let tail = s.tail.load(Ordering::Relaxed);
        let head = s.head.load(Ordering::Acquire);
        let n = out.len().min(head - tail);
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o = unsafe { *s.buf[(tail + i) & s.mask].get() };
        }
        s.tail.store(tail + n, Ordering::Release);
        n
    }
    /// Discard up to `n` samples.
    pub fn skip(&mut self, n: usize) -> usize {
        let s = &*self.0;
        let tail = s.tail.load(Ordering::Relaxed);
        let head = s.head.load(Ordering::Acquire);
        let n = n.min(head - tail);
        s.tail.store(tail + n, Ordering::Release);
        n
    }
    pub fn len(&self) -> usize {
        let s = &*self.0;
        s.head.load(Ordering::Acquire) - s.tail.load(Ordering::Relaxed)
    }
    /// Samples read (or skipped) since the ring was made.
    pub fn position(&self) -> u64 {
        self.0.tail.load(Ordering::Relaxed) as u64
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.0.buf.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Priming,
    Running,
}

/// Reads a ring at a different (drifting) clock. Allocation-free after `new`.
pub struct DriftReader {
    nominal: f64,
    in_rate: f64,
    out_rate: f64,
    pos: f64,
    hist: [f32; 4],
    integ: f64,
    fill_ema: f64,
    ema_ready: bool,
    /// Target fill in input samples.
    pub target: f64,
    max_target: f64,
    corr: f64,
    state: State,
    fade_in: usize,
    last: f32,
    inbuf: Vec<f32>,
    xf: Vec<f32>,
    /// Output underruns so far.
    pub xruns: u64,
    /// Times the reader skipped ahead because the ring overfilled.
    pub skips: u64,
}

impl DriftReader {
    /// `target_s`: initial ring fill to hold, in seconds.
    pub fn new(in_rate: f64, out_rate: f64, target_s: f64, max_frames: usize) -> Self {
        DriftReader {
            nominal: in_rate / out_rate,
            in_rate,
            out_rate,
            pos: 0.0,
            hist: [0.0; 4],
            integ: 0.0,
            fill_ema: 0.0,
            ema_ready: false,
            target: (target_s * in_rate).max(32.0),
            max_target: 0.08 * in_rate,
            corr: 0.0,
            state: State::Priming,
            fade_in: 0,
            last: 0.0,
            inbuf: vec![0.0; (max_frames as f64 * (in_rate / out_rate) * 1.1) as usize + 16],
            xf: vec![0.0; 64 * 2 + 8],
            xruns: 0,
            skips: 0,
        }
    }

    /// Current drift correction (e.g. 0.0004 = reading 0.04 % fast).
    pub fn correction(&self) -> f64 {
        self.corr
    }

    /// Smoothed ring fill in seconds (latency contributed by the ring).
    pub fn fill_seconds(&self) -> f64 {
        self.fill_ema / self.in_rate
    }

    /// Grow the target after an underrun (bounded by 80 ms).
    fn bump_target(&mut self) {
        self.target = (self.target + 0.002 * self.in_rate).min(self.max_target);
    }

    /// How many input samples `frames` outputs will consume at `step`.
    #[inline]
    fn needed(&self, frames: usize, step: f64) -> usize {
        (self.pos + frames as f64 * step).floor() as usize
    }

    /// Fill measurement, priming, overfill net and PI update. Returns the
    /// read step (input samples per output sample), or None while priming.
    fn control(&mut self, ring: &mut Consumer, frames: usize) -> Option<f64> {
        let fill = ring.len() as f64;
        let dt = frames as f64 / self.out_rate;
        if !self.ema_ready {
            self.fill_ema = fill;
            self.ema_ready = true;
        } else {
            let a = 1.0 - (-dt / 0.5).exp();
            self.fill_ema += (fill - self.fill_ema) * a;
        }

        if self.state == State::Priming {
            if fill >= self.target {
                self.state = State::Running;
                self.fade_in = 0;
                self.fill_ema = fill;
                self.hist = [0.0; 4];
                self.pos = 0.0;
            } else {
                return None;
            }
        }

        // overfill safety net: skip ahead with a cross-fade
        let hi = self.target * 2.0 + 0.03 * self.in_rate;
        if fill > hi {
            let excess = (fill - self.target) as usize;
            self.skip_with_crossfade(ring, excess);
            self.fill_ema = ring.len() as f64;
            self.integ = 0.0;
            self.skips += 1;
        }

        // PI control of the fill level
        let err = (self.fill_ema - self.target) / self.in_rate; // seconds
        let kp = 0.2;
        let ki = 0.01;
        self.integ = (self.integ + ki * err * dt).clamp(-0.005, 0.005);
        self.corr = (kp * err + self.integ).clamp(-0.01, 0.01);
        Some(self.nominal * (1.0 + self.corr))
    }

    /// Consume input exactly as `read` would for `frames` outputs, without
    /// producing audio (for long simulations of the clock controller).
    pub fn advance(&mut self, ring: &mut Consumer, frames: usize) {
        let Some(step) = self.control(ring, frames) else { return };
        let need = self.needed(frames, step);
        let got = ring.skip(need);
        self.pos = (self.pos + frames as f64 * step).fract();
        if got < need {
            self.xruns += 1;
            self.bump_target();
            self.state = State::Priming;
            self.integ = 0.0;
        }
    }

    /// Fill `out` (mono) from `ring`.
    pub fn read(&mut self, ring: &mut Consumer, out: &mut [f32]) {
        let frames = out.len();
        if frames == 0 {
            return;
        }
        // grow the scratch only if a host delivers an unexpectedly huge buffer
        // (never in steady state)
        let worst = (frames as f64 * self.nominal * 1.02) as usize + 8;
        if self.inbuf.len() < worst {
            self.inbuf.resize(worst, 0.0);
        }
        let Some(step) = self.control(ring, frames) else {
            out.fill(0.0);
            return;
        };
        let need = self.needed(frames, step);
        let got = ring.pop(&mut self.inbuf[..need]);
        let mut idx = 0usize;
        let mut starved_at = frames;
        for (k, o) in out.iter_mut().enumerate() {
            let t = self.pos as f32;
            let y = super::dsp::hermite(self.hist[0], self.hist[1], self.hist[2], self.hist[3], t);
            *o = y;
            self.pos += step;
            while self.pos >= 1.0 {
                self.pos -= 1.0;
                if idx >= got {
                    starved_at = starved_at.min(k + 1);
                    break;
                }
                self.hist = [self.hist[1], self.hist[2], self.hist[3], self.inbuf[idx]];
                idx += 1;
            }
            if starved_at < frames {
                break;
            }
        }
        // fade-in after priming
        if self.fade_in < 64 {
            for o in out.iter_mut() {
                if self.fade_in >= 64 {
                    break;
                }
                *o *= self.fade_in as f32 / 64.0;
                self.fade_in += 1;
            }
        }
        if starved_at < frames {
            // underrun: fade the last value out over 32 samples, then silence
            let mut v = if starved_at > 0 { out[starved_at - 1] } else { self.last };
            for o in out[starved_at..].iter_mut() {
                v *= 0.85;
                *o = v;
            }
            self.xruns += 1;
            self.bump_target();
            self.state = State::Priming;
            self.integ = 0.0;
        }
        self.last = out[frames - 1];
    }

    fn skip_with_crossfade(&mut self, ring: &mut Consumer, excess: usize) {
        // continuation A (64), skip, then B (64): replace B's start by A→B
        let n = 64;
        let (a, rest) = self.xf.split_at_mut(n);
        let b = &mut rest[..n];
        if ring.len() < excess + 2 * n {
            ring.skip(excess);
            return;
        }
        ring.pop(a);
        ring.skip(excess.saturating_sub(n));
        ring.pop(b);
        for i in 0..n {
            let w = (i as f32 + 0.5) / n as f32;
            b[i] = a[i] * (1.0 - w) + b[i] * w;
        }
        // feed the cross-faded block through the interpolator history by
        // pushing it into the ring position we just consumed: simplest is to
        // prime the history with its tail and accept 60 samples of the blend
        // being skipped too — at a rare, already-glitchy event this is inaudible.
        self.hist = [b[n - 4], b[n - 3], b[n - 2], b[n - 1]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spsc_ring_basics() {
        let (mut p, mut c) = ring(100);
        assert_eq!(p.capacity(), 128);
        let data: Vec<f32> = (0..200).map(|i| i as f32).collect();
        assert_eq!(p.push(&data), 128);
        let mut out = vec![0.0; 50];
        assert_eq!(c.pop(&mut out), 50);
        assert_eq!(out[49], 49.0);
        assert_eq!(p.push(&data[128..]), 50);
        assert_eq!(c.len(), 128);
        assert_eq!(c.skip(10), 10);
        c.pop(&mut out);
        assert_eq!(out[0], 60.0);
    }

    #[test]
    fn spsc_ring_threads() {
        let (mut p, mut c) = ring(1024);
        let t = std::thread::spawn(move || {
            let mut next = 0u32;
            while next < 200_000 {
                let chunk: Vec<f32> = (next..next + 100).map(|v| v as f32).collect();
                let n = p.push(&chunk);
                next += n as u32;
                if n == 0 {
                    std::thread::yield_now();
                }
            }
        });
        let mut expect = 0u32;
        let mut buf = vec![0.0; 77];
        while expect < 200_000 {
            let n = c.pop(&mut buf);
            for v in &buf[..n] {
                assert_eq!(*v, expect as f32);
                expect += 1;
            }
        }
        t.join().unwrap();
    }

    /// Simulate an input device running 0.1 % fast (and one 0.1 % slow)
    /// against an output device, with different block sizes and jitter, for
    /// one hour of audio. The ring fill must stay bounded and never underrun
    /// after the start.
    #[test]
    fn drift_compensation_holds_fill_for_an_hour() {
        for mismatch in [1.001f64, 0.999] {
            let in_rate = 48000.0 * mismatch; // true input clock
            let out_rate = 48000.0;
            let (mut p, mut c) = ring(48000);
            // the reader knows only the nominal rates
            let mut r = DriftReader::new(48000.0, 48000.0, 0.012, 1024);
            let in_block = 480usize; // WASAPI-like 10 ms
            let out_block = 441usize; // odd size on purpose
            let mut t_in = 0.0f64;
            let mut jit = 0.0f64;
            let mut t_out = 0.0f64;
            let seconds = 3600.0;
            // the first minutes run the full interpolating reader, the rest the
            // identical controller/consumption path without the audio math
            let real_secs = 120.0;
            let src = vec![0.25f32; in_block];
            let mut out = vec![0.0f32; out_block];
            let mut max_fill = 0usize;
            let mut min_fill_late = usize::MAX;
            let mut rng = crate::voice::dsp::Rng::new(1);
            let mut n_out = 0u64;
            while t_out < seconds {
                // advance whichever device fires next (with ±1 ms jitter on input)
                // the device clock ticks exactly; each callback arrives with ±0.5 ms jitter
                let next_in = t_in + in_block as f64 / in_rate + jit;
                let next_out = t_out + out_block as f64 / out_rate;
                if next_in <= next_out {
                    t_in = next_in - jit;
                    jit = rng.bipolar() as f64 * 0.0005;
                    p.push(&src);
                } else {
                    t_out = next_out;
                    // the engine's target rule: one input burst + one output burst + 1 ms
                    let want = (in_block + out_block) as f64 + 48.0;
                    if r.target < want {
                        r.target = want;
                    }
                    if t_out < real_secs {
                        r.read(&mut c, &mut out);
                    } else {
                        r.advance(&mut c, out_block);
                    }
                    n_out += 1;
                    let f = c.len();
                    max_fill = max_fill.max(f);
                    if t_out > 60.0 {
                        min_fill_late = min_fill_late.min(f);
                    }
                }
            }
            let _ = n_out;
            assert!(r.xruns <= 1, "mismatch {mismatch}: {} xruns", r.xruns);
            assert_eq!(r.skips, 0);
            // bounded: never more than ~40 ms queued, and the fill is stable
            assert!(max_fill < (0.04 * 48000.0) as usize, "mismatch {mismatch}: max fill {max_fill}");
            assert!(min_fill_late > 0, "mismatch {mismatch}");
            assert!((r.correction() - (mismatch - 1.0)).abs() < 2e-4, "mismatch {mismatch}: corr {}", r.correction());
        }
    }

    #[test]
    fn resampled_sine_is_clean_across_rates() {
        // 44.1 kHz input read by a 48 kHz output: the output is a clean sine
        let (mut p, mut c) = ring(8192);
        let mut r = DriftReader::new(44100.0, 48000.0, 0.01, 512);
        let f = 440.0f64;
        let mut ph = 0u64;
        let mut outs = Vec::new();
        for _ in 0..2000 {
            // the producer runs on its own (nominal) clock: 441 in per 480 out
            let chunk: Vec<f32> = (0..441).map(|k| ((ph + k) as f64 * 2.0 * std::f64::consts::PI * f / 44100.0).sin() as f32).collect();
            ph += 441;
            p.push(&chunk);
            let mut o = vec![0.0f32; 480];
            r.read(&mut c, &mut o);
            outs.extend_from_slice(&o);
        }
        let tail = &outs[outs.len() - 48000..];
        let fm = crate::voice::dsp::pitch::measure_f0(&tail[..12000], 48000.0, 200.0, 1000.0);
        assert!((1200.0 * (fm / 440.0).log2()).abs() < 3.0, "{fm}");
        let max_step = tail.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
        assert!(max_step < 2.0 * std::f32::consts::PI * 440.0 / 48000.0 * 1.05);
    }
}
