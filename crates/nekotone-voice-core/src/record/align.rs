//! Putting several capture clocks on one timeline.
//!
//! Every capture device runs on its own crystal: a USB microphone at a
//! nominal 48 kHz may really deliver 48 004 Hz while the sound card's
//! loopback delivers 47 996 Hz. Over an hour that is several seconds of
//! drift between "me" and "them". The recorder therefore treats the
//! system's monotonic clock as the master and, per source:
//!
//! 1. a **delay-locked loop** ([`Dll`]) follows "which source frame was
//!    captured when" from the (jittery) capture timestamps, giving a smooth
//!    estimate of the source's true period in seconds per frame;
//! 2. a **reader** walks the source's samples at `true rate / output rate`
//!    frames per output frame, nudged towards where the DLL says it should
//!    be, and interpolates with a windowed-sinc kernel ([`Kernel`]), so
//!    rate conversion (44.1 ↔ 48 kHz) and drift correction are one step;
//! 3. **gaps** (a loopback device that sends nothing while nothing plays, a
//!    device that stalls) start a new *run*: the time between runs is
//!    rendered as silence and the tracks stay aligned.
//!
//! A single source needs none of this: it is written sample for sample
//! (or through a fixed-ratio resampler when its rate differs from the
//! file's), see [`Lane::sequential`].

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Windowed-sinc interpolation kernel
// ---------------------------------------------------------------------------

/// A band-limited interpolation kernel: `2·half` taps, Kaiser window,
/// tabulated at `PHASES` fractional offsets (linear between them).
#[derive(Debug, Clone)]
pub(crate) struct Kernel {
    half: usize,
    table: Vec<f32>,
}

const PHASES: usize = 512;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..50 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}

impl Kernel {
    /// `half` taps each side (32 = studio quality, flat to ~0.9 × Nyquist
    /// with ~90 dB stop band); `cutoff` relative to the input's Nyquist
    /// frequency (use `< 1` when the output rate is lower).
    pub fn new(half: usize, cutoff: f64) -> Kernel {
        let half = half.max(2);
        let cutoff = cutoff.clamp(0.01, 1.0);
        let taps = 2 * half;
        let beta = if half >= 16 { 9.0 } else { 6.0 };
        let i0b = bessel_i0(beta);
        let mut table = vec![0f32; (PHASES + 1) * taps];
        for p in 0..=PHASES {
            let phase = p as f64 / PHASES as f64;
            let mut sum = 0.0;
            let row = &mut table[p * taps..(p + 1) * taps];
            for (j, v) in row.iter_mut().enumerate() {
                // Tap j covers input sample floor(pos) - half + 1 + j.
                let t = (j as f64 - half as f64 + 1.0) - phase;
                let x = t * cutoff;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let r = t / half as f64;
                let w = if r.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b };
                let h = cutoff * sinc * w;
                *v = h as f32;
                sum += h;
            }
            // Unity DC gain at every phase (no amplitude ripple from the table).
            if sum.abs() > 1e-9 {
                for v in row.iter_mut() {
                    *v = (*v as f64 / sum) as f32;
                }
            }
        }
        Kernel { half, table }
    }

    pub fn half(&self) -> usize {
        self.half
    }

    /// Interpolate channel `ch` of interleaved `buf` (`channels` wide,
    /// frame 0 = absolute frame `base`) at absolute fractional frame `pos`.
    /// Frames outside the buffer count as silence.
    #[inline]
    pub fn at(&self, buf: &VecDeque<f32>, channels: usize, base: u64, pos: f64, out: &mut [f32]) {
        let taps = 2 * self.half;
        let ipos = pos.floor();
        let frac = pos - ipos;
        let fp = frac * PHASES as f64;
        let p0 = (fp as usize).min(PHASES - 1);
        let a = (fp - p0 as f64) as f32;
        let row0 = &self.table[p0 * taps..(p0 + 1) * taps];
        let row1 = &self.table[(p0 + 1) * taps..(p0 + 2) * taps];
        let first = ipos as i64 - self.half as i64 + 1 - base as i64;
        let frames = (buf.len() / channels) as i64;
        for o in out.iter_mut().take(channels) {
            *o = 0.0;
        }
        for j in 0..taps {
            let f = first + j as i64;
            if f < 0 || f >= frames {
                continue;
            }
            let w = row0[j] + (row1[j] - row0[j]) * a;
            let idx = f as usize * channels;
            for (c, o) in out.iter_mut().enumerate().take(channels) {
                *o += buf[idx + c] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed-ratio streaming resampler (single source, and 16 kHz for Whisper)
// ---------------------------------------------------------------------------

/// Streaming resampler with a fixed ratio; pass-through when the rates match.
#[derive(Debug)]
pub(crate) struct Resampler {
    from: u32,
    to: u32,
    channels: usize,
    kernel: Option<Kernel>,
    buf: VecDeque<f32>,
    base: u64,
    /// Next output frame's position in input frames.
    pos: f64,
    step: f64,
    received: u64,
}

impl Resampler {
    pub fn new(from: u32, to: u32, channels: usize, half: usize) -> Resampler {
        let step = from as f64 / to.max(1) as f64;
        let kernel = if from == to { None } else { Some(Kernel::new(half, (1.0 / step).min(1.0) * 0.96)) };
        Resampler { from, to, channels: channels.max(1), kernel, buf: VecDeque::new(), base: 0, pos: 0.0, step, received: 0 }
    }

    /// Feed interleaved input; append the output that is ready.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let Some(kernel) = &self.kernel else {
            out.extend_from_slice(input);
            return;
        };
        self.buf.extend(input.iter().copied());
        self.received += (input.len() / self.channels) as u64;
        let need = kernel.half() as f64;
        let mut frame = vec![0f32; self.channels];
        while self.pos + need < self.received as f64 {
            kernel.at(&self.buf, self.channels, self.base, self.pos, &mut frame);
            out.extend_from_slice(&frame);
            self.pos += self.step;
        }
        self.trim();
    }

    /// Everything still buffered (zeros stand in for the future samples).
    #[allow(dead_code)]
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        let Some(kernel) = &self.kernel else { return };
        let mut frame = vec![0f32; self.channels];
        while self.pos < self.received as f64 {
            kernel.at(&self.buf, self.channels, self.base, self.pos, &mut frame);
            out.extend_from_slice(&frame);
            self.pos += self.step;
        }
        self.trim();
    }

    fn trim(&mut self) {
        let Some(kernel) = &self.kernel else { return };
        let keep_from = (self.pos.floor() as i64 - kernel.half() as i64 - 2).max(0) as u64;
        if keep_from > self.base {
            let drop = ((keep_from - self.base) as usize * self.channels).min(self.buf.len());
            self.buf.drain(..drop);
            self.base += (drop / self.channels) as u64;
        }
    }

    #[allow(dead_code)]
    pub fn rates(&self) -> (u32, u32) {
        (self.from, self.to)
    }
}
