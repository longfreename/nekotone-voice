//! Minimal feature helpers kept for Whisper STT.

use crate::audio::Clip;
use crate::Result;
use realfft::RealFftPlanner;

pub const ANALYSIS_RATE: u32 = 16_000;

/// Whisper-style log-mel spectrogram at 16 kHz, shaped as `frames × n_mels`.
pub fn whisper_mel(clip16k: &Clip, n_mels: usize) -> Result<Vec<Vec<f32>>> {
    const N: usize = 400;
    const H: usize = 160;
    if n_mels == 0 {
        return Err(crate::Error::Other(anyhow::anyhow!("whisper_mel: n_mels must be greater than 0")));
    }
    let x = if clip16k.sample_rate == ANALYSIS_RATE || clip16k.sample_rate == 0 {
        std::borrow::Cow::Borrowed(&clip16k.samples)
    } else {
        std::borrow::Cow::Owned(crate::audio::resample(clip16k, ANALYSIS_RATE)?.samples)
    };
    let len = x.len();
    let frames = len / H;
    if frames == 0 {
        return Ok(vec![]);
    }
    let pad = N / 2;
    let at = |i: usize| -> f32 {
        let j = i as isize - pad as isize;
        let j = if j < 0 { -j } else if j >= len as isize { 2 * (len as isize - 1) - j } else { j };
        if len == 1 { x[0] } else { x[j.clamp(0, len as isize - 1) as usize] }
    };
    let window: Vec<f32> = (0..N).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos()).collect();
    let fb = mel_filterbank(ANALYSIS_RATE as f32, N, n_mels, 0.0, 8000.0);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut fin = fft.make_input_vec();
    let mut fout = fft.make_output_vec();
    let mut power = vec![0.0f32; N / 2 + 1];
    let mut out = Vec::with_capacity(frames);
    let mut max = f32::MIN;
    for t in 0..frames {
        let s = t * H;
        for (i, (o, w)) in fin.iter_mut().zip(&window).enumerate() {
            *o = at(s + i) * w;
        }
        let _ = fft.process(&mut fin, &mut fout);
        for (p, c) in power.iter_mut().zip(&fout) {
            *p = c.norm_sqr();
        }
        let row: Vec<f32> = fb.iter().map(|f| {
            let v: f32 = f.iter().map(|&(k, w)| w * power[k]).sum();
            let l = v.max(1e-10).log10();
            max = max.max(l);
            l
        }).collect();
        out.push(row);
    }
    let floor = max - 8.0;
    for row in &mut out {
        for v in row.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }
    Ok(out)
}

fn hz_to_mel(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
}

fn mel_filterbank(sr: f32, n_fft: usize, n_mels: usize, fmin: f32, fmax: f32) -> Vec<Vec<(usize, f32)>> {
    let fft_freqs: Vec<f32> = (0..=n_fft / 2).map(|i| i as f32 * sr / n_fft as f32).collect();
    let (m_lo, m_hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_pts: Vec<f32> = (0..n_mels + 2).map(|i| mel_to_hz(m_lo + (m_hi - m_lo) * i as f32 / (n_mels + 1) as f32)).collect();
    let mut fb = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let (lo, mid, hi) = (mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]);
        let enorm = 2.0 / (hi - lo);
        let mut row = Vec::new();
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (mid - lo);
            let upper = (hi - f) / (hi - mid);
            let w = lower.min(upper).max(0.0);
            if w > 0.0 { row.push((k, w * enorm)); }
        }
        fb.push(row);
    }
    fb
}
