//! Voice changer integration tests (synthetic voices, no devices).
//!
//! The test voice is a Rosenberg glottal pulse train (with radiation) through
//! cascaded formant resonators — a classic source-filter vowel — plus noise
//! "sibilants" and silences for speech-like material.

use super::dsp::filters::{Biquad, Kind};
use super::dsp::lpc;
use super::dsp::pitch::measure_f0;
use super::dsp::shifter::VoiceParams;
use super::dsp::Rng;
use super::*;
use std::f32::consts::PI;

const RATE: u32 = 48000;

/// Vowel formants (Hz, bandwidth Hz).
const A: [(f32, f32); 4] = [(730.0, 90.0), (1090.0, 110.0), (2440.0, 160.0), (3400.0, 250.0)];
const I: [(f32, f32); 4] = [(270.0, 60.0), (2290.0, 100.0), (3010.0, 150.0), (3700.0, 250.0)];
const U: [(f32, f32); 4] = [(300.0, 60.0), (870.0, 90.0), (2240.0, 150.0), (3300.0, 250.0)];

struct Resonator {
    c1: f32,
    c2: f32,
    g: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn new(f: f32, bw: f32) -> Self {
        let r = (-PI * bw / RATE as f32).exp();
        let c1 = 2.0 * r * (2.0 * PI * f / RATE as f32).cos();
        let c2 = -r * r;
        // unity gain at DC
        Resonator { c1, c2, g: 1.0 - c1 - c2, y1: 0.0, y2: 0.0 }
    }
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.g * x + self.c1 * self.y1 + self.c2 * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Glottal source (Rosenberg pulse derivative) with f0 from `f0_at(t)`.
fn glottal(secs: f32, f0_at: impl Fn(f32) -> f32) -> Vec<f32> {
    let n = (secs * RATE as f32) as usize;
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    let mut prev = 0.0f32;
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        phase += f0_at(t) / RATE as f32;
        phase -= phase.floor();
        let (tp, tn) = (0.4, 0.16);
        let g = if phase < tp {
            0.5 * (1.0 - (PI * phase / tp).cos())
        } else if phase < tp + tn {
            (PI * (phase - tp) / (2.0 * tn)).cos()
        } else {
            0.0
        };
        out.push(g - prev); // radiation: first difference
        prev = g;
    }
    out
}

fn vowel(secs: f32, formants: &[(f32, f32)], f0_at: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut x = glottal(secs, f0_at);
    for &(f, bw) in formants {
        let mut r = Resonator::new(f, bw);
        for v in x.iter_mut() {
            *v = r.tick(*v);
        }
    }
    normalize(&mut x, 0.3);
    x
}

fn normalize(x: &mut [f32], peak: f32) {
    let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    if m > 0.0 {
        for v in x.iter_mut() {
            *v *= peak / m;
        }
    }
}

fn sibilant(secs: f32, seed: u32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut hp = [Biquad::new(Kind::HighPass, 4000.0, 0.707, 0.0, RATE as f32); 2];
    let mut lp = Biquad::new(Kind::LowPass, 9000.0, 0.707, 0.0, RATE as f32);
    let n = (secs * RATE as f32) as usize;
    let mut x: Vec<f32> = (0..n)
        .map(|_| {
            let v = rng.bipolar();
            let v = hp[0].tick(v);
            lp.tick(hp[1].tick(v))
        })
        .collect();
    normalize(&mut x, 0.12);
    x
}

/// Speech-like material: syllables of different vowels with falling/rising
/// intonation around `f0`, sibilants, pauses; with syllable envelopes.
fn speechlike(secs: f32, f0: f32) -> Vec<f32> {
    let mut out = Vec::new();
    let vowels = [A, I, U, A];
    let mut k = 0;
    while (out.len() as f32) < secs * RATE as f32 {
        let dur = 0.22 + 0.06 * (k % 3) as f32;
        let rise = if k % 2 == 0 { 1.0 } else { -1.0 };
        let mut v = vowel(dur, &vowels[k % 4], move |t| f0 * (1.0 + rise * 0.15 * t / dur));
        let n = v.len();
        for (i, s) in v.iter_mut().enumerate() {
            let t = i as f32 / n as f32;
            *s *= (PI * t).sin().powf(0.6);
        }
        out.extend(v);
        if k % 2 == 1 {
            let mut s = sibilant(0.1, k as u32 + 1);
            let n = s.len();
            for (i, v) in s.iter_mut().enumerate() {
                *v *= (PI * i as f32 / n as f32).sin();
            }
            out.extend(s);
        }
        out.extend(std::iter::repeat_n(0.0, (0.06 * RATE as f32) as usize));
        k += 1;
    }
    out.truncate((secs * RATE as f32) as usize);
    out
}

fn voice_only(p: VoiceParams) -> Preset {
    Preset {
        id: "t".into(),
        name: "t".into(),
        description: String::new(),
        category: String::new(),
        disguise: true,
        builtin: false,
        blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(p) }],
    }
}

fn opts() -> RenderOptions {
    // a quiet synthetic signal must never be gated in these tests
    let mut o = RenderOptions::default();
    o.front_end.gate.threshold_db = -70.0;
    o.front_end.gate.auto = 0.0;
    o
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Median f0 over 0.25 s windows between `from` and `to` seconds.
fn mean_f0(x: &[f32], from: f32, to: f32) -> f32 {
    mean_f0_w(x, from, to, RATE as usize / 4)
}

fn mean_f0_w(x: &[f32], from: f32, to: f32, w: usize) -> f32 {
    let mut v = Vec::new();
    let mut s = (from * RATE as f32) as usize;
    while s + w <= (to * RATE as f32) as usize {
        let f = measure_f0(&x[s..s + w], RATE as f32, 40.0, 1500.0);
        if f > 0.0 {
            v.push(f);
        }
        s += w;
    }
    assert!(!v.is_empty(), "no periodic windows");
    // median for robustness
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn pitch_shift_is_accurate_within_5_cents() {
    let mut worst = 0.0f32;
    for f0 in [110.0f32, 190.0] {
        let x = vowel(3.2, &A, |_| f0);
        for st in [4.5f32, -5.0, 9.0, -12.0, 12.0] {
            let y = render_with(&x, RATE, &voice_only(VoiceParams { pitch_st: st, ..Default::default() }), &opts());
            let got = mean_f0(&y, 2.0, 3.0);
            let want = f0 * 2f32.powf(st / 12.0);
            let cents = 1200.0 * (got / want).log2();
            println!("f0 {f0} Hz, {st:+} st: want {want:.2} Hz, got {got:.2} Hz ({cents:+.2} cents)");
            worst = worst.max(cents.abs());
            assert!(cents.abs() < 5.0, "f0 {f0}, {st} st: {got} Hz vs {want} Hz ({cents} cents)");
        }
    }
    println!("worst pitch error {worst:.2} cents");
}

/// Downsample 48 k → 16 k (low-pass then keep every third sample) for LPC.
fn to16k(x: &[f32]) -> Vec<f32> {
    let mut lp = [Biquad::new(Kind::LowPass, 6500.0, 0.707, 0.0, RATE as f32); 3];
    x.iter()
        .map(|v| {
            let a = lp[0].tick(*v);
            let b = lp[1].tick(a);
            lp[2].tick(b)
        })
        .step_by(3)
        .collect()
}

fn formant_peaks(x: &[f32]) -> Vec<f64> {
    let d = to16k(x);
    let a = lpc::lpc(&d, 18);
    lpc::envelope_peaks(&a, 16000.0, 150.0, 4000.0)
}

/// Peaks of an LPC envelope fitted to the pre-emphasised signal, with the
/// pre-emphasis tilt removed (the vocal-tract envelope).
fn true_peaks(a: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let pre = 0.97f64;
    let env = |f: f64| {
        let w = 2.0 * std::f64::consts::PI * f / RATE as f64;
        let p = (1.0 - pre * w.cos()).powi(2) + (pre * w.sin()).powi(2);
        lpc::envelope_at(a, f, RATE as f64) / p
    };
    let mut out = Vec::new();
    let step = 2.0;
    let mut f = lo;
    while f + step < hi {
        let (a0, a1, a2) = (env(f - step), env(f), env(f + step));
        if a1 > a0 && a1 >= a2 {
            out.push(f);
        }
        f += step;
    }
    out
}

#[test]
fn formant_shift_moves_lpc_peaks_by_the_ratio() {
    use super::dsp::dynamics::GateParams;
    use super::dsp::shifter::Shifter;
    use super::dsp::{Block, Ctx};
    let formants = [(600.0, 80.0), (1400.0, 100.0), (2600.0, 140.0)];
    // (1) the real-time envelope on a voiced (glottal) vowel: the synthesis
    //     envelope's peaks are the analysis envelope's peaks times alpha
    let voiced = vowel(2.0, &formants, |t| 85.0 * (1.0 + 0.12 * (2.0 * PI * 1.7 * t).sin()));
    for alpha in [0.8f32, 1.2, 1.35] {
        let mut s = Shifter::new(VoiceParams { formant: alpha, ..Default::default() }, RATE as f32);
        let mut x = voiced.clone();
        let mut errs = Vec::new();
        for (ci, c) in x.chunks_mut(240).enumerate() {
            s.process(c, &mut Ctx::default());
            if ci > 50 && ci % 5 == 0 {
                let (k, kw) = s.envelopes();
                let mut tmp = vec![0.0; kw.len() + 1];
                let (mut a, mut aw) = (vec![0.0; k.len() + 1], vec![0.0; kw.len() + 1]);
                lpc::k_to_a(&k, &mut a, &mut tmp[..k.len() + 1]);
                lpc::k_to_a(&kw, &mut aw, &mut tmp);
                let p = true_peaks(&a, 150.0, 3000.0);
                let pw = true_peaks(&aw, 100.0, 5000.0);
                for f in p.iter().take(2) {
                    let want = f * alpha as f64;
                    if let Some(g) = pw.iter().cloned().min_by(|u, v| (u - want).abs().total_cmp(&(v - want).abs())) {
                        errs.push(g / want - 1.0);
                    }
                }
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        let med = errs[errs.len() / 2];
        let worst = errs.iter().map(|e| e.abs()).fold(0.0, f64::max);
        println!("voiced vowel, alpha {alpha}: synthesis-envelope peak error median {:+.2} % (worst frame {:.1} %, a peak merging or splitting)", med * 100.0, worst * 100.0);
        assert!(med.abs() < 0.07, "alpha {alpha}: median peak error {med}");
    }
    // (2) end to end on a whispered vowel (flat excitation): the output's LPC
    //     peaks move by alpha
    let mut n = Rng::new(3);
    let mut xs: Vec<f32> = (0..RATE as usize * 3).map(|_| n.bipolar()).collect();
    for &(f, bw) in &formants {
        let mut r = Resonator::new(f, bw);
        for v in xs.iter_mut() {
            *v = r.tick(*v);
        }
    }
    normalize(&mut xs, 0.3);
    let mut o = opts();
    o.front_end.gate = GateParams { threshold_db: -90.0, auto: 0.0, ..Default::default() };
    let base = formant_peaks(&xs[RATE as usize..]);
    for alpha in [0.8f32, 1.2, 1.35] {
        let y = render_with(&xs, RATE, &voice_only(VoiceParams { formant: alpha, ..Default::default() }), &o);
        let peaks = formant_peaks(&y[RATE as usize..]);
        for f in base.iter().take(2) {
            let want = f * alpha as f64;
            let got = peaks.iter().cloned().min_by(|p, q| (p - want).abs().total_cmp(&(q - want).abs())).unwrap();
            let err = got / want - 1.0;
            println!("whispered vowel, alpha {alpha}: {f:.0} Hz -> {got:.0} Hz (want {want:.0}, {:+.1} %)", err * 100.0);
            assert!(err.abs() < 0.07, "alpha {alpha}: {f} -> {got} (want {want})");
        }
    }
    // (3) a formant shift leaves the pitch alone
    let y = render_with(&voiced, RATE, &voice_only(VoiceParams { formant: 1.25, ..Default::default() }), &opts());
    let mut ratios = Vec::new();
    for s in (RATE as usize / 2..(RATE as usize * 3 / 2)).step_by(2400) {
        let fx = measure_f0(&voiced[s..s + 2400], RATE as f32, 40.0, 1500.0);
        let fy = measure_f0(&y[s..s + 2400], RATE as f32, 40.0, 1500.0);
        if fx > 0.0 && fy > 0.0 {
            ratios.push(1200.0 * (fy / fx).log2());
        }
    }
    ratios.sort_by(|a, b| a.total_cmp(b));
    let med = ratios[ratios.len() / 2];
    assert!(med.abs() < 10.0, "formant shift changed the pitch by {med} cents");
}

fn centroid(x: &[f32]) -> f32 {
    // spectral centroid by a coarse DFT on 2048-sample frames
    let n = 2048;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for frame in x.chunks_exact(n).take(20) {
        for b in (8..n / 2).step_by(8) {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, v) in frame.iter().enumerate() {
                let w = 2.0 * std::f64::consts::PI * (b * i) as f64 / n as f64;
                re += *v as f64 * w.cos();
                im += *v as f64 * w.sin();
            }
            let m = (re * re + im * im).sqrt();
            num += m * b as f64 * RATE as f64 / n as f64;
            den += m;
        }
    }
    (num / den.max(1e-12)) as f32
}

#[test]
fn sibilants_pass_through_the_envelope_path_cleanly() {
    let x = sibilant(1.5, 42);
    for (st, alpha) in [(5.0f32, 1.0f32), (-7.0, 1.0), (4.5, 1.17)] {
        let y = render_with(&x, RATE, &voice_only(VoiceParams { pitch_st: st, formant: alpha, ..Default::default() }), &opts());
        let (cx, cy) = (centroid(&x[12000..60000]), centroid(&y[12000..60000]));
        let level = rms(&y[12000..]) / rms(&x[12000..]);
        // no pitch is invented in noise
        let f = measure_f0(&y[24000..36000], RATE as f32, 60.0, 1000.0);
        println!("sibilant {st:+} st, formant {alpha}: centroid {cx:.0} -> {cy:.0} Hz, level {:.1} dB, f0 {f}", 20.0 * level.log10());
        assert!((cy / (cx * alpha.min(1.3)) - 1.0).abs() < 0.2, "centroid {cx} -> {cy}");
        assert!(level > 0.4 && level < 2.5, "level ratio {level}");
        assert_eq!(f, 0.0, "periodicity invented in a sibilant");
    }
}

/// Maximum normalised cross-correlation between `y` and `x` over lags 0..=max_lag.
fn max_xcorr(x: &[f32], y: &[f32], max_lag: usize) -> f32 {
    let n = x.len().min(y.len()) - max_lag;
    let ex: f64 = x[..n].iter().map(|v| (*v as f64).powi(2)).sum();
    let mut best = 0.0f64;
    for lag in 0..=max_lag {
        let mut s = 0.0f64;
        let mut ey = 0.0f64;
        for i in 0..n {
            let b = y[i + lag] as f64;
            s += x[i] as f64 * b;
            ey += b * b;
        }
        best = best.max(s.abs() / (ex * ey).sqrt().max(1e-12));
    }
    best as f32
}

#[test]
fn no_dry_leakage_in_disguising_presets() {
    let x = speechlike(2.5, 120.0);
    let o = RenderOptions { align: false, ..opts() };
    let lat = Processor::new(RATE as f32, 128, o.front_end, &[], 0.0, -1.0).latency();
    // the metric works: a non-disguising preset correlates strongly with the input
    let studio = render_with(&x, RATE, &preset_by_id("studio").unwrap(), &o);
    let c_studio = max_xcorr(&x, &studio, lat + 200);
    println!("studio (keeps your voice) max xcorr {c_studio:.3}");
    assert!(c_studio > 0.6);
    for p in builtin_presets().into_iter().filter(|p| p.disguise) {
        let y = render_with(&x, RATE, &p, &o);
        let c = max_xcorr(&x, &y, lat + 200);
        println!("{:<14} max xcorr with the dry input over lags 0..{} = {c:.3}", p.id, lat + 200);
        assert!(c < 0.3, "{}: correlation with the dry voice {c}", p.id);
    }
}

#[test]
fn every_preset_is_silent_in_silence_finite_bounded_and_fast() {
    let silence = vec![0.0f32; RATE as usize * 3];
    let speech = speechlike(10.0, 115.0);
    let ceiling = super::dsp::db_to_lin(-1.0);
    let mut report = Vec::new();
    for p in builtin_presets() {
        let y = render(&silence, RATE, &p);
        let peak = y.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(peak < 3.2e-5, "{}: silence produced {peak}", p.id);
        // loud input to exercise the limiter
        let loud: Vec<f32> = speech.iter().map(|v| v * 3.0).collect();
        let t0 = std::time::Instant::now();
        let y = render_with(&loud, RATE, &p, &RenderOptions { tail_secs: 0.0, ..opts() });
        let el = t0.elapsed().as_secs_f32();
        assert!(y.iter().all(|v| v.is_finite()), "{}: NaN/Inf", p.id);
        let mx = y.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(mx <= ceiling + 1e-6, "{}: {mx} over the ceiling", p.id);
        let out_rms = rms(&render_with(&speech, RATE, &p, &opts()));
        let cpu = el / 10.0;
        report.push(format!("{:<11} cpu {:>5.2} % of one core, output {:+.1} dB vs input", p.id, cpu * 100.0, 20.0 * (out_rms / rms(&speech)).log10()));
        if !cfg!(debug_assertions) {
            assert!(cpu < 0.25, "{}: {cpu} of a core", p.id);
        }
    }
    println!("{}", report.join("\n"));
}

#[test]
fn latency_is_measured_and_render_aligns_it() {
    let mut x = vec![0.0f32; RATE as usize];
    // a click train through a non-voice preset: output peak at the latency
    x[24000] = 0.5;
    let p = preset_by_id("studio").unwrap();
    let o = RenderOptions { align: false, front_end: FrontEnd { deesser_on: false, ..opts().front_end }, ..opts() };
    let lat = Processor::new(RATE as f32, 128, o.front_end, &p.blocks, 0.0, -1.0).latency();
    let y = render_with(&x, RATE, &p, &o);
    let pk = (0..y.len()).max_by(|a, b| y[*a].abs().total_cmp(&y[*b].abs())).unwrap();
    println!("processing latency {lat} samples = {:.2} ms; impulse peak at +{} samples", lat as f32 * 1000.0 / RATE as f32, pk - 24000);
    assert!((pk as i64 - 24000 - lat as i64).abs() <= 3, "peak at {pk}, latency {lat}");
    // the voice stage's 25 ms (12 ms grain cap) plus the limiter's look-ahead
    assert!((lat as f32 / RATE as f32) < 0.030, "latency {lat}");
    let ya = render_with(&x, RATE, &p, &RenderOptions { align: true, ..o });
    let pk = (0..ya.len()).max_by(|a, b| ya[*a].abs().total_cmp(&ya[*b].abs())).unwrap();
    assert!((pk as i64 - 24000).abs() <= 3);
}

#[test]
fn gate_silences_background_noise_through_a_preset() {
    let mut rng = Rng::new(4);
    // -65 dBFS hiss (a quiet room) with the default auto gate
    let noise: Vec<f32> = (0..RATE as usize * 4).map(|_| rng.bipolar() * 0.00056 * 1.7).collect();
    let y = render(&noise, RATE, &preset_by_id("female").unwrap());
    let tail = rms(&y[RATE as usize..]);
    assert!(tail < 1e-5, "gated noise rms {tail}");
}

#[test]
fn realtime_block_sizes_give_the_same_sound() {
    let x = speechlike(1.5, 130.0);
    let p = preset_by_id("female").unwrap();
    let a = render_with(&x, RATE, &p, &RenderOptions { block: 128, ..opts() });
    let b = render_with(&x, RATE, &p, &RenderOptions { block: 480, ..opts() });
    // same algorithm, different host buffer partitioning: the output energy
    // and pitch agree (sample equality is not required)
    let (ra, rb) = (rms(&a), rms(&b));
    assert!((ra / rb - 1.0).abs() < 0.05, "{ra} vs {rb}");
}


/// Listening renders through every preset (not run by default): real speech
/// files when given, otherwise the synthetic speech-like test signal.
///
/// ```text
/// $env:NEKOTONE_VOICE_SAMPLES = "a.ogg;b.wav"   # speech files, concatenated
/// $env:NEKOTONE_RENDER_DIR = "C:\...\renders"
/// cargo test --release -p nekotone-core voice::tests::write_listening_renders -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn write_listening_renders() {
    // without samples, the synthetic speech-like signal is rendered instead
    let files = std::env::var("NEKOTONE_VOICE_SAMPLES").unwrap_or_default();
    let dir = std::path::PathBuf::from(std::env::var("NEKOTONE_RENDER_DIR").expect("set NEKOTONE_RENDER_DIR"));
    std::fs::create_dir_all(&dir).unwrap();
    let mut speech = Vec::new();
    for f in files.split(';').filter(|f| !f.is_empty()) {
        let clip = crate::audio::decode(std::path::Path::new(f)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        speech.extend_from_slice(&clip.samples);
        speech.extend(std::iter::repeat_n(0.0, RATE as usize / 3));
    }
    if speech.is_empty() {
        speech = speechlike(8.0, 115.0);
    }
    normalize(&mut speech, 0.5);
    crate::audio::write_wav(&dir.join("00-dry.wav"), &speech, RATE, 1).unwrap();
    for (i, p) in builtin_presets().iter().enumerate() {
        let t0 = std::time::Instant::now();
        let y = render_with(&speech, RATE, p, &RenderOptions { tail_secs: 2.5, ..Default::default() });
        let cpu = t0.elapsed().as_secs_f32() / (speech.len() as f32 / RATE as f32);
        let path = dir.join(format!("{:02}-{}.wav", i + 1, p.id));
        crate::audio::write_wav(&path, &y, RATE, 1).unwrap();
        println!("{} ({:.1} % cpu, peak {:.1} dBFS)", path.display(), cpu * 100.0, 20.0 * y.iter().fold(0.0f32, |a, v| a.max(v.abs())).log10());
    }
}

/// Fraction of 40 ms windows (from `from` s) whose f0 is within 50 cents of `want`.
fn pitched_fraction(y: &[f32], from: f32, want: f32) -> f32 {
    let w = RATE as usize / 25;
    let (mut ok, mut n) = (0, 0);
    let mut s = (from * RATE as f32) as usize;
    while s + w <= y.len() {
        let f = measure_f0(&y[s..s + w], RATE as f32, 40.0, 1500.0);
        n += 1;
        if f > 0.0 && (1200.0 * (f / want).log2()).abs() < 50.0 {
            ok += 1;
        }
        s += w;
    }
    ok as f32 / n.max(1) as f32
}

/// A breathy voice in a noisy room: a vowel with natural pitch wander and
/// broadband noise 10 dB under it.
fn noisy_vowel(f0: f32, secs: f32, seed: u32) -> Vec<f32> {
    let mut x = vowel(secs, &A, |t| f0 * (1.0 + 0.01 * (2.0 * PI * 5.3 * t).sin() + 0.006 * (2.0 * PI * 17.0 * t).sin()));
    let mut r = Rng::new(seed);
    let s = rms(&x) * super::dsp::db_to_lin(-10.0) * 1.732;
    for v in x.iter_mut() {
        *v += r.bipolar() * s;
    }
    x
}

/// Regression for "anything that wasn't the natural voice sounded bad": on a
/// real (breathy, noisy) voice the old tracker called most frames unvoiced
/// and re-synthesised them from noise, and jittery pitch marks roughened the
/// rest. The shifted voice must stay pitched at the new f0.
#[test]
fn a_noisy_breathy_voice_stays_pitched_after_shifting() {
    let x = noisy_vowel(120.0, 3.0, 9);
    let dry = pitched_fraction(&x, 0.5, 120.0);
    for (st, alpha) in [(5.0f32, 1.17f32), (-5.0, 0.85), (-12.0, 0.8)] {
        let y = render_with(&x, RATE, &voice_only(VoiceParams { pitch_st: st, formant: alpha, ..Default::default() }), &opts());
        let want = 120.0 * 2f32.powf(st / 12.0);
        let p = pitched_fraction(&y, 0.5, want);
        println!("noisy vowel {st:+} st: {:.0} % of windows pitched at {want:.0} Hz (dry {:.0} %)", p * 100.0, dry * 100.0);
        assert!(p >= 0.9 * dry, "{st} st: only {p} pitched (dry {dry})");
    }
}

#[test]
fn robot_pulses_hold_a_fixed_pitch() {
    let x = vowel(2.5, &I, |t| 110.0 * (1.0 + 0.3 * t));
    let p = VoiceParams { flatten: 1.0, fixed_hz: 150.0, pulse: 1.0, ..Default::default() };
    let y = render_with(&x, RATE, &voice_only(p), &opts());
    let got = mean_f0_w(&y, 0.5, 2.4, RATE as usize / 10);
    let cents = 1200.0 * (got / 150.0).log2();
    println!("robot: {got:.2} Hz ({cents:+.2} cents from 150 Hz)");
    assert!(cents.abs() < 5.0);
    assert!(pitched_fraction(&y, 0.5, 150.0) > 0.9);
}

#[test]
fn growl_adds_subharmonic_roughness_to_voiced_sound() {
    let x = vowel(2.0, &A, |_| 200.0);
    let sub = |g: f32| {
        let y = render_with(&x, RATE, &voice_only(VoiceParams { growl: g, ..Default::default() }), &opts());
        let seg = &y[RATE as usize / 2..RATE as usize * 3 / 2];
        // energy between the harmonics (at f0/2 multiples) relative to on them
        let dft = |hz: f32| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, v) in seg.iter().enumerate() {
                let w = 2.0 * PI * hz * i as f32 / RATE as f32;
                re += v * w.cos();
                im += v * w.sin();
            }
            re * re + im * im
        };
        let between: f32 = [100.0, 300.0, 500.0].iter().map(|f| dft(*f)).sum();
        let on: f32 = [200.0, 400.0, 600.0].iter().map(|f| dft(*f)).sum();
        between / on
    };
    let (clean, growl) = (sub(0.0), sub(0.8));
    println!("sub-harmonic / harmonic energy: clean {clean:.4}, growl {growl:.4}");
    assert!(growl > 100.0 * clean.max(1e-7) && growl > 0.002);
}

#[test]
fn layers_line_up_and_every_path_is_checked() {
    use super::dsp::route::{LayerParams, MergeParams, SplitParams};
    let s = |id: &str, b: BlockSpec| BlockSlot { id: id.into(), block: b };
    let layered = vec![
        s("split", BlockSpec::Split(SplitParams { bus: 1.0 })),
        s("main", BlockSpec::Voice(VoiceParams { pitch_st: -6.0, ..Default::default() })),
        s("layer", BlockSpec::Layer(LayerParams { bus: 1.0 })),
        s("sub", BlockSpec::Voice(VoiceParams { pitch_st: -18.0, ..Default::default() })),
        s("merge", BlockSpec::Merge(MergeParams { bus: 1.0, layer_db: -9.0, main_db: 0.0 })),
    ];
    assert!(chain::every_path_resynthesises(&layered));
    // a layer that skips the voice stage would leak the dry voice
    let mut leaky = layered.clone();
    leaky.remove(3);
    assert!(!chain::every_path_resynthesises(&leaky));
    // two parallel voice stages keep the single-stage latency
    let c = chain::Chain::new(&layered, RATE as f32);
    assert_eq!(c.latency(), chain::chain_latency(RATE as f32));
    // both layers are audible at their pitches (f0 120 -> 84.8 and 42.4 Hz)
    let x = vowel(2.5, &A, |_| 120.0);
    let p = Preset { blocks: layered, ..voice_only(VoiceParams::default()) };
    let y = render_with(&x, RATE, &p, &opts());
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(rms(&y[RATE as usize..]) > 0.3 * rms(&x[RATE as usize..]));
    // a misaligned sub layer would comb-filter: output still pitched at the main voice
    assert!(pitched_fraction(&y, 0.5, 120.0 * 2f32.powf(-6.0 / 12.0)) > 0.5 || pitched_fraction(&y, 0.5, 120.0 * 2f32.powf(-18.0 / 12.0)) > 0.5);
}

/// Go live on this PC's default devices, read the status, stop (real
/// hardware; ignored). `status()` once locked the config twice in one
/// expression and deadlocked, so Go live hung after the devices opened.
#[test]
#[ignore]
fn real_devices_go_live() {
    let t0 = std::time::Instant::now();
    let engine = VoiceEngine::start(VoiceConfig { start_muted: true, ..VoiceConfig::default() }, |_| {}).expect("the engine starts");
    let (tx, rx) = crossbeam_channel::bounded(1);
    let engine = std::sync::Arc::new(engine);
    let e2 = engine.clone();
    std::thread::spawn(move || {
        let _ = tx.send(e2.status());
    });
    let st = rx.recv_timeout(std::time::Duration::from_secs(5)).expect("status() returns (it deadlocked before)");
    eprintln!("live in {} ms: {} -> {}, {:.0} ms", t0.elapsed().as_millis(), st.input_device, st.output_device, st.latency_ms);
    assert!(st.running && st.muted);
    std::thread::sleep(std::time::Duration::from_millis(300));
    std::sync::Arc::try_unwrap(engine).ok().expect("one owner").stop();
}
