//! Voices with targets, checked by measurement: different speakers through
//! the same preset must come out at the preset's target pitch and vocal
//! tract, and the same as each other. The old fixed-shift presets are
//! measured alongside for comparison.
//!
//! Real recordings (listening renders + the same table):
//! ```text
//! NEKOTONE_ADAPT_WAVS="man=a.wav;woman=b.wav" NEKOTONE_ADAPT_OUT=renders \
//!   cargo test --release -p nekotone-core --lib voice::adaptive_tests -- --ignored --nocapture
//! ```

use super::dsp::profile::tests::talker_truth;
use super::dsp::profile::{measure, Profile};
use super::*;

const RATE: u32 = 48000;

fn opts() -> RenderOptions {
    let mut o = RenderOptions::default();
    // synthetic voices: never gated
    o.front_end.gate.threshold_db = -70.0;
    o.front_end.gate.auto = 0.0;
    o
}

fn cents(a: f32, b: f32) -> f32 {
    1200.0 * (a / b).log2()
}

/// The main voice stage's targets (the first voice block).
fn targets(p: &Preset) -> Option<(f32, f32)> {
    p.blocks.iter().find_map(|b| match b.block {
        BlockSpec::Voice(v) if v.target_hz > 0.0 => Some((v.target_hz, v.target_tract)),
        _ => None,
    })
}

/// Pitch ratio an accent's phrase melody settles at on a long phrase (the
/// test talkers speak in long phrases, so the rise or fall sits at its cap;
/// the lilt averages out).
fn accent_melody_ratio(p: &Preset) -> f32 {
    use super::dsp::accent::{accent_index, melody};
    p.blocks
        .iter()
        .find_map(|b| match b.block {
            BlockSpec::Voice(v) if v.accent > 0.0 => {
                let m = melody(accent_index(v.accent), v.accent_amount);
                Some(2f32.powf(m.cap_st * m.slope_st_s.signum() / 12.0))
            }
            _ => None,
        })
        .unwrap_or(1.0)
}

fn layered(p: &Preset) -> bool {
    p.blocks.iter().any(|b| matches!(b.block, BlockSpec::Split(_)))
}

fn whispered(p: &Preset) -> bool {
    p.blocks.iter().any(|b| matches!(b.block, BlockSpec::Voice(v) if v.whisper > 0.5))
}

/// Profile of a render's settled part once the adaptive voice has locked.
///
/// Profile of a render's settled part: from 3 s (the voice has adapted)
/// to 12 s (the talkers' length).
fn settled(y: &[f32]) -> Profile {
    measure(&y[(3 * RATE) as usize..(12 * RATE) as usize], RATE as f32)
}

/// The fixed-shift version of a preset (as before targets), for comparison.
fn legacy(id: &str) -> Option<(f32, f32)> {
    Some(match id {
        "female" => (5.0, 1.17),
        "male" => (-5.0, 0.85),
        "child" => (6.0, 1.25),
        "mouse" => (9.0, 1.35),
        "giant" => (-9.0, 0.72),
        "narrator" => (-3.0, 0.93),
        "alien" => (3.0, 1.25),
        "wuuwuu" => (2.0, 1.1),
        _ => return None,
    })
}

fn with_fixed_shift(p: &Preset, st: f32, formant: f32) -> Preset {
    let mut q = p.clone();
    for b in q.blocks.iter_mut() {
        if let BlockSpec::Voice(v) = &mut b.block {
            v.target_hz = 0.0;
            v.target_tract = 0.0;
            v.pitch_st = st;
            v.formant = formant;
            break;
        }
    }
    q
}

#[test]
fn every_speaker_lands_on_the_voice_target() {
    let r = RATE as f32;
    let (man, man_f0) = talker_truth(r, 110.0, 1.0, 12.0, 1);
    let (woman, woman_f0) = talker_truth(r, 210.0, 1.17, 12.0, 2);
    // what the adaptive shift is based on (the whole take) and the measured window's own median
    let (dm, dw) = (measure(&man, r), measure(&woman, r));
    let (wm, ww) = (settled(&man), settled(&woman));
    // the output of a speaker in the window should be target × (window median / profile)
    let expect = |t: f32, dry: &Profile, win: &Profile| t * win.f0_hz / dry.f0_hz;
    let mut report = vec![format!(
        "speakers: man {man_f0:.0} Hz (window {:.0}), woman {woman_f0:.0} Hz (window {:.0}); dry tract {:.2}, {:.2}",
        wm.f0_hz, ww.f0_hz, dm.tract, dw.tract
    )];
    let mut failures = Vec::new();
    let mut checked = 0;
    for p in builtin_presets() {
        let Some((t_hz, t_tract)) = targets(&p) else { continue };
        checked += 1;
        let (pm, pw) = (settled(&render_with(&man, RATE, &p, &opts())), settled(&render_with(&woman, RATE, &p, &opts())));
        let acc = accent_melody_ratio(&p);
        let (em, ew) = (expect(t_hz, &dm, &wm) * acc, expect(t_hz, &dw, &ww) * acc);
        let gap = cents(pm.f0_hz / em, pw.f0_hz / ew);
        let line = format!(
            "{:<9} target {:>3.0} Hz / {:.2}: man {:>5.1} Hz ({:>+4.0} c), woman {:>5.1} Hz ({:>+4.0} c), apart {:>+4.0} c",
            p.id, t_hz, t_tract, pm.f0_hz, cents(pm.f0_hz, em), pw.f0_hz, cents(pw.f0_hz, ew), gap
        );
        report.push(line);
        if whispered(&p) || pm.voiced_secs < 1.0 || pw.voiced_secs < 1.0 {
            // mostly noise: nothing to hold a pitch to
            continue;
        }
        checked += 1;
        if gap.abs() > 100.0 {
            failures.push(format!("{}: man and woman {gap:.0} cents apart", p.id));
        }
        if layered(&p) {
            // two voices an octave apart: the mixture's f0 is ambiguous; agreement is checked above
            continue;
        }
        // (the tract read back from processed audio is only informative: the
        // estimator is thrown by F1 near low harmonics and F4 at its band's
        // edge; the tract path is checked link by link in the tests below)
        for (who, got, e) in [("man", pm, em), ("woman", pw, ew)] {
            if cents(got.f0_hz, e).abs() > 100.0 {
                failures.push(format!("{}: {who} at {:.1} Hz, expected {e:.1}", p.id, got.f0_hz));
            }
        }
    }
    println!("{}", report.join("\n"));
    assert!(checked >= 8, "only {checked} presets checked:\n{}", report.join("\n"));
    assert!(failures.is_empty(), "{}\n\n{}", failures.join("\n"), report.join("\n"));
}

#[test]
fn a_calibration_is_used_from_the_first_word() {
    // With a saved profile the very first second is already on target.
    let (woman, _) = talker_truth(RATE as f32, 210.0, 1.17, 3.0, 5);
    let cal = measure(&talker_truth(RATE as f32, 210.0, 1.17, 6.0, 6).0, RATE as f32);
    let p = preset_by_id("male").unwrap();
    let y = render_with(&woman, RATE, &p, &RenderOptions { profile: Some(cal), ..opts() });
    let first = measure(&y[..(1.5 * RATE as f32) as usize], RATE as f32);
    assert!(cents(first.f0_hz, 110.0).abs() < 150.0, "first 1.5 s at {:.1} Hz", first.f0_hz);
}

#[test]
fn fixed_shift_voices_are_unchanged() {
    // No target: the shift is what the preset says, whoever speaks.
    let (man, man_f0) = talker_truth(RATE as f32, 110.0, 1.0, 12.0, 1);
    let p = Preset {
        id: "fixed".into(),
        name: "Fixed".into(),
        description: String::new(),
        category: String::new(),
        disguise: true,
        builtin: false,
        blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(super::dsp::shifter::VoiceParams { pitch_st: 7.0, ..Default::default() }) }],
    };
    let got = settled(&render_with(&man, RATE, &p, &opts()));
    assert!(cents(got.f0_hz, man_f0 * 2f32.powf(7.0 / 12.0)).abs() < 60.0, "{got:?}");
}

/// Real recordings through every adaptive voice: renders to listen to and the same table.
#[test]
#[ignore]
fn adaptive_renders_of_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let out = std::path::PathBuf::from(std::env::var("NEKOTONE_ADAPT_OUT").unwrap_or_else(|_| "adaptive-renders".into()));
    std::fs::create_dir_all(&out).unwrap();
    let mut inputs = Vec::new();
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        let prof = measure(&clip.samples, RATE as f32);
        println!("{name:<8} {:>5.1} Hz, spread {:.1} st, tract {:.2}", prof.f0_hz, prof.spread_st, prof.tract);
        crate::audio::write_wav(&out.join(format!("{name}__dry.wav")), &clip.samples, RATE, 1).unwrap();
        inputs.push((name.to_string(), clip.samples));
    }
    let real = RenderOptions { tail_secs: 1.0, ..Default::default() };
    // NEKOTONE_ADAPT_ONLY=female,male,… renders just those voices
    let only: Option<Vec<String>> = std::env::var("NEKOTONE_ADAPT_ONLY").ok().map(|s| s.split(',').map(|v| v.trim().to_string()).collect());
    for p in builtin_presets() {
        if only.as_ref().is_some_and(|o| !o.contains(&p.id)) {
            continue;
        }
        let Some((t_hz, t_tract)) = targets(&p) else { continue };
        let mut row = format!("{:<9} target {:>3.0} Hz / {:.2}:", p.id, t_hz, t_tract);
        for (name, x) in &inputs {
            let y = render_with(x, RATE, &p, &real);
            crate::audio::write_wav(&out.join(format!("{}__{name}.wav", p.id)), &y, RATE, 1).unwrap();
            let got = measure(&y[(y.len() / 4)..], RATE as f32);
            row += &format!("  {name} → {:>5.1} Hz / {:.2}", got.f0_hz, got.tract);
            if let Some((st, f)) = legacy(&p.id) {
                let q = with_fixed_shift(&p, st, f);
                let yl = render_with(x, RATE, &q, &real);
                crate::audio::write_wav(&out.join(format!("{}-fixed__{name}.wav", p.id)), &yl, RATE, 1).unwrap();
                let lg = measure(&yl[(yl.len() / 4)..], RATE as f32);
                row += &format!(" (fixed {:>5.1})", lg.f0_hz);
            }
        }
        println!("{row}");
    }
    println!("renders in {}", out.display());
}



/// Link 2 of the tract path: the voice stage applies target / measured tract.
#[test]
fn the_formant_ratio_is_target_over_the_speakers_tract() {
    use super::dsp::shifter::{Shifter, VoiceParams};
    use super::dsp::{Block, Ctx};
    for tract in [0.9f32, 1.0, 1.17, 1.3] {
        let mut s = Shifter::new(VoiceParams { target_tract: 1.17, ..Default::default() }, 48000.0);
        let mut ctx = Ctx { rate: 48000.0, profile: Profile { tract, voiced_secs: 10.0, ..Profile::NEUTRAL }, ..Default::default() };
        let mut buf = vec![0.0f32; 256];
        for _ in 0..800 {
            s.process(&mut buf, &mut ctx);
        }
        let (_, alpha) = s.adaptive_shift();
        assert!((alpha - 1.17 / tract).abs() < 0.005, "tract {tract}: ratio {alpha}");
    }
}

/// Link 3: the warp moves the formants by that ratio, for low and high voices
/// (the analysis window once left most of a 210 Hz voice's formants in the
/// residual, where x1.24 moved them by 0 %). F2 and F3 are the reliable ones.
#[test]
fn the_warp_moves_f2_and_f3_by_the_ratio_for_low_and_high_voices() {
    use super::dsp::lpc::{envelope_peaks, lpc};
    use super::dsp::shifter::VoiceParams;
    let r = RATE as f32;
    let formants = [730.0f32, 1090.0, 2440.0, 3500.0];
    for f0 in [110.0f32, 210.0] {
        let x: Vec<f32> = {
            let n = (3.0 * r) as usize;
            let mut ph = 0.0f32;
            let mut st = [[0.0f32; 2]; 4];
            let mut out = vec![0.0; n];
            for v in out.iter_mut() {
                ph += f0 / r;
                let mut y = 0.0;
                if ph >= 1.0 {
                    ph -= 1.0;
                    y = 1.0;
                }
                for (k, fr) in formants.iter().enumerate() {
                    let rr = (-std::f32::consts::PI * (60.0 + 0.05 * fr) / r).exp();
                    let c = 2.0 * rr * (2.0 * std::f32::consts::PI * fr / r).cos();
                    let yn = y + c * st[k][0] - rr * rr * st[k][1];
                    st[k][1] = st[k][0];
                    st[k][0] = yn;
                    y = yn * (1.0 - rr);
                }
                *v = y;
            }
            out
        };
        // formant peaks of a steady stretch (decimated to 12 kHz, order 14)
        let peaks = |y: &[f32]| {
            let seg = &y[(1.5 * r) as usize..(1.5 * r) as usize + 12000];
            let d: Vec<f32> = seg.chunks(4).map(|c| c.iter().sum::<f32>() / 4.0).collect();
            let pre: Vec<f32> = d.windows(2).map(|w| w[1] - 0.9 * w[0]).collect();
            envelope_peaks(&lpc(&pre, 14), 12000.0, 180.0, 4800.0)
        };
        let near = |ps: &[f64], want: f32| ps.iter().map(|p| *p as f32).min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs())).unwrap_or(0.0);
        for a in [0.8f32, 1.24] {
            let p = Preset {
                id: "x".into(),
                name: "x".into(),
                description: String::new(),
                category: String::new(),
                disguise: true,
                builtin: false,
                blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(VoiceParams { formant: a, ..Default::default() }) }],
            };
            let ps = peaks(&render_with(&x, RATE, &p, &opts()));
            // F3 exactly; F2 within 6 % (read next to F1, an order-14 fit
            // pulls a close pair apart a little)
            for (want, tol) in [(formants[1] * a, 0.06), (formants[2] * a, 0.03)] {
                let got = near(&ps, want);
                assert!((got / want - 1.0).abs() < tol, "f0 {f0}, x{a}: peak near {want:.0} Hz at {got:.0} ({ps:?})");
            }
        }
    }
}

/// Speech level of every adaptive voice on real recordings vs the dry level (for trims).
#[test]
#[ignore]
fn adaptive_levels_of_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let mut inputs = Vec::new();
    for item in list.split(';') {
        let (_, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        inputs.push(clip.samples);
    }
    let level = |x: &[f32]| super::dsp::timbre::tests::balance(x, RATE as f32).0;
    let dry: f32 = inputs.iter().map(|x| level(x)).sum::<f32>() / inputs.len() as f32;
    println!("dry speech level {dry:.1} dBFS");
    for p in builtin_presets() {
        let outs: Vec<f32> = inputs.iter().map(|x| level(&render_with(x, RATE, &p, &RenderOptions::default())[(RATE * 3) as usize..])).collect();
        let mean = outs.iter().sum::<f32>() / outs.len() as f32;
        let spread = outs.iter().cloned().fold(f32::MIN, f32::max) - outs.iter().cloned().fold(f32::MAX, f32::min);
        println!("{:<11} {:>+5.1} dB vs dry (spread across speakers {spread:.1} dB){}", p.id, mean - dry, if targets(&p).is_some() { "  [adaptive]" } else { "" });
    }
}

/// JSON of the built-in presets and block types (for UI checks outside the app).
#[test]
#[ignore]
fn dump_catalogue() {
    let dir = std::env::var("NEKOTONE_DUMP_DIR").expect("set NEKOTONE_DUMP_DIR");
    std::fs::write(format!("{dir}/presets.json"), serde_json::to_string(&builtin_presets()).unwrap()).unwrap();
    std::fs::write(format!("{dir}/blocks.json"), serde_json::to_string(&block_types()).unwrap()).unwrap();
}

/// Long-term average power spectrum (2048-point Hann, hop 1024): power per bin.
fn ltas(y: &[f32]) -> Vec<f64> {
    use realfft::RealFftPlanner;
    let n = 2048;
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(n);
    let win: Vec<f32> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos()).collect();
    let mut acc = vec![0.0f64; n / 2 + 1];
    let (mut buf, mut spec) = (fft.make_input_vec(), fft.make_output_vec());
    let mut pos = 0;
    while pos + n <= y.len() {
        for i in 0..n {
            buf[i] = y[pos + i] * win[i];
        }
        fft.process(&mut buf, &mut spec).unwrap();
        for (a, c) in acc.iter_mut().zip(spec.iter()) {
            *a += c.norm_sqr() as f64;
        }
        pos += n / 2;
    }
    acc
}

/// Energy (dB) of `lo..hi` Hz in a long-term spectrum.
fn band_db(spec: &[f64], lo: f32, hi: f32) -> f32 {
    let hz = RATE as f32 / 2048.0;
    let e: f64 = spec.iter().enumerate().filter(|(i, _)| (*i as f32 * hz) >= lo && (*i as f32 * hz) < hi).map(|(_, v)| *v).sum();
    10.0 * e.max(1e-30).log10() as f32
}

/// Harmonics-to-noise ratio (dB) in 1.5-5 kHz, the band where breath lives:
/// normalised autocorrelation at the pitch period (from a tracker on the
/// full signal) over 40 ms voiced frames, median.
fn hnr_high_db(y: &[f32]) -> f32 {
    super::dsp::style::hnr_high(y, RATE as f32).0
}

/// Identity beyond pitch and tract: breathy speakers are breathier, gravelly
/// ones rougher, the nasal one more nasal, measured on the output.
#[test]
fn speakers_differ_in_more_than_pitch() {
    let r = RATE as f32;
    // a steady pitch: harmonics-to-noise needs one (a gliding melody masks it)
    let (man, _) = super::dsp::profile::tests::talker_with(r, 110.0, 1.0, 10.0, 1, 0.0);
    let cues = |id: &str| -> (f32, f32) {
        let p = preset_by_id(id).unwrap();
        let y = render_with(&man, RATE, &p, &opts());
        let y = &y[(3 * RATE) as usize..];
        let sp = ltas(y);
        (hnr_high_db(y), band_db(&sp, 1100.0, 1600.0) - band_db(&sp, 400.0, 2500.0))
    };
    let ids = ["emma", "olivia", "margaret", "nina", "james", "walter", "liam", "neil"];
    let c: std::collections::BTreeMap<&str, (f32, f32)> = ids.into_iter().map(|id| (id, cues(id))).collect();
    for (id, (hnr, nasal)) in &c {
        println!("{id:<9} HNR 1.5-5 kHz {hnr:>5.1} dB   1.1-1.6 kHz re mids {nasal:+.1} dB");
    }
    // On this clean synthetic source breath shows as 1.4-2.1 dB less
    // periodicity; on recorded voices breath 0.5 is 3-4 dB
    // (identity_controls_on_recordings).
    assert!(c["margaret"].0 < c["emma"].0 - 1.0, "Margaret should be breathier than Emma");
    assert!(c["nina"].0 < c["olivia"].0 - 1.0, "Nina should be breathier than Olivia");
    assert!(c["walter"].0 < c["james"].0 - 1.0, "Walter should be rougher than James");
    assert!(c["neil"].1 > c["liam"].1 + 3.0, "Neil should be more nasal than Liam");
}


/// Warble on real recordings: short-term f0 wobble of the output (median
/// |Δf0| per 10 ms, cents) against the input's, with pulse snapping off/on.
#[test]
#[ignore]
fn warble_of_recordings() {
    use super::dsp::pitch::PitchTracker;
    use super::dsp::shifter::{Tuning, TUNE};
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let wobble = |y: &[f32]| {
        let mut t = PitchTracker::new(RATE as f32);
        let (mut prev, mut d) = (0.0f32, Vec::new());
        for (i, v) in y.iter().enumerate() {
            t.push(*v);
            if i % 480 == 479 {
                let e = t.analyze();
                if e.voiced && prev > 0.0 {
                    d.push((1200.0 * (e.f0 / prev).log2()).abs());
                }
                prev = if e.voiced { e.f0 } else { 0.0 };
            }
        }
        d.sort_by(|a, b| a.total_cmp(b));
        (d[d.len() / 2], d[d.len() * 9 / 10], hnr_high_db(y))
    };
    for item in list.split(';') {
        let (name, file) = item.split_once('=').unwrap();
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        let (m, p90, h) = wobble(&clip.samples);
        println!("{name:<7} dry: wobble median {m:.1} c, 90% {p90:.1} c, HNR {h:.1} dB");
        for id in ["female", "male", "james", "emma"] {
            let p = preset_by_id(id).unwrap();
            let mut row = format!("   {id:<7}");
            for snap in [0.0f32, 0.06] {
                *TUNE.lock().unwrap() = Some(Tuning { snap, ..Tuning::default() });
                let y = render_with(&clip.samples, RATE, &p, &RenderOptions::default());
                let (m, p90, h) = wobble(&y[(RATE * 2) as usize..]);
                row += &format!(" | snap {snap:.2}: wobble {m:.1}/{p90:.1} c, HNR {h:.1} dB");
            }
            *TUNE.lock().unwrap() = None;
            println!("{row}");
        }
    }
}

/// What the identity controls do to recorded voices (breath, growl, jitter).
#[test]
#[ignore]
fn identity_controls_on_recordings() {
    use super::dsp::shifter::VoiceParams;
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let mk = |v: VoiceParams| Preset { id: "x".into(), name: "x".into(), description: String::new(), category: String::new(), disguise: true, builtin: false, blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(v) }] };
    for item in list.split(';') {
        let (name, file) = item.split_once('=').unwrap();
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        let base = VoiceParams { target_hz: 150.0, target_tract: 1.08, ..Default::default() };
        let mut row = format!("{name:<7}");
        for (label, v) in [("plain", base), ("breath .3", VoiceParams { breath: 0.3, ..base }), ("breath .6", VoiceParams { breath: 0.6, ..base }), ("breath 1", VoiceParams { breath: 1.0, ..base }), ("growl .15", VoiceParams { growl: 0.15, ..base })] {
            let y = render_with(&clip.samples, RATE, &mk(v), &RenderOptions::default());
            row += &format!(" | {label}: HNR {:.1}", hnr_high_db(&y[(RATE * 2) as usize..]));
        }
        println!("{row}");
    }
}

/// A steady vowel (pulses at `f0` through four formants), 3 s.
fn steady_vowel(f0: f32, formants: [f32; 4]) -> Vec<f32> {
    let r = RATE as f32;
    let n = (3.0 * r) as usize;
    let mut ph = 0.0f32;
    let mut st = [[0.0f32; 2]; 4];
    let y: Vec<f32> = (0..n)
        .map(|_| {
            ph += f0 / r;
            let mut y = 0.0;
            if ph >= 1.0 {
                ph -= 1.0;
                y = 1.0;
            }
            for (k, fr) in formants.iter().enumerate() {
                let rr = (-std::f32::consts::PI * (60.0 + 0.05 * fr) / r).exp();
                let c = 2.0 * rr * (2.0 * std::f32::consts::PI * fr / r).cos();
                let yn = y + c * st[k][0] - rr * rr * st[k][1];
                st[k][1] = st[k][0];
                st[k][0] = yn;
                y = yn * (1.0 - rr);
            }
            y
        })
        .collect();
    // at a speaking level (rms 0.1)
    let rms = (y.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt().max(1e-12);
    y.iter().map(|v| v * 0.1 / rms).collect()
}

fn accented(accent: f32) -> Preset {
    use super::dsp::shifter::VoiceParams;
    Preset { id: "x".into(), name: "x".into(), description: String::new(), category: String::new(), disguise: true, builtin: false, blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(VoiceParams { accent, ..Default::default() }) }] }
}

/// Peaks of a steady stretch of `y` (12 kHz, order 14).
fn formants_of(y: &[f32]) -> Vec<f32> {
    use super::dsp::lpc::{envelope_peaks, lpc};
    let r = RATE as f32;
    let seg = &y[(1.5 * r) as usize..(1.5 * r) as usize + 12000];
    let d: Vec<f32> = seg.chunks(4).map(|c| c.iter().sum::<f32>() / 4.0).collect();
    let pre: Vec<f32> = d.windows(2).map(|w| w[1] - 0.9 * w[0]).collect();
    envelope_peaks(&lpc(&pre, 14), 12000.0, 180.0, 4800.0).into_iter().map(|f| f as f32).collect()
}

fn near(ps: &[f32], want: f32) -> f32 {
    ps.iter().copied().min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs())).unwrap_or(0.0)
}

#[test]
fn accent_colour_moves_vowels() {
    let goose = steady_vowel(110.0, [300.0, 870.0, 2240.0, 3500.0]);
    let plain = formants_of(&render_with(&goose, RATE, &accented(0.0), &opts()));
    let aus = formants_of(&render_with(&goose, RATE, &accented(2.0), &opts()));
    let f2_plain = near(&plain, 870.0);
    let f2_aus = near(&aus, 870.0 * 1.35);
    println!("GOOSE F2: plain {f2_plain:.0} Hz, Australian {f2_aus:.0} Hz ({plain:?} → {aus:?})");
    // The envelope moves F2 by the full ×1.35 (lpc::tests::formant_map_…);
    // through the whole chain about a third of it shows, because part of a
    // formant stays in the residual, which the warp cannot move. With 5 ms
    // grains: 882 → 1028 Hz (+17 %); with the 12 ms grains that stopped low
    // voices buzzing, more of the residual's formant rides in each grain:
    // 879 → 981 Hz (+11.6 %), and F1 reads 347 → 427 Hz. A weaker accent
    // was the price of a clean voice.
    assert!(f2_aus > f2_plain * 1.10, "Australian should front GOOSE");
    let f1 = (near(&plain, 300.0), near(&aus, 300.0));
    assert!(f1.1 < f1.0 * 1.3, "F1 should stay: {f1:?}");
}

#[test]
fn accent_melodies_rise_or_fall_along_a_phrase() {
    use super::dsp::pitch::PitchTracker;
    // one long voiced phrase at a steady 120 Hz
    let x = steady_vowel(120.0, [700.0, 1200.0, 2500.0, 3500.0]);
    let contour = |accent: f32| {
        let y = render_with(&x, RATE, &accented(accent), &opts());
        let mut t = PitchTracker::new(RATE as f32);
        // (time, f0) of every voiced 10 ms frame
        let mut frames = Vec::new();
        for (i, s) in y.iter().enumerate() {
            t.push(*s);
            if i % 480 == 479 {
                let e = t.analyze();
                if e.voiced {
                    frames.push((i as f32 / RATE as f32, e.f0));
                }
            }
        }
        let at = |from: f32, to: f32| {
            let mut v: Vec<f32> = frames.iter().filter(|(ts, _)| *ts >= from && *ts < to).map(|(_, f)| *f).collect();
            assert!(!v.is_empty(), "no voiced frames in {from}..{to} s");
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let (early, late) = (at(0.25, 0.45), at(2.0, 2.6));
        12.0 * (late / early).log2()
    };
    let (none, aus, rp) = (contour(0.0), contour(2.0), contour(1.0));
    println!("pitch change along the phrase: none {none:+.2} st, Australian {aus:+.2} st, RP {rp:+.2} st");
    assert!(none.abs() < 0.3, "no accent: flat");
    assert!(aus > 1.5, "Australian rises");
    assert!(rp < -1.0, "RP falls");
}

/// KNOWN GAP (not yet working): an r-coloured vowel has F3 only ~340 Hz
/// above F2, and neither the voice stage's formant finder (12 poles over
/// 0-6 kHz) nor this test's analysis resolves the two, so the vowel is never
/// recognised as r-coloured and the non-rhotic accents leave the r in.
/// Measured: NURSE [490, 1350, 1690] is seen as [536, 1350, 3185]. Needs a
/// finer finder (e.g. roots of the fitted polynomial, or a higher order over
/// a narrower band) before this can pass.
#[test]
#[ignore = "known gap: F2/F3 of r-coloured vowels are not resolved (see doc)"]
fn accent_takes_the_r_out() {
    // r-coloured NURSE: F3 near F2
    let nurse_r = steady_vowel(110.0, [490.0, 1350.0, 1690.0, 3300.0]);
    let plain = formants_of(&render_with(&nurse_r, RATE, &accented(0.0), &opts()));
    let rp = formants_of(&render_with(&nurse_r, RATE, &accented(1.0), &opts()));
    println!("NURSE: plain {plain:?} → RP {rp:?}");
    assert!(near(&plain, 1690.0) < 1850.0, "the test vowel is r-coloured");
    assert!(!rp.iter().any(|f| (1500.0..2000.0).contains(f)), "RP should take the r-colour out: {rp:?}");
    assert!(rp.iter().any(|f| (2200.0..2800.0).contains(f)), "F3 moves up: {rp:?}");
}

/// Your own voice, relative (no targets), for the voice controls.
fn you_preset() -> Preset {
    Preset { id: "me".into(), name: "Me".into(), description: String::new(), category: String::new(), disguise: false, builtin: false, blocks: vec![BlockSlot { id: "voice".into(), block: BlockSpec::Voice(super::dsp::shifter::VoiceParams::default()) }] }
}

#[test]
fn voice_controls_move_the_rendered_voice() {
    use super::identity::{apply_controls, VoiceControls};
    let r = RATE as f32;
    let (man, _) = talker_truth(r, 110.0, 1.0, 12.0, 1);
    let render = |c: VoiceControls| render_with(&man, RATE, &apply_controls(&you_preset(), &c), &opts());
    let plain = render(VoiceControls::default());
    let base = settled(&plain);
    // gender: pitch up about 4 st and a smaller tract
    let fem = settled(&render(VoiceControls { gender: 1.0, ..Default::default() }));
    let st = 12.0 * (fem.f0_hz / base.f0_hz).log2();
    println!("gender +1: {:.1} -> {:.1} Hz ({st:+.2} st), tract {:.2} -> {:.2}", base.f0_hz, fem.f0_hz, base.tract, fem.tract);
    assert!((st - 4.0).abs() < 0.6, "gender +1 should raise the voice ~4 st, got {st:.2}");
    assert!(fem.tract > base.tract * 1.03, "and shrink the tract");
    // confidence and energy: a wider melody (and confidence a little lower)
    let conf = settled(&render(VoiceControls { confidence: 1.0, ..Default::default() }));
    let calm = settled(&render(VoiceControls { energy: -1.0, ..Default::default() }));
    println!("spread: plain {:.2} st, confident {:.2} st, low energy {:.2} st", base.spread_st, conf.spread_st, calm.spread_st);
    assert!(conf.spread_st > base.spread_st * 1.12, "confidence widens the melody");
    assert!(calm.spread_st < base.spread_st * 0.8, "low energy flattens it");
    assert!(conf.f0_hz < base.f0_hz, "confidence sits a little lower");
    // warmth: more body, less air; clarity: more presence
    let (sp, sw, sc) = (ltas(&plain[(3 * RATE) as usize..]), ltas(&render(VoiceControls { warmth: 1.0, ..Default::default() })[(3 * RATE) as usize..]), ltas(&render(VoiceControls { clarity: 1.0, ..Default::default() })[(3 * RATE) as usize..]));
    let tilt = |s: &[f64]| band_db(s, 5000.0, 10000.0) - band_db(s, 100.0, 300.0);
    let pres = |s: &[f64]| band_db(s, 2000.0, 4500.0) - band_db(s, 300.0, 1500.0);
    println!("tilt (air - body): plain {:.1}, warm {:.1} dB; presence: plain {:.1}, clear {:.1} dB", tilt(&sp), tilt(&sw), pres(&sp), pres(&sc));
    // shelves at 300 Hz and 5 kHz cover the bands partly: ~80 % of the 7 dB shows
    assert!(tilt(&sw) < tilt(&sp) - 3.0, "warmth tilts towards body");
    assert!(pres(&sc) > pres(&sp) + 2.0, "clarity adds presence");
    // age: older is rougher and breathier (lower harmonics-to-noise ratio),
    // measured on a clean vowel (the synthetic talker is already noisy up
    // there: even breath 0.6 moves it only 1.4 dB)
    let vowel = steady_vowel(110.0, [700.0, 1200.0, 2500.0, 3500.0]);
    let on_vowel = |c: VoiceControls| render_with(&vowel, RATE, &apply_controls(&you_preset(), &c), &opts());
    let (h0, h_old) = (hnr_high_db(&on_vowel(VoiceControls::default())), hnr_high_db(&on_vowel(VoiceControls { age: 1.0, ..Default::default() })));
    println!("HNR 1.5-5 kHz: plain {h0:.1} dB, older {h_old:.1} dB");
    assert!(h_old < h0 - 1.5, "older is rougher and breathier");
}

/// The full identity (profile and style) of recordings, as calibration measures it.
#[test]
#[ignore = "needs recordings: NEKOTONE_ADAPT_WAVS=name=file;..."]
fn identity_of_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let p = super::dsp::profile::measure_identity(&clip.samples, clip.sample_rate as f32);
        println!("{name}: {:.1} s of audio, {:.1} Hz, spread {:.2} st, tract {:.2}, {:.1} s voiced\n  style {:?}", clip.samples.len() as f32 / clip.sample_rate as f32, p.f0_hz, p.spread_st, p.tract, p.voiced_secs, p.style);
    }
}

/// Pitch glitches: voiced 10 ms frames more than 7 semitones from the median
/// of the 200 ms around them (a tracker error turned into an audible jump).
fn glitch_rate(y: &[f32]) -> (f32, usize) {
    use super::dsp::pitch::PitchTracker;
    let mut t = PitchTracker::new(RATE as f32);
    let mut f = Vec::new();
    for (i, s) in y.iter().enumerate() {
        t.push(*s);
        if i % 480 == 479 {
            let e = t.analyze();
            f.push(if e.voiced { e.f0 } else { 0.0 });
        }
    }
    let (mut bad, mut n) = (0usize, 0usize);
    for i in 10..f.len().saturating_sub(10) {
        if f[i] <= 0.0 {
            continue;
        }
        let mut w: Vec<f32> = f[i - 10..=i + 10].iter().copied().filter(|v| *v > 0.0).collect();
        if w.len() < 8 {
            continue;
        }
        w.sort_by(|a, b| a.total_cmp(b));
        let med = w[w.len() / 2];
        n += 1;
        if (12.0 * (f[i] / med).log2()).abs() > 7.0 {
            bad += 1;
        }
    }
    (100.0 * bad as f32 / n.max(1) as f32, n)
}

#[test]
#[ignore = "needs recordings: NEKOTONE_ADAPT_WAVS=name=file;..."]
fn glitches_of_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        let (d, n) = glitch_rate(&clip.samples);
        println!("{name}: dry {d:.2} % of {n} frames");
        for id in ["female", "male", "james", "emma", "confident-me", "anonymous-me", "dragon"] {
            let p = super::preset_by_id(id).unwrap();
            let mut res = Vec::new();
            for guard in [0.0f32, 7.0] {
                *super::dsp::shifter::TUNE.lock().unwrap() = Some(super::dsp::shifter::Tuning { guard_st: guard, ..Default::default() });
                let y = render_with(&clip.samples, RATE, &p, &opts());
                res.push(glitch_rate(&y));
            }
            *super::dsp::shifter::TUNE.lock().unwrap() = None;
            println!("  {id:<14} guard off {:.2} %, on {:.2} % (of {} frames)", res[0].0, res[1].0, res[1].1);
        }
    }
}

#[test]
fn the_realism_guard_lets_a_real_jump_through() {
    use super::dsp::pitch::PitchTracker;
    // a steady vowel at 110 Hz, then a real jump up an octave that lasts
    let a = steady_vowel(110.0, [700.0, 1200.0, 2500.0, 3500.0]);
    let b = steady_vowel(220.0, [700.0, 1200.0, 2500.0, 3500.0]);
    let mut x = a[..(1.5 * RATE as f32) as usize].to_vec();
    x.extend_from_slice(&b[..(1.5 * RATE as f32) as usize]);
    let y = render_with(&x, RATE, &you_preset(), &opts());
    let mut t = PitchTracker::new(RATE as f32);
    let mut after = Vec::new();
    for (i, s) in y.iter().enumerate() {
        t.push(*s);
        // from 150 ms after the jump
        if i % 480 == 479 && i as f32 > 1.65 * RATE as f32 {
            let e = t.analyze();
            if e.voiced {
                after.push(e.f0);
            }
        } else if i % 480 == 479 {
            t.analyze();
        }
    }
    after.sort_by(|p, q| p.total_cmp(q));
    let med = after[after.len() / 2];
    println!("after the jump: {med:.1} Hz");
    assert!((12.0 * (med / 220.0).log2()).abs() < 0.5, "the output follows a real jump: {med:.1} Hz");
}

/// The resonance tamer on recordings: the largest change it makes to any
/// third-octave band 1.5-10 kHz of each preset's render (it should act only
/// where a preset rings).
#[test]
#[ignore = "needs recordings: NEKOTONE_ADAPT_WAVS=name=file;..."]
fn tamer_on_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let bands: Vec<(f32, f32)> = (0..9).map(|i| (1500.0 * 2f32.powf(i as f32 / 3.0), 1500.0 * 2f32.powf((i + 1) as f32 / 3.0))).collect();
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        println!("{name}:");
        for id in ["female", "male", "emma", "james", "child", "giant", "robot", "confident-me", "dragon"] {
            let p = super::preset_by_id(id).unwrap();
            let mut o = opts();
            o.front_end.tamer_on = false;
            let off = ltas(&render_with(&clip.samples, RATE, &p, &o));
            o.front_end.tamer_on = true;
            let on = ltas(&render_with(&clip.samples, RATE, &p, &o));
            let worst = bands.iter().map(|&(lo, hi)| band_db(&on, lo, hi) - band_db(&off, lo, hi)).fold(0.0f32, |m, d| if d.abs() > m.abs() { d } else { m });
            println!("  {id:<14} largest band change {worst:+.2} dB");
        }
    }
}

/// How much of a recording's vowel energy the voice stage treats as
/// unvoiced (noise instead of grains: heard as breathiness). Ignored: needs
/// NEKOTONE_ADAPT_WAVS. Frames are 5 ms blocks whose energy is mostly below
/// 1.4 kHz (vowels); prints the energy share classed unvoiced and the
/// aperiodicity distribution the tracker saw on them.
#[test]
#[ignore]
fn voicing_of_recordings() {
    use super::dsp::shifter::{Shifter, VoiceParams};
    use super::dsp::{Block, Ctx};
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        let mut s = Shifter::new(VoiceParams::default(), RATE as f32);
        let mut ctx = Ctx { rate: RATE as f32, ..Default::default() };
        let (mut e_vowel, mut e_unv) = (0f64, 0f64);
        let mut aps = Vec::new();
        for block in clip.samples.chunks(240) {
            let mut b = block.to_vec();
            s.process(&mut b, &mut ctx);
            let est = s.input_pitch();
            let e: f64 = block.iter().map(|v| (*v as f64).powi(2)).sum();
            if est.lf_share > 0.8 && est.rms > 0.003 {
                e_vowel += e;
                aps.push(est.aperiodicity);
                if !est.voiced {
                    e_unv += e;
                }
            }
        }
        aps.sort_by(|a, b| a.total_cmp(b));
        let q = |f: f32| aps.get(((aps.len() as f32 - 1.0) * f) as usize).copied().unwrap_or(f32::NAN);
        println!("{name}: vowel energy classed unvoiced {:.1} %; aperiodicity p10 {:.2} p50 {:.2} p90 {:.2} ({} vowel blocks)", 100.0 * e_unv / e_vowel.max(1e-12), q(0.1), q(0.5), q(0.9), aps.len());
    }
}

/// Any presets over real recordings, for listening and measuring (ignored):
/// NEKOTONE_ADAPT_WAVS as above, NEKOTONE_RENDER_IDS=relaxed-me,female,…,
/// NEKOTONE_ADAPT_OUT for the folder (dry copies are written too).
#[test]
#[ignore]
fn render_presets_of_recordings() {
    let list = std::env::var("NEKOTONE_ADAPT_WAVS").expect("set NEKOTONE_ADAPT_WAVS=name=file;name=file");
    let ids = std::env::var("NEKOTONE_RENDER_IDS").expect("set NEKOTONE_RENDER_IDS=id,id");
    let out = std::path::PathBuf::from(std::env::var("NEKOTONE_ADAPT_OUT").unwrap_or_else(|_| "renders".into()));
    std::fs::create_dir_all(&out).unwrap();
    let real = RenderOptions { tail_secs: 1.0, ..Default::default() };
    // NEKOTONE_TUNE=cap_ms=8,hard=0,…: shifter tuning for experiments
    if let Ok(spec) = std::env::var("NEKOTONE_TUNE") {
        let mut tn = super::dsp::shifter::Tuning::default();
        for kv in spec.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').expect("key=value");
            let f: f32 = v.parse().unwrap();
            match k {
                "cap_ms" => tn.cap_ms = f,
                "hard" => tn.hard = f != 0.0,
                "v_tau_ms" => tn.v_tau_ms = f,
                "on" => tn.on = f,
                "off" => tn.off = f,
                "forward" => tn.forward = f != 0.0,
                "xcorr" => tn.xcorr = f != 0.0,
                "two_period" => tn.two_period = f != 0.0,
                other => panic!("unknown tuning {other}"),
            }
        }
        *super::dsp::shifter::TUNE.lock().unwrap() = Some(tn);
    }
    for item in list.split(';') {
        let (name, file) = item.split_once('=').expect("name=file");
        let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
        let clip = if clip.sample_rate != RATE { crate::audio::resample(&clip, RATE).unwrap() } else { clip };
        crate::audio::write_wav(&out.join(format!("{name}__dry.wav")), &clip.samples, RATE, 1).unwrap();
        for spec in ids.split(',').map(str::trim) {
            // `id` or `id:slot+slot` (render without those blocks)
            let (id, drop) = spec.split_once(':').unwrap_or((spec, ""));
            // `@voice`: one voice stage at its defaults (a pass-through through the shifter)
            let mut p = if id == "@voice" {
                let mut p = preset_by_id("female").unwrap();
                p.id = "@voice".into();
                p.blocks = vec![super::BlockSlot { id: "voice".into(), block: BlockSpec::Voice(super::dsp::shifter::VoiceParams::default()) }];
                p
            } else {
                preset_by_id(id).unwrap_or_else(|| panic!("no preset {id}"))
            };
            let drop: Vec<&str> = drop.split('+').filter(|s| !s.is_empty()).collect();
            p.blocks.retain(|b| !drop.contains(&b.id.as_str()));
            let y = render_with(&clip.samples, RATE, &p, &real);
            let tag = spec.replace([':', '+'], "-");
            crate::audio::write_wav(&out.join(format!("{tag}__{name}.wav")), &y, RATE, 1).unwrap();
        }
    }
    println!("renders in {}", out.display());
}

