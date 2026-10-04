//! Chatterbox: sampling, token and text plumbing with fake logits (always
//! run) and, `#[ignore]`d, the real model (speed per engine, round trip
//! through Whisper, speaker similarity).

use super::*;

fn logits_with(n: usize, hot: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![-5.0f32; n];
    for &(i, x) in hot {
        v[i] = x;
    }
    v
}

#[test]
fn greedy_picks_the_largest_logit_after_the_penalty() {
    let s = Sampling { temperature: 0.0, ..Default::default() };
    let mut rng = Rng::new(1);
    let mut l = logits_with(10, &[(3, 2.0), (7, 1.9)]);
    assert_eq!(sample_next(&mut l, &[], &s, &mut rng), 3);
    // 3 was already said: 2.0 / 1.2 = 1.67 < 1.9
    let mut l = logits_with(10, &[(3, 2.0), (7, 1.9)]);
    assert_eq!(sample_next(&mut l, &[3], &s, &mut rng), 7);
}

#[test]
fn repetition_penalty_divides_positive_and_multiplies_negative_once() {
    let mut l = vec![2.4, -1.0, 0.5];
    apply_repetition_penalty(&mut l, &[0, 1, 0, 0], 1.2);
    assert!((l[0] - 2.0).abs() < 1e-6, "{l:?}");
    assert!((l[1] + 1.2).abs() < 1e-6, "{l:?}");
    assert_eq!(l[2], 0.5);
    // out-of-range history entries are ignored
    apply_repetition_penalty(&mut l, &[99, -1], 1.2);
    assert_eq!(l[2], 0.5);
}

#[test]
fn top_k_one_is_greedy_and_top_p_keeps_the_head() {
    let mut rng = Rng::new(7);
    let s = Sampling { top_k: 1, repetition_penalty: 1.0, ..Default::default() };
    for _ in 0..50 {
        let mut l = logits_with(100, &[(42, 3.0), (5, 2.9)]);
        assert_eq!(sample_next(&mut l, &[], &s, &mut rng), 42);
    }
    // one token holds 99 % of the mass: top-p 0.9 keeps only it
    let s = Sampling { top_k: 0, top_p: 0.9, temperature: 1.0, repetition_penalty: 1.0, seed: 0 };
    for _ in 0..50 {
        let mut l = logits_with(100, &[(9, 20.0)]);
        assert_eq!(sample_next(&mut l, &[], &s, &mut rng), 9);
    }
}

#[test]
fn sampling_follows_the_distribution() {
    // two tokens at logits ln 3 and 0 (temperature 1): 75 % / 25 %
    let s = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0 };
    let mut rng = Rng::new(12345);
    let mut count = [0usize; 2];
    for _ in 0..20_000 {
        let mut l = vec![f32::NEG_INFINITY; 4];
        l[1] = 3f32.ln();
        l[2] = 0.0;
        let t = sample_next(&mut l, &[], &s, &mut rng);
        assert!(t == 1 || t == 2, "{t}");
        count[(t - 1) as usize] += 1;
    }
    let p = count[0] as f32 / 20_000.0;
    assert!((p - 0.75).abs() < 0.015, "{p}");
    // a lower temperature sharpens it
    let s = Sampling { temperature: 0.5, ..s };
    let mut hi = 0;
    for _ in 0..20_000 {
        let mut l = vec![f32::NEG_INFINITY; 4];
        l[1] = 3f32.ln();
        l[2] = 0.0;
        hi += usize::from(sample_next(&mut l, &[], &s, &mut rng) == 1);
    }
    let p = hi as f32 / 20_000.0; // 9 / 10
    assert!((p - 0.9).abs() < 0.015, "{p}");
}

#[test]
fn broken_logits_never_win_and_all_masked_stops() {
    let s = Sampling::default();
    let mut rng = Rng::new(3);
    for _ in 0..200 {
        let mut l = logits_with(20, &[(4, 1.0)]);
        l[0] = f32::NAN;
        l[1] = f32::INFINITY;
        let t = sample_next(&mut l, &[], &s, &mut rng);
        assert!(t != 0 && t != 1, "{t}");
    }
    let mut l = vec![f32::NEG_INFINITY; 8];
    assert_eq!(sample_next(&mut l, &[], &s, &mut rng), STOP_SPEECH_TOKEN);
    assert_eq!(sample_next(&mut [], &[], &s, &mut rng), STOP_SPEECH_TOKEN);
}

#[test]
fn same_seed_same_tokens() {
    let s = Sampling::default();
    let run = |seed| {
        let mut rng = Rng::new(seed);
        (0..64)
            .map(|k| {
                let mut l: Vec<f32> = (0..300).map(|i| ((i * 37 + k * 11) % 101) as f32 / 20.0).collect();
                sample_next(&mut l, &[], &s, &mut rng)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
    let mut r = Rng::new(9);
    assert!((0..1000).map(|_| r.next_f32()).all(|v| (0.0..1.0).contains(&v)));
}

#[test]
fn decoder_tokens_add_prompt_and_silence_and_drop_markers() {
    let v = decoder_tokens(&[10, 11], &[START_SPEECH_TOKEN, 5, 6, STOP_SPEECH_TOKEN, 7]);
    assert_eq!(v, vec![10, 11, 5, 6, 7, SILENCE_TOKEN, SILENCE_TOKEN, SILENCE_TOKEN]);
}

#[test]
fn punctuation_is_cleaned_like_upstream() {
    assert_eq!(punc_norm("hello   world"), "Hello world.");
    assert_eq!(punc_norm("Wait… what: “this”"), "Wait, what, \"this\".");
    assert_eq!(punc_norm("Really?"), "Really?");
    assert_eq!(punc_norm("It’s fine — honestly"), "It's fine - honestly.");
    assert_eq!(punc_norm(""), "");
}

#[test]
fn plan_spells_numbers_and_cuts_long_sentences() {
    let p = plan("It costs $4.99. See you at 3:30!", MAX_PIECE_CHARS);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].phonemes.to_lowercase().contains("four dollars"), "{p:?}");
    assert!(p[0].phonemes.ends_with('.'));
    assert!(p[1].phonemes.ends_with('!'));
    let long = "This is a rather long sentence, with several clauses in it, that goes on and on, because people do talk like this sometimes, especially when they are excited about something.";
    let p = plan(long, 60);
    assert!(p.len() >= 2, "{p:?}");
    assert!(p[0].phonemes.chars().count() <= 62, "{:?}", p[0]);
    assert!(p.iter().all(|u| u.phonemes.chars().count() <= MAX_PIECE_CHARS + 1));
    assert!(plan("  ...  ", 60).is_empty());
}

#[test]
fn token_limit_grows_with_text_and_is_capped() {
    let a = max_tokens_for("Hi.");
    let b = max_tokens_for("This sentence is about fifty characters long, ok.");
    assert!(a >= 75 && b > a, "{a} {b}");
    // fifty characters at ordinary pace is ~3.5 s = ~90 tokens: plenty of room
    assert!(b >= 180, "{b}");
    assert_eq!(max_tokens_for(&"x".repeat(10_000)), 1000);
}

#[test]
fn voice_print_round_trips_and_rejects_damage() {
    let p = VoicePrint {
        audio_features: (0..2 * 3 * 4).map(|i| i as f32 * 0.5).collect(),
        features_shape: [1, 6, 4],
        audio_tokens: vec![1, 2, 3, 6000],
        speaker_embeddings: vec![0.25; 192],
        speaker_features: (0..10 * 80).map(|i| -(i as f32)).collect(),
        speaker_features_shape: [1, 10, 80],
        info: VoicePrintInfo { name: "My voice".into(), reference_secs: 9.5, created: 1, source: "recorded sample".into(), model: PRINT_MODEL.into() },
    };
    let dir = std::env::temp_dir().join(format!("nekotone-vp-{}", std::process::id()));
    let path = print_path(&dir, "mine");
    assert!(path.to_string_lossy().ends_with("mine.nkvoice"));
    p.save(&path).unwrap();
    let q = VoicePrint::load(&path).unwrap();
    assert_eq!(p, q);
    assert!((p.similarity(&q) - 1.0).abs() < 1e-6);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 3);
    std::fs::write(&path, &bytes).unwrap();
    let e = VoicePrint::load(&path).unwrap_err().to_string();
    assert!(e.contains("damaged"), "{e}");
    std::fs::write(&path, b"hello").unwrap();
    assert!(VoicePrint::load(&path).is_err());
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(print_path(Path::new("x"), "a b/c"), Path::new("x").join("a_b_c.nkvoice"));
}

#[test]
fn reference_is_trimmed_levelled_and_checked() {
    // 1 s silence, 7 s of a buzzy "voice" at 48 kHz, 1 s silence
    let rate = 48_000u32;
    let mut x = vec![0.0f32; rate as usize];
    x.extend((0..7 * rate as usize).map(|i| {
        let t = i as f32 / rate as f32;
        0.02 * ((2.0 * std::f32::consts::PI * 120.0 * t).sin() + 0.5 * (2.0 * std::f32::consts::PI * 240.0 * t).sin()) * (1.0 + (t * 3.0).sin()) * 0.5
    }));
    x.extend(vec![0.0f32; rate as usize]);
    let y = prepare_reference(&x, rate).unwrap();
    let secs = y.len() as f32 / SAMPLE_RATE as f32;
    assert!((6.9..7.5).contains(&secs), "{secs}");
    let l = crate::process::integrated_lufs(&y, 1, SAMPLE_RATE).unwrap();
    assert!((l - REFERENCE_LUFS).abs() < 1.0, "{l}");
    // too short
    let e = prepare_reference(&x[..3 * rate as usize], rate).unwrap_err().to_string();
    assert!(e.contains("more than 5"), "{e}");
    // long takes are cut to 15 s
    let long: Vec<f32> = x[rate as usize..8 * rate as usize].iter().cycle().take(30 * rate as usize).copied().collect();
    let y = prepare_reference(&long, rate).unwrap();
    assert_eq!(y.len(), (MAX_REFERENCE_SECS * SAMPLE_RATE as f32) as usize);
}

#[test]
fn catalogue_pins_every_file_once() {
    let files = model_files();
    assert_eq!(files.len(), 2 + 2 * GRAPHS.len());
    for f in &files {
        assert_eq!(f.sha256.len(), 64, "{}", f.name);
        assert!(f.size_bytes > 0 && f.url.contains("d21799bd0354adb85e348b8a0442a8405110a2cf"), "{}", f.name);
        assert_eq!(files.iter().filter(|g| g.name == f.name).count(), 1, "{}", f.name);
    }
    // every graph's external data sits next to it under the name the graph refers to
    for (_, name) in GRAPHS {
        assert!(files.iter().any(|f| f.name == format!("{name}_data")), "{name}");
    }
}

// ───────────────────────── real model (ignored) ─────────────────────────

/// Folder with the Chatterbox files: `NEKOTONE_CHATTERBOX_DIR`, else the installed model.
#[cfg(feature = "ml")]
pub(crate) fn chatterbox_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("NEKOTONE_CHATTERBOX_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    crate::models::ModelManager::default().path(crate::models::ModelId::TtsChatterbox)
}

/// The reference voice for the real-model tests: `NEKOTONE_CLONE_REF`, else
/// `%TEMP%\nekotone-notes-speech.wav` (Windows TTS, made for the notes tests).
#[cfg(feature = "ml")]
pub(crate) fn reference_clip() -> crate::audio::Clip {
    let p = std::env::var("NEKOTONE_CLONE_REF").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join("nekotone-notes-speech.wav"));
    crate::audio::decode(&p).unwrap_or_else(|e| panic!("reference {}: {e}", p.display()))
}

pub(crate) const SPEED_SENTENCES: &[&str] = &[
    "Hello, this is my own voice, speaking for me.",
    "I'll be there in about ten minutes, so start without me.",
    "The quick brown fox jumps over the lazy dog, and then it takes a nap in the sun.",
];

/// Speed of each engine: `NEKOTONE_GPU=0` (CPU), `=directml`, unset (auto);
/// `NEKOTONE_CHATTERBOX_GRAPHS=lm=q4,dec=fp16,…` picks precisions.
/// `cargo test -p nekotone-core --release tts::chatterbox::tests::speed -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn speed_per_engine() {
    let dir = chatterbox_dir().expect("Chatterbox is not installed (or set NEKOTONE_CHATTERBOX_DIR)");
    let t0 = std::time::Instant::now();
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let load = t0.elapsed().as_secs_f32();
    let clip = reference_clip();
    let t1 = std::time::Instant::now();
    let print = cb.encode_voice(&clip.samples, clip.sample_rate, "test", "nekotone-notes-speech.wav").unwrap();
    let enc = t1.elapsed().as_secs_f32();
    let graphs: Vec<String> = Role::ALL.iter().map(|r| graph_for(&dir, *r).map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).unwrap_or_default()).collect();
    println!("graphs {graphs:?}\nengines {:?}\nload {load:.1} s, encode {enc:.2} s ({:.1} s reference, cond {:?}, prompt {} tokens)", cb.engines(), print.info.reference_secs, print.features_shape, print.audio_tokens.len());
    let s = Sampling::default();
    // warm-up
    let _ = cb.synthesize_piece("Ready.", &print, &s, &|| false).unwrap();
    let out = std::env::temp_dir().join("nekotone-chatterbox");
    std::fs::create_dir_all(&out).unwrap();
    let (mut audio, mut work) = (0f32, 0f32);
    let mut first = Vec::new();
    for (k, text) in SPEED_SENTENCES.iter().enumerate() {
        let t = std::time::Instant::now();
        let a = cb.synthesize_piece(&punc_norm(text), &print, &s, &|| false).unwrap();
        let w = t.elapsed().as_secs_f32();
        let st = cb.last_stats();
        audio += a.len() as f32 / SAMPLE_RATE as f32;
        work += w;
        first.push(w);
        println!("{k}: {:.2} s audio in {w:.2} s ({:.2}x real time): {} text + {} speech tokens, prefill {:.0} ms, {:.1} ms/token, decode {:.0} ms{}", a.len() as f32 / SAMPLE_RATE as f32, a.len() as f32 / SAMPLE_RATE as f32 / w, st.text_tokens, st.speech_tokens, st.prefill_ms, st.step_ms, st.decode_ms, if st.hit_limit { " (hit the limit)" } else { "" });
        crate::audio::write_wav(&out.join(format!("speed-{}-{k}.wav", cb.device())), &a, SAMPLE_RATE, 1).unwrap();
    }
    // first audio of a short line (what Speak for me waits for)
    let t = std::time::Instant::now();
    let p = plan("Sure, give me a second to check.", FIRST_PIECE_CHARS);
    let _ = cb.synthesize_piece(&p[0].phonemes, &print, &s, &|| false).unwrap();
    println!("TOTAL {audio:.2} s audio in {work:.2} s = {:.2}x real time on {}; short line first audio {:.2} s", audio / work, cb.device(), t.elapsed().as_secs_f32());
    println!("WAVs in {}", out.display());
}

/// Quality proxies for the cloned voice:
/// * intelligibility: the first 10 round-trip sentences of the Kokoro test,
///   spoken in the reference's voice, transcribed by Whisper (small if
///   installed), word error rate;
/// * likeness: cosine between the reference's speaker x-vector (Chatterbox's
///   CAM++ speaker encoder) and that of the clone's output, next to the same
///   cosine for the 8 featured Kokoro voices saying the same sentences
///   (the baseline: a stock voice).
///
/// Writes the takes and `clone-report.txt` to `%LOCALAPPDATA%\Nekotone\speak-samples\clone`.
/// `cargo test -p nekotone-core --release tts::chatterbox::tests::round_trip -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn round_trip_and_likeness() {
    use crate::stt::Transcriber;
    use crate::tts::tests::{wer, wer_words, WER_SENTENCES};
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let clip = reference_clip();
    let reference = cb.encode_voice(&clip.samples, clip.sample_rate, "reference", "reference").unwrap();
    let mm = crate::models::ModelManager::default();
    let id = [crate::models::ModelId::WhisperSmall, crate::models::ModelId::WhisperBase].into_iter().find(|m| mm.path(*m).is_some()).expect("a Whisper model");
    let stt = crate::stt::WhisperOnnx::load(&mm, id).unwrap();
    let opts = crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() };
    let out_dir = crate::data_dir().join("speak-samples").join("clone");
    std::fs::create_dir_all(&out_dir).unwrap();
    let sentences = &WER_SENTENCES[..10];
    let s = Sampling::default();
    let hear = |a: &[f32]| -> String {
        let c = crate::audio::resample(&crate::audio::Clip { samples: a.to_vec(), sample_rate: SAMPLE_RATE, source_channels: 1 }, 16_000).unwrap();
        stt.transcribe(&c, &opts, &mut |_| {}).unwrap().text()
    };
    // likeness of everything a voice said (at least 5 s, so the encoder accepts it)
    let likeness = |takes: &[Vec<f32>]| -> f32 {
        let joined: Vec<f32> = takes.iter().flat_map(|t| t.iter().copied().chain(std::iter::repeat_n(0.0, 2400))).collect();
        let p = cb.encode_voice(&joined, SAMPLE_RATE, "take", "take").unwrap();
        p.similarity(&reference)
    };
    let mut report = format!("Chatterbox clone of {} ({:.1} s), graphs {:?}, engines {:?}, {} for the round trip\n", std::env::var("NEKOTONE_CLONE_REF").unwrap_or_else(|_| "nekotone-notes-speech.wav".into()), reference.info.reference_secs, Role::ALL.iter().map(|r| graph_for(&dir, *r).and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_default()).collect::<Vec<_>>(), cb.engines(), id.name());
    let _ = cb.synthesize_piece("Warm up.", &reference, &s, &|| false);
    let (mut errs, mut words, mut synth_s, mut audio_s) = (0usize, 0usize, 0f32, 0f32);
    let mut takes = Vec::new();
    for (k, text) in sentences.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let a = cb.synthesize(text, &reference, &s).unwrap();
        synth_s += t0.elapsed().as_secs_f32();
        audio_s += a.len() as f32 / SAMPLE_RATE as f32;
        crate::audio::write_wav(&out_dir.join(format!("clone-{:02}.wav", k + 1)), &a, SAMPLE_RATE, 1).unwrap();
        let heard = hear(&a);
        let (e, n) = wer(&wer_words(text), &wer_words(&heard));
        errs += e;
        words += n;
        let line = format!("{:>2}: {e}/{n} {text:?} → {heard:?}\n", k + 1);
        print!("{line}");
        report.push_str(&line);
        takes.push(a);
    }
    let clone_sim = likeness(&takes);
    let summary = format!(
        "clone: WER {:.1} % ({errs}/{words} words), {:.2}x real time, likeness to the reference {clone_sim:.3}\n",
        100.0 * errs as f32 / words.max(1) as f32,
        audio_s / synth_s.max(1e-6)
    );
    print!("{summary}");
    report.push_str(&summary);
    // baseline: the stock Kokoro voices
    if let Some(kdir) = crate::tts::tests::kokoro_dir() {
        let tts = crate::tts::Tts::load_dir(&kdir, crate::models::Accelerator::Cpu).unwrap();
        let mut sims = Vec::new();
        for v in crate::tts::voices().into_iter().filter(|v| v.featured) {
            let takes: Vec<Vec<f32>> = sentences.iter().map(|t| tts.synthesize(t, v.id, 1.0).unwrap()).collect();
            let sim = likeness(&takes);
            let line = format!("kokoro {:<11} likeness {sim:.3}\n", v.id);
            print!("{line}");
            report.push_str(&line);
            sims.push(sim);
        }
        sims.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let line = format!("kokoro baseline: best {:.3}, median {:.3}; clone {clone_sim:.3}\n", sims.last().copied().unwrap_or(0.0), sims.get(sims.len() / 2).copied().unwrap_or(0.0));
        print!("{line}");
        report.push_str(&line);
    }
    std::fs::write(out_dir.join("clone-report.txt"), &report).unwrap();
    println!("samples and report in {}", out_dir.display());
}

/// "Make my voice" on the recorded sample Voice Studio keeps
/// (`%LOCALAPPDATA%\Nekotone\voice-sample.wav`, or `NEKOTONE_CLONE_SAMPLE`),
/// on the app's default engines. Ignored: needs the model and the sample.
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn make_voice_from_the_recorded_sample() {
    let path = std::env::var("NEKOTONE_CLONE_SAMPLE").map(std::path::PathBuf::from).unwrap_or_else(|_| crate::data_dir().join("voice-sample.wav"));
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let clip = crate::audio::decode(&path).expect("the sample decodes");
    println!("sample {:.1} s at {} Hz", clip.samples.len() as f32 / clip.sample_rate as f32, clip.sample_rate);
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    println!("engines {:?}", cb.engines());
    let t = std::time::Instant::now();
    let print = cb.encode_voice(&clip.samples, clip.sample_rate, "My voice", "recorded sample");
    match &print {
        Ok(p) => println!("encoded in {:.2} s: {:.1} s reference, features {:?}, {} prompt tokens", t.elapsed().as_secs_f32(), p.info.reference_secs, p.features_shape, p.audio_tokens.len()),
        Err(e) => println!("ENCODE ERROR: {e}"),
    }
    let print = print.unwrap();
    use crate::stt::Transcriber;
    let mm = crate::models::ModelManager::default();
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let tag = std::env::var("NEKOTONE_CHATTERBOX_LM").unwrap_or_else(|_| "default".into());
    for (k, line) in ["Hello, this is my own voice, speaking for me.", "The meeting moved to three thirty, so I will be a little late.", "Thanks, that sounds great."].iter().enumerate() {
        let t = std::time::Instant::now();
        let a = cb.synthesize_piece(&punc_norm(line), &print, &Sampling::default(), &|| false).unwrap();
        let secs = t.elapsed().as_secs_f32();
        let st = cb.last_stats();
        let out = std::env::temp_dir().join(format!("nekotone-my-voice-{tag}-{k}.wav"));
        crate::audio::write_wav(&out, &a, SAMPLE_RATE, 1).unwrap();
        let clip = crate::audio::Clip { samples: a.clone(), sample_rate: SAMPLE_RATE, source_channels: 1 };
        let tr = w.transcribe(&clip, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap();
        let heard: String = tr.segments.iter().map(|s| s.text.trim()).collect::<Vec<_>>().join(" ");
        println!("LINE {k}: {:.2} s audio in {secs:.2} s; {} speech tokens, {:.1} ms/token, prefill {:.0} ms, decode {:.0} ms{}
  said: {line}
  heard: {heard}", a.len() as f32 / SAMPLE_RATE as f32, st.speech_tokens, st.step_ms, st.prefill_ms, st.decode_ms, if st.hit_limit { " (HIT THE LIMIT)" } else { "" });
    }
}

/// Voice conversion (ignored: needs the model, the recorded sample and
/// `%TEMP%\nekotone-notes-speech.wav`): the notes test speech (a Windows TTS
/// voice) said in the owner's voice. Words must survive (Whisper), and the
/// output must sound more like the owner than like the source (x-vector cosine).
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn convert_keeps_the_words_and_takes_the_voice() {
    use crate::stt::Transcriber;
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let me = crate::audio::decode(&std::env::var("NEKOTONE_CLONE_SAMPLE").map(std::path::PathBuf::from).unwrap_or_else(|_| crate::data_dir().join("voice-sample.wav"))).expect("sample");
    let src = crate::audio::decode(&std::env::temp_dir().join("nekotone-notes-speech.wav")).expect("source speech");
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let print = cb.encode_voice(&me.samples, me.sample_rate, "me", "sample").unwrap();
    let t = std::time::Instant::now();
    let out = cb.convert(&src.samples, src.sample_rate, &print).unwrap();
    let secs = t.elapsed().as_secs_f32();
    let dur_in = src.samples.len() as f32 / src.sample_rate as f32;
    crate::audio::write_wav(&std::env::temp_dir().join("nekotone-converted.wav"), &out, SAMPLE_RATE, 1).unwrap();
    let mm = crate::models::ModelManager::default();
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let say = |a: &[f32], r: u32| {
        let c = crate::audio::Clip { samples: a.to_vec(), sample_rate: r, source_channels: 1 };
        w.transcribe(&c, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ")
    };
    let (heard_src, heard_out) = (say(&src.samples, src.sample_rate), say(&out, SAMPLE_RATE));
    let v_me = cb.speaker_vector(&me.samples, me.sample_rate).unwrap();
    let v_src = cb.speaker_vector(&src.samples, src.sample_rate).unwrap();
    let v_out = cb.speaker_vector(&out, SAMPLE_RATE).unwrap();
    let (to_me, to_src, src_me) = (cosine(&v_out, &v_me), cosine(&v_out, &v_src), cosine(&v_src, &v_me));
    println!("converted {dur_in:.1} s in {secs:.2} s -> {:.1} s", out.len() as f32 / SAMPLE_RATE as f32);
    println!("  source said: {heard_src}\n  output said: {heard_out}");
    println!("  likeness: output~me {to_me:.3}, output~source {to_src:.3} (source~me {src_me:.3})");
    let (a, b) = (crate::tts::tests::wer_words(&heard_src), crate::tts::tests::wer_words(&heard_out));
    let (errs, n) = crate::tts::tests::wer(&a, &b);
    let wer = errs as f32 / n.max(1) as f32;
    println!("  word error vs the source's own transcript {:.1} %", wer * 100.0);
    assert!(wer < 0.25, "the words did not survive the conversion");
    assert!(to_me > to_src, "the output does not sound more like the target than the source");
}

/// First audio of a long typed line (ignored: needs the model and the
/// recorded sample): what Speak for me waits for before your voice starts.
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn first_audio_of_a_long_line() {
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let me = crate::audio::decode(&crate::data_dir().join("voice-sample.wav")).expect("the recorded sample");
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let print = cb.encode_voice(&me.samples, me.sample_rate, "me", "sample").unwrap();
    let s = Sampling::default();
    let _ = cb.synthesize_piece("Ready.", &print, &s, &|| false).unwrap();
    let line = "Thanks for waiting, I just checked the calendar and it looks like Thursday afternoon works for everyone on the team.";
    let pieces = plan(line, FIRST_PIECE_CHARS);
    let t = std::time::Instant::now();
    let a = cb.synthesize_piece(&pieces[0].phonemes, &print, &s, &|| false).unwrap();
    println!("first piece {:?} ({} chars): {:.2} s of audio after {:.2} s; {} pieces", pieces[0].phonemes, pieces[0].phonemes.chars().count(), a.len() as f32 / SAMPLE_RATE as f32, t.elapsed().as_secs_f32(), pieces.len());
}

/// Is a longer recording a better voice? (ignored: needs the model and
/// NEKOTONE_CLONE_LONG = a long clip of one person, NEKOTONE_CLONE_HELDOUT =
/// other recordings of the same person, `;`-separated). Prints are made from
/// (1) the recorded 10 s sample, (2) the first 15 s of the long clip, (3) its
/// most representative 15 s window (x-vector closest to the clip's mean),
/// (4) that window with the clip's mean x-vector; each says three lines, and
/// the output is scored against the held-out recordings (x-vector cosine)
/// and by Whisper (word errors).
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn longer_references_make_a_closer_voice() {
    use crate::stt::Transcriber;
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let long = crate::audio::decode(std::path::Path::new(&std::env::var("NEKOTONE_CLONE_LONG").expect("NEKOTONE_CLONE_LONG"))).unwrap();
    let long = crate::audio::resample(&long, SAMPLE_RATE).unwrap().samples;
    let held: Vec<Vec<f32>> = std::env::var("NEKOTONE_CLONE_HELDOUT").expect("NEKOTONE_CLONE_HELDOUT").split(';').map(|p| {
        let c = crate::audio::decode(std::path::Path::new(p)).unwrap();
        cb.speaker_vector(&c.samples, c.sample_rate).unwrap()
    }).collect();
    let sample = crate::audio::decode(&crate::data_dir().join("voice-sample.wav")).expect("the recorded sample");
    // windows of 15 s every 3 s over the long clip, their x-vectors and mean
    let w = (MAX_REFERENCE_SECS * SAMPLE_RATE as f32) as usize;
    let step = 3 * SAMPLE_RATE as usize;
    let mut wins = Vec::new();
    let mut s = 0;
    while s + w <= long.len() {
        if let Ok(v) = cb.speaker_vector(&long[s..s + w], SAMPLE_RATE) {
            wins.push((s, v));
        }
        s += step;
    }
    let dim = wins[0].1.len();
    let mut mean = vec![0f32; dim];
    for (_, v) in &wins {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x / wins.len() as f32;
        }
    }
    let n = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    mean.iter_mut().for_each(|x| *x /= n);
    let best = wins.iter().max_by(|a, b| cosine(&a.1, &mean).total_cmp(&cosine(&b.1, &mean))).unwrap().0;
    println!("{} windows; most representative starts at {:.0} s (cosine to the mean {:.3})", wins.len(), best as f32 / SAMPLE_RATE as f32, cosine(&wins.iter().find(|x| x.0 == best).unwrap().1, &mean));
    let p_sample = cb.encode_voice(&sample.samples, sample.sample_rate, "sample", "10 s").unwrap();
    let p_first = cb.encode_voice(&long[..w], SAMPLE_RATE, "first", "first 15 s").unwrap();
    let p_best = cb.encode_voice(&long[best..best + w], SAMPLE_RATE, "best", "best 15 s").unwrap();
    let mut p_mean = p_best.clone();
    // the x-vector has a scale: keep the window's norm, take the mean's direction
    let bn = p_best.speaker_embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
    p_mean.speaker_embeddings = mean.iter().map(|x| x * bn).collect();
    let mm = crate::models::ModelManager::default();
    let wh = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let lines = ["The meeting moved to three thirty, so I will be a little late.", "Could you send me the notes from yesterday before lunch?", "Honestly, I think the second option sounds much better to me."];
    for (name, p) in [("10 s sample", &p_sample), ("first 15 s", &p_first), ("best 15 s", &p_best), ("best + mean x-vector", &p_mean)] {
        let (mut errs, mut words) = (0usize, 0usize);
        let mut all = Vec::new();
        for line in lines {
            let a = cb.synthesize_piece(&punc_norm(line), p, &Sampling::default(), &|| false).unwrap();
            all.extend_from_slice(&a);
            let c = crate::audio::Clip { samples: a, sample_rate: SAMPLE_RATE, source_channels: 1 };
            let heard = wh.transcribe(&c, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ");
            let (e, n) = crate::tts::tests::wer(&crate::tts::tests::wer_words(line), &crate::tts::tests::wer_words(&heard));
            errs += e;
            words += n;
        }
        // the three lines together (one line is under the encoder's 5 s minimum)
        let v = cb.speaker_vector(&all, SAMPLE_RATE).unwrap();
        let like = held.iter().map(|h| cosine(&v, h)).sum::<f32>() / held.len() as f32;
        println!("{name:<22} likeness to held-out recordings {like:.3}; words wrong {errs}/{words}");
    }
}

/// One to three words said alone (ignored: needs the models). Cuts short
/// phrases out of a recording of the owner (`NEKOTONE_CLONE_SAMPLE`, at
/// least 20 s) by Whisper's word times, converts each into the same voice
/// and counts how many Whisper still understands. With
/// `NEKOTONE_CONVERT_NO_PAD` set, the short input is not padded (before 0.4.5).
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn convert_short_phrases() {
    use crate::stt::Transcriber;
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let path = std::env::var("NEKOTONE_CLONE_SAMPLE").expect("NEKOTONE_CLONE_SAMPLE");
    let me = crate::audio::resample(&crate::audio::decode(std::path::Path::new(&path)).unwrap(), 16_000).unwrap();
    let x = &me.samples[..me.samples.len().min(16_000 * 90)];
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let print = cb.encode_voice(&x[..16_000 * 15], 16_000, "me", "sample").unwrap();
    let mm = crate::models::ModelManager::default();
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let tr = w.transcribe(&crate::audio::Clip { samples: x.to_vec(), sample_rate: 16_000, source_channels: 1 }, &crate::stt::TranscribeOptions { language: Some("en".into()), word_timestamps: true, ..Default::default() }, &mut |_| {}).unwrap();
    let words: Vec<crate::stt::Word> = tr.segments.iter().flat_map(|s| s.words.iter().cloned()).collect();
    let say = |a: &[f32], r: u32| {
        let c = crate::audio::Clip { samples: a.to_vec(), sample_rate: r, source_channels: 1 };
        w.transcribe(&c, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ")
    };
    let norm = |s: &str| crate::tts::tests::wer_words(s);
    let (mut tried, mut kept, mut empty, mut last_kept) = (0, 0, 0, 0);
    let mut i = 0;
    while i < words.len() && tried < 40 {
        let n = 1 + tried % 6;
        if i + n > words.len() {
            break;
        }
        let (a, b) = (words[i].start - 0.08, words[i + n - 1].end + 0.08);
        i += n + 2;
        let (s0, s1) = (((a.max(0.0)) * 16_000.0) as usize, ((b * 16_000.0) as usize).min(x.len()));
        let clip = &x[s0..s1];
        let src = say(clip, 16_000);
        if norm(&src).is_empty() {
            continue;
        }
        tried += 1;
        let out = cb.convert(clip, 16_000, &print).unwrap();
        let heard = if out.is_empty() { String::new() } else { say(&out, SAMPLE_RATE) };
        let (e, m) = crate::tts::tests::wer(&norm(&src), &norm(&heard));
        let ok = (e as f32) / (m.max(1) as f32) <= 0.5;
        kept += ok as usize;
        // the last word you said is still there (among the last two heard)
        let (sw, hw) = (norm(&src), norm(&heard));
        let last_ok = sw.last().map(|l| hw.iter().rev().take(2).any(|w| w == l)).unwrap_or(true);
        last_kept += last_ok as usize;
        empty += out.is_empty() as usize;
        println!("{:.2} s  {:<28} -> {:<28} {}", clip.len() as f32 / 16_000.0, src, heard, if ok { "" } else { "LOST" });
    }
    println!("{kept} of {tried} short phrases understood after conversion ({empty} came out empty); last word kept in {last_kept}");
}

/// Change my voice on the phrases Speak for me actually makes (ignored:
/// needs the models and a recording, NEKOTONE_CLONE_SAMPLE): the recording
/// is cut by the voice detector with Speak for me's settings, each phrase is
/// converted, and Whisper reads the phrase and the result. Reports whether
/// the last word survives, the length ratio, and the sound left in the last
/// 0.3 s (owner: "clipping off the end of the words … missing words at the
/// end"). NEKOTONE_CONVERT_TAIL_MS overrides the silence put after a phrase.
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn convert_keeps_phrase_endings() {
    use crate::record::vad::{Vad, VadConfig};
    use crate::stt::Transcriber;
    let dir = chatterbox_dir().expect("Chatterbox is not installed");
    let path = std::env::var("NEKOTONE_CLONE_SAMPLE").expect("NEKOTONE_CLONE_SAMPLE");
    let me = crate::audio::resample(&crate::audio::decode(std::path::Path::new(&path)).unwrap(), 16_000).unwrap();
    let x = &me.samples[..me.samples.len().min(16_000 * 120)];
    let cb = Chatterbox::load_dir(&dir, crate::models::accelerator()).unwrap();
    let print = cb.encode_voice(&x[..16_000 * 15], 16_000, "me", "sample").unwrap();
    let mm = crate::models::ModelManager::default();
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let say = |a: &[f32], r: u32| {
        let c = crate::audio::Clip { samples: a.to_vec(), sample_rate: r, source_channels: 1 };
        w.transcribe(&c, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ")
    };
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (gap, ctx_ms) = (env("NEKOTONE_GAP_FRAMES", 3), env("NEKOTONE_LEFT_CTX_MS", 0));
    let cfg = VadConfig { hangover_secs: 0.4, soft_max_secs: 2.5, hard_max_secs: 6.0, soft_gap_frames: gap, seamless_cuts: true, min_speech_secs: 0.15, ..VadConfig::default() };
    let mut vad = Vad::new(cfg);
    let mut utts = vec![];
    vad.feed(x, &mut utts);
    vad.flush(&mut utts);
    let rms = |a: &[f32]| (a.iter().map(|v| v * v).sum::<f32>() / a.len().max(1) as f32).sqrt();
    let norm = |s: &str| crate::tts::tests::wer_words(s);
    let (mut n, mut last_ok, mut ratio_sum, mut tail_in, mut tail_out) = (0, 0, 0.0f32, 0.0f32, 0.0f32);
    let (mut errs, mut words) = (0usize, 0usize);
    let mut prev_end: Option<(u64, Vec<f32>)> = None;
    for u in utts.iter().take(30) {
        // the previous phrase's last `ctx_ms` when this one carries straight on from it
        // NEKOTONE_CTX_ALWAYS=1: after a pause too (under 3 s), with 0.25 s of silence between
        let always = std::env::var_os("NEKOTONE_CTX_ALWAYS").is_some();
        let ctx: Vec<f32> = match &prev_end {
            Some((e, s)) if *e == u.start && ctx_ms > 0 => s[s.len().saturating_sub(ctx_ms * 16)..].to_vec(),
            Some((e, s)) if always && ctx_ms > 0 && u.start > *e && u.start - *e < 48_000 => s[s.len().saturating_sub(ctx_ms * 16)..].iter().copied().chain(std::iter::repeat_n(0.0, 4_000)).collect(),
            _ => Vec::new(),
        };
        prev_end = Some((u.end, u.samples.clone()));
        let src = say(&u.samples, 16_000);
        let sw = norm(&src);
        if sw.len() < 2 {
            continue;
        }
        let with: Vec<f32> = ctx.iter().chain(u.samples.iter()).copied().collect();
        let mut out = cb.convert(&with, 16_000, &print).unwrap();
        out.drain(..(ctx.len() * 3 / 2).min(out.len()));
        let heard = say(&out, SAMPLE_RATE);
        let hw = norm(&heard);
        let ok = hw.iter().rev().take(2).any(|wd| Some(wd) == sw.last());
        n += 1;
        last_ok += ok as usize;
        let (e, m) = crate::tts::tests::wer(&sw, &hw);
        errs += e;
        words += m;
        let (li, lo) = (u.samples.len() as f32 / 16_000.0, out.len() as f32 / SAMPLE_RATE as f32);
        ratio_sum += lo / li;
        let ti = rms(&u.samples[u.samples.len().saturating_sub(4_800)..]) / rms(&u.samples).max(1e-9);
        let to = rms(&out[out.len().saturating_sub(7_200)..]) / rms(&out).max(1e-9);
        tail_in += ti;
        tail_out += to;
        println!("{li:5.2} s -> {lo:5.2} s  tail {ti:.2} -> {to:.2}  {}  {src:<45} -> {heard}", if ok { "  " } else { "!!" });
    }
    println!("gap {gap} frames, left context {ctx_ms} ms{}: word errors", if std::env::var_os("NEKOTONE_CTX_ALWAYS").is_some() { " (after pauses too)" } else { "" }); println!("  word errors {:.0} % ({errs}/{words})", 100.0 * errs as f32 / words.max(1) as f32);
    println!("{n} phrases: last word kept in {last_ok}; length out/in {:.3}; sound in the last 0.3 s (vs the phrase) {:.2} in, {:.2} out", ratio_sum / n.max(1) as f32, tail_in / n.max(1) as f32, tail_out / n.max(1) as f32);
}
