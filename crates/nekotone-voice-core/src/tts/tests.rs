//! Tests of the TTS front end plumbing (no model) and, `#[ignore]`d, of
//! Kokoro itself: synthesis, speed, and the TTS → Whisper word error rate.

use super::*;

#[test]
fn sentences_split_after_terminators_and_keep_them() {
    let s = split_sentences("Hello there. How are you?! I'm fine… \"Really.\" Yes\nNew line");
    assert_eq!(s, vec!["Hello there.", "How are you?!", "I'm fine…", "\"Really.\"", "Yes", "New line"]);
    assert!(split_sentences("  ... !! ").is_empty());
    assert_eq!(split_sentences("3.5 is a number."), vec!["3.5 is a number."]);
}

#[test]
fn chunks_respect_the_limit_and_prefer_clauses() {
    let ps = "ðə kwˈɪk bɹˈWn fˈɑks, ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ; ænd ðˈɛn ɪt ɹˈʌnz əwˈA fˈæst.";
    let n = ps.chars().count();
    assert_eq!(chunk_phonemes(ps, MAX_TOKENS, MAX_TOKENS), vec![ps.to_string()]);
    let pieces = chunk_phonemes(ps, 40, 40);
    assert!(pieces.len() >= 2);
    for p in &pieces {
        assert!(p.chars().count() <= 40, "{p}");
        assert!(!p.starts_with(' ') && !p.ends_with(' '));
    }
    // clause cut: the first piece ends at the comma
    assert!(pieces[0].ends_with(','), "{pieces:?}");
    // nothing lost but spaces
    let joined: String = pieces.join(" ");
    assert_eq!(joined.chars().count(), n);
    // a long word without spaces is still cut
    let long: String = std::iter::repeat_n('a', 1200).collect();
    let p = chunk_phonemes(&long, MAX_TOKENS, 100);
    assert_eq!(p[0].len(), 100);
    assert!(p.iter().all(|x| x.len() <= MAX_TOKENS));
    assert_eq!(p.iter().map(|x| x.len()).sum::<usize>(), 1200);
}

#[test]
fn trim_keeps_a_margin_and_fades() {
    let rate = 24_000;
    let mut a = vec![0.0f32; rate as usize];
    for (i, v) in a.iter_mut().enumerate().skip(10_000).take(2_400) {
        *v = 0.5 * (i as f32 * 0.1).sin();
    }
    trim_silence(&mut a, rate, 0.03);
    let keep = (0.03 * rate as f32) as usize;
    assert!(a.len() <= 2_400 + 2 * keep + 2 && a.len() >= 2_300 + keep, "{}", a.len());
    assert_eq!(a[0], 0.0);
    let mut silent = vec![0.0f32; 1000];
    trim_silence(&mut silent, rate, 0.03);
    assert!(silent.is_empty());
}

#[test]
fn catalogue_lists_every_voice_once() {
    let files = model_files();
    assert_eq!(files.len(), 8 + VOICES.len());
    for vi in VOICES {
        assert!(files.iter().any(|f| f.name == format!("{}.bin", vi.id)));
        assert_eq!(vi.sha256.len(), 64);
        assert!(vi.gender == "female" || vi.gender == "male");
        assert!(vi.accent == "american" || vi.accent == "british");
    }
    assert!(voice(DEFAULT_VOICE).unwrap().featured);
    assert_eq!(voices().iter().filter(|v| v.featured).count(), 8);
    let total: u64 = files.iter().map(|f| f.size_bytes).sum();
    assert_eq!(total, 352_782_869, "the download size quoted in the docs");
}

#[test]
fn vocab_tokenises_and_skips_unknown() {
    let mut vocab = HashMap::new();
    vocab.insert('h', 50);
    vocab.insert('ə', 83);
    vocab.insert(' ', 16);
    assert_eq!(tokenize(&vocab, "hə ❓h"), vec![50, 83, 16, 50]);
}

// ───────────────────────── with the real model ─────────────────────────

/// Folder with the Kokoro files: `NEKOTONE_KOKORO_DIR`, else the installed model.
#[cfg(feature = "ml")]
pub(crate) fn kokoro_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NEKOTONE_KOKORO_DIR") {
        let p = PathBuf::from(d);
        if p.join(KOKORO_MODEL).is_file() {
            return Some(p);
        }
    }
    crate::models::ModelManager::default().path(crate::models::ModelId::TtsKokoro)
}

/// Engine speed on a fixed phoneme string (the Kokoro README's sentence),
/// on the CPU and (when available) DirectML; writes both WAVs to %TEMP%.
/// `cargo test -p nekotone-core --release tts::tests::engine_speed -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn engine_speed_cpu_and_gpu() {
    let dir = kokoro_dir().expect("Kokoro is not installed");
    let tokens: &[i64] = &[
        50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53,
        102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64,
        83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4,
    ];
    let vocab = read_vocab(&dir.join(KOKORO_TOKENIZER)).unwrap();
    let back: HashMap<i64, char> = vocab.iter().map(|(c, i)| (*i, *c)).collect();
    let ps: String = tokens.iter().map(|t| back[t]).collect();
    println!("phonemes: {ps}");
    for accel in [crate::models::Accelerator::Cpu, crate::models::Accelerator::Auto] {
        let t0 = std::time::Instant::now();
        let tts = Tts::load_dir(&dir, accel).unwrap();
        let load = t0.elapsed().as_secs_f32();
        if let Err(e) = tts.synthesize_phonemes(&ps, "af_heart", 1.0) {
            println!("{}: load {load:.2} s, synthesis FAILED: {e}", tts.device());
            continue;
        }
        // cost against length: is there a fixed overhead per call?
        let words: Vec<&str> = ps.split(' ').collect();
        for n in [2usize, 4, 8, 16, words.len()] {
            let part = words[..n.min(words.len())].join(" ");
            let mut best = f32::MAX;
            let mut len = 0;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                len = tts.synthesize_phonemes(&part, "af_heart", 1.0).unwrap().len();
                best = best.min(t0.elapsed().as_secs_f32());
            }
            println!("  {} tokens: {:.2} s of audio in {:.0} ms", tokenize(&vocab, &part).len(), len as f32 / SAMPLE_RATE as f32, best * 1000.0);
        }
        let mut best = f32::MAX;
        let mut a = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            a = tts.synthesize_phonemes(&ps, "af_heart", 1.0).unwrap();
            best = best.min(t0.elapsed().as_secs_f32());
        }
        let short = std::time::Instant::now();
        let _ = tts.synthesize_phonemes("həlˈO, wˈɜɹld.", "af_heart", 1.0).unwrap();
        let short = short.elapsed().as_secs_f32();
        let secs = a.len() as f32 / SAMPLE_RATE as f32;
        println!("{}: load {load:.2} s, {secs:.2} s of audio in {best:.3} s = {:.1}x real time; a 2-word line in {:.0} ms", tts.device(), secs / best, short * 1000.0);
        crate::audio::write_wav(&std::env::temp_dir().join(format!("nekotone-kokoro-{}.wav", tts.device())), &a, SAMPLE_RATE, 1).unwrap();
    }
}

/// The 20 sentences of the round-trip test: everyday talk, game chat,
/// numbers and money, contractions, questions, names and made-up words.
pub(crate) const WER_SENTENCES: [&str; 20] = [
    "Could you please pass me the salt?",
    "I'll meet you at the north gate in ten minutes.",
    "That was a great game, well played everyone.",
    "Let me check the map before we head out.",
    "It costs $4.99, which is cheaper than last year.",
    "The meeting moved to 3:30 pm on Thursday.",
    "We need about 250 more gold for the new armor.",
    "Don't worry, I've got the healing potions.",
    "My phone number changed last week, I'll text you.",
    "Can you hear me clearly, or is my microphone too quiet?",
    "The weather today is sunny with a light breeze.",
    "She bought three apples, two oranges and a melon.",
    "Turn left at the bakery and walk for two blocks.",
    "Thanks for waiting, I'm ready to go now.",
    "The dragon breathes fire, so stay behind the pillars.",
    "Please remember to save your work every few minutes.",
    "He finished the marathon in just under four hours.",
    "Our flight leaves at 7 in the morning on the 21st.",
    "I think the password is written on the blue notebook.",
    "Good luck out there, and have fun!",
];

/// A harder set (reported separately): names, rare words, heteronyms,
/// acronyms and figures. Whisper's spelling of a name counts against the
/// voice here, so this is a pessimistic number.
pub(crate) const HARD_SENTENCES: [&str; 10] = [
    "Kasimir and Siobhan flew from Reykjavik to Warsaw last Tuesday.",
    "The quinoa vendor's schedule was completely unpredictable.",
    "I read that book last year, and I will read it again soon.",
    "Please record the new record before the concert starts.",
    "Our CEO announced a 12.5% raise starting January 1st, 2027.",
    "The NASA probe sent back 3,400 gigabytes of data.",
    "My gamertag is ShadowFoxx, add me after the raid.",
    "The pharmacist suggested ibuprofen instead of acetaminophen.",
    "We should rendezvous near the old quay at dawn.",
    "Wednesday's colonel wore a bright yellow uniform.",
];

/// Lower-case words for word error rate: numbers spelled out on both
/// sides (so "3:30" and "three thirty" agree), punctuation dropped.
pub(crate) fn wer_words(text: &str) -> Vec<String> {
    // Whisper writes times British-style ("3.30 p.m."): same words as "3:30 pm"
    let chars: Vec<char> = text.chars().collect();
    let mut fixed = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        let time_dot = c == '.'
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())
            && chars.get(i + 2).is_some_and(|d| d.is_ascii_digit())
            && {
                let rest: String = chars[i + 3..].iter().collect::<String>().trim_start().to_lowercase();
                rest.starts_with("p.m") || rest.starts_with("pm") || rest.starts_with("a.m") || rest.starts_with("am")
            };
        fixed.push(if time_dot { ':' } else { c });
    }
    let n = normalize::normalize(&fixed, false).to_lowercase().replace(['-', '—'], " ");
    n.split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>())
        .filter(|w| !w.is_empty())
        .map(|w| match w.as_str() {
            "ok" => "okay".to_string(),
            "armour" => "armor".to_string(),
            _ => w,
        })
        .collect()
}

/// Word edit distance / reference length.
pub(crate) fn wer(reference: &[String], hypothesis: &[String]) -> (usize, usize) {
    let (n, m) = (reference.len(), hypothesis.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, v) in d[0].iter_mut().enumerate() {
        *v = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let sub = d[i - 1][j - 1] + usize::from(reference[i - 1] != hypothesis[j - 1]);
            d[i][j] = sub.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    (d[n][m], n)
}

#[test]
fn word_error_rate_counts_edits() {
    let r = wer_words("It costs $4.99, okay?");
    assert_eq!(r, vec!["it", "costs", "four", "dollars", "and", "ninety", "nine", "cents", "okay"]);
    let h = wer_words("It cost four dollars and ninety-nine cents, OK?");
    assert_eq!(wer(&r, &h), (1, 9));
    assert_eq!(wer(&r, &r), (0, 9));
    assert_eq!(wer(&r, &[]), (9, 9));
    assert_eq!(wer_words("at 3.30 p.m. on"), wer_words("at 3:30 pm on"));
    assert_eq!(wer_words("at 3.30pm"), wer_words("at 3:30 pm"));
    assert_eq!(wer_words("pi is 3.14"), vec!["pi", "is", "three", "point", "one", "four"]);
}

/// The objective quality proxy: every featured voice reads the 20
/// sentences, Whisper (small if installed, else base) transcribes them, and
/// the word error rate is reported per voice. Also the speed. Writes every
/// take to `%LOCALAPPDATA%\Nekotone\speak-samples` for listening.
/// `cargo test -p nekotone-core --release tts::tests::round_trip -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn round_trip_word_error_rate() {
    use crate::stt::Transcriber;
    let dir = kokoro_dir().expect("Kokoro is not installed");
    let tts = Tts::load_dir(&dir, crate::models::Accelerator::Cpu).unwrap();
    let mm = crate::models::ModelManager::default();
    let id = [crate::models::ModelId::WhisperSmall, crate::models::ModelId::WhisperBase].into_iter().find(|m| mm.path(*m).is_some()).expect("a Whisper model");
    let stt = crate::stt::WhisperOnnx::load(&mm, id).unwrap();
    let out_dir = std::env::var("NEKOTONE_SAMPLES_DIR").map(PathBuf::from).unwrap_or_else(|_| crate::data_dir().join("speak-samples"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let only: Option<Vec<String>> = std::env::var("NEKOTONE_WER_VOICES").ok().map(|v| v.split(',').map(str::to_string).collect());
    let opts = crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() };
    let mut report = format!("Round trip Kokoro → {} (CPU), {} sentences per voice\n", id.name(), WER_SENTENCES.len());
    let mut worst: Vec<(f32, String, String, String)> = Vec::new();
    let _ = tts.synthesize("Warm up.", DEFAULT_VOICE, 1.0).unwrap();
    for v in VOICES.iter().filter(|v| v.featured && only.as_ref().map(|o| o.iter().any(|x| x == v.id)).unwrap_or(true)) {
        let (mut errs, mut words) = (0usize, 0usize);
        let (mut synth_s, mut audio_s) = (0f32, 0f32);
        for (k, s) in WER_SENTENCES.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let a = tts.synthesize(s, v.id, 1.0).unwrap();
            synth_s += t0.elapsed().as_secs_f32();
            audio_s += a.len() as f32 / SAMPLE_RATE as f32;
            if k < 5 || std::env::var_os("NEKOTONE_SAMPLES_ALL").is_some() {
                crate::audio::write_wav(&out_dir.join(format!("{}-{:02}.wav", v.id, k + 1)), &a, SAMPLE_RATE, 1).unwrap();
            }
            let clip = crate::audio::resample(&crate::audio::Clip { samples: a, sample_rate: SAMPLE_RATE, source_channels: 1 }, 16_000).unwrap();
            let heard = stt.transcribe(&clip, &opts, &mut |_| {}).unwrap().text();
            let (r, h) = (wer_words(s), wer_words(&heard));
            let (e, n) = wer(&r, &h);
            errs += e;
            words += n;
            if e > 0 {
                worst.push((e as f32 / n as f32, v.id.to_string(), s.to_string(), heard.clone()));
            }
        }
        let (mut herrs, mut hwords) = (0usize, 0usize);
        for s in HARD_SENTENCES {
            let a = tts.synthesize(s, v.id, 1.0).unwrap();
            let clip = crate::audio::resample(&crate::audio::Clip { samples: a, sample_rate: SAMPLE_RATE, source_channels: 1 }, 16_000).unwrap();
            let heard = stt.transcribe(&clip, &opts, &mut |_| {}).unwrap().text();
            let (e, n) = wer(&wer_words(s), &wer_words(&heard));
            herrs += e;
            hwords += n;
            if e > 0 {
                worst.push((e as f32 / n as f32, format!("{} (hard)", v.id), s.to_string(), heard.clone()));
            }
        }
        let line = format!(
            "{:<12} {:>3} grade  WER {:>5.1} %  ({errs}/{words} words)  hard set {:>5.1} % ({herrs}/{hwords})  {:.1}x real time\n",
            v.id,
            v.grade,
            100.0 * errs as f32 / words as f32,
            100.0 * herrs as f32 / hwords as f32,
            audio_s / synth_s
        );
        print!("{line}");
        report.push_str(&line);
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    report.push_str("\nSentences with errors (worst first):\n");
    for (w, v, s, h) in worst.iter().take(40) {
        report.push_str(&format!("{:>5.1} %  {v}: {s:?} → {h:?}\n", w * 100.0));
    }
    std::fs::write(out_dir.join("wer-report.txt"), &report).unwrap();
    println!("{report}");
    println!("samples and report in {}", out_dir.display());
}

/// Which graph × optimisation level gives finite audio, and how fast (CPU).
/// `NEKOTONE_KOKORO_GRAPHS` = extra graph files to compare (`;`-separated).
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn graph_variants() {
    use ort::session::builder::GraphOptimizationLevel as L;
    use ort::value::Tensor;
    let dir = kokoro_dir().expect("Kokoro is not installed");
    let vocab = read_vocab(&dir.join(KOKORO_TOKENIZER)).unwrap();
    let ps = "hˌaʊ kʊd aɪ nˈoʊ? ɪts ɐn ʌnˈænsɚɹəbəl kwˈɛstʃən. lˈaɪk ˈæskɪŋ ɐn ʌnbˈɔːɹn tʃˈaɪld ɪf ðeɪl lˈiːd ɐ ɡˈʊd lˈaɪf. ðeɪ hˈævənt ˈiːvən bˌɪn bˈɔːɹn.";
    let ids = tokenize(&vocab, ps);
    let style_all: Vec<f32> = std::fs::read(dir.join("af_heart.bin")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let style = style_all[ids.len() * 256..(ids.len() + 1) * 256].to_vec();
    let mut graphs = vec![dir.join(KOKORO_MODEL)];
    if let Ok(extra) = std::env::var("NEKOTONE_KOKORO_GRAPHS") {
        graphs.extend(extra.split(';').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    for g in graphs {
        for (name, lvl) in [("disable", L::Disable), ("level1", L::Level1), ("level2", L::Level2), ("level3", L::Level3)] {
            let mut s = ort::session::Session::builder().unwrap().with_optimization_level(lvl).unwrap().with_intra_threads(crate::models::onnx_threads()).unwrap().commit_from_file(&g).unwrap();
            let mut input = vec![0i64];
            input.extend(&ids);
            input.push(0);
            let n = input.len();
            let mut best = f32::MAX;
            let mut stats = (0usize, 0usize, 0f32);
            for _ in 0..2 {
                let t0 = std::time::Instant::now();
                let out = s
                    .run(ort::inputs![
                        "input_ids" => Tensor::from_array(([1usize, n], input.clone())).unwrap(),
                        "style" => Tensor::from_array(([1usize, 256], style.clone())).unwrap(),
                        "speed" => Tensor::from_array(([1usize], vec![1.0f32])).unwrap()
                    ])
                    .unwrap();
                best = best.min(t0.elapsed().as_secs_f32());
                let v = out["waveform"].try_extract_array::<f32>().unwrap();
                stats = (v.len(), v.iter().filter(|x| !x.is_finite()).count(), v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs())));
            }
            let secs = stats.0 as f32 / 24_000.0;
            println!("{} {name}: {} samples, {} not finite, peak {:.3}, {:.2} s → {:.1}x real time", g.file_name().unwrap().to_string_lossy(), stats.0, stats.1, stats.2, best, secs / best);
        }
    }
}

/// `cargo test -p nekotone-core --release tts::tests::speaks -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn speaks_a_sentence_on_the_cpu() {
    let dir = kokoro_dir().expect("Kokoro is not installed (nekotone models get tts-kokoro, or set NEKOTONE_KOKORO_DIR)");
    let t0 = std::time::Instant::now();
    let tts = Tts::load_dir(&dir, crate::models::Accelerator::Cpu).unwrap();
    println!("load {:.2} s on {}", t0.elapsed().as_secs_f32(), tts.device());
    let text = "Hello! This is Nekotone speaking for you. It costs $5.20, and it's 3:45 pm.";
    let plan = tts.plan(text, "af_heart", MAX_TOKENS).unwrap();
    for u in &plan {
        println!("{:?} -> {}", u.text, u.phonemes);
    }
    let _ = tts.synthesize("Warm up.", "af_heart", 1.0).unwrap();
    let t0 = std::time::Instant::now();
    let a = tts.synthesize(text, "af_heart", 1.0).unwrap();
    let el = t0.elapsed().as_secs_f32();
    let secs = a.len() as f32 / SAMPLE_RATE as f32;
    println!("{secs:.2} s of audio in {el:.2} s = {:.1}x real time", secs / el);
    assert!(secs > 2.0 && secs < 12.0);
    let peak = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.05, "silent output");
    let out = std::env::temp_dir().join("nekotone-tts-hello.wav");
    crate::audio::write_wav(&out, &a, SAMPLE_RATE, 1).unwrap();
    println!("wrote {}", out.display());
}
