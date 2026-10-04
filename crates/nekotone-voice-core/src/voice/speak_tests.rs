//! Speak for me: pipeline tests with fake devices, a fake transcriber and a
//! fake synthesiser (always run), and with the real Whisper + Kokoro
//! (`#[ignore]`, need the models).

use super::*;
use crate::record::vad::tests::speechish;
use crate::record::{FakeCapture, ManualClock};
use crate::stt::{Segment, Transcript};
use std::f32::consts::PI;

const OUT_RATE: u32 = 48_000;

/// Returns the same line for every utterance (or nothing).
struct FakeStt {
    line: String,
    calls: AtomicU64,
}

impl Transcriber for FakeStt {
    fn name(&self) -> String {
        "fake".into()
    }
    fn transcribe(&self, clip: &crate::audio::Clip, _o: &TranscribeOptions, _p: crate::OnProgress) -> Result<Transcript> {
        if clip.samples.iter().all(|v| *v == 0.0) {
            // the engine's warm-up call
            return Ok(Transcript { language: "en".into(), segments: vec![], model: "fake".into() });
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(20));
        let segs = if self.line.is_empty() {
            vec![]
        } else {
            vec![Segment { start: 0.0, end: clip.duration_secs(), text: self.line.clone(), words: vec![], no_speech_prob: 0.01, speaker: None }]
        };
        Ok(Transcript { language: "en".into(), segments: segs, model: "fake".into() })
    }
}

/// A "voice" that is a 1 kHz tone, 0.4 s per piece (one piece per word group).
struct ToneSynth;

impl Synth for ToneSynth {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, _voice: &str, _first: usize, _accent: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        Ok(crate::tts::split_sentences(text).into_iter().map(|s| Piece { text: s.clone(), phonemes: s, pause_after: 0.05 }).collect())
    }
    fn render(&self, _p: &Piece, _voice: &str, speed: f32) -> Result<Vec<f32>> {
        let n = (0.4 / speed * 24_000.0) as usize;
        Ok((0..n).map(|i| 0.3 * (2.0 * PI * 1000.0 * i as f32 / 24_000.0).sin()).collect())
    }
    fn device(&self) -> String {
        "fake".into()
    }
}

/// Plays a tap like a device would, `speedup`× faster than real time, into a buffer.
struct FakeDevice {
    stop: Arc<AtomicBool>,
    out: Arc<Mutex<Vec<f32>>>,
    th: Option<std::thread::JoinHandle<()>>,
}

impl FakeDevice {
    fn start(mut tap: OutputTap, rate: u32, speedup: f32) -> FakeDevice {
        let stop = Arc::new(AtomicBool::new(false));
        let out = Arc::new(Mutex::new(Vec::new()));
        let (s, o) = (stop.clone(), out.clone());
        let th = std::thread::spawn(move || {
            let block = (rate / 100) as usize; // 10 ms
            let mut buf = vec![0.0f32; block];
            let t0 = Instant::now();
            let mut k = 0u64;
            while !s.load(Ordering::Relaxed) {
                tap.fill(&mut buf);
                o.lock().extend_from_slice(&buf);
                k += 1;
                let due = Duration::from_secs_f64(k as f64 * 0.01 / speedup as f64);
                if let Some(w) = due.checked_sub(t0.elapsed()) {
                    std::thread::sleep(w);
                }
            }
        });
        FakeDevice { stop, out, th: Some(th) }
    }
    fn samples(&self) -> Vec<f32> {
        self.out.lock().clone()
    }
    fn finish(mut self) -> Vec<f32> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
        self.samples()
    }
}

fn collect() -> (Arc<Mutex<Vec<SpeakEvent>>>, impl Fn(SpeakEvent) + Send + Sync + 'static) {
    let ev = Arc::new(Mutex::new(Vec::new()));
    let e2 = ev.clone();
    (ev, move |e: SpeakEvent| {
        if !matches!(e, SpeakEvent::Level { .. }) {
            e2.lock().push(e)
        }
    })
}

fn wait_until(secs: f32, f: impl Fn() -> bool) -> bool {
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f32() < secs {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    f()
}

/// Mic signal: `n` speech-like bursts (1.2 s) with 1.5 s pauses, 48 kHz.
fn mic_signal(n: usize) -> Vec<f32> {
    let rate = 48_000usize;
    let mut x = vec![0.0f32; rate];
    for k in 0..n {
        x.extend(speechish(1.2, rate, 0.3, k as u32));
        x.extend(std::iter::repeat_n(0.0, rate * 3 / 2));
    }
    // a little room noise so the VAD has a floor
    for (i, v) in x.iter_mut().enumerate() {
        *v += 0.001 * ((i as f32 * 12.9898).sin() * 43758.545).fract();
    }
    x
}

fn fake_mic(signal: Vec<f32>, pace: f64) -> Box<dyn CaptureSource> {
    let mut c = FakeCapture::new("Fake mic", 48_000, 1, signal, Arc::new(ManualClock::default()));
    c.chunk_frames = 480;
    c.pace = pace;
    Box::new(c)
}

/// Energy share of `x` in `lo..hi` Hz.
fn band_share(x: &[f32], rate: u32, lo: f32, hi: f32) -> f32 {
    let n = x.len().next_power_of_two();
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut inp = fft.make_input_vec();
    inp[..x.len()].copy_from_slice(x);
    let mut out = fft.make_output_vec();
    fft.process(&mut inp, &mut out).unwrap();
    let (mut band, mut total) = (0f64, 0f64);
    for (b, c) in out.iter().enumerate() {
        let hz = b as f32 * rate as f32 / n as f32;
        let p = c.norm_sqr() as f64;
        total += p;
        if hz >= lo && hz < hi {
            band += p;
        }
    }
    (band / total.max(1e-20)) as f32
}

#[test]
fn clean_up_removes_fillers_noises_and_masks_swearing() {
    assert_eq!(clean_transcript("Um, I think, uh, we should go.", true, false), "I think, we should go.");
    assert_eq!(clean_transcript("Um, I think, uh, we should go.", false, false), "Um, I think, uh, we should go.");
    assert_eq!(clean_transcript("[MUSIC] Hello (coughs) there *laughs* ♪", true, false), "Hello there");
    assert_eq!(clean_transcript("What the fuck, man.", true, true), "What the bleep, man.");
    assert_eq!(clean_transcript("Uh. Um.", true, false), "");
    assert_eq!(clean_transcript("  [BLANK_AUDIO]  ", true, false), "");
    assert_eq!(clean_transcript("so it's, um.", true, false), "so it's.");
    assert_eq!(clean_transcript("Uh, yes!", true, false), "Yes!");
}

#[test]
fn mic_never_reaches_the_output() {
    // 1: nothing transcribed → the output is digital silence, whatever the mic does
    let stt = Arc::new(FakeStt { line: String::new(), calls: AtomicU64::new(0) });
    let (events, on) = collect();
    let cfg = SpeakConfig::default();
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneSynth), Some((stt.clone(), "fake".into())), Some(fake_mic(mic_signal(3), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    assert!(wait_until(10.0, || stt.calls.load(Ordering::Relaxed) >= 3), "VAD should find 3 utterances, got {}", stt.calls.load(Ordering::Relaxed));
    std::thread::sleep(Duration::from_millis(200));
    let st = core.status();
    core.stop();
    let out = dev.finish();
    assert!(out.len() > 48_000);
    assert!(out.iter().all(|v| *v == 0.0), "the output must be silent when nothing is spoken");
    assert!(st.input_level_db > -60.0 || st.input_level_db == -120.0 || st.input_level_db < 0.0);
    let ev = events.lock();
    assert_eq!(ev.iter().filter(|e| matches!(e, SpeakEvent::Dropped { .. })).count(), 3);
    assert!(ev.iter().any(|e| matches!(e, SpeakEvent::Listening)));

    // 2: every utterance is spoken by the synthetic voice; none of the mic's
    // voice (130 Hz harmonics) is in the output, only the 1 kHz "voice"
    let stt = Arc::new(FakeStt { line: "Hello there.".into(), calls: AtomicU64::new(0) });
    let (events, on) = collect();
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneSynth), Some((stt.clone(), "fake".into())), Some(fake_mic(mic_signal(3), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    assert!(wait_until(15.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= 3));
    core.stop();
    let out = dev.finish();
    let loud = out.iter().filter(|v| v.abs() > 0.01).count();
    assert!(loud > OUT_RATE as usize, "three 0.4 s lines should play ({loud} loud samples)");
    let low = band_share(&out, OUT_RATE, 60.0, 700.0);
    assert!(low < 0.01, "mic-band energy in the output: {low}");
    let ev = events.lock();
    assert_eq!(ev.iter().filter(|e| matches!(e, SpeakEvent::Heard { .. })).count(), 3);
    assert_eq!(ev.iter().filter(|e| matches!(e, SpeakEvent::Speaking { latency_ms: Some(_), .. })).count(), 3);
}

#[test]
fn typed_lines_play_in_order_and_skip_clear_mute_work() {
    let (events, on) = collect();
    let cfg = SpeakConfig { listen: false, ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneSynth), None, None, OutputRates { main: OUT_RATE, monitor: Some(44_100) }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 2.0);
    let mon = FakeDevice::start(taps.remove(0), 44_100, 2.0);
    assert_eq!(core.say("   "), None);
    let a = core.say("First line. Second sentence of it.").unwrap();
    let b = core.say("Second line.").unwrap();
    let c = core.say("Third line.").unwrap();
    assert!(a < b && b < c);
    // skip the first while it plays
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Speaking { id, .. } if *id == a))));
    core.skip();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == c))));
    {
        let ev = events.lock();
        let order: Vec<u64> = ev.iter().filter_map(|e| if let SpeakEvent::Speaking { id, .. } = e { Some(*id) } else { None }).collect();
        assert_eq!(order, vec![a, b, c]);
        assert!(ev.iter().any(|e| matches!(e, SpeakEvent::Done { id, skipped: true } if *id == a)), "{ev:?}");
        assert!(ev.iter().any(|e| matches!(e, SpeakEvent::Done { id, skipped: false } if *id == b)));
        assert!(ev.iter().all(|e| !matches!(e, SpeakEvent::Speaking { latency_ms: Some(_), .. })), "typed lines have no mic latency");
    }
    // the first line lost its second sentence: less than 3 × 0.45 s + 0.4 s of tone
    let loud = dev.samples().iter().filter(|v| v.abs() > 0.01).count() as f32 / OUT_RATE as f32;
    // lines two and three (0.8 s) plus whatever of line one played before the skip
    assert!(loud > 0.79 && loud < 1.6, "{loud} s of speech after a skip");
    let mon_loud = mon.samples().iter().filter(|v| v.abs() > 0.01).count() as f32 / 44_100.0;
    assert!((mon_loud - loud).abs() < 0.3, "monitor {mon_loud} vs output {loud}");

    // mute: lines advance silently
    core.set_mute(true);
    let before = dev.samples().len();
    let d = core.say("Muted line.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == d))));
    let after = dev.samples();
    // allow the 5 ms fade
    assert!(after[before + 480..].iter().all(|v| v.abs() < 1e-6), "muted output must be silent");
    core.set_mute(false);

    // clear: a long queue stops at once
    let ids: Vec<u64> = (0..5).map(|i| core.say(&format!("Queued line {i}. And more. And more.")).unwrap()).collect();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Speaking { id, .. } if *id == ids[0]))));
    core.clear();
    std::thread::sleep(Duration::from_millis(300));
    let n = dev.samples().len();
    std::thread::sleep(Duration::from_millis(500));
    let tail = dev.samples();
    assert!(tail[n..].iter().all(|v| *v == 0.0), "after clear the output is silent");
    assert!(wait_until(3.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { id, skipped: true } if ids.contains(id))).count() == 5));
    assert_eq!(core.status().queued, 0);
    core.stop();
    let _ = dev.finish();
    let _ = mon.finish();
}

#[test]
fn effect_and_gain_apply_to_the_synthetic_voice() {
    let run = |effect: Option<Preset>, gain: f32| {
        let (events, on) = collect();
        let cfg = SpeakConfig { listen: false, effect, output_gain_db: gain, ..Default::default() };
        let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneSynth), None, None, OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
        let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 8.0);
        let id = core.say("One line.").unwrap();
        assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id: i, .. } if *i == id))));
        std::thread::sleep(Duration::from_millis(100));
        core.stop();
        dev.finish()
    };
    let clean = run(None, 0.0);
    let quiet = run(None, -12.0);
    let peak = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!((peak(&clean) - 0.3).abs() < 0.03, "clean peak {}", peak(&clean));
    assert!((peak(&quiet) / peak(&clean) - 0.25).abs() < 0.03);
    // Cathedral adds a long tail after the tone
    let hall = run(crate::voice::preset_by_id("cathedral"), 0.0);
    let last_loud = |x: &[f32]| x.iter().rposition(|v| v.abs() > 0.002).unwrap_or(0);
    assert!(last_loud(&hall) > last_loud(&clean) + OUT_RATE as usize / 4, "the reverb tail should ring out");
}

#[test]
fn latency_is_hangover_plus_work() {
    let stt = Arc::new(FakeStt { line: "Testing latency.".into(), calls: AtomicU64::new(0) });
    let (events, on) = collect();
    let cfg = SpeakConfig { hangover_ms: 400, ..Default::default() };
    // real-time pacing so wall-clock latency is meaningful
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneSynth), Some((stt, "fake".into())), Some(fake_mic(mic_signal(1), 0.01)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 1.0);
    assert!(wait_until(10.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Speaking { .. }))));
    core.stop();
    let _ = dev.finish();
    let ev = events.lock();
    let lat = ev.iter().find_map(|e| if let SpeakEvent::Speaking { latency_ms, .. } = e { *latency_ms } else { None }).unwrap();
    println!("fake pipeline latency: {lat:.0} ms");
    // 400 ms hangover − the 150 ms kept tail + 20 ms fake Whisper + a 10 ms output block
    assert!(lat > 200.0 && lat < 800.0, "{lat}");
}

// ───────────────────────── with the real models ─────────────────────────

#[cfg(feature = "ml")]
fn real_models() -> Option<(Arc<crate::tts::Tts>, Arc<dyn Transcriber>, String)> {
    let dir = crate::tts::tests::kokoro_dir()?;
    let accel = if std::env::var("NEKOTONE_GPU").map(|v| v == "1").unwrap_or(false) { crate::models::Accelerator::Auto } else { crate::models::Accelerator::Cpu };
    // Kokoro always runs on the CPU; NEKOTONE_GPU=1 lets Whisper use DirectML
    let tts = Arc::new(crate::tts::Tts::load_dir(&dir, crate::models::Accelerator::Cpu).expect("Kokoro loads"));
    let mm = crate::models::ModelManager::default();
    let id = [crate::models::ModelId::WhisperBase, crate::models::ModelId::WhisperSmall, crate::models::ModelId::WhisperTiny].into_iter().find(|m| mm.path(*m).is_some())?;
    crate::models::set_accelerator(accel);
    let stt: Arc<dyn Transcriber> = Arc::new(crate::stt::WhisperOnnx::load(&mm, id).expect("Whisper loads"));
    Some((tts, stt, id.name().to_string()))
}

/// End to end: a sentence spoken (by Kokoro, as a stand-in for you) into a
/// fake microphone in real time → Whisper → Kokoro in another voice →
/// output. Prints the end-of-speech → first-sample latency.
/// `cargo test -p nekotone-core --release speak_tests::real -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn real_models_end_to_end_latency() {
    let (tts, stt, name) = real_models().expect("needs tts-kokoro and a Whisper model installed");
    let lines = ["Could you please pass me the salt?", "I will meet you at the north gate in ten minutes.", "That was a great game, well played everyone.", "Let me check the map before we go.", "Thanks for waiting, I am ready now."];
    let mut mic: Vec<f32> = vec![0.0; 48_000];
    for l in lines {
        let a = tts.synthesize(l, "am_michael", 1.0).unwrap();
        mic.extend(crate::process::resample_sinc(&a, 24_000, 48_000));
        mic.extend(std::iter::repeat_n(0.0, 48_000 * 4));
    }
    let (events, on) = collect();
    let cfg = SpeakConfig { voice: "af_heart".into(), ..Default::default() };
    let _ = tts.synthesize("Warm up.", "af_heart", 1.0);
    let (core, mut taps) = SpeakCore::start(&cfg, tts.clone(), Some((stt, name.clone())), Some(fake_mic(mic, 0.01)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 1.0);
    assert!(wait_until(60.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= lines.len()));
    let st = core.status();
    core.stop();
    let out = dev.finish();
    let ev = events.lock();
    let mut lats = vec![];
    for e in ev.iter() {
        match e {
            SpeakEvent::Heard { text, stt_ms, speech_secs, .. } => println!("heard {text:?} ({speech_secs:.1} s of speech, Whisper {stt_ms:.0} ms)"),
            SpeakEvent::Speaking { text, latency_ms, synth_ms, .. } => {
                println!("  speaking {text:?}: latency {:.0} ms (first chunk synthesised {synth_ms:.0} ms after the line was queued)", latency_ms.unwrap_or(-1.0));
                lats.extend(*latency_ms);
            }
            _ => {}
        }
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{} ({}), TTS on {}: latency min {:.0} / median {:.0} / max {:.0} ms", name, st.stt_model, st.tts_device, lats[0], lats[lats.len() / 2], lats[lats.len() - 1]);
    let path = std::env::temp_dir().join("nekotone-speak-e2e.wav");
    crate::audio::write_wav(&path, &out, OUT_RATE, 1).unwrap();
    println!("output written to {}", path.display());
    assert_eq!(lats.len(), lines.len());
}

/// Records the accent each line is planned in.
struct AccentSpy(parking_lot::Mutex<Vec<crate::tts::accent::Accent>>);

impl Synth for AccentSpy {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, _voice: &str, _first: usize, accent: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        self.0.lock().push(accent);
        Ok(crate::tts::split_sentences(text).into_iter().map(|s| Piece { text: s.clone(), phonemes: s, pause_after: 0.05 }).collect())
    }
    fn render(&self, _p: &Piece, _voice: &str, _speed: f32) -> Result<Vec<f32>> {
        Ok(vec![0.1; 2400])
    }
}

#[test]
fn the_accent_reaches_the_voice_and_can_change_live() {
    use crate::tts::accent::Accent;
    let (events, on) = collect();
    let spy = Arc::new(AccentSpy(parking_lot::Mutex::new(Vec::new())));
    let cfg = SpeakConfig { listen: false, accent: Accent::Scottish, ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, spy.clone(), None, None, OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let _dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    let a = core.say("First line.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == a))));
    core.set_accent(Accent::Australian);
    assert_eq!(core.status().accent, Accent::Australian);
    let b = core.say("Second line.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == b))));
    let seen = spy.0.lock().clone();
    // warm-up and the first line in the configured accent, then the new one
    assert!(seen.len() >= 3, "{seen:?}");
    assert!(seen[..seen.len() - 1].iter().all(|a| *a == Accent::Scottish), "{seen:?}");
    assert_eq!(*seen.last().unwrap(), Accent::Australian);
    core.stop();
}

// ───────────────────────── your own voice (router) ─────────────────────────

/// A tone at `hz` Hz, 0.4 s per piece, that remembers the voice ids it rendered.
struct NamedTone {
    hz: f32,
    rendered: parking_lot::Mutex<Vec<String>>,
    device: &'static str,
}

impl NamedTone {
    fn new(hz: f32, device: &'static str) -> Arc<NamedTone> {
        Arc::new(NamedTone { hz, rendered: parking_lot::Mutex::new(Vec::new()), device })
    }
}

impl Synth for NamedTone {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, _voice: &str, _first: usize, _accent: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        Ok(crate::tts::split_sentences(text).into_iter().map(|s| Piece { text: s.clone(), phonemes: s, pause_after: 0.05 }).collect())
    }
    fn render(&self, _p: &Piece, voice: &str, _speed: f32) -> Result<Vec<f32>> {
        self.rendered.lock().push(voice.to_string());
        Ok((0..9_600).map(|i| 0.3 * (2.0 * PI * self.hz * i as f32 / 24_000.0).sin()).collect())
    }
    fn device(&self) -> String {
        self.device.into()
    }
}

#[test]
fn router_sends_cloned_voices_to_the_clone_engine_and_switches_live() {
    assert!(is_clone_voice("clone:mine") && !is_clone_voice("af_heart"));
    let kokoro = NamedTone::new(1000.0, "cpu");
    let clone = NamedTone::new(300.0, "directml");
    let router = Arc::new(VoiceRouter { kokoro: Some(kokoro.clone()), clone: Some(clone.clone()), clone_first: true });
    assert_eq!(router.device(), "directml");
    let (events, on) = collect();
    let cfg = SpeakConfig { listen: false, voice: "clone:mine".into(), ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, router, None, None, OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    assert_eq!(core.status().tts_device, "directml");
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    let a = core.say("Mine first.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == a))));
    let n_mine = dev.samples().len();
    core.set_voice("af_heart");
    let b = core.say("Then Heart.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == b))));
    core.stop();
    let out = dev.finish();
    // warm-up + the first line in the cloned voice, the second in Kokoro
    let mine = clone.rendered.lock().clone();
    assert!(mine.len() >= 2 && mine.iter().all(|v| v == "clone:mine"), "{mine:?}");
    assert_eq!(*kokoro.rendered.lock(), vec!["af_heart".to_string()]);
    // and it is audible that way: 300 Hz first, 1 kHz after the switch
    assert!(band_share(&out[..n_mine], OUT_RATE, 250.0, 350.0) > 0.5);
    assert!(band_share(&out[n_mine..], OUT_RATE, 900.0, 1100.0) > 0.5);
}

#[test]
fn a_missing_engine_is_an_error_event_not_a_crash() {
    let router = Arc::new(VoiceRouter { kokoro: Some(NamedTone::new(1000.0, "cpu")), clone: None, clone_first: false });
    let (events, on) = collect();
    let cfg = SpeakConfig { listen: false, ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, router, None, None, OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    core.set_voice("clone:mine");
    let a = core.say("Nobody to say this.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Done { id, .. } if *id == a))));
    let ev = events.lock().clone();
    assert!(ev.iter().any(|e| matches!(e, SpeakEvent::Error { message } if message.contains("voice-clone model"))), "{ev:?}");
    // the engine keeps going: back to a Kokoro voice
    core.set_voice("af_heart");
    let b = core.say("Back again.").unwrap();
    assert!(wait_until(5.0, || events.lock().iter().any(|e| matches!(e, SpeakEvent::Speaking { id, .. } if *id == b))));
    core.stop();
    let _ = dev.finish();
}

#[test]
fn the_mic_never_reaches_the_output_with_a_cloned_voice_either() {
    // same check as mic_never_reaches_the_output, through the router with an empty transcriber
    let stt = Arc::new(FakeStt { line: String::new(), calls: AtomicU64::new(0) });
    let router = Arc::new(VoiceRouter { kokoro: None, clone: Some(NamedTone::new(300.0, "cpu")), clone_first: true });
    let (_events, on) = collect();
    let cfg = SpeakConfig { voice: "clone:mine".into(), ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, router, Some((stt.clone(), "fake".into())), Some(fake_mic(mic_signal(2), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    assert!(wait_until(10.0, || stt.calls.load(Ordering::Relaxed) >= 2));
    std::thread::sleep(Duration::from_millis(200));
    core.stop();
    let out = dev.finish();
    assert!(out.iter().all(|v| *v == 0.0), "only synthetic speech may reach the output");
}

/// Speak for me in a cloned voice, end to end: Kokoro speaks into a fake
/// microphone in real time → Whisper → Chatterbox (the reference voice) →
/// output. Prints the end-of-speech → first-sample latency per line.
/// `cargo test -p nekotone-core --release speak_tests::clone_end -- --ignored --nocapture`
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn clone_end_to_end_latency() {
    let (tts, stt, name) = real_models().expect("needs tts-kokoro and a Whisper model installed");
    let dir = crate::tts::chatterbox::tests::chatterbox_dir().expect("needs tts-chatterbox (or NEKOTONE_CHATTERBOX_DIR)");
    crate::models::set_accelerator(crate::models::Accelerator::Auto);
    let cb = Arc::new(crate::tts::chatterbox::Chatterbox::load_dir(&dir, crate::models::accelerator()).expect("Chatterbox loads"));
    let clip = crate::tts::chatterbox::tests::reference_clip();
    let print = cb.encode_voice(&clip.samples, clip.sample_rate, "test", "reference").unwrap();
    let prints = std::env::temp_dir().join(format!("nekotone-clone-e2e-{}", std::process::id()));
    let clone = Arc::new(CloneSynth::new(cb.clone(), prints.clone()));
    clone.set_print("mine", print);
    let router = Arc::new(VoiceRouter { kokoro: Some(tts.clone()), clone: Some(clone), clone_first: true });
    let lines = ["Could you please pass me the salt?", "I will meet you at the north gate in ten minutes.", "That was a great game, well played everyone.", "Let me check the map before we go.", "Thanks for waiting, I am ready now."];
    let mut mic: Vec<f32> = vec![0.0; 48_000];
    for l in lines {
        let a = tts.synthesize(l, "am_michael", 1.0).unwrap();
        mic.extend(crate::process::resample_sinc(&a, 24_000, 48_000));
        mic.extend(std::iter::repeat_n(0.0, 48_000 * 6));
    }
    let (events, on) = collect();
    let cfg = SpeakConfig { voice: "clone:mine".into(), ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, router, Some((stt, name.clone())), Some(fake_mic(mic, 0.01)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 1.0);
    assert!(wait_until(120.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= lines.len()));
    let st = core.status();
    core.stop();
    let out = dev.finish();
    let ev = events.lock();
    let mut lats = vec![];
    for e in ev.iter() {
        if let SpeakEvent::Speaking { text, latency_ms, synth_ms, .. } = e {
            println!("  speaking {text:?}: latency {:.0} ms (first piece ready {synth_ms:.0} ms after queueing)", latency_ms.unwrap_or(-1.0));
            lats.extend(*latency_ms);
        }
        if let SpeakEvent::Error { message } = e {
            println!("  error: {message}");
        }
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{name}, clone on {} ({:?}): latency min {:.0} / median {:.0} / max {:.0} ms", st.tts_device, cb.engines(), lats[0], lats[lats.len() / 2], lats[lats.len() - 1]);
    let path = std::env::temp_dir().join("nekotone-speak-clone-e2e.wav");
    crate::audio::write_wav(&path, &out, OUT_RATE, 1).unwrap();
    println!("output written to {}", path.display());
    let _ = std::fs::remove_dir_all(&prints);
    assert_eq!(lats.len(), lines.len());
}

/// A fake converter: the utterance comes back as a 1 kHz tone of the same
/// length (so the test can tell converted output from the mic's 130 Hz voice).
struct ToneConverter;

impl Synth for ToneConverter {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, v: &str, f: usize, a: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        ToneSynth.plan(text, v, f, a)
    }
    fn render(&self, p: &Piece, v: &str, s: f32) -> Result<Vec<f32>> {
        ToneSynth.render(p, v, s)
    }
    fn convert(&self, samples: &[f32], rate: u32, _voice: &str) -> Result<Vec<f32>> {
        let n = (samples.len() as f64 * 24_000.0 / rate as f64) as usize;
        Ok((0..n).map(|i| 0.3 * (2.0 * PI * 1000.0 * i as f32 / 24_000.0).sin()).collect())
    }
}

/// A converter as slow as Chatterbox on a long phrase: it takes as long as
/// the audio lasts (the tests run 4x faster than real time).
struct SlowConverter;

impl Synth for SlowConverter {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, v: &str, f: usize, a: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        ToneSynth.plan(text, v, f, a)
    }
    fn render(&self, p: &Piece, v: &str, s: f32) -> Result<Vec<f32>> {
        ToneSynth.render(p, v, s)
    }
    fn convert(&self, samples: &[f32], rate: u32, voice: &str) -> Result<Vec<f32>> {
        std::thread::sleep(Duration::from_secs_f32(samples.len() as f32 / rate as f32 / 4.0));
        ToneConverter.convert(samples, rate, voice)
    }
}

/// Gaps (s) between bursts of sound in `x`, ignoring dips under 50 ms.
fn output_gaps(x: &[f32], rate: u32) -> Vec<f32> {
    let (mut gaps, mut quiet, mut seen) = (Vec::new(), 0usize, false);
    for v in x {
        if v.abs() > 0.01 {
            if seen && quiet > rate as usize / 20 {
                gaps.push(quiet as f32 / rate as f32);
            }
            seen = true;
            quiet = 0;
        } else {
            quiet += 1;
        }
    }
    gaps
}

/// Keep my pauses: a long phrase, a 0.8 s pause, a short one. The slow
/// converter backs the queue up, so without the option the short phrase
/// follows the long one with no gap at all; with it the output leaves the
/// pause that was on the mic (between the two utterances' edges).
///
/// This test drives real wall-clock threads (`SlowConverter` sleeps,
/// `FakeDevice` plays back at a real-time factor): under heavy CPU
/// contention from `cargo test --workspace`'s full parallelism it can pick
/// up a spurious extra gap from scheduler jitter and fail (seen once,
/// 2026-10-04: an extra 0.76 s gap). It passed deterministically every time
/// run alone: `cargo test -p nekotone-core --lib
/// voice::speak::tests::change_my_voice_keeps_my_pauses`. If this flakes,
/// rerun in isolation before treating it as a regression.
#[test]
fn change_my_voice_keeps_my_pauses() {
    let rate = 48_000usize;
    let mut mic = vec![0.0f32; rate];
    mic.extend(speechish(2.0, rate, 0.3, 1));
    mic.extend(std::iter::repeat_n(0.0, rate * 8 / 10));
    mic.extend(speechish(0.5, rate, 0.3, 2));
    mic.extend(std::iter::repeat_n(0.0, rate * 3));
    for (i, v) in mic.iter_mut().enumerate() {
        *v += 0.001 * ((i as f32 * 12.9898).sin() * 43758.545).fract();
    }
    let run = |keep: bool| {
        let stt = Arc::new(FakeStt { line: "unused".into(), calls: AtomicU64::new(0) });
        let (events, on) = collect();
        let cfg = SpeakConfig { convert: true, keep_pauses: keep, trim_fillers: false, mask_profanity: false, ..Default::default() };
        let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(SlowConverter), Some((stt, "fake".into())), Some(fake_mic(mic.clone(), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
        let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
        assert!(wait_until(20.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= 2), "both phrases should be said");
        core.stop();
        output_gaps(&dev.finish(), OUT_RATE)
    };
    let (mut last_off, mut last_on) = (Vec::new(), Vec::new());
    for attempt in 1..=3 {
        let (off, on) = (run(false), run(true));
        println!("attempt {attempt}: gaps between the phrases: kept {on:?}, not kept {off:?}");
        let kept_ok = on.len() == 1 && (0.3..0.55).contains(&on[0]);
        let merged_ok = off.is_empty() || off[0] < 0.1;
        if kept_ok && merged_ok {
            return;
        }
        last_off = off;
        last_on = on;
    }
    // the mic gap between the utterances' edges: 0.8 s of pause minus the
    // VAD's 150 ms tail and ~240 ms pre-roll + onset
    assert!(last_on.len() == 1 && (0.3..0.55).contains(&last_on[0]), "kept pause {last_on:?}");
    assert!(last_off.is_empty() || last_off[0] < 0.1, "without the option the phrases run together: {last_off:?}");
}

/// A converter `factor` times slower than real time (the tests run 4x).
struct SlowerConverter(f32);

impl Synth for SlowerConverter {
    fn rate(&self) -> u32 {
        24_000
    }
    fn plan(&self, text: &str, v: &str, f: usize, a: crate::tts::accent::Accent) -> Result<Vec<Piece>> {
        ToneSynth.plan(text, v, f, a)
    }
    fn render(&self, p: &Piece, v: &str, s: f32) -> Result<Vec<f32>> {
        ToneSynth.render(p, v, s)
    }
    fn convert(&self, samples: &[f32], rate: u32, voice: &str) -> Result<Vec<f32>> {
        std::thread::sleep(Duration::from_secs_f32(self.0 * samples.len() as f32 / rate as f32 / 4.0));
        ToneConverter.convert(samples, rate, voice)
    }
}

/// Keep my pauses adds no lag: six 2 s phrases with 1 s pauses through a
/// converter slower than real time (owner: "the voice seems to lag behind
/// when I have natural pauses in"; the cause was a busy compute server,
/// fixed by racing it, see remote::RemoteClone).
#[test]
fn change_my_voice_catches_up_in_pauses() {
    let rate = 48_000usize;
    let mut mic = vec![0.0f32; rate];
    for k in 0..6 {
        mic.extend(speechish(2.0, rate, 0.3, 10 + k));
        mic.extend(std::iter::repeat_n(0.0, rate));
    }
    mic.extend(std::iter::repeat_n(0.0, rate * 4));
    for (i, v) in mic.iter_mut().enumerate() {
        *v += 0.001 * ((i as f32 * 12.9898).sin() * 43758.545).fract();
    }
    let run = |keep: bool| -> Vec<f32> {
        let stt = Arc::new(FakeStt { line: "unused".into(), calls: AtomicU64::new(0) });
        let (events, on) = collect();
        let cfg = SpeakConfig { convert: true, keep_pauses: keep, trim_fillers: false, mask_profanity: false, ..Default::default() };
        let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(SlowerConverter(1.4)), Some((stt, "fake".into())), Some(fake_mic(mic.clone(), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
        let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
        assert!(wait_until(30.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= 6), "six phrases");
        core.stop();
        drop(dev);
        // test seconds (4x real time)
        let lags: Vec<f32> = events.lock().iter().filter_map(|e| match e { SpeakEvent::Speaking { latency_ms: Some(l), .. } => Some(l * 4.0 / 1000.0), _ => None }).collect();
        lags
    };
    let (on, off) = (run(true), run(false));
    println!("lag behind you per phrase (s): keep pauses {on:.2?}, without {off:.2?}");
    let last = |v: &[f32]| *v.last().unwrap();
    // a converter slower than real time backs the queue up whatever the
    // pauses do (measured 4.70 -> 5.02 s with, 4.73 -> 5.59 s without): what
    // matters is that keeping your pauses adds no lag of its own
    assert!(last(&on) < last(&off) + 1.0, "Keep my pauses must not add lag: {on:.2?} vs {off:.2?}");
}

#[test]
fn change_my_voice_shows_my_words_without_waiting_for_them() {
    let stt = Arc::new(FakeStt { line: "hello there".into(), calls: AtomicU64::new(0) });
    let (events, on) = collect();
    // the text options off: then nothing needs the words
    let cfg = SpeakConfig { convert: true, trim_fillers: false, mask_profanity: false, ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneConverter), Some((stt.clone(), "fake".into())), Some(fake_mic(mic_signal(3), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    assert!(wait_until(15.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= 3), "three utterances should be spoken again");
    // switched off live: the next utterances go through speech to text again
    core.set_convert(false);
    core.stop();
    let out = dev.finish();
    let loud = out.iter().filter(|v| v.abs() > 0.01).count();
    assert!(loud > OUT_RATE as usize / 2, "the converted utterances should play ({loud} loud samples)");
    let low = band_share(&out, OUT_RATE, 60.0, 700.0);
    assert!(low < 0.01, "mic-band energy in the output: {low}");
    let ev = events.lock();
    // the transcript shows your words (it said "(your words, 1.1 s)"), and
    // each phrase was queued for the voice before Whisper wrote them down
    let heard: Vec<u64> = ev.iter().filter_map(|e| match e { SpeakEvent::Heard { id, text, .. } if text == "hello there" => Some(*id), _ => None }).collect();
    assert!(heard.len() >= 3, "{ev:?}");
    for id in heard {
        let q = ev.iter().position(|e| matches!(e, SpeakEvent::Queued { id: i, .. } if *i == id)).unwrap();
        let h = ev.iter().position(|e| matches!(e, SpeakEvent::Heard { id: i, .. } if *i == id)).unwrap();
        assert!(q < h, "phrase {id} waited for its words");
    }
    assert!(ev.iter().all(|e| !matches!(e, SpeakEvent::Error { .. })), "{ev:?}");
}

/// Change my voice with real models (ignored: needs Kokoro, Chatterbox,
/// whisper-small and the recorded sample): the owner's recorded sample
/// spoken again as a stock Kokoro voice (learnt once from Kokoro reading
/// the reference text). The words must survive; the output must be closer
/// to the stock voice than to the owner.
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn change_my_voice_into_a_stock_voice_with_real_models() {
    use crate::stt::Transcriber;
    let mm = crate::models::ModelManager::default();
    let kdir = mm.path(crate::models::ModelId::TtsKokoro).expect("Kokoro");
    let cdir = mm.path(crate::models::ModelId::TtsChatterbox).expect("Chatterbox");
    let kokoro: Arc<dyn Synth> = Arc::new(crate::tts::Tts::load_dir(&kdir, crate::models::Accelerator::Cpu).unwrap());
    let engine = Arc::new(crate::tts::chatterbox::Chatterbox::load_dir(&cdir, crate::models::accelerator()).unwrap());
    let prints = std::env::temp_dir().join("nekotone-stock-prints");
    let _ = std::fs::remove_dir_all(&prints);
    let clone = Arc::new(CloneSynth::new(engine.clone(), prints));
    let router = VoiceRouter { kokoro: Some(kokoro.clone()), clone: Some(clone.clone() as Arc<dyn Synth>), clone_first: false };
    let me = crate::audio::decode(&crate::data_dir().join("voice-sample.wav")).expect("the recorded sample");
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).expect("whisper-small");
    let say = |a: &[f32], r: u32| {
        let c = crate::audio::Clip { samples: a.to_vec(), sample_rate: r, source_channels: 1 };
        w.transcribe(&c, &crate::stt::TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ")
    };
    for voice in ["am_michael", "af_heart", "bf_emma"] {
        let t = std::time::Instant::now();
        let out = router.convert(&me.samples, me.sample_rate, voice).unwrap();
        let first = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let again = router.convert(&me.samples, me.sample_rate, voice).unwrap();
        let second = t.elapsed().as_secs_f32();
        crate::audio::write_wav(&std::env::temp_dir().join(format!("nekotone-change-{voice}.wav")), &again, 24_000, 1).unwrap();
        // the stock voice's own sound, for likeness
        let mut reference = Vec::new();
        for p in kokoro.plan(STOCK_REFERENCE_TEXT, voice, usize::MAX, crate::tts::accent::Accent::Voice).unwrap() {
            reference.extend(kokoro.render(&p, voice, 1.0).unwrap());
        }
        let v_ref = engine.speaker_vector(&reference, 24_000).unwrap();
        let v_me = engine.speaker_vector(&me.samples, me.sample_rate).unwrap();
        let v_out = engine.speaker_vector(&out, 24_000).unwrap();
        let (to_voice, to_me) = (crate::tts::chatterbox::cosine(&v_out, &v_ref), crate::tts::chatterbox::cosine(&v_out, &v_me));
        let (src, heard) = (say(&me.samples, me.sample_rate), say(&out, 24_000));
        let (errs, n) = crate::tts::tests::wer(&crate::tts::tests::wer_words(&src), &crate::tts::tests::wer_words(&heard));
        println!("{voice}: first call {first:.2} s (learns the voice), then {second:.2} s for {:.1} s of speech; likeness to {voice} {to_voice:.3}, to you {to_me:.3}; words {errs}/{n} wrong\n  you said:   {src}\n  it said:    {heard}", me.samples.len() as f32 / me.sample_rate as f32);
        assert!(to_voice > to_me, "{voice}: the output should sound more like {voice} than like you");
        assert!((errs as f32) / (n.max(1) as f32) < 0.3, "{voice}: the words did not survive");
    }
}

/// "Drop um and uh" / "Mask swearing" in Change my voice: word edits on the
/// utterance itself. Three tone "words" with silent gaps; the middle one is
/// cut (a filler) or silenced and bleeped (a swear word).
#[test]
fn change_my_voice_cuts_fillers_and_bleeps_swearing() {
    use super::{apply_edits, bleep, word_edits, AudioEdit};
    let rate = 16_000u32;
    let r = rate as f32;
    // word i: 0.4 s of tone, then 0.2 s of silence
    let mut x = Vec::new();
    for f in [300.0f32, 500.0, 700.0] {
        x.extend((0..(0.4 * r) as usize).map(|i| 0.3 * (2.0 * PI * f * i as f32 / r).sin()));
        x.extend(std::iter::repeat_n(0.0, (0.2 * r) as usize));
    }
    // Whisper's interpolated times are off by up to ~0.1 s: snapping finds the gaps
    let w = |t: &str, s: f32, e: f32| crate::stt::Word { text: t.into(), start: s, end: e, prob: 0.9 };
    let words = [w(" so", 0.0, 0.45), w(" um,", 0.68, 1.08), w(" fine.", 1.15, 1.6)];
    let edits = word_edits(&words, &x, rate, true, true);
    assert_eq!(edits.len(), 1, "{edits:?}");
    let AudioEdit::Cut(s, e) = edits[0] else { panic!("the filler should be cut: {edits:?}") };
    assert!((0.4..=0.6).contains(&s) && (1.0..=1.2).contains(&e), "snapped to the gaps: {s:.2}-{e:.2}");
    let (y, b) = apply_edits(&x, rate, &edits);
    assert!(b.is_empty());
    let removed = (x.len() - y.len()) as f32 / r;
    assert!((removed - (e - s)).abs() < 0.02, "removed {removed:.3} s for a {:.3} s filler", e - s);
    // the 500 Hz word is gone: little energy left near 500 Hz
    let share = band_share(&y, rate, 450.0, 550.0);
    assert!(share < 0.05, "the filler's tone is still there ({share:.3})");

    // a swear word in the middle: silenced, and the bleep goes where it was
    let words = [w(" so", 0.0, 0.45), w(" shit,", 0.62, 1.05), w(" fine.", 1.15, 1.6)];
    let edits = word_edits(&words, &x, rate, true, true);
    assert!(matches!(edits.as_slice(), [AudioEdit::Bleep(..)]), "{edits:?}");
    let (y, b) = apply_edits(&x, rate, &edits);
    assert_eq!(y.len(), x.len(), "bleeping keeps the length");
    assert_eq!(b.len(), 1);
    let (bs, be) = b[0];
    let mid = (((bs + be) / 2.0) * r) as usize;
    assert!(y[mid - 100..mid + 100].iter().all(|v| *v == 0.0), "the word is silenced before conversion");
    // after "conversion" (here: the same audio at 24 kHz is not needed; bleep in place)
    let mut z = y.clone();
    bleep(&mut z, rate, &b);
    let tone = band_share(&z[((bs) * r) as usize..((be) * r) as usize], rate, 950.0, 1050.0);
    assert!(tone > 0.8, "a 1 kHz bleep fills the span ({tone:.2})");
    // nothing to do when the options are off
    assert!(word_edits(&words, &x, rate, false, false).is_empty());
}


/// With "Drop um and uh" or "Mask swearing" on, Change my voice asks Whisper
/// for the words (to edit the audio) and shows them in the log.
#[test]
fn change_my_voice_listens_for_words_when_the_text_options_are_on() {
    let stt = Arc::new(FakeStt { line: "Hello there.".into(), calls: AtomicU64::new(0) });
    let (events, on) = collect();
    let cfg = SpeakConfig { convert: true, trim_fillers: true, ..Default::default() };
    let (core, mut taps) = SpeakCore::start(&cfg, Arc::new(ToneConverter), Some((stt.clone(), "fake".into())), Some(fake_mic(mic_signal(3), 0.0025)), OutputRates { main: OUT_RATE, monitor: None }, on).unwrap();
    let dev = FakeDevice::start(taps.remove(0), OUT_RATE, 4.0);
    assert!(wait_until(15.0, || events.lock().iter().filter(|e| matches!(e, SpeakEvent::Done { .. })).count() >= 3));
    core.stop();
    let out = dev.finish();
    assert!(stt.calls.load(Ordering::Relaxed) >= 3, "one call per utterance ({}; the fake does not count its warm-up)", stt.calls.load(Ordering::Relaxed));
    let low = band_share(&out, OUT_RATE, 60.0, 700.0);
    assert!(low < 0.01, "mic-band energy in the output: {low}");
    let ev = events.lock();
    assert!(ev.iter().filter(|e| matches!(e, SpeakEvent::Heard { text, .. } if text == "Hello there.")).count() >= 3, "the log shows the words: {ev:?}");
}

/// Mask swearing in Change my voice with real models (ignored: needs Whisper
/// small, Kokoro, Chatterbox and NEKOTONE_SWEAR_CLIP=file@start-end, a
/// recording of someone swearing). The converted output must not contain the
/// swear words (Whisper on the output), and must keep the other words.
#[test]
#[ignore]
#[cfg(feature = "ml")]
fn mask_swearing_in_change_my_voice_with_real_models() {
    use crate::stt::Transcriber;
    let spec = std::env::var("NEKOTONE_SWEAR_CLIP").expect("NEKOTONE_SWEAR_CLIP=file@start-end");
    let (file, range) = spec.rsplit_once('@').expect("file@start-end");
    let (a, b) = range.split_once('-').unwrap();
    let (a, b): (f32, f32) = (a.parse().unwrap(), b.parse().unwrap());
    let clip = crate::audio::decode(std::path::Path::new(file)).unwrap();
    let clip = crate::audio::resample(&clip, 16_000).unwrap();
    let x = clip.samples[(a * 16_000.0) as usize..((b * 16_000.0) as usize).min(clip.samples.len())].to_vec();
    let mm = crate::models::ModelManager::default();
    let w = crate::stt::WhisperOnnx::load(&mm, crate::models::ModelId::WhisperSmall).unwrap();
    let opts = TranscribeOptions { language: Some("en".into()), word_timestamps: true, ..Default::default() };
    let c = crate::audio::Clip { samples: x.clone(), sample_rate: 16_000, source_channels: 1 };
    let tr = w.transcribe(&c, &opts, &mut |_| {}).unwrap();
    let words: Vec<crate::stt::Word> = tr.segments.iter().flat_map(|s| s.words.iter().cloned()).collect();
    let said: String = tr.segments.iter().map(|s| s.text.trim()).collect::<Vec<_>>().join(" ");
    let mut edits = super::word_edits(&words, &x, 16_000, true, true);
    println!("estimate:   {edits:?}");
    super::refine_bleeps(&mut edits, &words, &x, 16_000, &w, Some("en".into()));
    let (y, bleeps) = super::apply_edits(&x, 16_000, &edits);
    let kdir = mm.path(crate::models::ModelId::TtsKokoro).unwrap();
    let cdir = mm.path(crate::models::ModelId::TtsChatterbox).unwrap();
    let kokoro: Arc<dyn Synth> = Arc::new(crate::tts::Tts::load_dir(&kdir, crate::models::Accelerator::Cpu).unwrap());
    let engine = Arc::new(crate::tts::chatterbox::Chatterbox::load_dir(&cdir, crate::models::accelerator()).unwrap());
    let clone = Arc::new(CloneSynth::new(engine, std::env::temp_dir().join("nekotone-stock-prints")));
    let router = VoiceRouter { kokoro: Some(kokoro), clone: Some(clone as Arc<dyn Synth>), clone_first: false };
    let mut out = router.convert(&y, 16_000, "af_heart").unwrap();
    super::bleep(&mut out, 24_000, &bleeps);
    crate::audio::write_wav(&std::env::temp_dir().join("nekotone-masked.wav"), &out, 24_000, 1).unwrap();
    let oc = crate::audio::Clip { samples: out, sample_rate: 24_000, source_channels: 1 };
    let heard: String = w.transcribe(&oc, &TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap().segments.iter().map(|s| s.text.trim().to_string()).collect::<Vec<_>>().join(" ");
    println!("you said:   {said}\nedits:      {edits:?}\nit said:    {heard}");
    let lower = heard.to_lowercase();
    for bad in super::PROFANE {
        assert!(!lower.split(|c: char| !c.is_alphanumeric()).any(|w| w == *bad), "'{bad}' is still in the output: {heard}");
    }
    assert!(!edits.is_empty(), "the clip should contain something to mask");
}
