#![cfg_attr(not(feature = "server"), allow(unused_imports))]

use super::*;
use crate::voice::speak::Synth;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn addresses_get_a_scheme_and_the_default_port() {
    assert_eq!(normalise_url("brainchild").as_deref(), Some("http://brainchild:8199"));
    assert_eq!(normalise_url(" brainchild.bennfry.info:9000/ ").as_deref(), Some("http://brainchild.bennfry.info:9000"));
    assert_eq!(normalise_url("http://10.0.0.5").as_deref(), Some("http://10.0.0.5:8199"));
    assert_eq!(normalise_url("https://gpu.example:443").as_deref(), Some("https://gpu.example:443"));
    assert_eq!(normalise_url("   "), None);
    assert_eq!(normalise_url("http://"), None);
}

#[test]
fn samples_cross_the_wire_exactly() {
    let x = vec![0.0f32, -1.0, 0.5, f32::MIN_POSITIVE, 123.456];
    assert_eq!(from_le(&to_le(&x)), x);
}

/// Echoes the input as 24 kHz-length audio; counts calls; `delay_ms` per synth.
#[cfg(feature = "server")]
struct FakeEngine {
    converts: AtomicUsize,
    synths: AtomicUsize,
    delay_ms: AtomicUsize,
}

#[cfg(feature = "server")]
impl CloneEngine for FakeEngine {
    fn backend(&self) -> String {
        "fake".into()
    }
    fn convert(&self, samples: &[f32], rate: u32, print: &Path) -> Result<Vec<f32>> {
        assert!(print.is_file());
        self.converts.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(self.delay_ms.load(Ordering::Relaxed) as u64));
        let n = (samples.len() as f64 * CLONE_RATE as f64 / rate as f64) as usize;
        Ok((0..n).map(|i| samples[i * samples.len() / n.max(1)] * 0.5).collect())
    }
    fn synth(&self, text: &str, _print: &Path) -> Result<Vec<f32>> {
        self.synths.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(self.delay_ms.load(Ordering::Relaxed) as u64));
        Ok(vec![0.25; text.len() * 100])
    }
}

#[cfg(feature = "server")]
fn start(token: Option<&str>) -> (u16, Arc<FakeEngine>, tempfile::TempDir) {
    let engine = Arc::new(FakeEngine { converts: AtomicUsize::new(0), synths: AtomicUsize::new(0), delay_ms: AtomicUsize::new(0) });
    let dir = tempfile::tempdir().unwrap();
    let l = bind("127.0.0.1:0").unwrap();
    let port = l.port();
    let (e, d, t) = (engine.clone() as Arc<dyn CloneEngine>, dir.path().join("prints"), token.map(str::to_string));
    std::thread::spawn(move || l.run(e, &d, t, 2, None));
    (port, engine, dir)
}

#[cfg(feature = "server")]
fn a_print(dir: &Path, name: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let p = crate::tts::chatterbox::print_path(dir, name);
    std::fs::write(&p, format!("not a real print: {name}")).unwrap();
    p
}

#[cfg(feature = "server")]
#[test]
fn the_server_converts_and_speaks_and_takes_each_print_once() {
    let (port, engine, tmp) = start(None);
    let s = ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap();
    let info = s.info().unwrap();
    assert_eq!((info.server.as_str(), info.backend.as_str()), ("nekotone", "fake"));
    assert!(info.features.contains(&"convert".to_string()));
    let print = a_print(&tmp.path().join("mine"), "mine");
    let x: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin()).collect();
    let y = s.convert(&x, 16_000, &print).unwrap();
    assert_eq!(y.len(), 24_000, "24 kHz out");
    assert!((y[100] - x[(100 * 16_000) / 24_000] * 0.5).abs() < 1e-6, "the server's audio comes back exactly");
    let a = s.synth("Hello there", &print).unwrap();
    assert_eq!(a.len(), 1100);
    // the print is kept: the server lost it (restart) → uploaded again once
    for f in std::fs::read_dir(tmp.path().join("prints")).unwrap() {
        std::fs::remove_file(f.unwrap().path()).unwrap();
    }
    assert!(s.convert(&x, 16_000, &print).is_ok(), "a lost print is uploaded again");
    assert_eq!((engine.converts.load(Ordering::Relaxed), engine.synths.load(Ordering::Relaxed)), (2, 1));
    assert!(s.usable());
}

#[cfg(feature = "server")]
#[test]
fn a_token_is_checked() {
    let (port, _, _tmp) = start(Some("s3cret"));
    let bad = ComputeServer::new(&format!("127.0.0.1:{port}"), Some("wrong".into())).unwrap();
    let e = bad.info().unwrap_err().to_string();
    assert!(e.contains("refused the token"), "{e}");
    assert!(!bad.usable(), "a failed server is skipped for a while");
    let good = ComputeServer::new(&format!("127.0.0.1:{port}"), Some("s3cret".into())).unwrap();
    assert!(good.info().is_ok());
}

#[cfg(feature = "server")]
#[test]
fn a_busy_server_is_set_aside_for_a_while() {
    let (port, engine, tmp) = start(None);
    let s = ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap();
    let print = a_print(&tmp.path().join("mine"), "mine");
    s.synth("Quick.", &print).unwrap();
    assert!(s.usable(), "a quick answer keeps the server");
    // 0.05 s of speech that takes 1.6 s: slower than real time and the floor
    engine.delay_ms.store(1600, Ordering::Relaxed);
    let a = s.synth("Slow.", &print).unwrap();
    assert_eq!(a.len(), 500, "the slow answer is still used");
    assert!(!s.usable(), "a busy server is skipped for a while");
}

/// The local engine: counts calls.
#[cfg(feature = "server")]
struct LocalClone(AtomicUsize);

#[cfg(feature = "server")]
impl Synth for LocalClone {
    fn rate(&self) -> u32 {
        CLONE_RATE
    }
    fn plan(&self, _t: &str, _v: &str, _f: usize, _a: crate::tts::accent::Accent) -> Result<Vec<crate::tts::Utterance>> {
        unreachable!()
    }
    fn render(&self, _p: &crate::tts::Utterance, _v: &str, _s: f32) -> Result<Vec<f32>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(vec![1.0; 10])
    }
    fn convert(&self, _s: &[f32], _r: u32, _v: &str) -> Result<Vec<f32>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(vec![1.0; 10])
    }
}

#[cfg(feature = "server")]
#[test]
fn remote_first_falls_back_to_this_pc_and_loads_it_only_then() {
    let tmp = tempfile::tempdir().unwrap();
    let prints = tmp.path().join("voice-clone");
    a_print(&prints, "mine");
    let loads = Arc::new(AtomicUsize::new(0));
    let local = Arc::new(LocalClone(AtomicUsize::new(0)));
    let make = |server: Arc<ComputeServer>| {
        let (loads, local) = (loads.clone(), local.clone());
        RemoteClone::new(server, prints.clone(), move || {
            loads.fetch_add(1, Ordering::Relaxed);
            Ok(local.clone() as Arc<dyn Synth>)
        })
    };
    // a working server: the local engine is never loaded
    let (port, engine, _t) = start(None);
    let up = Arc::new(ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap());
    up.info().unwrap();
    let rc = make(up);
    assert_eq!(rc.convert(&[0.1; 1600], 16_000, "clone:mine").unwrap().len(), 2400);
    assert_eq!(rc.render(&crate::tts::chatterbox::plan("Hi there.", 90)[0], "clone:mine", 1.0).unwrap().len(), "Hi there.".len() * 100);
    assert_eq!(engine.converts.load(Ordering::Relaxed), 1);
    assert_eq!(loads.load(Ordering::Relaxed), 0, "no local engine with a working server");
    assert!(rc.device().contains("fake"), "{}", rc.device());
    // a server that went away: this PC answers, and is loaded once
    let gone = Arc::new(ComputeServer::new("127.0.0.1:9", None).unwrap());
    let rc = make(gone.clone());
    assert_eq!(rc.convert(&[0.1; 1600], 16_000, "clone:mine").unwrap(), vec![1.0; 10]);
    assert!(!gone.usable(), "skipped after the failure");
    assert_eq!(rc.convert(&[0.1; 1600], 16_000, "clone:mine").unwrap(), vec![1.0; 10]);
    assert_eq!(loads.load(Ordering::Relaxed), 1);
    assert_eq!(local.0.load(Ordering::Relaxed), 2);
    // a stock voice not learnt yet has no print: this PC (which learns it)
    assert!(!rc.knows_voice("af_heart"));
}

#[cfg(feature = "server")]
#[test]
fn a_late_server_is_raced_by_this_pc() {
    let tmp = tempfile::tempdir().unwrap();
    let prints = tmp.path().join("voice-clone");
    a_print(&prints, "mine");
    let local = Arc::new(LocalClone(AtomicUsize::new(0)));
    let (port, engine, _t) = start(None);
    engine.delay_ms.store(3000, Ordering::Relaxed);
    let server = Arc::new(ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap());
    let l2 = local.clone();
    let rc = RemoteClone::new(server.clone(), prints.clone(), move || Ok(l2.clone() as Arc<dyn Synth>));
    rc.preload();
    // 1 s of speech: the server gets 0.6 s, then this PC answers
    let t0 = Instant::now();
    let a = rc.convert(&[0.1; 16_000], 16_000, "clone:mine").unwrap();
    let took = t0.elapsed();
    assert_eq!(a, vec![1.0; 10], "this PC's answer");
    assert!(took < Duration::from_millis(1500), "not held back by the server: {took:?}");
    assert!(!server.usable(), "a late server is set aside");
    // set aside: the next call goes straight to this PC
    let t0 = Instant::now();
    rc.convert(&[0.1; 16_000], 16_000, "clone:mine").unwrap();
    assert!(t0.elapsed() < Duration::from_millis(200));
    assert_eq!(local.0.load(Ordering::Relaxed), 2);
}

/// A real server (ignored): NEKOTONE_SERVER=brainchild. Your saved voice
/// (`voice-clone/mine.nkvoice`) reads two lines and converts the notes
/// speech clip there; times include this PC's network round trip.
#[test]
#[ignore]
fn a_real_server_from_this_pc() {
    let url = std::env::var("NEKOTONE_SERVER").expect("NEKOTONE_SERVER=host[:port]");
    let s = ComputeServer::new(&url, std::env::var("NEKOTONE_SERVER_TOKEN").ok()).unwrap();
    println!("{:?}", s.info().unwrap());
    let print = crate::tts::chatterbox::print_path(&crate::data_dir().join("voice-clone"), "mine");
    for line in ["Hello, this is my own voice, speaking for me.", "The meeting moved to three thirty, so I will be a little late."] {
        let t = Instant::now();
        let a = s.synth(line, &print).unwrap();
        println!("synth {:.2} s of audio in {:.2} s: {line}", a.len() as f32 / CLONE_RATE as f32, t.elapsed().as_secs_f32());
    }
    let src = crate::audio::decode(&std::env::temp_dir().join("nekotone-notes-speech.wav")).unwrap();
    let t = Instant::now();
    let y = s.convert(&src.samples, src.sample_rate, &print).unwrap();
    println!("convert {:.1} s in {:.2} s", y.len() as f32 / CLONE_RATE as f32, t.elapsed().as_secs_f32());
}


/// A local engine that takes `0` ms per call.
#[cfg(feature = "server")]
struct SlowLocal(u64);

#[cfg(feature = "server")]
impl Synth for SlowLocal {
    fn rate(&self) -> u32 {
        CLONE_RATE
    }
    fn plan(&self, _t: &str, _v: &str, _f: usize, _a: crate::tts::accent::Accent) -> Result<Vec<crate::tts::Utterance>> {
        unreachable!()
    }
    fn render(&self, _p: &crate::tts::Utterance, _v: &str, _s: f32) -> Result<Vec<f32>> {
        std::thread::sleep(Duration::from_millis(self.0));
        Ok(vec![1.0; 10])
    }
    fn convert(&self, _s: &[f32], _r: u32, _v: &str) -> Result<Vec<f32>> {
        std::thread::sleep(Duration::from_millis(self.0));
        Ok(vec![1.0; 10])
    }
}

#[cfg(feature = "server")]
#[test]
fn both_rails_run_together_and_the_quicker_side_speaks() {
    let tmp = tempfile::tempdir().unwrap();
    let prints = tmp.path().join("voice-clone");
    a_print(&prints, "mine");
    let (port, engine, _t) = start(None);
    let url = format!("127.0.0.1:{port}");
    let make = |local_ms: u64| {
        let server = Arc::new(ComputeServer::new(&url, None).unwrap());
        server.info().unwrap();
        RemoteClone::new(server, prints.clone(), move || Ok(Arc::new(SlowLocal(local_ms)) as Arc<dyn Synth>)).with_rails(Rails::Both)
    };
    // the server is quick, this PC slow: the server's answer, without waiting for this PC
    let rc = make(1500);
    let t0 = Instant::now();
    assert_eq!(rc.convert(&[0.1; 16_000], 16_000, "clone:mine").unwrap().len(), 24_000, "the server's audio");
    assert!(t0.elapsed() < Duration::from_millis(1000), "{:?}", t0.elapsed());
    assert_eq!(rc.wins(), (1, 0));
    // the server is busy, this PC quick: this PC's answer at once (no budget to wait out)
    engine.delay_ms.store(2000, Ordering::Relaxed);
    let rc = make(50);
    let t0 = Instant::now();
    assert_eq!(rc.convert(&[0.1; 16_000], 16_000, "clone:mine").unwrap(), vec![1.0; 10]);
    assert!(t0.elapsed() < Duration::from_millis(500), "{:?}", t0.elapsed());
    assert_eq!(rc.wins(), (0, 1));
    assert!(rc.device().contains("this PC 1"), "{}", rc.device());
    // the server gone: this PC
    let gone = Arc::new(ComputeServer::new("127.0.0.1:9", None).unwrap());
    let rc = RemoteClone::new(gone, prints.clone(), || Ok(Arc::new(SlowLocal(10)) as Arc<dyn Synth>)).with_rails(Rails::Both);
    assert_eq!(rc.convert(&[0.1; 1600], 16_000, "clone:mine").unwrap(), vec![1.0; 10]);
    assert_eq!(Rails::from_name("server-first"), Rails::ServerFirst);
    assert_eq!(Rails::from_name("both"), Rails::Both);
    assert_eq!(Rails::from_name(""), Rails::Auto);
}


#[cfg(feature = "server")]
#[test]
fn auto_settles_on_the_quicker_side_and_checks_back() {
    let tmp = tempfile::tempdir().unwrap();
    let prints = tmp.path().join("voice-clone");
    a_print(&prints, "mine");
    let (port, engine, _t) = start(None);
    let server = Arc::new(ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap());
    server.info().unwrap();
    // this PC is slow (300 ms), the server quick: both for 6 phrases, then the server alone
    let locals = Arc::new(AtomicUsize::new(0));
    let l2 = locals.clone();
    let rc = RemoteClone::new(server.clone(), prints.clone(), move || {
        l2.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(SlowLocal(300)) as Arc<dyn Synth>)
    })
    .with_rails(Rails::Auto);
    let x = [0.1f32; 16_000];
    for _ in 0..ADAPT_STREAK {
        rc.convert(&x, 16_000, "clone:mine").unwrap();
    }
    assert!(rc.device().contains("now"), "settled on the server: {}", rc.device());
    let t0 = Instant::now();
    for _ in 0..3 {
        rc.convert(&x, 16_000, "clone:mine").unwrap();
    }
    assert!(t0.elapsed() < Duration::from_millis(600), "the server alone answers quickly: {:?}", t0.elapsed());
    // the server becomes busy: the guard hands the phrase to this PC and Auto goes back to both
    engine.delay_ms.store(2000, Ordering::Relaxed);
    assert_eq!(rc.convert(&x, 16_000, "clone:mine").unwrap(), vec![1.0; 10]);
    assert!(rc.device().contains("(both)"), "back to both after a late answer: {}", rc.device());
    // still busy: this PC wins 6 in a row and works alone
    for _ in 0..ADAPT_STREAK {
        rc.convert(&x, 16_000, "clone:mine").unwrap();
    }
    assert!(rc.device().contains("this PC now"), "{}", rc.device());
    // the server is quick again: the next probe finds it and both run again
    engine.delay_ms.store(0, Ordering::Relaxed);
    for _ in 0..PROBE_LOCAL_EVERY {
        rc.convert(&x, 16_000, "clone:mine").unwrap();
    }
    assert!(rc.device().contains("(both)"), "the probe found the server again: {}", rc.device());
}


#[cfg(feature = "server")]
#[test]
fn a_server_can_be_stopped() {
    let engine = Arc::new(FakeEngine { converts: AtomicUsize::new(0), synths: AtomicUsize::new(0), delay_ms: AtomicUsize::new(0) });
    let dir = tempfile::tempdir().unwrap();
    let l = bind("127.0.0.1:0").unwrap();
    let (port, stop) = (l.port(), l.stopper());
    let d = dir.path().join("prints");
    let h = std::thread::spawn(move || l.run(engine, &d, None, 3, Some("box".into())));
    let s = ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap();
    assert_eq!(s.info().unwrap().host, "box", "the name given is shown");
    stop.stop();
    let t0 = Instant::now();
    h.join().unwrap().unwrap();
    assert!(t0.elapsed() < Duration::from_secs(2), "run returns when stopped");
    assert!(ComputeServer::new(&format!("127.0.0.1:{port}"), None).unwrap().info().is_err(), "nothing answers after");
}
