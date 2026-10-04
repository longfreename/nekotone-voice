//! Compute server: heavy models on another machine (a GPU box on the
//! network, e.g. a DGX Spark running `nekotone serve`), with this PC as the
//! fallback.
//!
//! Optional everywhere: nothing needs a server, and every remote call that
//! fails (unreachable, timeout, an error answer) is retried locally by the
//! caller; after a failure the server is skipped for [`RETRY_AFTER`] so a
//! machine that went away costs one timeout, not one per sentence.
//!
//! Protocol (HTTP/1.1, plain; meant for a home network or a VPN):
//! - `GET /v1/info` → JSON [`ServerInfo`].
//! - `PUT /v1/prints/<sha256>` with a voice print file (`.nkvoice`) as the
//!   body; the server keeps it under that hash. Prints are uploaded once,
//!   on first use (the server answers 404 to an unknown hash).
//! - `POST /v1/convert?print=<sha256>&rate=<hz>`: body = mono f32 LE
//!   samples at `rate`; answer = the same speech in the print's voice, f32
//!   LE at 24 kHz (Change my voice).
//! - `POST /v1/synth?print=<sha256>`: body = one piece of text (UTF-8);
//!   answer = f32 LE at 24 kHz (your voice reading typed text).
//!
//! With a token set on the server, every request needs
//! `Authorization: Bearer <token>`.

use crate::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// After a failed call the server is not tried again for this long.
pub const RETRY_AFTER: Duration = Duration::from_secs(30);
/// A call slower than this many times the speech it made (and slower than
/// [`SLOW_FLOOR_SECS`]) means the server is busy: this PC is used for
/// [`RETRY_AFTER`]. Measured: a busy shared GB10 took 5x real time; this PC
/// (RTX 4070 Ti) makes speech at about 0.7x real time.
pub const SLOW_FACTOR: f32 = 1.0;
pub const SLOW_FLOOR_SECS: f32 = 1.5;
/// The port `nekotone serve` listens on unless told otherwise.
pub const DEFAULT_PORT: u16 = 8199;
/// Sample rate of the voice-clone answers.
pub const CLONE_RATE: u32 = 24_000;

/// What a server offers (`GET /v1/info`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Always "nekotone".
    pub server: String,
    pub version: String,
    /// Host name of the machine.
    pub host: String,
    /// Where the voice clone runs there ("cuda", "cpu"…).
    pub backend: String,
    /// "convert", "synth".
    pub features: Vec<String>,
}

/// A compute server this PC may use.
pub struct ComputeServer {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
    /// Print hashes the server is known to have.
    uploaded: Mutex<HashSet<String>>,
    down_until: Mutex<Option<Instant>>,
    info: Mutex<Option<ServerInfo>>,
}

/// `host`, `host:port` or a URL → `http://host:port`.
pub fn normalise_url(s: &str) -> Option<String> {
    let s = s.trim();
    let (scheme, rest) = s.split_once("://").unwrap_or(("http", s));
    let rest = rest.trim_end_matches('/');
    if rest.is_empty() || scheme.is_empty() {
        return None;
    }
    let has_port = rest.rsplit_once(':').map(|(_, p)| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())).unwrap_or(false);
    Some(if has_port { format!("{scheme}://{rest}") } else { format!("{scheme}://{rest}:{DEFAULT_PORT}") })
}

impl ComputeServer {
    /// `url`: anything [`normalise_url`] accepts.
    pub fn new(url: &str, token: Option<String>) -> Result<ComputeServer> {
        let base = normalise_url(url).ok_or_else(|| Error::Other(anyhow::anyhow!("\"{url}\" is not a server address (a host name, or host:port; default port {DEFAULT_PORT})")))?;
        let cfg = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(2)))
            .timeout_global(Some(Duration::from_secs(60)))
            .user_agent(concat!("nekotone/", env!("CARGO_PKG_VERSION")))
            .build();
        Ok(ComputeServer {
            base,
            token: token.filter(|t| !t.trim().is_empty()),
            agent: ureq::Agent::new_with_config(cfg),
            uploaded: Mutex::new(HashSet::new()),
            down_until: Mutex::new(None),
            info: Mutex::new(None),
        })
    }

    pub fn url(&self) -> &str {
        &self.base
    }

    /// False for [`RETRY_AFTER`] after a failed call.
    pub fn usable(&self) -> bool {
        self.down_until.lock().map(|t| Instant::now() >= t).unwrap_or(true)
    }

    /// A call that worked but took longer than the speech it made: the
    /// server is busy (a shared GPU), and this PC is quicker for now.
    fn check_speed(&self, audio: &[f32], took: Duration) {
        let secs = audio.len() as f32 / CLONE_RATE as f32;
        if took.as_secs_f32() > SLOW_FLOOR_SECS.max(secs * SLOW_FACTOR) {
            log::warn!("compute server {}: {:.1} s of speech took {:.1} s (busy?); using this PC for {} s", self.base, secs, took.as_secs_f32(), RETRY_AFTER.as_secs());
            *self.down_until.lock() = Some(Instant::now() + RETRY_AFTER);
        }
    }

    /// Skip the server for [`RETRY_AFTER`] (it was too slow this time).
    pub fn set_aside(&self) {
        *self.down_until.lock() = Some(Instant::now() + RETRY_AFTER);
    }

    fn failed(&self, e: &Error) {
        log::warn!("compute server {}: {e}; using this PC for {} s", self.base, RETRY_AFTER.as_secs());
        *self.down_until.lock() = Some(Instant::now() + RETRY_AFTER);
    }

    fn auth<B>(&self, r: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(t) => r.header("Authorization", &format!("Bearer {t}")),
            None => r,
        }
    }

    fn net(&self, what: &str, e: ureq::Error) -> Error {
        match e {
            ureq::Error::StatusCode(401) => Error::Other(anyhow::anyhow!("the compute server {} refused the token", self.base)),
            ureq::Error::StatusCode(c) => Error::Other(anyhow::anyhow!("the compute server {} answered {c} to {what}", self.base)),
            ureq::Error::Timeout(_) => Error::Other(anyhow::anyhow!("the compute server {} did not answer {what} in time", self.base)),
            other => Error::Other(anyhow::anyhow!("could not reach the compute server {} ({other})", self.base)),
        }
    }

    /// What the server offers (2 s timeout); remembered.
    pub fn info(&self) -> Result<ServerInfo> {
        let r = (|| {
            let mut resp = self.auth(self.agent.get(&format!("{}/v1/info", self.base))).config().timeout_global(Some(Duration::from_secs(3))).build().call().map_err(|e| self.net("info", e))?;
            let info: ServerInfo = resp.body_mut().read_json().map_err(|e| Error::Other(anyhow::anyhow!("the compute server {} sent a bad answer ({e})", self.base)))?;
            if info.server != "nekotone" {
                return Err(Error::Other(anyhow::anyhow!("{} is not a Nekotone server", self.base)));
            }
            Ok(info)
        })();
        match &r {
            Ok(i) => {
                *self.info.lock() = Some(i.clone());
                *self.down_until.lock() = None;
            }
            Err(e) => self.failed(e),
        }
        r
    }

    /// The last [`ComputeServer::info`] that worked.
    pub fn known(&self) -> Option<ServerInfo> {
        self.info.lock().clone()
    }

    /// Make sure the server has this print; its hash.
    fn ensure_print(&self, print: &Path) -> Result<String> {
        let bytes = std::fs::read(print).map_err(|e| Error::Other(anyhow::anyhow!("could not read the voice print {} ({e})", print.display())))?;
        let hash = sha256(&bytes);
        if self.uploaded.lock().contains(&hash) {
            return Ok(hash);
        }
        self.auth(self.agent.put(&format!("{}/v1/prints/{hash}", self.base))).send(&bytes[..]).map_err(|e| self.net("the voice upload", e))?;
        self.uploaded.lock().insert(hash.clone());
        Ok(hash)
    }

    /// POST f32 or text, get f32 back; a 404 (the server lost the print)
    /// uploads it again once.
    fn audio_call(&self, path_query: impl Fn(&str) -> String, print: &Path, body: &[u8], what: &str) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let r = (|| {
            for attempt in 0..2 {
                let hash = self.ensure_print(print)?;
                let resp = self.auth(self.agent.post(&format!("{}{}", self.base, path_query(&hash)))).send(body);
                match resp {
                    Ok(mut resp) => {
                        let bytes = resp.body_mut().with_config().limit(512 * 1024 * 1024).read_to_vec().map_err(|e| self.net(what, e))?;
                        return Ok(from_le(&bytes));
                    }
                    Err(ureq::Error::StatusCode(404)) if attempt == 0 => {
                        self.uploaded.lock().remove(&hash);
                    }
                    Err(e) => return Err(self.net(what, e)),
                }
            }
            Err(Error::Other(anyhow::anyhow!("the compute server {} keeps losing the voice", self.base)))
        })();
        match &r {
            Ok(a) => self.check_speed(a, t0.elapsed()),
            Err(e) => self.failed(e),
        }
        r
    }

    /// Change my voice on the server: `samples` (mono, `rate`) in the
    /// print's voice, at [`CLONE_RATE`].
    pub fn convert(&self, samples: &[f32], rate: u32, print: &Path) -> Result<Vec<f32>> {
        self.audio_call(|h| format!("/v1/convert?print={h}&rate={rate}"), print, &to_le(samples), "the voice change")
    }

    /// One piece of text in the print's voice, at [`CLONE_RATE`].
    pub fn synth(&self, text: &str, print: &Path) -> Result<Vec<f32>> {
        self.audio_call(|h| format!("/v1/synth?print={h}"), print, text.as_bytes(), "the speech")
    }
}

pub(crate) fn sha256(b: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(b))
}

pub(crate) fn to_le(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(crate) fn from_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

// ───────────────────────── the voice clone, remote first ─────────────────────────

/// Loads the clone engine on this PC.
type LocalLoader = Arc<dyn Fn() -> Result<Arc<dyn crate::voice::speak::Synth>> + Send + Sync>;

/// How the compute server and this PC share the voice clone's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Rails {
    /// Adapts (the default): both sides at first; when one side wins
    /// [`ADAPT_STREAK`] phrases in a row it alone does the work (the server
    /// still with this PC as a guard when it is late), with a phrase on both
    /// sides now and then to check; a late or failed answer goes back to both.
    #[default]
    Auto,
    /// Both start every phrase; the first answer is used.
    Both,
    /// The server first; this PC only when it is late or fails.
    ServerFirst,
}

impl Rails {
    /// "auto" | "both" | "server-first" (anything else: Auto).
    pub fn from_name(s: &str) -> Rails {
        match s.trim().to_ascii_lowercase().as_str() {
            "server-first" => Rails::ServerFirst,
            "both" => Rails::Both,
            _ => Rails::Auto,
        }
    }
}

/// Wins in a row by one side before Auto lets it work alone.
pub const ADAPT_STREAK: u32 = 6;
/// Auto with the server alone: one phrase on both sides every this many.
pub const PROBE_SERVER_EVERY: u32 = 12;
/// Auto with this PC alone: one phrase on both sides every this many.
pub const PROBE_LOCAL_EVERY: u32 = 8;

/// Where Auto sends the next phrase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    Both,
    Server,
    Local,
}

#[derive(Debug)]
struct Adapt {
    lane: Lane,
    /// (side won by the server?, wins in a row)
    streak: (bool, u32),
    since_probe: u32,
}

impl Default for Adapt {
    fn default() -> Self {
        Adapt { lane: Lane::Both, streak: (true, 0), since_probe: 0 }
    }
}

/// Your cloned voices (and Change my voice) on the compute server and this
/// PC: both at once ([`Rails::Auto`]), or this PC racing the server only
/// when it is late ([`Rails::ServerFirst`]).
///
/// Server first:
/// Each call goes to the server first. If no answer comes within a budget
/// (about half the length of the speech), this PC does the same work and
/// whichever finishes first is used; the server is then set aside for
/// [`RETRY_AFTER`]. So a server that is busy with other work (a shared GPU
/// box: 5x slower than real time measured) never holds the voice back, and a
/// fast one (0.07x) answers before the race starts. [`RemoteClone::preload`]
/// loads this PC's engine in the background so taking over never waits for
/// a model to load (8 s measured).
pub struct RemoteClone {
    server: Arc<ComputeServer>,
    prints_dir: PathBuf,
    loader: LocalLoader,
    loaded: Arc<Mutex<Option<Arc<dyn crate::voice::speak::Synth>>>>,
    loading: Arc<Mutex<()>>,
    rails: Rails,
    adapt: Mutex<Adapt>,
    /// Answers used from the server and from this PC (for the status line).
    wins: Arc<(std::sync::atomic::AtomicUsize, std::sync::atomic::AtomicUsize)>,
}

/// This PC's engine, loaded once (several threads may ask at once).
fn load_local(loader: &LocalLoader, loaded: &Mutex<Option<Arc<dyn crate::voice::speak::Synth>>>, loading: &Mutex<()>) -> Result<Arc<dyn crate::voice::speak::Synth>> {
    let _g = loading.lock();
    if let Some(s) = loaded.lock().clone() {
        return Ok(s);
    }
    let s = loader()?;
    *loaded.lock() = Some(s.clone());
    Ok(s)
}

impl RemoteClone {
    /// `local` loads the clone engine on this PC (called at most once).
    pub fn new(server: Arc<ComputeServer>, prints_dir: PathBuf, local: impl Fn() -> Result<Arc<dyn crate::voice::speak::Synth>> + Send + Sync + 'static) -> RemoteClone {
        RemoteClone { server, prints_dir, loader: Arc::new(local), loaded: Arc::new(Mutex::new(None)), loading: Arc::new(Mutex::new(())), rails: Rails::ServerFirst, adapt: Mutex::new(Adapt::default()), wins: Default::default() }
    }

    /// How the two sides share the work (default here: server first).
    pub fn with_rails(mut self, rails: Rails) -> RemoteClone {
        self.rails = rails;
        self
    }

    /// Answers used so far: (server, this PC).
    pub fn wins(&self) -> (usize, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.wins.0.load(Relaxed), self.wins.1.load(Relaxed))
    }

    fn won(&self, server: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        if server { &self.wins.0 } else { &self.wins.1 }.fetch_add(1, Relaxed);
    }

    /// Auto: which lane this phrase takes (a probe on both now and then).
    fn lane(&self) -> Lane {
        let mut a = self.adapt.lock();
        a.since_probe += 1;
        match a.lane {
            Lane::Server if a.since_probe >= PROBE_SERVER_EVERY => Lane::Both,
            Lane::Local if a.since_probe >= PROBE_LOCAL_EVERY => Lane::Both,
            l => l,
        }
    }

    /// Auto: learn from how a phrase went. `took`: the lane used; `server`:
    /// who answered; `clean`: no failure and no late answer.
    fn learn(&self, took: Lane, server: bool, clean: bool) {
        let mut a = self.adapt.lock();
        let before = a.lane;
        if !clean {
            a.lane = Lane::Both;
            a.streak = (server, 0);
        } else if took == Lane::Both {
            a.since_probe = 0;
            a.streak = if a.streak.0 == server { (server, a.streak.1 + 1) } else { (server, 1) };
            match a.lane {
                // a probe the lone side lost: both again
                Lane::Server if !server => a.lane = Lane::Both,
                Lane::Local if server => a.lane = Lane::Both,
                Lane::Both if a.streak.1 >= ADAPT_STREAK => a.lane = if server { Lane::Server } else { Lane::Local },
                _ => {}
            }
        }
        if a.lane != before {
            log::warn!("voice clone: {} from now on (auto)", match a.lane { Lane::Both => "the compute server and this PC together", Lane::Server => "the compute server", Lane::Local => "this PC" });
        }
    }

    /// One phrase the Auto way.
    fn adaptive(&self, remote: impl FnOnce(&ComputeServer) -> Result<Vec<f32>> + Send + 'static, local: impl FnOnce(&dyn crate::voice::speak::Synth) -> Result<Vec<f32>> + Send + 'static, budget: Duration) -> Result<Vec<f32>> {
        use std::sync::atomic::Ordering::Relaxed;
        let lane = self.lane();
        let before = (self.wins.0.load(Relaxed), self.wins.1.load(Relaxed));
        let r = match lane {
            Lane::Both => self.both(remote, local),
            Lane::Server => self.race(budget, remote, local),
            Lane::Local => {
                let r = self.local().and_then(|s| local(&*s));
                if r.is_ok() {
                    self.won(false);
                }
                r
            }
        };
        let server = self.wins.0.load(Relaxed) > before.0;
        // the server lane is clean when the server itself answered in time
        let clean = r.is_ok() && (lane != Lane::Server || server);
        self.learn(lane, server, clean);
        r
    }

    /// Load this PC's engine now, on another thread.
    pub fn preload(&self) {
        let (loader, loaded, loading) = (self.loader.clone(), self.loaded.clone(), self.loading.clone());
        std::thread::spawn(move || {
            let _g = loading.lock();
            if loaded.lock().is_none() {
                match loader() {
                    Ok(s) => *loaded.lock() = Some(s),
                    Err(e) => log::warn!("the voice clone could not be loaded on this PC ({e}); only the compute server can speak"),
                }
            }
        });
    }

    fn local(&self) -> Result<Arc<dyn crate::voice::speak::Synth>> {
        load_local(&self.loader, &self.loaded, &self.loading)
    }

    /// The print file for `voice`, when there is one on this PC.
    fn print_file(&self, voice: &str) -> Option<PathBuf> {
        let name = match voice.strip_prefix(crate::tts::chatterbox::CLONE_PREFIX) {
            Some(n) => n.to_string(),
            None => format!("stock-{voice}"),
        };
        let p = crate::tts::chatterbox::print_path(&self.prints_dir, &name);
        p.is_file().then_some(p)
    }

    /// Where it runs, for the status line: "gpubox (cuda)", the local
    /// engine's, or with both "gpubox 12 · this PC 3" (answers used).
    pub fn where_(&self) -> String {
        let local = self.loaded.lock().as_ref().map(|s| s.device()).unwrap_or_else(|| "this PC".into());
        match self.server.known() {
            Some(i) if self.rails != Rails::ServerFirst => {
                let (s, l) = self.wins();
                let now = match (self.rails, self.adapt.lock().lane) {
                    (Rails::Auto, Lane::Server) => format!("{} now", i.host),
                    (Rails::Auto, Lane::Local) => "this PC now".to_string(),
                    _ => "both".to_string(),
                };
                if s + l == 0 { format!("{} ({}) + this PC", i.host, i.backend) } else { format!("{} {s} · this PC {l} ({now})", i.host) }
            }
            Some(i) if self.server.usable() => format!("{} ({})", i.host, i.backend),
            _ => local,
        }
    }

    /// Both sides at once; the first answer that works is used.
    fn both(&self, remote: impl FnOnce(&ComputeServer) -> Result<Vec<f32>> + Send + 'static, local: impl FnOnce(&dyn crate::voice::speak::Synth) -> Result<Vec<f32>> + Send + 'static) -> Result<Vec<f32>> {
        let (tx, rx) = crossbeam_channel::bounded::<(bool, Result<Vec<f32>>)>(2);
        let server = self.server.clone();
        let tx2 = tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send((true, remote(&server)));
        });
        let (loader, loaded, loading) = (self.loader.clone(), self.loaded.clone(), self.loading.clone());
        std::thread::spawn(move || {
            let r = load_local(&loader, &loaded, &loading).and_then(|s| local(&*s));
            let _ = tx2.send((false, r));
        });
        let mut last_err = None;
        for _ in 0..2 {
            match rx.recv_timeout(Duration::from_secs(120)) {
                Ok((from_server, Ok(a))) => {
                    self.won(from_server);
                    return Ok(a);
                }
                Ok((_, Err(e))) => last_err = Some(e),
                Err(_) => break,
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Other(anyhow::anyhow!("neither the compute server nor this PC answered"))))
    }

    /// The server's answer if it comes within `budget`, else this PC's
    /// (or the server's after all, if this PC fails).
    fn race(&self, budget: Duration, remote: impl FnOnce(&ComputeServer) -> Result<Vec<f32>> + Send + 'static, local: impl FnOnce(&dyn crate::voice::speak::Synth) -> Result<Vec<f32>>) -> Result<Vec<f32>> {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let server = self.server.clone();
        std::thread::spawn(move || {
            let _ = tx.send(remote(&server));
        });
        match rx.recv_timeout(budget) {
            Ok(Ok(a)) => {
                self.won(true);
                return Ok(a);
            }
            Ok(Err(_)) => {
                self.won(false);
                return local(&*self.local()?);
            }
            Err(_) => {}
        }
        log::warn!("compute server {}: no answer within {:.1} s; this PC takes over for {} s", self.server.url(), budget.as_secs_f32(), RETRY_AFTER.as_secs());
        self.server.set_aside();
        match self.local().and_then(|s| local(&*s)) {
            Ok(a) => {
                self.won(false);
                Ok(a)
            }
            // this PC failed: wait for the server after all
            Err(e) => rx.recv_timeout(Duration::from_secs(60)).ok().and_then(|r| r.ok()).ok_or(e),
        }
    }
}

/// How long the server may take before this PC starts too: half the speech,
/// at least `floor`.
fn budget(speech_secs: f32, floor: f32) -> Duration {
    Duration::from_secs_f32((0.5 * speech_secs).max(floor))
}

impl crate::voice::speak::Synth for RemoteClone {
    fn rate(&self) -> u32 {
        CLONE_RATE
    }
    fn plan(&self, text: &str, _voice: &str, _first_max: usize, _accent: crate::tts::accent::Accent) -> Result<Vec<crate::tts::Utterance>> {
        Ok(crate::tts::chatterbox::plan(text, crate::tts::chatterbox::FIRST_PIECE_CHARS))
    }
    fn render(&self, piece: &crate::tts::Utterance, voice: &str, speed: f32) -> Result<Vec<f32>> {
        if let (Rails::Auto | Rails::Both, Some(p)) = (self.rails, self.print_file(voice)) {
            let (text, piece2, voice2) = (piece.phonemes.clone(), piece.clone(), voice.to_string());
            let b = budget(text.chars().count() as f32 / 15.0, 1.0);
            let (remote, local) = (move |s: &ComputeServer| s.synth(&text, &p), move |l: &dyn crate::voice::speak::Synth| l.render(&piece2, &voice2, speed));
            return if self.rails == Rails::Both { self.both(remote, local) } else { self.adaptive(remote, local, b) };
        }
        if let (true, Some(p)) = (self.server.usable(), self.print_file(voice)) {
            let text = piece.phonemes.clone();
            // about 15 characters a second of speech
            let b = budget(text.chars().count() as f32 / 15.0, 1.0);
            return self.race(b, move |s| s.synth(&text, &p), |l| l.render(piece, voice, speed));
        }
        self.local()?.render(piece, voice, speed)
    }
    fn device(&self) -> String {
        self.where_()
    }
    fn convert(&self, samples: &[f32], rate: u32, voice: &str) -> Result<Vec<f32>> {
        if let (Rails::Auto | Rails::Both, Some(p)) = (self.rails, self.print_file(voice)) {
            let (x, x2, voice2) = (samples.to_vec(), samples.to_vec(), voice.to_string());
            let b = budget(samples.len() as f32 / rate.max(1) as f32, 0.6);
            let (remote, local) = (move |s: &ComputeServer| s.convert(&x, rate, &p), move |l: &dyn crate::voice::speak::Synth| l.convert(&x2, rate, &voice2));
            return if self.rails == Rails::Both { self.both(remote, local) } else { self.adaptive(remote, local, b) };
        }
        if let (true, Some(p)) = (self.server.usable(), self.print_file(voice)) {
            let x = samples.to_vec();
            let b = budget(samples.len() as f32 / rate.max(1) as f32, 0.6);
            return self.race(b, move |s| s.convert(&x, rate, &p), |l| l.convert(samples, rate, voice));
        }
        self.local()?.convert(samples, rate, voice)
    }
    fn knows_voice(&self, voice: &str) -> bool {
        self.print_file(voice).is_some()
    }
    fn learn_voice(&self, voice: &str, reference: &[f32], rate: u32) -> Result<()> {
        // learnt on this PC once (it writes the print the server then gets)
        self.local()?.learn_voice(voice, reference, rate)
    }
}

// ───────────────────────── the server ─────────────────────────

/// What the server does with audio; the voice clone in practice, a fake in tests.
pub trait CloneEngine: Send + Sync {
    /// Where it runs ("cuda", "cpu").
    fn backend(&self) -> String;
    fn convert(&self, samples: &[f32], rate: u32, print: &Path) -> Result<Vec<f32>>;
    fn synth(&self, text: &str, print: &Path) -> Result<Vec<f32>>;
}

/// The voice clone as a [`CloneEngine`]; prints are read once and kept.
pub struct ChatterboxEngine {
    cb: Arc<crate::tts::chatterbox::Chatterbox>,
    prints: Mutex<std::collections::HashMap<PathBuf, Arc<crate::tts::chatterbox::VoicePrint>>>,
    sampling: crate::tts::chatterbox::Sampling,
}

impl ChatterboxEngine {
    pub fn new(cb: Arc<crate::tts::chatterbox::Chatterbox>) -> ChatterboxEngine {
        ChatterboxEngine { cb, prints: Mutex::new(Default::default()), sampling: Default::default() }
    }

    fn print(&self, p: &Path) -> Result<Arc<crate::tts::chatterbox::VoicePrint>> {
        if let Some(v) = self.prints.lock().get(p) {
            return Ok(v.clone());
        }
        let v = Arc::new(crate::tts::chatterbox::VoicePrint::load(p)?);
        self.prints.lock().insert(p.to_path_buf(), v.clone());
        Ok(v)
    }
}

impl CloneEngine for ChatterboxEngine {
    fn backend(&self) -> String {
        // the graphics side when any graph runs there (on Windows only the
        // decoder does: "cpu" alone undersold it)
        let e = self.cb.engines();
        [e.language_model, e.decoder].into_iter().find(|b| b != "cpu" && !b.is_empty()).unwrap_or_else(|| "cpu".into())
    }
    fn convert(&self, samples: &[f32], rate: u32, print: &Path) -> Result<Vec<f32>> {
        self.cb.convert(samples, rate, &*self.print(print)?)
    }
    fn synth(&self, text: &str, print: &Path) -> Result<Vec<f32>> {
        self.cb.synthesize_piece(text, &*self.print(print)?, &self.sampling, &|| false)
    }
}

/// A bound server socket (`bind`, then `run`).
#[cfg(feature = "server")]
pub struct Listener {
    server: Arc<tiny_http::Server>,
}

/// Ends a running server: [`Listener::run`] returns once its requests are done.
#[cfg(feature = "server")]
#[derive(Clone)]
pub struct Stopper(Arc<tiny_http::Server>);

#[cfg(feature = "server")]
impl Stopper {
    pub fn stop(&self) {
        // one wake-up per worker (each waits in incoming_requests)
        for _ in 0..64 {
            self.0.unblock();
        }
    }
}

/// This machine's address on the local network (the one a route to the
/// internet would use; nothing is sent), for "connect to …" hints.
pub fn lan_address() -> Option<std::net::IpAddr> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    s.local_addr().ok().map(|a| a.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified())
}

/// Listen on `addr` ("0.0.0.0:8199"; port 0 picks a free one).
#[cfg(feature = "server")]
pub fn bind(addr: &str) -> Result<Listener> {
    let server = tiny_http::Server::http(addr).map_err(|e| Error::Other(anyhow::anyhow!("could not listen on {addr} ({e})")))?;
    Ok(Listener { server: Arc::new(server) })
}

#[cfg(feature = "server")]
impl Listener {
    pub fn port(&self) -> u16 {
        self.server.server_addr().to_ip().map(|a| a.port()).unwrap_or(0)
    }

    /// Stops [`Listener::run`] from another thread.
    pub fn stopper(&self) -> Stopper {
        Stopper(self.server.clone())
    }

    /// Serve until the process ends. `prints_dir` keeps uploaded prints;
    /// `threads` requests run at once (the engine serialises its own models);
    /// `name` is what clients show (default: the host name).
    pub fn run(self, engine: Arc<dyn CloneEngine>, prints_dir: &Path, token: Option<String>, threads: usize, name: Option<String>) -> Result<()> {
        std::fs::create_dir_all(prints_dir).map_err(|e| Error::Other(anyhow::anyhow!("could not create {} ({e})", prints_dir.display())))?;
        let host = name.filter(|n| !n.trim().is_empty()).unwrap_or_else(hostname);
        let mut workers = Vec::new();
        for _ in 0..threads.max(1) {
            let (server, engine, dir, token, host) = (self.server.clone(), engine.clone(), prints_dir.to_path_buf(), token.clone(), host.clone());
            workers.push(std::thread::spawn(move || {
                for req in server.incoming_requests() {
                    handle(req, engine.as_ref(), &dir, token.as_deref(), &host);
                }
            }));
        }
        for w in workers {
            let _ = w.join();
        }
        Ok(())
    }
}

#[cfg(feature = "server")]
fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "server".into())
}

#[cfg(feature = "server")]
fn handle(mut req: tiny_http::Request, engine: &dyn CloneEngine, dir: &Path, token: Option<&str>, host: &str) {
    use std::io::Read;
    let respond = |req: tiny_http::Request, code: u16, body: Vec<u8>, json: bool| {
        let ct = if json { "application/json" } else { "application/octet-stream" };
        let mut r = tiny_http::Response::from_data(body).with_status_code(code);
        if let Ok(h) = tiny_http::Header::from_bytes(&b"Content-Type"[..], ct.as_bytes()) {
            r = r.with_header(h);
        }
        let _ = req.respond(r);
    };
    let text = |req: tiny_http::Request, code: u16, msg: &str| respond(req, code, msg.as_bytes().to_vec(), false);
    if let Some(t) = token {
        let ok = req.headers().iter().any(|h| h.field.equiv("Authorization") && h.value.as_str() == format!("Bearer {t}"));
        if !ok {
            return text(req, 401, "a token is needed");
        }
    }
    let url = req.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    let arg = |k: &str| query.split('&').find_map(|kv| kv.split_once('=').filter(|(a, _)| *a == k).map(|(_, v)| v.to_string()));
    let hash_ok = |h: &str| h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit());
    let mut body = Vec::new();
    if req.as_reader().take(512 * 1024 * 1024).read_to_end(&mut body).is_err() {
        return text(req, 400, "could not read the request");
    }
    let t0 = Instant::now();
    match (req.method().as_str(), path) {
        ("GET", "/v1/info") => {
            let info = ServerInfo { server: "nekotone".into(), version: env!("CARGO_PKG_VERSION").into(), host: host.into(), backend: engine.backend(), features: vec!["convert".into(), "synth".into()] };
            respond(req, 200, serde_json::to_vec(&info).unwrap_or_default(), true)
        }
        ("PUT", p) if p.starts_with("/v1/prints/") => {
            let h = &p["/v1/prints/".len()..];
            if !hash_ok(h) || sha256(&body) != h.to_ascii_lowercase() {
                return text(req, 400, "the print does not match its hash");
            }
            match std::fs::write(dir.join(format!("{h}.nkvoice")), &body) {
                Ok(()) => text(req, 200, "ok"),
                Err(e) => text(req, 500, &format!("could not keep the print ({e})")),
            }
        }
        ("POST", "/v1/convert") | ("POST", "/v1/synth") => {
            let Some(h) = arg("print").filter(|h| hash_ok(h)) else { return text(req, 400, "print=<sha256> is needed") };
            let print = dir.join(format!("{}.nkvoice", h.to_ascii_lowercase()));
            if !print.is_file() {
                return text(req, 404, "unknown print: upload it first");
            }
            let r = if path == "/v1/convert" {
                let rate: u32 = arg("rate").and_then(|r| r.parse().ok()).unwrap_or(16_000);
                engine.convert(&from_le(&body), rate, &print)
            } else {
                match String::from_utf8(body) {
                    Ok(s) => engine.synth(&s, &print),
                    Err(_) => return text(req, 400, "the text is not UTF-8"),
                }
            };
            match r {
                Ok(a) => {
                    log::info!("{path}: {:.2} s of audio in {:.0} ms", a.len() as f32 / CLONE_RATE as f32, t0.elapsed().as_secs_f32() * 1000.0);
                    respond(req, 200, to_le(&a), false)
                }
                Err(e) => text(req, 500, &e.to_string()),
            }
        }
        _ => text(req, 404, "no such endpoint"),
    }
}

#[cfg(test)]
#[path = "remote_tests.rs"]
mod tests;
