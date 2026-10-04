//! "Speak for me": your words, in another voice, and never your voice
//! (owner: the speak package).
//!
//! ```text
//! mic ─ capture ─ mono ─ 16 kHz ─ VAD (utterances, 400 ms hangover) ─▶ Whisper (transcribe or translate → English)
//!                                                                     ─▶ clean-up (fillers, bracketed noises, optional profanity mask)
//!                                                                     ─▶ queue ◀─ say(text) (type to speak)
//! queue ─▶ Kokoro, sentence by sentence (the first sentence of a long reply is cut at a clause so it starts sooner)
//!       ─▶ 24 kHz → output rate (band-limited sinc) ─▶ optional Voice Studio preset (Cathedral, Radio, Dragon…)
//!       ─▶ gain ─▶ lookahead limiter ─┬─ ring ─ output (virtual cable / speakers)
//!                                     └─ ring ─ monitor (headphones)
//! ```
//!
//! Guarantee: **the output only ever carries synthesised speech or
//! silence.** The microphone feeds nothing but the voice-activity detector
//! and the transcriber; the output rings are written only by the synthesis
//! thread (`speak_loop`). There is no code path from a captured sample to
//! an output sample (`mic_never_reaches_the_output` checks it with fake
//! devices).
//!
//! Latency (end of your sentence → first synthetic sample) = VAD hangover
//! (`hangover_ms`, default 275) + Whisper on the utterance + Kokoro on the
//! first chunk + the output buffer. Everything after the first chunk is
//! synthesised while earlier audio plays, so long replies do not add to it.
//! `Speaking.latency_ms` reports the measured value for every utterance.
//!
//! Voices: Kokoro (`tts::Tts`) or your own cloned voice (Chatterbox,
//! [`CloneSynth`], voice ids `clone:<name>`); [`VoiceRouter`] holds both so
//! the voice can change while on air. Chatterbox adds its generation time
//! per sentence (see HANDOFF §5h for the measured first-audio latency).
//!
//! Split for testing: [`SpeakCore`] is the device-free pipeline (any
//! [`CaptureSource`] in, [`OutputTap`]s out, any [`Synth`] and
//! [`Transcriber`]); [`SpeakEngine`] puts it on real devices (cpal).

use super::chain::{FrontEnd, Processor};
use super::dsp::dynamics::GateParams;
use super::dsp::lin_to_db;
use super::presets::Preset;
use super::ring::{ring, Consumer, Producer};
use crate::record::align::Resampler;
use crate::record::capture::{CaptureSink, CaptureSource, Chunk, RealClock};
use crate::record::vad::{Vad, VadConfig, RATE as VAD_RATE};
use crate::stt::{TranscribeOptions, Transcriber};
use crate::tts::accent::Accent;
use crate::tts::Utterance as Piece;
use crate::{Error, Result};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ───────────────────────── configuration, events ─────────────────────────

/// How to run Speak for me.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeakConfig {
    /// Microphone by name (`None` = system default).
    pub input: Option<String>,
    /// Output device by name (`None` = system default). Ignored when `virtual_mic` is on.
    pub output: Option<String>,
    /// Send the voice to the detected virtual cable (the same rule as the voice changer).
    pub virtual_mic: bool,
    /// Also play the voice here (headphones). `Some("")` = system default output.
    pub monitor: Option<String>,
    /// Kokoro voice id, e.g. "af_heart".
    pub voice: String,
    /// 0.5–2.0 (1 = natural pace).
    pub speed: f32,
    /// A Voice Studio preset applied to the synthetic voice (None = clean).
    pub effect: Option<Preset>,
    /// The accent the voice speaks in (any voice can speak any accent).
    pub accent: Accent,
    /// Listen to the microphone (false = type-to-speak only).
    pub listen: bool,
    /// Whisper `translate`: any language in, English out.
    pub translate: bool,
    /// Language you speak (ISO 639-1); None = detect each utterance.
    pub language: Option<String>,
    /// Names and words to expect (Whisper's prompt).
    pub prompt: Option<String>,
    /// Drop "um", "uh", "erm"… before speaking.
    pub trim_fillers: bool,
    /// Replace common swear words with "bleep".
    pub mask_profanity: bool,
    /// Ignore the microphone while the voice plays (speakers instead of headphones: stops it hearing itself).
    pub half_duplex: bool,
    /// Silence that ends an utterance (ms, 250–1000); with `auto_pause`
    /// the start value.
    pub hangover_ms: u32,
    /// Learn the end-of-sentence pause from how you speak (the 90th
    /// percentile of your recent pauses, 250-900 ms). Measured on the owner's
    /// recordings at three speaking paces (record::vad tests,
    /// learnt_hangover_on_real_speech): ~280 ms when talking quickly, 500-680
    /// ms when slow; overall 29 % early cuts at a 473 ms mean wait, against
    /// 27 % at 525 ms and 39 % at 400 ms fixed.
    pub auto_pause: bool,
    pub output_gain_db: f32,
    pub monitor_gain_db: f32,
    /// Start with the output muted.
    pub start_muted: bool,
    /// Change my voice (neural): each utterance is spoken again in the chosen
    /// voice with your words, timing and delivery (voice conversion, no
    /// speech-to-text). Needs the voice-clone model.
    pub convert: bool,
    /// Change my voice: leave the pause you left before each phrase (up to
    /// 2 s) instead of speaking it as soon as it is converted.
    pub keep_pauses: bool,
}

impl Default for SpeakConfig {
    fn default() -> Self {
        SpeakConfig {
            input: None,
            output: None,
            virtual_mic: false,
            monitor: None,
            voice: crate::tts::DEFAULT_VOICE.into(),
            speed: 1.0,
            effect: None,
            accent: Accent::Voice,
            listen: true,
            translate: false,
            language: None,
            prompt: None,
            trim_fillers: true,
            mask_profanity: false,
            half_duplex: false,
            // measured on the owner's speech (123 pauses over five recordings):
            // 70 % of pauses are under 250 ms (inside a phrase), 17 % 250-400
            // (clause breaks), 11 % over 500 (sentence ends); 275 ms answers at
            // clause and sentence breaks without cutting a phrase. Was 400.
            hangover_ms: 275,
            auto_pause: true,
            output_gain_db: 0.0,
            monitor_gain_db: -6.0,
            start_muted: false,
            convert: false,
            keep_pauses: true,
        }
    }
}

/// Where a line to speak came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Mic,
    Typed,
}

/// Events for the UI (from the pipeline's threads).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SpeakEvent {
    Started { input: Option<String>, output: String, monitor: Option<String>, sample_rate: u32 },
    /// You started talking.
    Listening,
    /// You stopped; Whisper is working on `speech_secs` of audio.
    Transcribing { speech_secs: f32 },
    /// What Whisper heard (`original`) and what will be spoken (`text`, after clean-up).
    Heard { id: u64, text: String, original: String, language: String, stt_ms: f32, speech_secs: f32 },
    /// An utterance produced no words (noise, a cough).
    Dropped { reason: String },
    /// A line joined the queue.
    Queued { id: u64, text: String, source: Source },
    /// A line started playing. `latency_ms`: end of your speech → first
    /// synthetic sample at the output (None for typed lines).
    Speaking { id: u64, text: String, latency_ms: Option<f32>, synth_ms: f32 },
    /// A line finished playing (or was skipped).
    Done { id: u64, skipped: bool },
    /// About ten times a second.
    Level { input_db: f32, output_db: f32, listening: bool, speaking: bool },
    Error { message: String },
    DeviceLost { role: &'static str, message: String },
    /// A lost device reconnected (`name`: the device now used).
    DeviceRestored { role: &'static str, name: String },
    Stopped,
}

/// A snapshot for polling.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SpeakStatus {
    pub running: bool,
    pub muted: bool,
    pub listening_enabled: bool,
    /// You are talking right now.
    pub hearing: bool,
    /// Synthetic speech is playing.
    pub speaking: bool,
    /// Lines waiting (not counting the one playing).
    pub queued: usize,
    pub input_level_db: f32,
    pub output_level_db: f32,
    /// Last measured end-of-speech → first-sample latency (ms, 0 = none yet).
    pub last_latency_ms: f32,
    /// Mean of the measured latencies.
    pub mean_latency_ms: f32,
    pub utterances: u64,
    pub voice: String,
    pub speed: f32,
    pub effect_id: Option<String>,
    /// The accent in use (`voice` = the voice's own).
    pub accent: Accent,
    pub sample_rate: u32,
    pub input_device: Option<String>,
    pub output_device: String,
    pub monitor_device: Option<String>,
    /// Where Kokoro runs ("cpu" / "directml").
    pub tts_device: String,
    pub stt_model: String,
    /// The end-of-sentence pause in use (ms): learnt with auto_pause.
    pub pause_ms: u32,
    /// Pauses it was learnt from (0: still the start value).
    pub pauses_heard: u32,
}

// ───────────────────────── the synthesiser seam ─────────────────────────

/// What the pipeline needs from a TTS engine (Kokoro in the app; tests plug in fakes).
pub trait Synth: Send + Sync {
    /// Output sample rate.
    fn rate(&self) -> u32;
    /// Text → pieces in `accent`; `first_max` bounds the first piece's phonemes.
    fn plan(&self, text: &str, voice: &str, first_max: usize, accent: Accent) -> Result<Vec<Piece>>;
    /// One piece → audio (without its pause).
    fn render(&self, piece: &Piece, voice: &str, speed: f32) -> Result<Vec<f32>>;
    /// "cpu" / "directml" / "fake".
    fn device(&self) -> String {
        "cpu".into()
    }
    /// Change my voice: `samples` (mono, `rate`) spoken again in `voice`
    /// with the same words and timing, at [`Synth::rate`].
    fn convert(&self, _samples: &[f32], _rate: u32, _voice: &str) -> Result<Vec<f32>> {
        Err(Error::Model("Change my voice needs the voice-clone model (tts-chatterbox); download it in Voice Studio → Speak for me".into()))
    }
    /// A print for a stock voice exists already (Change my voice).
    fn knows_voice(&self, _voice: &str) -> bool {
        false
    }
    /// Learn a stock voice for Change my voice from a reference recording.
    fn learn_voice(&self, _voice: &str, _reference: &[f32], _rate: u32) -> Result<()> {
        Err(Error::Model("this engine cannot learn voices".into()))
    }
}

#[cfg(feature = "ml")]
impl Synth for crate::tts::Tts {
    fn rate(&self) -> u32 {
        crate::tts::SAMPLE_RATE
    }
    fn plan(&self, text: &str, voice: &str, first_max: usize, accent: Accent) -> Result<Vec<Piece>> {
        crate::tts::Tts::plan_accent(self, text, voice, first_max, accent)
    }
    fn render(&self, piece: &Piece, voice: &str, speed: f32) -> Result<Vec<f32>> {
        self.synthesize_phonemes(&piece.phonemes, voice, speed)
    }
    fn device(&self) -> String {
        crate::tts::Tts::device(self).into()
    }
}

/// Speak in a cloned voice (Chatterbox): voice ids are `clone:<name>`, and
/// `<name>` is a voice print saved in `dir` (see `tts::chatterbox::print_path`).
/// Prints are read on first use and kept. Speed is not supported by the
/// model (the line is spoken at the reference's pace); accents are ignored.
#[cfg(feature = "ml")]
pub struct CloneSynth {
    engine: Arc<crate::tts::chatterbox::Chatterbox>,
    dir: std::path::PathBuf,
    prints: Mutex<std::collections::HashMap<String, Arc<crate::tts::chatterbox::VoicePrint>>>,
    sampling: crate::tts::chatterbox::Sampling,
}

#[cfg(feature = "ml")]
impl CloneSynth {
    pub fn new(engine: Arc<crate::tts::chatterbox::Chatterbox>, prints_dir: std::path::PathBuf) -> CloneSynth {
        CloneSynth { engine, dir: prints_dir, prints: Mutex::new(Default::default()), sampling: Default::default() }
    }

    /// Use this print for `clone:<name>` (replaces a cached one, e.g. after re-recording).
    pub fn set_print(&self, name: &str, print: crate::tts::chatterbox::VoicePrint) {
        self.prints.lock().insert(name.to_string(), Arc::new(print));
    }

    /// Forget a cached print (it is read from disk again on next use).
    pub fn forget(&self, name: &str) {
        self.prints.lock().remove(name);
    }

    pub fn engine(&self) -> &Arc<crate::tts::chatterbox::Chatterbox> {
        &self.engine
    }

    /// The print name for a voice: `clone:<name>` → `<name>`, a stock voice
    /// (a Kokoro id) → `stock-<id>`.
    fn print_name(voice: &str) -> String {
        match voice.strip_prefix(crate::tts::chatterbox::CLONE_PREFIX) {
            Some(n) => n.to_string(),
            None => format!("stock-{voice}"),
        }
    }

    fn print(&self, voice: &str) -> Result<Arc<crate::tts::chatterbox::VoicePrint>> {
        let owned = Self::print_name(voice);
        let name = owned.as_str();
        if let Some(p) = self.prints.lock().get(name) {
            return Ok(p.clone());
        }
        let path = crate::tts::chatterbox::print_path(&self.dir, name);
        if !path.is_file() {
            return Err(Error::Model("there is no saved voice of yours yet: record a sample and choose Make my voice".into()));
        }
        let p = Arc::new(crate::tts::chatterbox::VoicePrint::load(&path)?);
        self.prints.lock().insert(name.to_string(), p.clone());
        Ok(p)
    }
}

#[cfg(feature = "ml")]
impl Synth for CloneSynth {
    fn rate(&self) -> u32 {
        crate::tts::chatterbox::SAMPLE_RATE
    }
    fn plan(&self, text: &str, _voice: &str, _first_max: usize, _accent: Accent) -> Result<Vec<Piece>> {
        Ok(crate::tts::chatterbox::plan(text, crate::tts::chatterbox::FIRST_PIECE_CHARS))
    }
    fn render(&self, piece: &Piece, voice: &str, _speed: f32) -> Result<Vec<f32>> {
        let print = self.print(voice)?;
        self.engine.synthesize_piece(&piece.phonemes, &print, &self.sampling, &|| false)
    }
    fn device(&self) -> String {
        self.engine.device()
    }
    fn convert(&self, samples: &[f32], rate: u32, voice: &str) -> Result<Vec<f32>> {
        let print = self.print(voice)?;
        // no silence trimming: bleeps are placed by the utterance's timing
        self.engine.convert(samples, rate, &print)
    }
    fn knows_voice(&self, voice: &str) -> bool {
        let name = Self::print_name(voice);
        self.prints.lock().contains_key(&name) || crate::tts::chatterbox::print_path(&self.dir, &name).is_file()
    }
    fn learn_voice(&self, voice: &str, reference: &[f32], rate: u32) -> Result<()> {
        let name = Self::print_name(voice);
        let print = self.engine.encode_voice(reference, rate, voice, "Kokoro reference")?;
        let path = crate::tts::chatterbox::print_path(&self.dir, &name);
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        print.save(&path)?;
        self.prints.lock().insert(name, Arc::new(print));
        Ok(())
    }
}

/// What a stock voice reads once so Change my voice can learn it (~15 s:
/// statements, a question, varied vowels and consonants).
pub const STOCK_REFERENCE_TEXT: &str = "Hello there. I wanted to tell you about my week. On Monday I walked along the river, and the weather was lovely, bright and cool. Later we cooked dinner together and laughed about old stories. What do you think we should do next time?";

/// True for a cloned-voice id (`clone:<name>`).
pub fn is_clone_voice(voice: &str) -> bool {
    voice.starts_with(crate::tts::chatterbox::CLONE_PREFIX)
}

/// Kokoro voices and cloned voices behind one [`Synth`]: each line goes to
/// the engine its voice id belongs to, so switching between a Kokoro voice
/// and "My voice" works while on air. Either side may be absent (its voices
/// then fail with a sentence saying what to download).
pub struct VoiceRouter {
    pub kokoro: Option<Arc<dyn Synth>>,
    pub clone: Option<Arc<dyn Synth>>,
    /// Report the clone engine's device (the line that started is a cloned voice).
    pub clone_first: bool,
}

impl VoiceRouter {
    fn pick(&self, voice: &str) -> Result<&Arc<dyn Synth>> {
        let (side, what) = if is_clone_voice(voice) {
            (&self.clone, "your own voice needs the voice-clone model (tts-chatterbox); download it in Voice Studio → Speak for me")
        } else {
            (&self.kokoro, "the voices are not downloaded yet (tts-kokoro); download them in Voice Studio → Speak for me")
        };
        side.as_ref().ok_or_else(|| Error::Model(what.into()))
    }
}

impl Synth for VoiceRouter {
    fn rate(&self) -> u32 {
        // both engines speak at 24 kHz; the pipeline resamples by `rate()` per call anyway
        self.kokoro.as_ref().or(self.clone.as_ref()).map(|s| s.rate()).unwrap_or(24_000)
    }
    fn plan(&self, text: &str, voice: &str, first_max: usize, accent: Accent) -> Result<Vec<Piece>> {
        self.pick(voice)?.plan(text, voice, first_max, accent)
    }
    fn render(&self, piece: &Piece, voice: &str, speed: f32) -> Result<Vec<f32>> {
        self.pick(voice)?.render(piece, voice, speed)
    }
    fn device(&self) -> String {
        let first = if self.clone_first { self.clone.as_ref().or(self.kokoro.as_ref()) } else { self.kokoro.as_ref().or(self.clone.as_ref()) };
        first.map(|s| s.device()).unwrap_or_else(|| "cpu".into())
    }
    /// Change my voice: the clone engine does the conversion; a Kokoro voice
    /// is learnt once (Kokoro reads [`STOCK_REFERENCE_TEXT`], the print is
    /// kept on disk) and then works like "My voice".
    fn convert(&self, samples: &[f32], rate: u32, voice: &str) -> Result<Vec<f32>> {
        let clone = self.clone.as_ref().ok_or_else(|| Error::Model("Change my voice needs the voice-clone model (tts-chatterbox); download it in Voice Studio → Speak for me".into()))?;
        if !is_clone_voice(voice) && !clone.knows_voice(voice) {
            let kokoro = self.kokoro.as_ref().ok_or_else(|| Error::Model("this voice needs the Speak for me voices (tts-kokoro) once, to learn it".into()))?;
            let mut reference = Vec::new();
            for piece in kokoro.plan(STOCK_REFERENCE_TEXT, voice, usize::MAX, Accent::Voice)? {
                reference.extend(kokoro.render(&piece, voice, 1.0)?);
                reference.extend(std::iter::repeat_n(0.0, (piece.pause_after * kokoro.rate() as f32) as usize));
            }
            clone.learn_voice(voice, &reference, kokoro.rate())?;
        }
        clone.convert(samples, rate, voice)
    }
}

/// Phonemes allowed in the first piece of a line (~3 s of speech, ~0.7 s
/// of Kokoro on the CPU): the voice starts sooner and the rest of a long
/// sentence is synthesised while the first piece plays. Kokoro costs about
/// 0.2 s per call plus ~10 ms per phoneme, so much smaller pieces gain
/// little and break the intonation more often.
pub const FIRST_PIECE_PHONEMES: usize = 60;

// ───────────────────────── text clean-up ─────────────────────────

const FILLERS: &[&str] = &["um", "umm", "ummm", "uh", "uhh", "uhm", "erm", "er", "err", "ah", "ahh", "hmm", "hm", "hmmm", "mm", "mmm", "mhm", "eh"];
pub(crate) const PROFANE: &[&str] = &[
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fucks", "motherfucker", "motherfucking", "shit", "shitty", "bullshit", "shits", "bitch", "bitches",
    "cunt", "cunts", "asshole", "assholes", "arsehole", "bastard", "bastards", "dick", "dickhead", "cock", "pussy", "wanker", "twat", "prick", "damn", "goddamn",
    "slut", "whore",
];

/// Clean a transcript before it is spoken: drop bracketed annotations
/// ("[music]", "(coughs)", "*laughs*", "♪"), optionally the filler words,
/// and optionally mask swear words as "bleep". Tidies the punctuation and
/// capitalisation that removals leave behind. Empty when nothing is left.
pub fn clean_transcript(text: &str, trim_fillers: bool, mask_profanity: bool) -> String {
    // bracketed annotations
    let mut s = String::with_capacity(text.len());
    let mut depth: i32 = 0;
    let mut star = false;
    for c in text.chars() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = (depth - 1).max(0),
            '*' => star = !star,
            '♪' | '♫' | '♬' => {}
            _ if depth == 0 && !star => s.push(c),
            _ => {}
        }
    }
    // words, keeping their punctuation
    let mut out: Vec<String> = Vec::new();
    let mut cap_next = false;
    for raw in s.split_whitespace() {
        let core: String = raw.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>().to_lowercase();
        let tail: String = raw.chars().rev().take_while(|c| !c.is_alphanumeric()).collect::<Vec<_>>().into_iter().rev().collect();
        let starts_upper = raw.chars().find(|c| c.is_alphabetic()).map(|c| c.is_uppercase()).unwrap_or(false);
        if trim_fillers && FILLERS.contains(&core.as_str()) {
            // keep a sentence end the filler carried ("… it, um." → "… it.")
            let end = tail.chars().find(|c| matches!(c, '.' | '!' | '?'));
            if let Some(e) = end {
                if let Some(last) = out.last_mut() {
                    let t = last.trim_end_matches([',', ';', ':']).to_string();
                    *last = format!("{t}{e}");
                }
            }
            if starts_upper || out.is_empty() {
                cap_next = true;
            }
            continue;
        }
        let mut w = if mask_profanity && PROFANE.contains(&core.trim_matches('\'')) {
            let lead: String = raw.chars().take_while(|c| !c.is_alphanumeric()).collect();
            format!("{lead}bleep{tail}")
        } else {
            raw.to_string()
        };
        if cap_next {
            let mut cs = w.chars();
            if let Some(f) = cs.next() {
                w = f.to_uppercase().collect::<String>() + cs.as_str();
            }
            cap_next = false;
        }
        if w.chars().any(|c| c.is_alphanumeric()) || !out.is_empty() {
            out.push(w);
        }
    }
    let mut joined = out.join(" ");
    // punctuation left dangling at the start or doubled
    joined = joined.trim_start_matches([',', ';', ':', '.', '-', ' ']).to_string();
    while joined.contains(",,") || joined.contains(", ,") || joined.contains(",.") {
        joined = joined.replace(", ,", ",").replace(",,", ",").replace(",.", ".");
    }
    if joined.chars().any(|c| c.is_alphanumeric()) {
        joined.trim().to_string()
    } else {
        String::new()
    }
}

// ───────────────────────── shared state ─────────────────────────

#[derive(Default)]
struct AF32(AtomicU32);
impl AF32 {
    fn set(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
    fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    fn max(&self, v: f32) {
        let _ = self.0.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| (v > f32::from_bits(b)).then_some(v.to_bits()));
    }
    fn take(&self) -> f32 {
        f32::from_bits(self.0.swap(0, Ordering::Relaxed))
    }
}

struct Job {
    id: u64,
    text: String,
    /// When your speech ended (mic lines).
    eos: Option<Instant>,
    /// Your utterance (16 kHz) to speak again in the voice (Change my voice).
    audio: Option<Vec<f32>>,
    /// Swear words to bleep in the converted audio (seconds into the utterance).
    bleeps: Vec<(f32, f32)>,
    /// Silence on the mic before this utterance (s; None = first, or typed).
    gap: Option<f32>,
    /// Audio before it to convert along and drop again (see CONTEXT_SECS).
    context: Vec<f32>,
}

/// Where a line sits on the output timeline (ring positions, in samples).
struct Mark {
    id: u64,
    text: String,
    start: Option<u64>,
    end: Option<u64>,
    eos: Option<Instant>,
    synth_ms: f32,
    announced: bool,
    skipped: bool,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    marks: VecDeque<Mark>,
    /// Job being synthesised.
    current: Option<u64>,
    /// Stop synthesising this job.
    cancel: Option<u64>,
    /// Stop everything (panic / clear).
    cancel_all: bool,
    next_id: u64,
}

/// Settings that can change while running.
#[derive(Clone)]
struct Live {
    voice: String,
    speed: f32,
    effect: Option<Preset>,
    accent: Accent,
    output_gain_db: f32,
    translate: bool,
    language: Option<String>,
    prompt: Option<String>,
    trim_fillers: bool,
    mask_profanity: bool,
    half_duplex: bool,
    convert: bool,
    keep_pauses: bool,
}

struct Shared {
    stop: AtomicBool,
    muted: AtomicBool,
    listen: AtomicBool,
    hearing: AtomicBool,
    live: Mutex<Live>,
    queue: Mutex<Queue>,
    wake: Condvar,
    /// Samples written to the output ring(s).
    written: AtomicU64,
    /// Samples the main output has consumed (its ring position).
    played: AtomicU64,
    /// Outputs discard everything before this ring position (skip / panic).
    skip_to: AtomicU64,
    /// Start of the next line not yet heard, and when the main output reached it.
    watch_start: AtomicU64,
    reached_at_ns: AtomicU64,
    epoch: Instant,
    in_peak: AF32,
    out_peak: AF32,
    in_ui: AF32,
    out_ui: AF32,
    monitor_gain: AF32,
    out_latency: AF32,
    last_latency: AF32,
    lat_sum: Mutex<(f64, u64)>,
    utterances: AtomicU64,
    /// The end-of-sentence pause in use (ms) and how many pauses it was learnt from.
    pause_ms: AtomicU64,
    pauses_heard: AtomicU64,
    stream_failed: Mutex<Option<(&'static str, String)>>,
    /// Silence the main output has played since it ran out of audio (ring
    /// samples; 0 while it plays). Counted in samples, not time, so it is
    /// exact whatever the device's clock does.
    dry_samples: AtomicU64,
    rate: u32,
}

impl Shared {
    fn speaking(&self) -> bool {
        self.played.load(Ordering::Relaxed) < self.written.load(Ordering::Relaxed)
    }
}

// ───────────────────────── output side ─────────────────────────

/// The consumer end of one output (main or monitor), driven by a device
/// callback or a test. Allocation- and lock-free.
pub struct OutputTap {
    ring: Consumer,
    shared: Arc<Shared>,
    monitor: bool,
    gain: f32,
    step: f32,
    /// Fixed-ratio reader for a monitor at another rate (Hermite).
    ratio: f64,
    frac: f64,
    hist: [f32; 4],
    scratch: Vec<f32>,
}

impl OutputTap {
    fn new(ring: Consumer, shared: Arc<Shared>, monitor: bool, device_rate: u32) -> OutputTap {
        let ratio = shared.rate as f64 / device_rate.max(1) as f64;
        let muted = shared.muted.load(Ordering::Relaxed);
        OutputTap {
            ring,
            monitor,
            gain: if muted { 0.0 } else { 1.0 },
            step: 1.0 / (0.005 * device_rate.max(1) as f32),
            ratio,
            frac: 0.0,
            hist: [0.0; 4],
            scratch: vec![0.0; 16384],
            shared,
        }
    }

    /// Ring position (samples consumed so far). Every ring carries the
    /// same sample stream from position 0, so positions are comparable.
    fn position(&self) -> u64 {
        self.ring.position()
    }

    /// Fill `out` (mono, device rate) with what is queued, silence otherwise.
    pub fn fill(&mut self, out: &mut [f32]) {
        let pos = self.position();
        let target = self.shared.skip_to.load(Ordering::Relaxed);
        if target > pos {
            self.ring.skip((target - pos) as usize);
            self.hist = [0.0; 4];
        }
        let before = self.position();
        if (self.ratio - 1.0).abs() < 1e-9 {
            let n = self.ring.pop(out);
            out[n..].iter_mut().for_each(|v| *v = 0.0);
        } else {
            for o in out.iter_mut() {
                while self.frac >= 1.0 {
                    let mut x = [0.0f32];
                    let _ = self.ring.pop(&mut x);
                    self.hist = [self.hist[1], self.hist[2], self.hist[3], x[0]];
                    self.frac -= 1.0;
                }
                let t = self.frac as f32;
                let [y0, y1, y2, y3] = self.hist;
                let c1 = 0.5 * (y2 - y0);
                let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                *o = ((c3 * t + c2) * t + c1) * t + y1;
                self.frac += self.ratio;
            }
        }
        let user = if self.monitor { self.shared.monitor_gain.get() } else { 1.0 };
        let want = if self.shared.muted.load(Ordering::Relaxed) { 0.0 } else { 1.0 };
        let mut peak = 0f32;
        let _ = self.scratch.len();
        for v in out.iter_mut() {
            if self.gain < want {
                self.gain = (self.gain + self.step).min(want);
            } else if self.gain > want {
                self.gain = (self.gain - self.step).max(want);
            }
            *v = (*v * self.gain * user).clamp(-1.0, 1.0);
            peak = peak.max(v.abs());
        }
        if !self.monitor {
            let after = self.position();
            self.shared.played.store(after, Ordering::Relaxed);
            if after >= self.shared.written.load(Ordering::Acquire) {
                let asked = (out.len() as f64 * self.ratio) as u64;
                self.shared.dry_samples.fetch_add(asked.saturating_sub(after - before), Ordering::Relaxed);
            } else {
                self.shared.dry_samples.store(0, Ordering::Relaxed);
            }
            self.shared.out_peak.max(peak);
            let w = self.shared.watch_start.load(Ordering::Relaxed);
            if before <= w && after > w && self.shared.reached_at_ns.load(Ordering::Relaxed) == 0 {
                let ns = self.shared.epoch.elapsed().as_nanos() as u64;
                self.shared.reached_at_ns.store(ns.max(1), Ordering::Relaxed);
            }
        }
    }
}

// ───────────────────────── the device-free pipeline ─────────────────────────

/// Outputs to create: the main one's device rate and the monitor's (if any).
#[derive(Debug, Clone, Copy)]
pub struct OutputRates {
    pub main: u32,
    pub monitor: Option<u32>,
}

/// The pipeline without devices. Dropping it stops its threads.
pub struct SpeakCore {
    shared: Arc<Shared>,
    capture: Option<Box<dyn CaptureSource>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    info: Mutex<SpeakStatus>,
    on_event: OnEvent,
}

type OnEvent = Arc<dyn Fn(SpeakEvent) + Send + Sync>;

impl SpeakCore {
    /// Start listening to `input` (None = type-to-speak only) and speaking
    /// into the returned taps (main first, then the monitor). The synthesis
    /// runs at `rates.main`.
    pub fn start(
        cfg: &SpeakConfig,
        synth: Arc<dyn Synth>,
        stt: Option<(Arc<dyn Transcriber>, String)>,
        input: Option<Box<dyn CaptureSource>>,
        rates: OutputRates,
        on_event: impl Fn(SpeakEvent) + Send + Sync + 'static,
    ) -> Result<(SpeakCore, Vec<OutputTap>)> {
        let on_event: OnEvent = Arc::new(on_event);
        let mut effect = cfg.effect.clone();
        if let Some(p) = effect.as_mut() {
            p.sanitize();
        }
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            muted: AtomicBool::new(cfg.start_muted),
            listen: AtomicBool::new(cfg.listen),
            hearing: AtomicBool::new(false),
            live: Mutex::new(Live {
                voice: cfg.voice.clone(),
                speed: cfg.speed.clamp(0.5, 2.0),
                effect,
                accent: cfg.accent,
                output_gain_db: cfg.output_gain_db,
                translate: cfg.translate,
                language: cfg.language.clone().filter(|l| !l.is_empty()),
                prompt: cfg.prompt.clone().filter(|l| !l.trim().is_empty()),
                trim_fillers: cfg.trim_fillers,
                mask_profanity: cfg.mask_profanity,
                half_duplex: cfg.half_duplex,
                convert: cfg.convert,
                keep_pauses: cfg.keep_pauses,
            }),
            queue: Mutex::new(Queue { next_id: 1, ..Default::default() }),
            wake: Condvar::new(),
            written: AtomicU64::new(0),
            played: AtomicU64::new(0),
            skip_to: AtomicU64::new(0),
            watch_start: AtomicU64::new(u64::MAX),
            reached_at_ns: AtomicU64::new(0),
            epoch: Instant::now(),
            in_peak: AF32::default(),
            out_peak: AF32::default(),
            in_ui: AF32::default(),
            out_ui: AF32::default(),
            monitor_gain: AF32(AtomicU32::new(super::dsp::db_to_lin(cfg.monitor_gain_db).to_bits())),
            out_latency: AF32::default(),
            last_latency: AF32::default(),
            lat_sum: Mutex::new((0.0, 0)),
            utterances: AtomicU64::new(0),
            pause_ms: AtomicU64::new(cfg.hangover_ms as u64),
            pauses_heard: AtomicU64::new(0),
            stream_failed: Mutex::new(None),
            dry_samples: AtomicU64::new(0),
            rate: rates.main,
        });
        // rings: ~40 s of audio (8 MB at 48 kHz); a longer paragraph only
        // makes the synthesiser wait for room, it never drops audio
        let cap = rates.main as usize * 40;
        let (p_main, c_main) = ring(cap);
        let mut producers = vec![p_main];
        let mut taps = vec![OutputTap::new(c_main, shared.clone(), false, rates.main)];
        if let Some(mr) = rates.monitor {
            let (p, c) = ring(cap);
            producers.push(p);
            taps.push(OutputTap::new(c, shared.clone(), true, mr));
        }
        let mut threads = Vec::new();
        // synthesis
        {
            let (shared, synth, ev) = (shared.clone(), synth.clone(), on_event.clone());
            threads.push(
                std::thread::Builder::new()
                    .name("nekotone-speak-tts".into())
                    .spawn(move || speak_loop(shared, synth, producers, ev))
                    .map_err(Error::Io)?,
            );
        }
        // transcription + listening
        let mut capture = None;
        let stt_name = stt.as_ref().map(|s| s.1.clone()).unwrap_or_default();
        if let (Some(mut src), Some((stt, _))) = (input, stt) {
            let (utt_tx, utt_rx) = bounded::<Heard16k>(16);
            {
                let (shared, ev) = (shared.clone(), on_event.clone());
                threads.push(
                    std::thread::Builder::new()
                        .name("nekotone-speak-stt".into())
                        .spawn(move || stt_loop(shared, stt, utt_rx, ev))
                        .map_err(Error::Io)?,
                );
            }
            let fmt = src.format();
            let (tx, rx) = bounded::<Chunk>(512);
            let (pool_tx, pool_rx) = bounded::<Vec<f32>>(64);
            {
                let (shared, ev) = (shared.clone(), on_event.clone());
                let hang = cfg.hangover_ms.clamp(250, 1000) as f32 / 1000.0;
                // Change my voice speaks each utterance after it ends, so a long
                // one leaves the output silent while it is said: cut it at the
                // first small pause after 2.5 s (at most 6 s). Conversion keeps
                // the timing, so the cut is not heard (an owner's recording had
                // 11 silent gaps of 0.5-3 s mid-speech with the 8 s cut).
                let cut_secs = if cfg.convert { (2.5, 6.0) } else { (8.0, 20.0) };
                let auto_pause = cfg.auto_pause;
                let (rate, ch) = (fmt.sample_rate, fmt.channels.max(1) as usize);
                threads.push(
                    std::thread::Builder::new()
                        .name("nekotone-speak-listen".into())
                        .spawn(move || listen_loop(shared, rx, pool_tx, rate, ch, hang, auto_pause, cut_secs, utt_tx, ev))
                        .map_err(Error::Io)?,
                );
            }
            let sink = CaptureSink {
                source: 0,
                tx,
                pool: pool_rx,
                clock: Arc::new(RealClock::new()),
                dropped: Arc::new(AtomicU64::new(0)),
                blocking: false,
            };
            if let Err(e) = src.start(sink) {
                shared.stop.store(true, Ordering::Relaxed);
                shared.wake.notify_all();
                for t in threads {
                    let _ = t.join();
                }
                return Err(e);
            }
            capture = Some(src);
        }
        // events / marks
        {
            let (shared, ev) = (shared.clone(), on_event.clone());
            threads.push(std::thread::Builder::new().name("nekotone-speak-watch".into()).spawn(move || watch_loop(shared, ev)).map_err(Error::Io)?);
        }
        let live = shared.live.lock().clone();
        let info = SpeakStatus {
            running: true,
            voice: live.voice.clone(),
            speed: live.speed,
            effect_id: live.effect.as_ref().map(|p| p.id.clone()),
            accent: live.accent,
            sample_rate: rates.main,
            tts_device: synth.device(),
            stt_model: stt_name,
            ..Default::default()
        };
        Ok((SpeakCore { shared, capture, threads, info: Mutex::new(info), on_event }, taps))
    }

    /// Queue `text` to be spoken (type to speak). Returns its id.
    pub fn say(&self, text: &str) -> Option<u64> {
        let t = text.trim();
        if !t.chars().any(|c| c.is_alphanumeric()) {
            return None;
        }
        Some(enqueue(&self.shared, &self.on_event, t.to_string(), None, Source::Typed))
    }

    /// Stop the line that is playing (the next one starts).
    pub fn skip(&self) {
        let mut q = self.shared.queue.lock();
        let played = self.shared.played.load(Ordering::Relaxed);
        // the audible line, else the one being synthesised
        let audible = q.marks.iter().find(|m| m.start.map(|s| s <= played).unwrap_or(false) && m.end.map(|e| e > played).unwrap_or(true)).map(|m| (m.id, m.end));
        let target = audible.or_else(|| q.current.map(|c| (c, None)));
        if let Some((id, end)) = target {
            if let Some(m) = q.marks.iter_mut().find(|m| m.id == id) {
                m.skipped = true;
            }
            match end {
                Some(e) => {
                    self.shared.skip_to.fetch_max(e, Ordering::Relaxed);
                }
                None => q.cancel = Some(id),
            }
        }
        drop(q);
        self.shared.wake.notify_all();
    }

    /// Drop the queue and silence what is playing now (the panic button's second half).
    pub fn clear(&self) {
        let mut q = self.shared.queue.lock();
        let dropped: Vec<u64> = q.jobs.drain(..).map(|j| j.id).collect();
        let played = self.shared.played.load(Ordering::Relaxed);
        for m in q.marks.iter_mut() {
            if m.end.map(|e| e > played).unwrap_or(true) {
                m.skipped = true;
            }
        }
        if q.current.is_some() {
            q.cancel_all = true;
        } else {
            self.shared.skip_to.fetch_max(self.shared.written.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        drop(q);
        for id in dropped {
            (self.on_event)(SpeakEvent::Done { id, skipped: true });
        }
        self.shared.wake.notify_all();
    }

    /// Mute: the outputs fade to silence in 5 ms (lines keep advancing silently).
    pub fn set_mute(&self, mute: bool) {
        self.shared.muted.store(mute, Ordering::Relaxed);
    }

    /// Panic: mute, clear the queue and cut what is playing.
    pub fn panic(&self) {
        self.set_mute(true);
        self.clear();
    }

    /// Pause or resume listening to the microphone.
    pub fn set_listening(&self, on: bool) {
        self.shared.listen.store(on, Ordering::Relaxed);
    }

    pub fn set_voice(&self, voice: &str) {
        self.shared.live.lock().voice = voice.to_string();
        self.info.lock().voice = voice.to_string();
    }

    pub fn set_speed(&self, speed: f32) {
        let s = speed.clamp(0.5, 2.0);
        self.shared.live.lock().speed = s;
        self.info.lock().speed = s;
    }

    /// Accent of the synthetic voice; applies from the next line.
    pub fn set_accent(&self, accent: Accent) {
        self.shared.live.lock().accent = accent;
        self.info.lock().accent = accent;
    }

    /// Effect on the synthetic voice (None = clean); applies from the next line.
    pub fn set_effect(&self, effect: Option<Preset>) {
        let mut e = effect;
        if let Some(p) = e.as_mut() {
            p.sanitize();
        }
        self.info.lock().effect_id = e.as_ref().map(|p| p.id.clone());
        self.shared.live.lock().effect = e;
    }

    /// Change my voice (neural) on or off, for utterances from now on.
    pub fn set_convert(&self, on: bool) {
        self.shared.live.lock().convert = on;
    }

    pub fn set_translate(&self, translate: bool, language: Option<String>) {
        let mut l = self.shared.live.lock();
        l.translate = translate;
        l.language = language.filter(|s| !s.is_empty());
    }

    pub fn set_text_options(&self, trim_fillers: bool, mask_profanity: bool, half_duplex: bool) {
        let mut l = self.shared.live.lock();
        l.trim_fillers = trim_fillers;
        l.mask_profanity = mask_profanity;
        l.half_duplex = half_duplex;
    }

    pub fn set_output_gain_db(&self, db: f32) {
        self.shared.live.lock().output_gain_db = db.clamp(-24.0, 24.0);
    }

    pub fn set_monitor_gain_db(&self, db: f32) {
        self.shared.monitor_gain.set(super::dsp::db_to_lin(db.clamp(-60.0, 12.0)));
    }

    /// Record the device output latency (added to the measured latency).
    pub fn set_output_latency(&self, secs: f32) {
        self.shared.out_latency.set(secs);
    }

    fn set_devices(&self, input: Option<String>, output: String, monitor: Option<String>) {
        let mut i = self.info.lock();
        i.input_device = input;
        i.output_device = output;
        i.monitor_device = monitor;
    }

    pub fn status(&self) -> SpeakStatus {
        let s = &self.shared;
        let mut st = self.info.lock().clone();
        st.running = !s.stop.load(Ordering::Relaxed);
        st.muted = s.muted.load(Ordering::Relaxed);
        st.listening_enabled = s.listen.load(Ordering::Relaxed);
        st.hearing = s.hearing.load(Ordering::Relaxed);
        st.speaking = s.speaking();
        st.queued = s.queue.lock().jobs.len();
        st.input_level_db = lin_to_db(s.in_ui.get());
        st.output_level_db = lin_to_db(s.out_ui.get());
        st.last_latency_ms = s.last_latency.get();
        let (sum, n) = *s.lat_sum.lock();
        st.mean_latency_ms = if n > 0 { (sum / n as f64) as f32 } else { 0.0 };
        st.utterances = s.utterances.load(Ordering::Relaxed);
        st.pause_ms = s.pause_ms.load(Ordering::Relaxed) as u32;
        st.pauses_heard = s.pauses_heard.load(Ordering::Relaxed) as u32;
        st
    }

    /// Stop everything and join the threads.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(mut c) = self.capture.take() {
            c.stop();
        }
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for SpeakCore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn enqueue(shared: &Arc<Shared>, on_event: &OnEvent, text: String, eos: Option<Instant>, source: Source) -> u64 {
    let mut q = shared.queue.lock();
    let id = q.next_id;
    q.next_id += 1;
    q.jobs.push_back(Job { id, text: text.clone(), eos, audio: None, bleeps: Vec::new(), gap: None, context: Vec::new() });
    drop(q);
    shared.wake.notify_all();
    on_event(SpeakEvent::Queued { id, text, source });
    id
}

/// One change to an utterance before it is spoken again (Change my voice).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioEdit {
    /// Remove this span (a filler word), seconds.
    Cut(f32, f32),
    /// Silence this span and bleep it in the output (a swear word), seconds.
    Bleep(f32, f32),
}

/// The quietest 10 ms within ±150 ms of `t` (seconds): Whisper's word times
/// here are interpolated across a segment, so a word edge is snapped to the
/// gap between words, where speech is quietest.
fn snap_to_quiet(x: &[f32], rate: u32, t: f32) -> f32 {
    let r = rate as f32;
    let w = (0.010 * r) as usize;
    let (lo, hi) = (((t - 0.15) * r).max(0.0) as usize, (((t + 0.15) * r) as usize).min(x.len().saturating_sub(w)));
    if hi <= lo {
        return t;
    }
    let mut best = (f32::MAX, t);
    let mut i = lo;
    while i <= hi {
        let e: f32 = x[i..i + w].iter().map(|v| v * v).sum();
        if e < best.0 {
            best = (e, (i + w / 2) as f32 / r);
        }
        i += w / 2;
    }
    best.1
}

/// First and last moment of speech in `x` (seconds): 10 ms frames within
/// 35 dB of the loudest one (and above -60 dBFS).
fn speech_bounds(x: &[f32], rate: u32) -> Option<(f32, f32)> {
    let n = (0.010 * rate as f32) as usize;
    let e: Vec<f32> = x.chunks(n.max(1)).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()).collect();
    let peak = e.iter().cloned().fold(0f32, f32::max);
    let thr = (peak * 10f32.powf(-35.0 / 20.0)).max(10f32.powf(-60.0 / 20.0));
    let first = e.iter().position(|&v| v > thr)?;
    let last = e.iter().rposition(|&v| v > thr)?;
    Some((first as f32 * n as f32 / rate as f32, (last + 1) as f32 * n as f32 / rate as f32))
}

/// The fillers to cut and swear words to bleep, from Whisper's words, with
/// edges snapped to the quiet between words. Whisper's word times here are
/// spread over its segment, silence included (a clip with 1 s of silence
/// first put "Bitch" on the silence), so they are first mapped onto where
/// speech actually is, then each edge is snapped to the nearest quiet.
pub fn word_edits(words: &[crate::stt::Word], x: &[f32], rate: u32, trim_fillers: bool, mask_profanity: bool) -> Vec<AudioEdit> {
    let mut out = Vec::new();
    let (w0, w1) = match (words.first(), words.last()) {
        (Some(a), Some(b)) if b.end > a.start => (a.start, b.end),
        _ => return out,
    };
    let (s0, s1) = speech_bounds(x, rate).unwrap_or((w0, w1));
    let map = |t: f32| s0 + (t - w0) / (w1 - w0) * (s1 - s0);
    for w in words {
        let w = crate::stt::Word { start: map(w.start), end: map(w.end), ..w.clone() };
        let w = &w;
        let core: String = w.text.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>().to_lowercase();
        let core = core.trim_matches('\'');
        let filler = trim_fillers && FILLERS.contains(&core);
        let profane = mask_profanity && PROFANE.contains(&core);
        if !(filler || profane) {
            continue;
        }
        let (s, e) = (snap_to_quiet(x, rate, w.start), snap_to_quiet(x, rate, w.end));
        if e <= s + 0.03 {
            continue;
        }
        out.push(if filler { AudioEdit::Cut(s, e) } else { AudioEdit::Bleep(s, e) });
    }
    out
}

/// Where a swear word really is: the shortest beginning of the utterance in
/// which Whisper hears it gives its end (binary search, to ~40 ms); its start
/// is that end minus a length from its letters, snapped to the quiet before
/// it. Each search costs a few Whisper passes, only on lines with swearing.
fn refine_bleeps(edits: &mut [AudioEdit], words: &[crate::stt::Word], x: &[f32], rate: u32, stt: &dyn Transcriber, language: Option<String>) {
    let norm = |s: &str| -> String { s.chars().filter(|c| c.is_alphanumeric() || *c == '\'').collect::<String>().to_lowercase().trim_matches('\'').to_string() };
    let profane: Vec<String> = words.iter().map(|w| norm(&w.text)).filter(|w| PROFANE.contains(&w.as_str())).collect();
    let mut k = 0;
    for e in edits.iter_mut() {
        let AudioEdit::Bleep(s, end) = *e else { continue };
        let Some(word) = profane.get(k).cloned() else { break };
        k += 1;
        let hears = |t: f32| -> Option<bool> {
            let n = ((t * rate as f32) as usize).min(x.len());
            let clip = crate::audio::Clip { samples: x[..n].to_vec(), sample_rate: rate, source_channels: 1 };
            let opts = TranscribeOptions { language: language.clone(), ..Default::default() };
            let tr = stt.transcribe(&clip, &opts, &mut |_| {}).ok()?;
            // count occurrences so a second swear word is searched past the first
            let n_said = tr.segments.iter().flat_map(|s| s.text.split_whitespace()).filter(|w| norm(w) == word).count();
            Some(n_said >= k)
        };
        let len = x.len() as f32 / rate as f32;
        // the true end lies within ~0.8 s of the estimate
        let (mut lo, mut hi) = ((s - 0.4).max(0.05), (end + 0.8).min(len));
        if hears(hi) != Some(true) {
            continue; // not found in the prefix: keep the estimate
        }
        if hears(lo) == Some(true) {
            lo = 0.05;
        }
        for _ in 0..6 {
            if hi - lo < 0.04 {
                break;
            }
            let mid = 0.5 * (lo + hi);
            match hears(mid) {
                Some(true) => hi = mid,
                Some(false) => lo = mid,
                None => break,
            }
        }
        // Whisper names a word once most of it is there: add a little after
        let new_end = (hi + 0.08).min(len);
        let dur = (0.075 * word.chars().count() as f32).clamp(0.22, 0.6);
        let new_start = snap_to_quiet(x, rate, (new_end - dur).max(0.0)).min(new_end - 0.12).max(0.0);
        *e = AudioEdit::Bleep(new_start, new_end);
    }
}

/// Apply the edits to an utterance: fillers are cut out (10 ms crossfade),
/// swear words silenced. Returns the edited audio and where the bleeps go in
/// it (seconds, after the cuts).
pub fn apply_edits(x: &[f32], rate: u32, edits: &[AudioEdit]) -> (Vec<f32>, Vec<(f32, f32)>) {
    let r = rate as f32;
    let fade = (0.010 * r) as usize;
    let mut out: Vec<f32> = Vec::with_capacity(x.len());
    let mut bleeps = Vec::new();
    let mut spans: Vec<AudioEdit> = edits.to_vec();
    spans.sort_by(|a, b| {
        let (sa, sb) = (match a { AudioEdit::Cut(s, _) | AudioEdit::Bleep(s, _) => *s }, match b { AudioEdit::Cut(s, _) | AudioEdit::Bleep(s, _) => *s });
        sa.total_cmp(&sb)
    });
    let mut pos = 0usize;
    for e in spans {
        let (s, t, cut) = match e {
            AudioEdit::Cut(s, t) => (s, t, true),
            AudioEdit::Bleep(s, t) => (s, t, false),
        };
        let (si, ti) = (((s * r) as usize).clamp(pos, x.len()), ((t * r) as usize).min(x.len()));
        if ti <= si {
            continue;
        }
        out.extend_from_slice(&x[pos..si]);
        if cut {
            // crossfade the end of what came before into what follows
            let n = fade.min(out.len()).min(x.len() - ti);
            let base = out.len() - n;
            for k in 0..n {
                let g = (k as f32 + 0.5) / n as f32;
                out[base + k] = out[base + k] * (1.0 - g) + x[ti + k] * g;
            }
            pos = ti + n;
        } else {
            let at = out.len() as f32 / r;
            out.extend(std::iter::repeat_n(0.0, ti - si));
            bleeps.push((at, at + (ti - si) as f32 / r));
            pos = ti;
        }
    }
    out.extend_from_slice(&x[pos.min(x.len())..]);
    (out, bleeps)
}

/// A 1 kHz bleep over each span of `y` (at the converted voice's level),
/// with 5 ms ramps; the span is padded 40 ms each side, because conversion
/// keeps timing only to a few tens of ms.
pub fn bleep(y: &mut [f32], rate: u32, spans: &[(f32, f32)]) {
    if spans.is_empty() || y.is_empty() {
        return;
    }
    let r = rate as f32;
    let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt().max(0.02);
    let amp = (rms * 1.4).min(0.5);
    let ramp = (0.005 * r) as usize;
    for &(s, e) in spans {
        let (a, b) = ((((s - 0.04) * r).max(0.0)) as usize, (((e + 0.04) * r) as usize).min(y.len()));
        for (j, v) in y[a..b].iter_mut().enumerate() {
            let k = j.min(b - a - 1 - j);
            let g = if k < ramp { k as f32 / ramp as f32 } else { 1.0 };
            *v = amp * g * (2.0 * std::f32::consts::PI * 1000.0 * j as f32 / r).sin();
        }
    }
}

/// Queue an utterance to be spoken again in the voice (Change my voice).
#[allow(clippy::too_many_arguments)]
fn enqueue_audio(shared: &Arc<Shared>, on_event: &OnEvent, label: String, eos: Instant, audio: Vec<f32>, bleeps: Vec<(f32, f32)>, gap: Option<f32>, context: Vec<f32>) -> u64 {
    let mut q = shared.queue.lock();
    let id = q.next_id;
    q.next_id += 1;
    q.jobs.push_back(Job { id, text: label.clone(), eos: Some(eos), audio: Some(audio), bleeps, gap, context });
    drop(q);
    shared.wake.notify_all();
    on_event(SpeakEvent::Queued { id, text: label, source: Source::Mic });
    id
}

// ───────────────────────── threads ─────────────────────────

struct Heard16k {
    samples: Vec<f32>,
    eos: Instant,
    /// Mic time between the previous utterance's end and this one's start
    /// (0 after a cut inside a phrase; None for the first).
    gap: Option<f32>,
    /// The end of the previous piece when this one carries straight on
    /// from it (a cut inside a phrase): context for the voice conversion.
    context: Vec<f32>,
}

/// Change my voice: the previous piece's last second goes along as context
/// when a phrase is cut while you keep talking, and its sound is dropped
/// again after conversion. On the owner's recording (tts::chatterbox tests,
/// convert_keeps_phrase_endings, 22-23 phrases cut as Speak for me cuts
/// them) Whisper misheard 46 % of the converted words with 2.5 s pieces cut
/// at any 60 ms gap; 34 % with this context and cuts only at 160 ms gaps
/// (37 % context alone, 53 % the wider gap alone): a word split at a cut
/// came out garbled on both sides.
const CONTEXT_SECS: f32 = 1.0;

#[allow(clippy::too_many_arguments)]
fn listen_loop(shared: Arc<Shared>, rx: Receiver<Chunk>, pool: Sender<Vec<f32>>, rate: u32, channels: usize, hangover: f32, auto_pause: bool, cut_secs: (f32, f32), out: Sender<Heard16k>, on_event: OnEvent) {
    crate::threads::raise_current_thread_priority();
    // a long phrase is cut into pieces with nothing lost between them (words
    // at a cut were dropped: the next piece waited for a fresh onset and a
    // short last piece fell under the 0.3 s minimum), and a soft cut waits
    // for a 60 ms gap between words rather than any quiet 20 ms frame. A
    // phrase needs 0.15 s of voiced speech (the gap is 160 ms since 0.4.14:
    // a 60 ms gap fell inside words, see CONTEXT_SECS) (was 0.3: "yes" or "okay" alone
    // could fall under it and was never said)
    let cfg = VadConfig { hangover_secs: hangover, soft_max_secs: cut_secs.0, hard_max_secs: cut_secs.1, soft_gap_frames: 8, seamless_cuts: true, min_speech_secs: 0.15, adapt_hangover: auto_pause.then(|| crate::record::vad::PauseLearner::new(0.90)), ..Default::default() };
    let mut vad = Vad::new(cfg);
    let mut last_end: Option<u64> = None;
    let mut prev_tail: Vec<f32> = Vec::new();
    let mut rs = Resampler::new(rate, VAD_RATE as u32, 1, 16);
    let mut mono = Vec::new();
    let mut x16 = Vec::new();
    let mut utts = Vec::new();
    let mut deaf_until = Instant::now();
    let mut was_active = false;
    while !shared.stop.load(Ordering::Relaxed) {
        let chunk = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        let samples = match chunk {
            Chunk::Audio { samples, .. } => samples,
            Chunk::Failed { message, .. } => {
                *shared.stream_failed.lock() = Some(("input", message.clone()));
                on_event(SpeakEvent::DeviceLost { role: "input", message });
                continue;
            }
            Chunk::Restored { name, .. } => {
                *shared.stream_failed.lock() = None;
                on_event(SpeakEvent::DeviceRestored { role: "input", name });
                continue;
            }
        };
        mono.clear();
        let mut peak = 0f32;
        for f in samples.chunks_exact(channels) {
            let v = f.iter().sum::<f32>() / channels as f32;
            peak = peak.max(v.abs());
            mono.push(v);
        }
        let _ = pool.try_send(samples);
        shared.in_peak.max(peak);
        let half = shared.live.lock().half_duplex;
        if half && shared.speaking() {
            deaf_until = Instant::now() + Duration::from_millis(300);
        }
        if !shared.listen.load(Ordering::Relaxed) || (half && Instant::now() < deaf_until) {
            mono.iter_mut().for_each(|v| *v = 0.0);
        }
        x16.clear();
        rs.process(&mono, &mut x16);
        vad.feed(&x16, &mut utts);
        shared.pause_ms.store((vad.hangover_secs() * 1000.0).round() as u64, Ordering::Relaxed);
        shared.pauses_heard.store(vad.pauses_heard() as u64, Ordering::Relaxed);
        let active = vad.is_active();
        if active && !was_active {
            shared.hearing.store(true, Ordering::Relaxed);
            on_event(SpeakEvent::Listening);
        }
        was_active = active;
        if !active {
            shared.hearing.store(false, Ordering::Relaxed);
        }
        for u in utts.drain(..) {
            // the VAD keeps 150 ms of the silence after the last speech frame
            let eos_pos = u.end.saturating_sub((0.15 * VAD_RATE as f64) as u64);
            let behind = vad.position().saturating_sub(eos_pos) as f64 / VAD_RATE as f64;
            let eos = Instant::now().checked_sub(Duration::from_secs_f64(behind)).unwrap_or_else(Instant::now);
            let secs = u.samples.len() as f32 / VAD_RATE as f32;
            on_event(SpeakEvent::Transcribing { speech_secs: secs });
            let gap = last_end.map(|e| u.start.saturating_sub(e) as f32 / VAD_RATE as f32);
            let context = if last_end == Some(u.start) { prev_tail.clone() } else { Vec::new() };
            last_end = Some(u.end);
            prev_tail = u.samples[u.samples.len().saturating_sub((CONTEXT_SECS * VAD_RATE as f32) as usize)..].to_vec();
            if out.try_send(Heard16k { samples: u.samples, eos, gap, context }).is_err() {
                on_event(SpeakEvent::Dropped { reason: "the transcriber is behind; an utterance was skipped".into() });
            }
        }
    }
    // end of input: close an open utterance
    let mut rest = Vec::new();
    vad.flush(&mut rest);
    for u in rest {
        let gap = last_end.map(|e| u.start.saturating_sub(e) as f32 / VAD_RATE as f32);
        let context = if last_end == Some(u.start) { prev_tail.clone() } else { Vec::new() };
        last_end = Some(u.end);
        let _ = out.try_send(Heard16k { samples: u.samples, eos: Instant::now(), gap, context });
    }
}

fn stt_loop(shared: Arc<Shared>, stt: Arc<dyn Transcriber>, rx: Receiver<Heard16k>, on_event: OnEvent) {
    // warm-up: the first Whisper call (graph allocation, DirectML shaders) is
    // several times slower than the rest; pay it now, not on your first sentence
    let silence = crate::audio::Clip { samples: vec![0.0; VAD_RATE / 2], sample_rate: VAD_RATE as u32, source_channels: 1 };
    let _ = stt.transcribe(&silence, &TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {});
    loop {
        let u = match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(u) => u,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if shared.stop.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
            Err(_) => break,
        };
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }
        let live = shared.live.lock().clone();
        if live.convert {
            // the words stay yours: the audio itself goes on. "Drop um and uh"
            // and "Mask swearing" need the words, so then Whisper listens too
            // (with word times) and the utterance is edited before conversion.
            let secs = u.samples.len() as f32 / VAD_RATE as f32;
            let mut audio = u.samples;
            let (mut label, mut original, mut language, mut stt_ms, mut bleeps) = (format!("(your words, {secs:.1} s)"), String::new(), String::new(), 0.0, Vec::new());
            if live.trim_fillers || live.mask_profanity {
                let t0 = Instant::now();
                let clip = crate::audio::Clip { samples: audio.clone(), sample_rate: VAD_RATE as u32, source_channels: 1 };
                let opts = TranscribeOptions { language: live.language.clone(), translate: false, word_timestamps: true, initial_prompt: live.prompt.clone(), ranges: None };
                if let Ok(tr) = stt.transcribe(&clip, &opts, &mut |_| {}) {
                    let words: Vec<crate::stt::Word> = tr.segments.iter().flat_map(|s| s.words.iter().cloned()).collect();
                    original = tr.segments.iter().map(|s| s.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
                    language = tr.language.clone();
                    let mut edits = word_edits(&words, &audio, VAD_RATE as u32, live.trim_fillers, live.mask_profanity);
                    // a swear word must be covered exactly: Whisper's word times
                    // are spread over the phrase (one landed 0.35 s late in the
                    // owner's recording, and the word was heard), so its end is
                    // found by listening to shorter and shorter beginnings
                    refine_bleeps(&mut edits, &words, &audio, VAD_RATE as u32, stt.as_ref(), live.language.clone());
                    let (cut, b) = apply_edits(&audio, VAD_RATE as u32, &edits);
                    audio = cut;
                    bleeps = b;
                    let shown = clean_transcript(&original, live.trim_fillers, live.mask_profanity);
                    if !shown.is_empty() {
                        label = shown;
                    }
                }
                stt_ms = t0.elapsed().as_secs_f32() * 1000.0;
                if audio.len() < VAD_RATE / 5 {
                    on_event(SpeakEvent::Dropped { reason: if original.is_empty() { "no words".into() } else { format!("only fillers: {original}") } });
                    continue;
                }
            }
            shared.utterances.fetch_add(1, Ordering::Relaxed);
            // the words for the transcript when Whisper did not listen above:
            // after the audio is queued, so the voice never waits for them,
            // and skipped while more speech is waiting (never falls behind)
            let for_words = original.is_empty().then(|| audio.clone());
            let id = enqueue_audio(&shared, &on_event, label.clone(), u.eos, audio, bleeps, u.gap, u.context);
            if let Some(a) = for_words.filter(|_| rx.is_empty()) {
                let t0 = Instant::now();
                let clip = crate::audio::Clip { samples: a, sample_rate: VAD_RATE as u32, source_channels: 1 };
                let opts = TranscribeOptions { language: live.language.clone(), initial_prompt: live.prompt.clone(), ..Default::default() };
                if let Ok(tr) = stt.transcribe(&clip, &opts, &mut |_| {}) {
                    let words = tr.segments.iter().filter(|s| !(s.no_speech_prob > 0.6 && s.text.split_whitespace().count() <= 3)).map(|s| s.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ");
                    if !words.is_empty() {
                        original = words.clone();
                        label = words;
                        language = tr.language.clone();
                        stt_ms = t0.elapsed().as_secs_f32() * 1000.0;
                    }
                }
            }
            on_event(SpeakEvent::Heard { id, text: label, original: if original.is_empty() { format!("(your words, {secs:.1} s)") } else { original }, language, stt_ms, speech_secs: secs });
            continue;
        }
        let opts = TranscribeOptions {
            language: live.language.clone(),
            translate: live.translate,
            word_timestamps: false,
            initial_prompt: live.prompt.clone(),
            ranges: None,
        };
        let secs = u.samples.len() as f32 / VAD_RATE as f32;
        let clip = crate::audio::Clip { samples: u.samples, sample_rate: VAD_RATE as u32, source_channels: 1 };
        let t0 = Instant::now();
        match stt.transcribe(&clip, &opts, &mut |_| {}) {
            Ok(tr) => {
                let original = tr
                    .segments
                    .iter()
                    .filter(|s| !(s.no_speech_prob > 0.6 && s.text.split_whitespace().count() <= 3))
                    .map(|s| s.text.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                let text = clean_transcript(&original, live.trim_fillers, live.mask_profanity);
                let stt_ms = t0.elapsed().as_secs_f32() * 1000.0;
                if text.is_empty() {
                    on_event(SpeakEvent::Dropped { reason: if original.is_empty() { "no words".into() } else { format!("only fillers or noises: {original}") } });
                    continue;
                }
                shared.utterances.fetch_add(1, Ordering::Relaxed);
                let id = enqueue(&shared, &on_event, text.clone(), Some(u.eos), Source::Mic);
                on_event(SpeakEvent::Heard { id, text, original, language: tr.language.clone(), stt_ms, speech_secs: secs });
            }
            Err(e) => on_event(SpeakEvent::Error { message: format!("speech to text failed: {e}") }),
        }
    }
}

/// Front end for the synthetic voice: rumble filter only (no gate, no de-esser: Kokoro is clean).
fn tts_front_end() -> FrontEnd {
    FrontEnd {
        hpf_hz: 40.0,
        gate: GateParams { threshold_db: -90.0, floor_db: 0.0, auto: 0.0, ..Default::default() },
        deesser_on: false,
        ..Default::default()
    }
}

fn build_processor(rate: u32, live: &Live) -> Processor {
    let blocks = live.effect.as_ref().map(|p| p.blocks.clone()).unwrap_or_default();
    Processor::new(rate as f32, 512, tts_front_end(), &blocks, live.output_gain_db, -1.0)
}

struct ConvertedPiece(Vec<f32>);
enum Either {
    Text(Vec<Piece>),
    Converted(Vec<ConvertedPiece>),
}
/// One piece of a job: text to render, or audio already converted.
enum Work {
    Text(Piece),
    Audio(Vec<f32>),
}

fn speak_loop(shared: Arc<Shared>, synth: Arc<dyn Synth>, mut outs: Vec<Producer>, on_event: OnEvent) {
    let rate = shared.rate;
    let mut key = String::new();
    let mut proc: Option<Processor> = None;
    // ring position where the last line's speech ended (Keep my pauses)
    let mut speech_end: Option<u64> = None;
    // warm-up (the first Kokoro call is slower); nothing is output
    {
        let (voice, accent) = {
            let l = shared.live.lock();
            (l.voice.clone(), l.accent)
        };
        if let Ok(pieces) = synth.plan("Ready.", &voice, FIRST_PIECE_PHONEMES, accent) {
            if let Some(p) = pieces.first() {
                let _ = synth.render(p, &voice, 1.0);
            }
        }
    }
    loop {
        // next job
        let job = {
            let mut q = shared.queue.lock();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(j) = q.jobs.pop_front() {
                    q.current = Some(j.id);
                    q.cancel = None;
                    q.cancel_all = false;
                    q.marks.push_back(Mark { id: j.id, text: j.text.clone(), start: None, end: None, eos: j.eos, synth_ms: 0.0, announced: false, skipped: false });
                    break j;
                }
                shared.wake.wait_for(&mut q, Duration::from_millis(50));
            }
        };
        let live = shared.live.lock().clone();
        let k = format!("{}|{}", serde_json::to_string(&live.effect).unwrap_or_default(), live.output_gain_db);
        if proc.is_none() || k != key {
            proc = Some(build_processor(rate, &live));
            key = k;
        }
        let p = proc.as_mut().expect("built above");
        let cancelled = |shared: &Shared, id: u64| {
            let q = shared.queue.lock();
            q.cancel_all || q.cancel == Some(id) || shared.stop.load(Ordering::Relaxed)
        };
        let t_job = Instant::now();
        let mut first = true;
        let mut failed = false;
        let planned = match &job.audio {
            // Change my voice: one piece, your utterance in the voice
            Some(audio) => {
                // with the end of the previous piece in front, dropped again after
                let with: Vec<f32> = job.context.iter().chain(audio.iter()).copied().collect();
                let drop = (job.context.len() as u64 * synth.rate() as u64 / VAD_RATE as u64) as usize;
                synth
                    .convert(&with, VAD_RATE as u32, &live.voice)
                    .map(|mut a| {
                        a.drain(..drop.min(a.len()));
                        bleep(&mut a, synth.rate(), &job.bleeps);
                        vec![ConvertedPiece(a)]
                    })
                    .map(Either::Converted)
            }
            None => synth.plan(&job.text, &live.voice, FIRST_PIECE_PHONEMES, live.accent).map(Either::Text),
        };
        let planned = planned.map(|p| match p {
            Either::Text(pieces) => pieces.into_iter().map(Work::Text).collect::<Vec<_>>(),
            Either::Converted(v) => v.into_iter().map(|c| Work::Audio(c.0)).collect(),
        });
        match planned {
            Err(e) => {
                on_event(SpeakEvent::Error { message: format!("could not read \"{}\" aloud: {e}", job.text) });
                failed = true;
            }
            Ok(pieces) => {
                for work in pieces {
                    if cancelled(&shared, job.id) {
                        break;
                    }
                    let a = match work {
                        Work::Audio(a) => Ok(a),
                        Work::Text(piece) => synth.render(&piece, &live.voice, live.speed).map(|mut a| {
                            a.extend(std::iter::repeat_n(0.0, (piece.pause_after / live.speed * synth.rate() as f32) as usize));
                            a
                        }),
                    };
                    let a = match a {
                        Ok(a) => a,
                        Err(e) => {
                            on_event(SpeakEvent::Error { message: format!("speech synthesis failed: {e}") });
                            failed = true;
                            break;
                        }
                    };
                    let mut y = if synth.rate() == rate { a } else { crate::process::resample_sinc(&a, synth.rate(), rate) };
                    if first && live.keep_pauses {
                        if let (Some(gap), Some(end)) = (job.gap, speech_end) {
                            let lead = pause_to_add(&shared, end, gap, rate);
                            if lead > 0 {
                                y.splice(0..0, std::iter::repeat_n(0.0, lead));
                            }
                        }
                    }
                    p.process(&mut y);
                    if first {
                        let start = shared.written.load(Ordering::Relaxed);
                        let mut q = shared.queue.lock();
                        if let Some(m) = q.marks.iter_mut().find(|m| m.id == job.id) {
                            m.start = Some(start);
                            m.synth_ms = t_job.elapsed().as_secs_f32() * 1000.0;
                        }
                        first = false;
                    }
                    if !push_all(&shared, &mut outs, &y, job.id) {
                        break;
                    }
                }
            }
        }
        speech_end = if failed || first { speech_end } else { Some(shared.written.load(Ordering::Relaxed)) };
        // let an effect's tail ring out when nothing follows
        let idle = shared.queue.lock().jobs.is_empty();
        if !failed && idle && !first && live.effect.is_some() && !cancelled(&shared, job.id) {
            let mut tail = vec![0.0f32; (rate as f32 * 1.2) as usize];
            p.process(&mut tail);
            let _ = push_all(&shared, &mut outs, &tail, job.id);
        }
        let mut q = shared.queue.lock();
        let end = shared.written.load(Ordering::Relaxed);
        let stop_all = q.cancel_all;
        let skipped = q.cancel == Some(job.id) || stop_all;
        if let Some(m) = q.marks.iter_mut().find(|m| m.id == job.id) {
            if m.start.is_none() {
                m.start = Some(end);
            }
            m.end = Some(end);
            m.skipped |= skipped;
        }
        if skipped {
            shared.skip_to.fetch_max(end, Ordering::Relaxed);
            // a fresh processor: no tail of the cut line leaks into the next
            proc = None;
            speech_end = None;
        }
        q.current = None;
        q.cancel = None;
        q.cancel_all = false;
        drop(q);
    }
}

/// Keep my pauses: how much silence (output samples) to put before a mic
/// line so the output leaves the pause `gap` (s, capped at 2 s) you left on
/// the mic before it. Silence already queued after the last line's speech
/// (`speech_end`, a ring position; an effect's tail counts) and silence
/// already played since the output ran dry both count, so a line that took
/// longer to convert than the pause starts at once. The output then stays a
/// steady delay behind you instead of the delay showing up as pauses.
fn pause_to_add(shared: &Shared, speech_end: u64, gap: f32, rate: u32) -> usize {
    let want = (gap.clamp(0.0, MAX_KEPT_PAUSE_SECS) * rate as f32) as u64;
    let written = shared.written.load(Ordering::Acquire);
    let played = shared.played.load(Ordering::Relaxed);
    let queued_silence = written.saturating_sub(speech_end);
    let played_silence = if played >= written { shared.dry_samples.load(Ordering::Relaxed) } else { 0 };
    want.saturating_sub(queued_silence + played_silence) as usize
}

/// Longest pause Keep my pauses leaves (a longer one is a new start).
const MAX_KEPT_PAUSE_SECS: f32 = 2.0;

/// Push to every output ring, waiting (in 5 ms steps) while they are full.
/// False when the job was cancelled meanwhile.
fn push_all(shared: &Shared, outs: &mut [Producer], y: &[f32], id: u64) -> bool {
    let mut done = vec![0usize; outs.len()];
    loop {
        for (o, d) in outs.iter_mut().zip(done.iter_mut()) {
            if *d < y.len() {
                *d += o.push(&y[*d..]);
            }
        }
        let min = done.iter().copied().min().unwrap_or(y.len());
        if min >= y.len() {
            shared.written.fetch_add(y.len() as u64, Ordering::Release);
            return true;
        }
        {
            let q = shared.queue.lock();
            if q.cancel_all || q.cancel == Some(id) || shared.stop.load(Ordering::Relaxed) {
                // what was pushed counts as written; rings that got less are
                // padded with silence so every ring keeps the same positions
                let most = done.iter().copied().max().unwrap_or(0);
                for (o, d) in outs.iter_mut().zip(done.iter()) {
                    let mut left = most - d;
                    let pad = [0.0f32; 1024];
                    while left > 0 {
                        let k = o.push(&pad[..left.min(1024)]);
                        if k == 0 {
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        left -= k;
                    }
                }
                shared.written.fetch_add(most as u64, Ordering::Release);
                return false;
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Announce lines as the main output reaches them; levels at 10 Hz.
fn watch_loop(shared: Arc<Shared>, on_event: OnEvent) {
    let mut last_level = Instant::now();
    while !shared.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(10));
        let played = shared.played.load(Ordering::Relaxed);
        let mut events = Vec::new();
        {
            let mut q = shared.queue.lock();
            // point the output's watch at the first unannounced line
            if let Some(m) = q.marks.iter().find(|m| !m.announced) {
                if let Some(s) = m.start {
                    if shared.watch_start.load(Ordering::Relaxed) != s {
                        shared.watch_start.store(s, Ordering::Relaxed);
                        shared.reached_at_ns.store(0, Ordering::Relaxed);
                    }
                }
            }
            let skip_to = shared.skip_to.load(Ordering::Relaxed);
            let out_lat = shared.out_latency.get();
            for m in q.marks.iter_mut() {
                let Some(start) = m.start else { break };
                let empty = m.end == Some(start);
                if !m.announced && (played > start || empty || skip_to > start) {
                    m.announced = true;
                    if empty {
                        continue;
                    }
                    let reached_ns = shared.reached_at_ns.swap(0, Ordering::Relaxed);
                    let reached = if reached_ns > 0 { shared.epoch + Duration::from_nanos(reached_ns) } else { Instant::now() };
                    let latency_ms = m.eos.map(|e| (reached.saturating_duration_since(e).as_secs_f32() + out_lat) * 1000.0);
                    if let Some(l) = latency_ms {
                        shared.last_latency.set(l);
                        let mut s = shared.lat_sum.lock();
                        s.0 += l as f64;
                        s.1 += 1;
                    }
                    events.push(SpeakEvent::Speaking { id: m.id, text: m.text.clone(), latency_ms, synth_ms: m.synth_ms });
                }
            }
            while let Some(m) = q.marks.front() {
                match (m.start, m.end) {
                    (Some(_), Some(e)) if m.announced && (played >= e || skip_to >= e) => {
                        events.push(SpeakEvent::Done { id: m.id, skipped: m.skipped });
                        q.marks.pop_front();
                    }
                    _ => break,
                }
            }
        }
        for e in events {
            on_event(e);
        }
        if last_level.elapsed() >= Duration::from_millis(100) {
            last_level = Instant::now();
            let ip = shared.in_peak.take();
            let op = shared.out_peak.take();
            shared.in_ui.set(ip);
            shared.out_ui.set(op);
            on_event(SpeakEvent::Level {
                input_db: lin_to_db(ip),
                output_db: lin_to_db(op),
                listening: shared.hearing.load(Ordering::Relaxed),
                speaking: shared.speaking(),
            });
        }
    }
    on_event(SpeakEvent::Stopped);
}

// ───────────────────────── on real devices ─────────────────────────

#[cfg(feature = "player")]
pub use real::SpeakEngine;

#[cfg(feature = "player")]
mod real {
    use super::*;
    use crate::voice::devices;
    use cpal::traits::{DeviceTrait, StreamTrait};

    /// Speak for me on audio devices. Dropping it stops it.
    pub struct SpeakEngine {
        core: Option<SpeakCore>,
        _output: cpal::Stream,
        _monitor: Option<cpal::Stream>,
        #[cfg(target_os = "linux")]
        _vmic: Option<crate::voice::virtual_mic::linux::VirtualMic>,
    }

    // cpal streams are !Send on some hosts; the engine lives behind a mutex
    // in the app and is only ever touched from one thread at a time.
    unsafe impl Send for SpeakEngine {}

    impl SpeakEngine {
        /// Open the devices and start. `stt` = (transcriber, model name);
        /// None = type-to-speak only.
        pub fn start(
            cfg: SpeakConfig,
            synth: Arc<dyn Synth>,
            stt: Option<(Arc<dyn Transcriber>, String)>,
            on_event: impl Fn(SpeakEvent) + Send + Sync + 'static,
        ) -> Result<SpeakEngine> {
            #[cfg(target_os = "linux")]
            let mut vmic = None;
            let out_name: Option<String> = if cfg.virtual_mic {
                #[cfg(target_os = "linux")]
                {
                    vmic = Some(crate::voice::virtual_mic::linux::VirtualMic::create()?);
                    None
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let st = crate::voice::virtual_mic::virtual_mic_status();
                    match st.output_device {
                        Some(d) => Some(d),
                        None => return Err(Error::Output(st.instructions)),
                    }
                }
            } else {
                cfg.output.clone()
            };
            let (out_dev, out_name_shown) = devices::find(false, out_name.as_deref())?;
            let (out_fmt, out_cfg) = devices::choose_config(&out_dev, false, Some(48_000), 480).map_err(|e| Error::Output(format!("{out_name_shown}: {e}")))?;
            let main_rate = out_cfg.sample_rate;
            let mon = match cfg.monitor.as_ref() {
                Some(m) => {
                    let (d, n) = devices::find(false, if m.is_empty() { None } else { Some(m.as_str()) })?;
                    let (f, c) = devices::choose_config(&d, false, Some(main_rate), 480)?;
                    Some((d, n, f, c))
                }
                None => None,
            };
            let input = if cfg.listen && stt.is_some() {
                let src = crate::record::CpalCapture::open(crate::record::SourceRole::Mic, cfg.input.as_deref())?;
                Some(Box::new(src) as Box<dyn CaptureSource>)
            } else {
                None
            };
            let in_name = input.as_ref().map(|i| i.format().name);
            let rates = OutputRates { main: main_rate, monitor: mon.as_ref().map(|m| m.3.sample_rate) };
            let on_event: OnEvent = Arc::new(on_event);
            let ev = on_event.clone();
            let (core, mut taps) = SpeakCore::start(&cfg, synth, stt, input, rates, move |e| ev(e))?;
            let main_tap = taps.remove(0);
            let shared = core.shared.clone();
            let output = build(&out_dev, out_fmt, out_cfg, main_tap, shared.clone(), "output").map_err(|e| Error::Output(format!("{out_name_shown}: could not open the output ({e})")))?;
            output.play().map_err(|e| Error::Output(format!("{out_name_shown}: could not start the output ({e})")))?;
            #[cfg(target_os = "linux")]
            if cfg.virtual_mic {
                let before: Vec<u32> = Vec::new();
                for _ in 0..20 {
                    if crate::voice::virtual_mic::linux::VirtualMic::capture_new_streams(&before) > 0 {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            let (monitor, mon_name) = match (mon, taps.pop()) {
                (Some((d, n, f, c)), Some(tap)) => {
                    let s = build(&d, f, c, tap, shared.clone(), "monitor").map_err(|e| Error::Output(format!("{n}: could not open the monitor ({e})")))?;
                    s.play().map_err(|e| Error::Output(format!("{n}: could not start the monitor ({e})")))?;
                    (Some(s), Some(n))
                }
                _ => (None, None),
            };
            let shown = if cfg.virtual_mic {
                let st = crate::voice::virtual_mic::virtual_mic_status();
                st.mic_name.map(|m| format!("{out_name_shown} → {m}")).unwrap_or(out_name_shown)
            } else {
                out_name_shown
            };
            core.set_devices(in_name.clone(), shown.clone(), mon_name.clone());
            on_event(SpeakEvent::Started { input: in_name, output: shown, monitor: mon_name, sample_rate: main_rate });
            Ok(SpeakEngine {
                core: Some(core),
                _output: output,
                _monitor: monitor,
                #[cfg(target_os = "linux")]
                _vmic: vmic,
            })
        }

        pub fn core(&self) -> &SpeakCore {
            self.core.as_ref().expect("running")
        }

        pub fn stop(mut self) {
            if let Some(c) = self.core.take() {
                c.stop();
            }
        }
    }

    impl Drop for SpeakEngine {
        fn drop(&mut self) {
            if let Some(c) = self.core.take() {
                c.stop();
            }
        }
    }

    fn build(dev: &cpal::Device, fmt: cpal::SampleFormat, cfg: cpal::StreamConfig, tap: OutputTap, shared: Arc<Shared>, role: &'static str) -> std::result::Result<cpal::Stream, String> {
        macro_rules! go {
            ($t:ty) => {
                stream::<$t>(dev, cfg, tap, shared, role).map_err(|e| e.to_string())
            };
        }
        match fmt {
            cpal::SampleFormat::F32 => go!(f32),
            cpal::SampleFormat::I16 => go!(i16),
            cpal::SampleFormat::I32 => go!(i32),
            cpal::SampleFormat::U16 => go!(u16),
            cpal::SampleFormat::F64 => go!(f64),
            cpal::SampleFormat::U8 => go!(u8),
            cpal::SampleFormat::I8 => go!(i8),
            other => Err(format!("sample format {other} is not supported")),
        }
    }

    fn stream<T>(dev: &cpal::Device, cfg: cpal::StreamConfig, mut tap: OutputTap, shared: Arc<Shared>, role: &'static str) -> std::result::Result<cpal::Stream, cpal::Error>
    where
        T: cpal::SizedSample + cpal::FromSample<f32>,
    {
        let ch = cfg.channels.max(1) as usize;
        let mut mono = vec![0.0f32; 8192];
        let mut first = true;
        let fail = shared.clone();
        let main = role == "output";
        dev.build_output_stream::<T, _, _>(
            cfg,
            move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                if first {
                    first = false;
                    crate::threads::raise_current_thread_priority();
                }
                let frames = data.len() / ch;
                let mut done = 0;
                while done < frames {
                    let n = (frames - done).min(mono.len());
                    tap.fill(&mut mono[..n]);
                    for (f, m) in data[done * ch..(done + n) * ch].chunks_exact_mut(ch).zip(mono[..n].iter()) {
                        for o in f.iter_mut() {
                            *o = T::from_sample(*m);
                        }
                    }
                    done += n;
                }
                if main {
                    let ts = info.timestamp();
                    if let Some(d) = ts.playback.checked_duration_since(ts.callback) {
                        shared.out_latency.set(d.as_secs_f32());
                    }
                }
            },
            move |e: cpal::Error| {
                if matches!(e.kind(), cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::StreamInvalidated | cpal::ErrorKind::BackendError) {
                    *fail.stream_failed.lock() = Some((role, e.to_string()));
                }
            },
            None,
        )
    }
}

#[cfg(test)]
#[path = "speak_tests.rs"]
mod tests;
