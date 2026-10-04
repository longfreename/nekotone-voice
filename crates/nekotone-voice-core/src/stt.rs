//! Speech to text. The [`Transcriber`] trait is what the rest of the crate
//! uses; [`WhisperOnnx`] implements it with the Whisper encoder/decoder ONNX
//! models through ONNX Runtime (feature `ml`).
//!
//! How a transcription runs (the same steps as OpenAI's reference
//! `transcribe`, greedy):
//!
//! 1. The clip is resampled to 16 kHz and cut into 30 s windows. With
//!    `opts.ranges` (the classifier's speech ranges) only those stretches
//!    are windowed.
//! 2. Each window becomes an 80-bin log-mel (`features::whisper_mel`, or the
//!    private fallback here while that lands) and goes through the encoder
//!    once.
//! 3. The decoder runs token by token with its KV cache: the prompt is
//!    `<|startofprev|> …previous text… <|startoftranscript|> <|lang|>
//!    <|transcribe|>` (timestamps on), the language is detected from the
//!    first window when not forced, Whisper's timestamp rules and token
//!    suppression are applied to every step, and `<|nospeech|>`'s probability
//!    at the start-of-transcript position is kept as `no_speech_prob`.
//! 4. A window whose text compresses too well (repetition) or has a low
//!    average log-probability is decoded again at a higher temperature
//!    (0.2 … 1.0); silence is skipped.
//! 5. Timestamp tokens split the window into [`Segment`]s; the next window
//!    starts at the last complete segment's end so nothing is cut mid-sentence.
//! 6. Word times are interpolated across each segment by character count
//!    (the exported graphs expose no cross-attention, so no DTW).
//!
//! Owner: the `stt` work package.

use crate::audio::Clip;
#[cfg(any(feature = "ml", test))]
use crate::models::ModelManager;
use crate::{OnProgress, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// 0..1.
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub words: Vec<Word>,
    /// Probability that this window had no speech (Whisper's estimate).
    pub no_speech_prob: f32,
    /// Who is talking (0-based, from `diarize`); None when not labelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    /// ISO 639-1, e.g. "en".
    pub language: String,
    pub segments: Vec<Segment>,
    /// Model name, e.g. "whisper-base".
    pub model: String,
}

impl Transcript {
    /// Plain text, segments joined by spaces.
    pub fn text(&self) -> String {
        self.segments.iter().map(|s| s.text.trim()).filter(|t| !t.is_empty()).collect::<Vec<_>>().join(" ")
    }
    pub fn is_empty(&self) -> bool {
        self.segments.iter().all(|s| s.text.trim().is_empty())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Force a language (ISO 639-1); None = detect.
    pub language: Option<String>,
    /// Translate to English instead of transcribing.
    pub translate: bool,
    pub word_timestamps: bool,
    /// Vocabulary hint (names, places), as Whisper's initial prompt.
    pub initial_prompt: Option<String>,
    /// Only transcribe these ranges (from the classifier); None = whole clip.
    pub ranges: Option<Vec<(f32, f32)>>,
}

pub trait Transcriber: Send + Sync {
    fn name(&self) -> String;
    fn transcribe(&self, clip: &Clip, opts: &TranscribeOptions, progress: OnProgress) -> Result<Transcript>;
}

/// Export formats for a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Txt,
    Srt,
    Vtt,
    Json,
    /// Timed lyrics (`[mm:ss.xx]` lines), for players.
    Lrc,
    /// One line per word with times, tab-separated.
    Words,
}

impl Format {
    pub fn from_name(s: &str) -> Option<Format> {
        Some(match s.to_ascii_lowercase().as_str() {
            "txt" | "text" => Format::Txt,
            "srt" => Format::Srt,
            "vtt" => Format::Vtt,
            "json" => Format::Json,
            "lrc" => Format::Lrc,
            "words" | "tsv" => Format::Words,
            _ => return None,
        })
    }
    pub fn extension(self) -> &'static str {
        match self {
            Format::Txt => "txt",
            Format::Srt => "srt",
            Format::Vtt => "vtt",
            Format::Json => "json",
            Format::Lrc => "lrc",
            Format::Words => "tsv",
        }
    }
}

fn stamp(secs: f32, sep: char) -> String {
    let ms = (secs.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
}

/// Render a transcript in `format`.
pub fn export(t: &Transcript, format: Format) -> String {
    match format {
        Format::Txt => t.text(),
        Format::Json => serde_json::to_string_pretty(t).unwrap_or_default(),
        Format::Srt => t
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}\n{} --> {}\n{}\n", i + 1, stamp(s.start, ','), stamp(s.end, ','), s.text.trim()))
            .collect::<Vec<_>>()
            .join("\n"),
        Format::Vtt => {
            let body = t
                .segments
                .iter()
                .map(|s| format!("{} --> {}\n{}\n", stamp(s.start, '.'), stamp(s.end, '.'), s.text.trim()))
                .collect::<Vec<_>>()
                .join("\n");
            format!("WEBVTT\n\n{body}")
        }
        Format::Lrc => t
            .segments
            .iter()
            .map(|s| {
                let cs = (s.start.max(0.0) * 100.0).round() as u64;
                format!("[{:02}:{:02}.{:02}]{}", cs / 6000, (cs / 100) % 60, cs % 100, s.text.trim())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Format::Words => t
            .segments
            .iter()
            .flat_map(|s| s.words.iter())
            .map(|w| format!("{:.3}\t{:.3}\t{:.2}\t{}", w.start, w.end, w.prob, w.text.trim()))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

// ---------------------------------------------------------------------------
// Whisper's fixed numbers and the model-free parts of decoding. Everything in
// this section runs without ONNX Runtime so it can be unit-tested.
// ---------------------------------------------------------------------------

/// Whisper's input rate.
pub const WHISPER_RATE: u32 = 16_000;
/// One window: 30 s of 16 kHz audio, 3000 mel frames (hop 160).
const WINDOW_SECS: f32 = 30.0;
const N_SAMPLES: usize = 480_000;
const N_FRAMES: usize = 3000;
const MEL_HOP: usize = 160;
const MEL_FFT: usize = 400;
/// Seconds per timestamp token.
const TIME_PRECISION: f32 = 0.02;
/// Temperatures tried in turn when a window decodes badly.
const TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const COMPRESSION_RATIO_THRESHOLD: f32 = 2.4;
const LOGPROB_THRESHOLD: f32 = -1.0;
const NO_SPEECH_THRESHOLD: f32 = 0.6;

/// Whisper's stock phrases on near-silence ("Thank you.", "Thanks for
/// watching!"): a very short segment in a window the model itself found
/// unlikely to hold speech is dropped rather than indexed as words.
fn is_hallucination(text: &str, no_speech_prob: f32) -> bool {
    let t = text.trim().trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase();
    let words = t.split_whitespace().count();
    const STOCK: &[&str] = &["thank you", "thanks for watching", "thank you for watching", "you", "bye", "the end", "subtitles by the amara org community", "so"];
    (no_speech_prob > 0.3 && words <= 3) || STOCK.contains(&t.as_str()) && no_speech_prob > 0.15
}

/// ISO 639-1 codes in the order Whisper assigns its language tokens
/// (`<|en|>` = sot + 1, `<|zh|>` = sot + 2, …).
#[rustfmt::skip]
const LANGUAGE_CODES: [&str; 99] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi",
    "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
    "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
];

/// The special tokens of a Whisper vocabulary and the decoding limits that
/// go with them. Read from `generation_config.json`; the defaults are the
/// multilingual v1/v2 layout (51 865 tokens) that tiny…medium share.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Vocab {
    eot: u32,
    sot: u32,
    translate: u32,
    transcribe: u32,
    sot_prev: u32,
    no_speech: u32,
    no_timestamps: u32,
    timestamp_begin: u32,
    vocab_size: usize,
    /// (code, token id), e.g. ("en", 50259).
    langs: Vec<(String, u32)>,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    max_initial_timestamp_index: u32,
    /// Prompt + generated tokens per window.
    max_length: usize,
}

impl Default for Vocab {
    fn default() -> Self {
        let sot = 50258;
        Vocab {
            eot: 50257,
            sot,
            translate: 50358,
            transcribe: 50359,
            sot_prev: 50361,
            no_speech: 50362,
            no_timestamps: 50363,
            timestamp_begin: 50364,
            vocab_size: 51865,
            langs: LANGUAGE_CODES.iter().enumerate().map(|(i, c)| (c.to_string(), sot + 1 + i as u32)).collect(),
            suppress: vec![
                1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542,
                873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183,
                4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
                15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161, 26435, 28279, 29464,
                31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254, 50258, 50358, 50359, 50360, 50361, 50362,
            ],
            begin_suppress: vec![220, 50257],
            max_initial_timestamp_index: 50,
            max_length: 448,
        }
    }
}

impl Vocab {
    /// Build from the JSON of `generation_config.json` (and the vocabulary
    /// size from `config.json`). Missing keys keep the defaults.
    pub(crate) fn from_generation_config(gen: &serde_json::Value, vocab_size: Option<usize>) -> Vocab {
        let mut v = Vocab::default();
        let num = |k: &str| gen.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
        if let Some(x) = num("eos_token_id") {
            v.eot = x;
        }
        if let Some(x) = num("decoder_start_token_id") {
            v.sot = x;
        }
        if let Some(x) = num("prev_sot_token_id") {
            v.sot_prev = x;
        }
        if let Some(x) = num("no_timestamps_token_id") {
            v.no_timestamps = x;
            v.timestamp_begin = x + 1;
            v.no_speech = x - 1;
        }
        if let Some(t) = gen.get("task_to_id") {
            if let Some(x) = t.get("transcribe").and_then(|x| x.as_u64()) {
                v.transcribe = x as u32;
            }
            if let Some(x) = t.get("translate").and_then(|x| x.as_u64()) {
                v.translate = x as u32;
            }
        }
        if let Some(l) = gen.get("lang_to_id").and_then(|x| x.as_object()) {
            let mut langs: Vec<(String, u32)> = l
                .iter()
                .filter_map(|(k, id)| Some((k.trim_start_matches("<|").trim_end_matches("|>").to_string(), id.as_u64()? as u32)))
                .collect();
            langs.sort_by_key(|(_, id)| *id);
            if !langs.is_empty() {
                v.langs = langs;
            }
        }
        let list = |k: &str| -> Option<Vec<u32>> {
            Some(gen.get(k)?.as_array()?.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect())
        };
        if let Some(s) = list("suppress_tokens") {
            v.suppress = s;
        }
        if let Some(s) = list("begin_suppress_tokens") {
            v.begin_suppress = s;
        }
        if let Some(x) = num("max_initial_timestamp_index") {
            v.max_initial_timestamp_index = x;
        }
        if let Some(x) = num("max_length") {
            v.max_length = x as usize;
        }
        if let Some(n) = vocab_size {
            v.vocab_size = n;
        }
        v
    }
    fn is_timestamp(&self, id: u32) -> bool {
        id >= self.timestamp_begin
    }
    /// Seconds of a timestamp token, relative to the window.
    fn timestamp_secs(&self, id: u32) -> Option<f32> {
        self.is_timestamp(id).then(|| (id - self.timestamp_begin) as f32 / (1.0 / TIME_PRECISION))
    }
    fn lang_token(&self, code: &str) -> Option<u32> {
        let code = code.trim().to_ascii_lowercase();
        self.langs.iter().find(|(c, _)| *c == code).map(|(_, id)| *id)
    }
    fn lang_code(&self, id: u32) -> Option<&str> {
        self.langs.iter().find(|(_, i)| *i == id).map(|(c, _)| c.as_str())
    }
    /// Longest prefix (previous text / initial prompt) that fits before the
    /// start-of-transcript tokens: half the context minus the `sot_prev`.
    fn max_prefix_len(&self) -> usize {
        (self.max_length / 2).saturating_sub(1)
    }
}

/// The decoder prompt for one window: `[<|startofprev|> prefix…]
/// <|startoftranscript|> <|lang|> <|transcribe|>` (or `<|translate|>`).
/// Timestamps stay enabled (no `<|notimestamps|>`). Returns the tokens and
/// the index of `<|startoftranscript|>` (where `no_speech_prob` is read).
pub(crate) fn build_prompt(v: &Vocab, prefix: &[u32], lang: u32, translate: bool) -> (Vec<u32>, usize) {
    let mut out = Vec::with_capacity(prefix.len() + 4);
    if !prefix.is_empty() {
        out.push(v.sot_prev);
        let keep = v.max_prefix_len();
        let start = prefix.len().saturating_sub(keep);
        out.extend_from_slice(&prefix[start..]);
    }
    let sot_index = out.len();
    out.push(v.sot);
    out.push(lang);
    out.push(if translate { v.translate } else { v.transcribe });
    (out, sot_index)
}

/// Whisper's logit filters for one decoding step, in place: token
/// suppression, blank suppression at the first step, and the timestamp
/// rules (segments start with a timestamp, timestamps come in pairs and
/// never go backwards, and a step whose timestamp mass beats the best word
/// must emit a timestamp). `sampled` are the tokens generated so far in
/// this window (after the prompt).
pub(crate) fn apply_rules(logits: &mut [f32], v: &Vocab, sampled: &[u32]) {
    let n = logits.len();
    let tb = v.timestamp_begin as usize;
    let eot = v.eot as usize;
    let ninf = f32::NEG_INFINITY;
    for &t in &v.suppress {
        if (t as usize) < n {
            logits[t as usize] = ninf;
        }
    }
    if (v.no_timestamps as usize) < n {
        logits[v.no_timestamps as usize] = ninf;
    }
    if sampled.is_empty() {
        for &t in &v.begin_suppress {
            if (t as usize) < n {
                logits[t as usize] = ninf;
            }
        }
    }
    let last_was_ts = sampled.last().is_some_and(|&t| v.is_timestamp(t));
    let penultimate_was_ts = sampled.len() < 2 || v.is_timestamp(sampled[sampled.len() - 2]);
    if last_was_ts {
        if penultimate_was_ts {
            // Two timestamps in a row: the next token must be text.
            for l in logits.iter_mut().skip(tb) {
                *l = ninf;
            }
        } else {
            // A segment's end must follow: no ordinary text.
            for l in logits.iter_mut().take(eot.min(n)) {
                *l = ninf;
            }
        }
    }
    if let Some(&last_ts) = sampled.iter().rev().find(|&&t| v.is_timestamp(t)) {
        let mut floor = last_ts as usize;
        if !last_was_ts || penultimate_was_ts {
            floor += 1;
        }
        for l in logits.iter_mut().take(floor.min(n)).skip(tb) {
            *l = ninf;
        }
    }
    if sampled.is_empty() {
        // The first token is a timestamp, and an early one.
        for l in logits.iter_mut().take(tb.min(n)) {
            *l = ninf;
        }
        let last_allowed = tb + v.max_initial_timestamp_index as usize;
        for l in logits.iter_mut().skip(last_allowed + 1) {
            *l = ninf;
        }
    }
    // Timestamp mass versus the best text token.
    let lp = log_softmax(logits);
    let ts_mass = log_sum_exp(&lp[tb.min(n)..]);
    let best_text = lp[..tb.min(n)].iter().copied().fold(ninf, f32::max);
    if ts_mass > best_text {
        for l in logits.iter_mut().take(tb.min(n)) {
            *l = ninf;
        }
    }
}

fn log_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![f32::NEG_INFINITY; x.len()];
    }
    let lse = max + x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    x.iter().map(|&v| v - lse).collect()
}

fn log_sum_exp(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return f32::NEG_INFINITY;
    }
    max + x.iter().map(|&v| (v - max).exp()).sum::<f32>().ln()
}

/// A tiny deterministic generator for temperature sampling (xorshift64*).
pub(crate) struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Pick the next token from filtered logits: argmax at temperature 0,
/// otherwise a draw from the tempered distribution. Returns the token and
/// its log-probability under the (untempered) filtered distribution.
pub(crate) fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> (u32, f32) {
    let lp = log_softmax(logits);
    let token = if temperature <= 0.0 {
        argmax(logits)
    } else {
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
        let p = log_softmax(&scaled);
        let mut r = rng.next_f32();
        let mut pick = argmax(logits);
        for (i, &l) in p.iter().enumerate() {
            let q = l.exp();
            if r < q {
                pick = i;
                break;
            }
            r -= q;
        }
        pick
    };
    (token as u32, lp[token])
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// One window's decoded tokens split at timestamp pairs, times relative
/// to the window start.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RawSegment {
    pub start: f32,
    pub end: f32,
    /// Text tokens only (no timestamps, no specials).
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f32>,
}

/// Split a window's sampled tokens into segments the way Whisper does and
/// say how far to advance (seconds) before the next window: to the end of
/// the last complete segment, or the whole window when the window ended
/// mid-segment or without timestamps. `window_secs` is the audio actually
/// in the window (≤ 30).
pub(crate) fn split_segments(v: &Vocab, tokens: &[u32], logprobs: &[f32], window_secs: f32) -> (Vec<RawSegment>, f32) {
    let is_ts: Vec<bool> = tokens.iter().map(|&t| v.is_timestamp(t)).collect();
    let n = tokens.len();
    let lp = |i: usize| logprobs.get(i).copied().unwrap_or(0.0);
    let text_of = |range: std::ops::Range<usize>| -> (Vec<u32>, Vec<f32>) {
        let mut toks = Vec::new();
        let mut lps = Vec::new();
        for i in range {
            if tokens[i] < v.eot {
                toks.push(tokens[i]);
                lps.push(lp(i));
            }
        }
        (toks, lps)
    };
    let mut segs = Vec::new();
    let consecutive: Vec<usize> = (1..n).filter(|&i| is_ts[i - 1] && is_ts[i]).collect();
    let single_ts_ending = n >= 2 && !is_ts[n - 2] && is_ts[n - 1];
    if !consecutive.is_empty() {
        let mut slices = consecutive.clone();
        if single_ts_ending {
            slices.push(n);
        }
        let mut last = 0;
        for cut in slices {
            let sl = &tokens[last..cut];
            let start = sl.first().and_then(|&t| v.timestamp_secs(t)).unwrap_or(0.0);
            let end = sl.last().and_then(|&t| v.timestamp_secs(t)).unwrap_or(window_secs);
            let (toks, lps) = text_of(last..cut);
            // Timestamp pairs with nothing between them carry no words.
            if !toks.is_empty() {
                segs.push(RawSegment { start, end: end.max(start), tokens: toks, logprobs: lps });
            }
            last = cut;
        }
        let advance = if single_ts_ending {
            window_secs
        } else {
            tokens.get(last.wrapping_sub(1)).and_then(|&t| v.timestamp_secs(t)).unwrap_or(window_secs)
        };
        (segs, advance)
    } else {
        let mut duration = window_secs;
        if let Some(&t) = tokens.iter().rev().find(|&&t| v.is_timestamp(t)) {
            if let Some(s) = v.timestamp_secs(t) {
                if s > 0.0 {
                    duration = s;
                }
            }
        }
        let (toks, lps) = text_of(0..n);
        if !toks.is_empty() {
            segs.push(RawSegment { start: 0.0, end: duration, tokens: toks, logprobs: lps });
        }
        (segs, window_secs)
    }
}

/// Group text tokens into words using a per-token decoder (byte-level BPE
/// can split a character over several tokens, so tokens are joined until
/// they decode cleanly), then spread `start..end` over the words by
/// character count. Word probability is the mean token probability.
pub(crate) fn words_by_interpolation(
    tokens: &[u32],
    logprobs: &[f32],
    start: f32,
    end: f32,
    decode: &dyn Fn(&[u32]) -> String,
) -> Vec<Word> {
    // 1. Tokens → clean units.
    let mut units: Vec<(String, Vec<f32>)> = Vec::new();
    let mut pending: Vec<u32> = Vec::new();
    let mut pending_lp: Vec<f32> = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        pending.push(t);
        pending_lp.push(logprobs.get(i).copied().unwrap_or(0.0));
        let s = decode(&pending);
        if !s.contains('\u{FFFD}') {
            units.push((s, std::mem::take(&mut pending_lp)));
            pending.clear();
        }
    }
    if !pending.is_empty() {
        units.push((decode(&pending), pending_lp));
    }
    // 2. Units → words (a unit that starts with a space starts a word; bare
    //    punctuation sticks to the previous word).
    let mut words: Vec<(String, Vec<f32>)> = Vec::new();
    for (s, lps) in units {
        if s.is_empty() {
            continue;
        }
        let starts_word = s.starts_with(' ') || words.is_empty();
        let punct_only = s.trim().chars().all(|c| c.is_ascii_punctuation()) && !s.trim().is_empty();
        if starts_word && (!punct_only || words.is_empty()) {
            words.push((s, lps));
        } else if let Some(last) = words.last_mut() {
            last.0.push_str(&s);
            last.1.extend(lps);
        } else {
            words.push((s, lps));
        }
    }
    // 3. Times by character share.
    let total: usize = words.iter().map(|(w, _)| w.trim().chars().count().max(1)).sum();
    let span = (end - start).max(0.0);
    let mut t = start;
    let mut out = Vec::with_capacity(words.len());
    for (w, lps) in words {
        let n = w.trim().chars().count().max(1);
        let dur = if total > 0 { span * n as f32 / total as f32 } else { 0.0 };
        let prob = if lps.is_empty() { 0.0 } else { (lps.iter().sum::<f32>() / lps.len() as f32).exp().clamp(0.0, 1.0) };
        out.push(Word { text: w.trim().to_string(), start: t, end: t + dur, prob });
        t += dur;
    }
    out
}

/// A stand-in for zlib's compression ratio (Whisper's repetition test):
/// bytes in / bytes of a greedy LZ77 parse. Repeated phrases give big
/// ratios; ordinary prose stays near 1.5–2.
pub(crate) fn compression_ratio(text: &str) -> f32 {
    let b = text.as_bytes();
    if b.is_empty() {
        return 1.0;
    }
    let mut i = 0;
    let mut cost = 0usize;
    while i < b.len() {
        let mut best_len = 0;
        let window_start = i.saturating_sub(4096);
        let mut j = window_start;
        while j < i {
            let mut l = 0;
            while i + l < b.len() && b[j + l] == b[i + l] && l < 255 {
                l += 1;
            }
            if l > best_len {
                best_len = l;
            }
            j += 1;
        }
        if best_len >= 4 {
            cost += 3; // (offset, length) pair
            i += best_len;
        } else {
            cost += 1;
            i += 1;
        }
    }
    b.len() as f32 / cost.max(1) as f32
}

/// Whisper's fixed 80-bin mel filter bank (librosa `mel(sr=16000,
/// n_fft=400, n_mels=80)`: Slaney scale, Slaney-normalised), as
/// `n_mels × 201`.
fn mel_filters(n_mels: usize) -> Vec<Vec<f32>> {
    let n_bins = MEL_FFT / 2 + 1;
    let sr = WHISPER_RATE as f64;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |hz: f64| if hz < min_log_hz { hz / f_sp } else { min_log_mel + (hz / min_log_hz).ln() / logstep };
    let mel_to_hz = |mel: f64| if mel < min_log_mel { mel * f_sp } else { min_log_hz * ((mel - min_log_mel) * logstep).exp() };
    let mel_max = hz_to_mel(sr / 2.0);
    let pts: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(mel_max * i as f64 / (n_mels + 1) as f64)).collect();
    let fft_freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / MEL_FFT as f64).collect();
    (0..n_mels)
        .map(|m| {
            let (lo, mid, hi) = (pts[m], pts[m + 1], pts[m + 2]);
            let enorm = 2.0 / (hi - lo);
            fft_freqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (mid - lo).max(1e-9);
                    let upper = (hi - f) / (hi - mid).max(1e-9);
                    (lower.min(upper).max(0.0) * enorm) as f32
                })
                .collect()
        })
        .collect()
}

/// Whisper log-mel of exactly 30 s of 16 kHz audio (`samples` is padded or
/// cut to 480 000), as `3000 × n_mels`. Used until `features::whisper_mel`
/// lands; the maths follow `whisper.audio.log_mel_spectrogram` (periodic
/// Hann, centred reflect-padded STFT, power, log10 floor 1e-10, clamp to
/// max − 8, then `(x + 4) / 4`).
pub(crate) fn fallback_mel(samples: &[f32], n_mels: usize) -> Vec<Vec<f32>> {
    use realfft::RealFftPlanner;
    let mut audio = samples.to_vec();
    audio.resize(N_SAMPLES, 0.0);
    let half = MEL_FFT / 2;
    let mut padded = Vec::with_capacity(N_SAMPLES + MEL_FFT);
    padded.extend((1..=half).rev().map(|i| audio[i]));
    padded.extend_from_slice(&audio);
    padded.extend((1..=half).map(|i| audio[N_SAMPLES - 1 - i]));
    let window: Vec<f32> = (0..MEL_FFT).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / MEL_FFT as f32).cos()).collect();
    let filters = mel_filters(n_mels);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(MEL_FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut power = vec![0f32; half + 1];
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(N_FRAMES);
    let mut max_val = f32::NEG_INFINITY;
    for f in 0..N_FRAMES {
        let off = f * MEL_HOP;
        for i in 0..MEL_FFT {
            input[i] = padded[off + i] * window[i];
        }
        if fft.process(&mut input, &mut spectrum).is_err() {
            spectrum.iter_mut().for_each(|c| *c = realfft::num_complex::Complex::new(0.0, 0.0));
        }
        for (p, c) in power.iter_mut().zip(spectrum.iter()) {
            *p = c.norm_sqr();
        }
        let row: Vec<f32> = filters
            .iter()
            .map(|w| {
                let e: f32 = w.iter().zip(power.iter()).map(|(a, b)| a * b).sum();
                let v = e.max(1e-10).log10();
                if v > max_val {
                    max_val = v;
                }
                v
            })
            .collect();
        frames.push(row);
    }
    let floor = max_val - 8.0;
    for row in &mut frames {
        for v in row.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
    }
    frames
}

/// Merge overlapping or touching ranges, clamp to the clip, drop empties.
fn normalise_ranges(ranges: Option<&[(f32, f32)]>, duration: f32) -> Vec<(f32, f32)> {
    let mut r: Vec<(f32, f32)> = match ranges {
        None => vec![(0.0, duration)],
        Some(rs) => rs
            .iter()
            .map(|&(a, b)| (a.max(0.0).min(duration), b.max(0.0).min(duration)))
            .filter(|&(a, b)| b - a > 0.05)
            .collect(),
    };
    r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<(f32, f32)> = Vec::new();
    for (a, b) in r {
        match out.last_mut() {
            Some(last) if a <= last.1 + 0.5 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The ONNX Runtime engine.
// ---------------------------------------------------------------------------

/// Whisper through ONNX Runtime.
#[cfg(feature = "ml")]
pub struct WhisperOnnx {
    id: crate::models::ModelId,
    encoder: parking_lot::Mutex<ort::session::Session>,
    decoder: parking_lot::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    vocab: Vocab,
    layers: usize,
    heads: usize,
    head_dim: usize,
    n_mels: usize,
    /// `past_key_values.{l}.decoder.key`, `.value`, `.encoder.key`, `.value` per layer.
    past_names: Vec<[String; 4]>,
    present_names: Vec<[String; 4]>,
}

#[cfg(feature = "ml")]
mod engine {
    use super::*;
    use crate::models::{ModelId, WHISPER_CONFIG, WHISPER_DECODER, WHISPER_ENCODER, WHISPER_GENERATION_CONFIG};
    use crate::{Error, Progress};
    use ort::session::{Session, SessionInputValue};
    use ort::value::{DynValue, Tensor};
    use std::borrow::Cow;

    fn ml_err(what: &str, e: ort::Error) -> Error {
        Error::Model(format!("Whisper {what}: {e}"))
    }

    /// The decoder's key/value cache between steps.
    struct Cache {
        /// Per layer: decoder key, decoder value (grow by one each step).
        decoder: Vec<[DynValue; 2]>,
        /// Per layer: encoder key, encoder value (fixed after step one).
        encoder: Vec<[DynValue; 2]>,
    }

    /// What one pass over a window produced.
    pub(super) struct WindowDecode {
        pub tokens: Vec<u32>,
        pub logprobs: Vec<f32>,
        pub no_speech_prob: f32,
        pub avg_logprob: f32,
        pub temperature: f32,
    }

    impl WhisperOnnx {
        /// Load the model `id` (a `ModelId::Whisper*`) from the manager; errors
        /// with `Error::ModelMissing` when it is not downloaded.
        pub fn load(models: &ModelManager, id: ModelId) -> Result<Self> {
            if !id.is_whisper() {
                return Err(Error::Model(format!("{} is not a speech-to-text model; pick whisper-tiny, -base, -small or -medium", id.name())));
            }
            let dir = models.path(id).ok_or_else(|| Error::ModelMissing { model: id.name().to_string() })?;
            let read_json = |name: &str| -> Result<serde_json::Value> {
                let text = std::fs::read_to_string(dir.join(name))
                    .map_err(|e| Error::Model(format!("could not read {} of {} ({e}); download the model again", name, id.name())))?;
                serde_json::from_str(&text).map_err(|e| Error::Model(format!("{} of {} is not valid JSON ({e}); download the model again", name, id.name())))
            };
            let config = read_json(WHISPER_CONFIG)?;
            let gen = read_json(WHISPER_GENERATION_CONFIG)?;
            let num = |k: &str, default: usize| config.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(default);
            let layers = num("decoder_layers", 6);
            let heads = num("decoder_attention_heads", 8);
            let d_model = num("d_model", 512);
            let n_mels = num("num_mel_bins", crate::models::whisper_mel_bins(id));
            let vocab = Vocab::from_generation_config(&gen, config.get("vocab_size").and_then(|v| v.as_u64()).map(|v| v as usize));
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join(crate::models::WHISPER_TOKENIZER))
                .map_err(|e| Error::Model(format!("could not load the tokenizer of {} ({e}); download the model again", id.name())))?;
            let encoder = crate::models::onnx_session(&dir.join(WHISPER_ENCODER))?;
            let decoder = crate::models::onnx_session(&dir.join(WHISPER_DECODER))?;
            let names = |prefix: &str| -> Vec<[String; 4]> {
                (0..layers)
                    .map(|l| {
                        [
                            format!("{prefix}.{l}.decoder.key"),
                            format!("{prefix}.{l}.decoder.value"),
                            format!("{prefix}.{l}.encoder.key"),
                            format!("{prefix}.{l}.encoder.value"),
                        ]
                    })
                    .collect()
            };
            Ok(WhisperOnnx {
                id,
                encoder: parking_lot::Mutex::new(encoder),
                decoder: parking_lot::Mutex::new(decoder),
                tokenizer,
                vocab,
                layers,
                heads,
                head_dim: d_model / heads.max(1),
                n_mels,
                past_names: names("past_key_values"),
                present_names: names("present"),
            })
        }

        pub fn model_id(&self) -> ModelId {
            self.id
        }

        fn decode_text(&self, tokens: &[u32]) -> String {
            let text: Vec<u32> = tokens.iter().copied().filter(|&t| t < self.vocab.eot).collect();
            self.tokenizer.decode(&text, true).unwrap_or_default()
        }

        fn encode_text(&self, text: &str) -> Vec<u32> {
            self.tokenizer.encode(text, false).map(|e| e.get_ids().to_vec()).unwrap_or_default()
        }

        /// 30 s of 16 kHz audio → encoder hidden states `[1, 1500, d_model]`.
        fn encode_window(&self, samples: &[f32]) -> Result<DynValue> {
            let mut clip = Clip { samples: samples.to_vec(), sample_rate: WHISPER_RATE, source_channels: 1 };
            clip.samples.resize(N_SAMPLES, 0.0);
            let mel = match crate::features::whisper_mel(&clip, self.n_mels) {
                Ok(m) if m.len() >= N_FRAMES.min(1) && m.iter().all(|r| r.len() == self.n_mels) => m,
                Ok(_) | Err(Error::NotImplemented(_)) => fallback_mel(&clip.samples, self.n_mels),
                Err(e) => return Err(e),
            };
            // frames × mels → [1, mels, 3000], padded or cut to 3000 frames.
            let mut data = vec![0f32; self.n_mels * N_FRAMES];
            let pad_value = mel.iter().flatten().copied().fold(f32::INFINITY, f32::min);
            for m in 0..self.n_mels {
                for f in 0..N_FRAMES {
                    data[m * N_FRAMES + f] = mel.get(f).map(|row| row[m]).unwrap_or(pad_value);
                }
            }
            let input = Tensor::from_array(([1usize, self.n_mels, N_FRAMES], data)).map_err(|e| ml_err("encoder input", e))?;
            let mut enc = self.encoder.lock();
            let mut out = enc.run(ort::inputs!["input_features" => input]).map_err(|e| ml_err("encoder", e))?;
            out.remove("last_hidden_state").ok_or_else(|| Error::Model("the Whisper encoder returned no `last_hidden_state`; the model file is not a Whisper encoder".into()))
        }

        /// One decoder pass. Without a cache `ids` is the whole prompt; with
        /// one, `ids` is the single new token. Returns the logits of every
        /// position (`n × vocab`) and the cache for the next step.
        fn decoder_step(&self, sess: &mut Session, ids: &[u32], enc: &DynValue, cache: Option<Cache>) -> Result<(Vec<Vec<f32>>, Cache)> {
            let n = ids.len();
            let id_tensor = Tensor::from_array(([1usize, n], ids.iter().map(|&t| t as i64).collect::<Vec<i64>>())).map_err(|e| ml_err("decoder input", e))?;
            let flag = Tensor::from_array(([1usize], vec![cache.is_some()])).map_err(|e| ml_err("decoder input", e))?;
            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(3 + 4 * self.layers);
            inputs.push(("input_ids".into(), id_tensor.into()));
            inputs.push(("encoder_hidden_states".into(), SessionInputValue::from(enc)));
            inputs.push(("use_cache_branch".into(), flag.into()));
            let old_encoder = match cache {
                None => {
                    for l in 0..self.layers {
                        for k in 0..4 {
                            let empty = Tensor::<f32>::from_array(([1usize, self.heads, 0, self.head_dim], Vec::new())).map_err(|e| ml_err("decoder input", e))?;
                            inputs.push((self.past_names[l][k].as_str().into(), empty.into()));
                        }
                    }
                    None
                }
                Some(c) => {
                    for (l, [k, v]) in c.decoder.into_iter().enumerate() {
                        inputs.push((self.past_names[l][0].as_str().into(), k.into()));
                        inputs.push((self.past_names[l][1].as_str().into(), v.into()));
                    }
                    Some(c.encoder)
                }
            };
            if let Some(encoder_kv) = &old_encoder {
                for (l, [k, v]) in encoder_kv.iter().enumerate() {
                    inputs.push((self.past_names[l][2].as_str().into(), SessionInputValue::from(k)));
                    inputs.push((self.past_names[l][3].as_str().into(), SessionInputValue::from(v)));
                }
            }
            let mut out = sess.run(inputs).map_err(|e| ml_err("decoder", e))?;
            let logits = {
                let view = out["logits"].try_extract_array::<f32>().map_err(|e| ml_err("decoder output", e))?;
                let shape = view.shape().to_vec();
                if shape.len() != 3 || shape[1] != n {
                    return Err(Error::Model(format!("the Whisper decoder returned logits of shape {shape:?}; expected [1, {n}, vocab]")));
                }
                (0..n)
                    .map(|i| view.index_axis(ndarray::Axis(0), 0).index_axis(ndarray::Axis(0), i).iter().copied().collect::<Vec<f32>>())
                    .collect::<Vec<Vec<f32>>>()
            };
            let mut take = |name: &str| -> Result<DynValue> {
                out.remove(name).ok_or_else(|| Error::Model(format!("the Whisper decoder returned no `{name}`; the model file is not a merged Whisper decoder")))
            };
            let mut decoder = Vec::with_capacity(self.layers);
            for l in 0..self.layers {
                decoder.push([take(&self.present_names[l][0])?, take(&self.present_names[l][1])?]);
            }
            let encoder = match old_encoder {
                Some(e) => e,
                None => {
                    let mut e = Vec::with_capacity(self.layers);
                    for l in 0..self.layers {
                        e.push([take(&self.present_names[l][2])?, take(&self.present_names[l][3])?]);
                    }
                    e
                }
            };
            Ok((logits, Cache { decoder, encoder }))
        }

        /// The language token with the highest probability right after
        /// `<|startoftranscript|>`.
        fn detect_language(&self, sess: &mut Session, enc: &DynValue) -> Result<u32> {
            let (logits, _) = self.decoder_step(sess, &[self.vocab.sot], enc, None)?;
            let row = &logits[0];
            let mut best = (f32::NEG_INFINITY, self.vocab.lang_token("en").unwrap_or(self.vocab.sot + 1));
            for &(_, id) in &self.vocab.langs {
                if let Some(&l) = row.get(id as usize) {
                    if l > best.0 {
                        best = (l, id);
                    }
                }
            }
            Ok(best.1)
        }

        /// Greedy (or tempered) decoding of one window with the KV cache.
        fn decode_window(&self, sess: &mut Session, enc: &DynValue, prompt: &[u32], sot_index: usize, temperature: f32, seed: u64) -> Result<WindowDecode> {
            let (logits, mut cache) = self.decoder_step(sess, prompt, enc, None)?;
            let no_speech_prob = {
                let lp = log_softmax(&logits[sot_index]);
                lp.get(self.vocab.no_speech as usize).map(|l| l.exp()).unwrap_or(0.0)
            };
            let mut current = logits.into_iter().next_back().unwrap_or_default();
            let mut rng = Rng::new(seed);
            let mut tokens: Vec<u32> = Vec::new();
            let mut logprobs: Vec<f32> = Vec::new();
            let mut sum_logprob = 0.0f32;
            let budget = self.vocab.max_length.saturating_sub(prompt.len());
            while tokens.len() < budget {
                apply_rules(&mut current, &self.vocab, &tokens);
                let (tok, lp) = sample(&current, temperature, &mut rng);
                sum_logprob += lp;
                if tok == self.vocab.eot {
                    break;
                }
                tokens.push(tok);
                logprobs.push(lp);
                if tokens.len() >= budget {
                    break;
                }
                let (next, new_cache) = self.decoder_step(sess, &[tok], enc, Some(cache))?;
                cache = new_cache;
                current = next.into_iter().next_back().unwrap_or_default();
            }
            let avg_logprob = sum_logprob / (tokens.len() as f32 + 1.0);
            Ok(WindowDecode { tokens, logprobs, no_speech_prob, avg_logprob, temperature })
        }

        /// Decode with temperature fallback. `Ok(None)` = the window is silence.
        fn decode_with_fallback(&self, sess: &mut Session, enc: &DynValue, prefix: &[u32], lang: u32, translate: bool, seed: u64) -> Result<Option<WindowDecode>> {
            let mut last: Option<WindowDecode> = None;
            for &t in &TEMPERATURES {
                // Above 0.5 the previous text is dropped from the prompt (it may be what misled the model).
                let (prompt, sot_index) = build_prompt(&self.vocab, if t > 0.5 { &[] } else { prefix }, lang, translate);
                let d = self.decode_window(sess, enc, &prompt, sot_index, t, seed)?;
                let text = self.decode_text(&d.tokens);
                let mut needs_fallback = false;
                if compression_ratio(&text) > COMPRESSION_RATIO_THRESHOLD {
                    needs_fallback = true;
                }
                if d.avg_logprob < LOGPROB_THRESHOLD {
                    needs_fallback = true;
                }
                if d.no_speech_prob > NO_SPEECH_THRESHOLD && d.avg_logprob < LOGPROB_THRESHOLD {
                    // Silence: nothing to retry.
                    return Ok(None);
                }
                last = Some(d);
                if !needs_fallback {
                    break;
                }
            }
            Ok(last)
        }

        pub(super) fn run(&self, clip: &Clip, opts: &TranscribeOptions, progress: OnProgress) -> Result<Transcript> {
            if clip.sample_rate == 0 || clip.samples.is_empty() {
                return Ok(Transcript { language: opts.language.clone().unwrap_or_default(), segments: vec![], model: self.id.name().into() });
            }
            let clip16k = if clip.sample_rate == WHISPER_RATE { clip.clone() } else { crate::audio::resample(clip, WHISPER_RATE)? };
            let duration = clip16k.duration_secs();
            let ranges = normalise_ranges(opts.ranges.as_deref(), duration);
            let total_secs: f32 = ranges.iter().map(|(a, b)| b - a).sum::<f32>().max(0.01);
            let n_windows_est = ranges.iter().map(|(a, b)| ((b - a) / WINDOW_SECS).ceil().max(1.0) as usize).sum::<usize>();
            let initial_tokens: Vec<u32> = match opts.initial_prompt.as_deref().map(str::trim) {
                Some(p) if !p.is_empty() => self.encode_text(&format!(" {p}")),
                _ => Vec::new(),
            };
            let forced_lang = match opts.language.as_deref().map(str::trim) {
                Some(code) if !code.is_empty() => Some(
                    self.vocab
                        .lang_token(code)
                        .ok_or_else(|| Error::Model(format!("Whisper does not know the language code `{code}`; use an ISO 639-1 code such as en, de or ja")))?,
                ),
                _ => None,
            };
            let mut lang: Option<u32> = forced_lang;
            let mut segments: Vec<Segment> = Vec::new();
            let mut previous_tokens: Vec<u32> = Vec::new();
            let mut done_secs = 0.0f32;
            let mut window_no = 0usize;
            let sr = WHISPER_RATE as f32;
            let mut decoder = self.decoder.lock();
            for &(range_start, range_end) in &ranges {
                let mut seek = range_start;
                while seek < range_end - 0.05 {
                    let win_end = (seek + WINDOW_SECS).min(range_end);
                    let a = (seek * sr) as usize;
                    let b = ((win_end * sr) as usize).min(clip16k.samples.len()).max(a);
                    let window = &clip16k.samples[a..b];
                    let window_secs = (b - a) as f32 / sr;
                    window_no += 1;
                    progress(Progress::new(
                        format!("Transcribing {:.0}–{:.0} s of {:.0} s (window {window_no}/{n_windows_est})", seek, win_end, duration),
                        Some((done_secs / total_secs).clamp(0.0, 1.0)),
                    ));
                    let mut advance = window_secs.max(0.5);
                    let silent = window.iter().all(|s| s.abs() < 1e-5);
                    if !silent {
                        let enc = self.encode_window(window)?;
                        let lang_id = match lang {
                            Some(l) => l,
                            None => {
                                let l = self.detect_language(&mut decoder, &enc)?;
                                lang = Some(l);
                                l
                            }
                        };
                        let mut prefix = initial_tokens.clone();
                        prefix.extend_from_slice(&previous_tokens);
                        if let Some(d) = self.decode_with_fallback(&mut decoder, &enc, &prefix, lang_id, opts.translate, window_no as u64)? {
                            let (raw, adv) = split_segments(&self.vocab, &d.tokens, &d.logprobs, window_secs);
                            advance = if adv.is_finite() && adv > 0.05 { adv.min(window_secs.max(0.5)) } else { window_secs.max(0.5) };
                            if d.temperature > 0.5 {
                                previous_tokens.clear();
                            }
                            let mut window_text_tokens = Vec::new();
                            for r in raw {
                                let text = self.decode_text(&r.tokens);
                                if text.trim().is_empty() || is_hallucination(&text, d.no_speech_prob) {
                                    continue;
                                }
                                let start = seek + r.start;
                                let end = (seek + r.end).min(seek + window_secs).max(start);
                                let words = if opts.word_timestamps {
                                    words_by_interpolation(&r.tokens, &r.logprobs, start, end, &|ids| self.decode_text(ids))
                                } else {
                                    Vec::new()
                                };
                                window_text_tokens.extend_from_slice(&r.tokens);
                                segments.push(Segment { start, end, text, words, no_speech_prob: d.no_speech_prob, speaker: None });
                            }
                            previous_tokens.extend(window_text_tokens);
                            let keep = self.vocab.max_prefix_len();
                            if previous_tokens.len() > keep {
                                previous_tokens.drain(..previous_tokens.len() - keep);
                            }
                        }
                    }
                    seek += advance;
                    done_secs += advance;
                }
            }
            progress(Progress::new("Transcription done", Some(1.0)));
            let language = lang.and_then(|l| self.vocab.lang_code(l)).unwrap_or("").to_string();
            Ok(Transcript { language, segments, model: self.id.name().into() })
        }
    }
}

#[cfg(feature = "ml")]
impl Transcriber for WhisperOnnx {
    fn name(&self) -> String {
        self.id.name().into()
    }
    fn transcribe(&self, clip: &Clip, opts: &TranscribeOptions, progress: OnProgress) -> Result<Transcript> {
        self.run(clip, opts, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Transcript {
        Transcript {
            language: "en".into(),
            model: "test".into(),
            segments: vec![
                Segment { start: 0.0, end: 1.5, text: " Hello there.".into(), words: vec![], no_speech_prob: 0.0, speaker: None },
                Segment { start: 1.5, end: 3.25, text: "General.".into(), words: vec![], no_speech_prob: 0.0, speaker: None },
            ],
        }
    }

    #[test]
    fn exports() {
        assert_eq!(export(&t(), Format::Txt), "Hello there. General.");
        assert!(export(&t(), Format::Srt).starts_with("1\n00:00:00,000 --> 00:00:01,500\nHello there."));
        assert!(export(&t(), Format::Vtt).starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:01.500"));
        assert_eq!(export(&t(), Format::Lrc).lines().nth(1), Some("[00:01.50]General."));
    }

    #[test]
    fn vocab_defaults_match_whisper_multilingual() {
        let v = Vocab::default();
        assert_eq!(v.lang_token("en"), Some(50259));
        assert_eq!(v.lang_token("ja"), Some(50266));
        assert_eq!(v.lang_token("su"), Some(50357));
        assert_eq!(v.lang_code(50261), Some("de"));
        assert_eq!(v.timestamp_secs(50364), Some(0.0));
        assert_eq!(v.timestamp_secs(51864), Some(30.0));
        assert_eq!(v.timestamp_secs(50257), None);
        assert_eq!(v.max_prefix_len(), 223);
    }

    #[test]
    fn vocab_from_generation_config() {
        let gen = serde_json::json!({
            "decoder_start_token_id": 50258, "eos_token_id": 50257, "no_timestamps_token_id": 50363,
            "prev_sot_token_id": 50361, "max_length": 448, "max_initial_timestamp_index": 50,
            "task_to_id": {"transcribe": 50359, "translate": 50358},
            "lang_to_id": {"<|en|>": 50259, "<|fr|>": 50265, "<|de|>": 50261},
            "suppress_tokens": [1, 2, 3], "begin_suppress_tokens": [220, 50257]
        });
        let v = Vocab::from_generation_config(&gen, Some(51865));
        assert_eq!(v.timestamp_begin, 50364);
        assert_eq!(v.no_speech, 50362);
        assert_eq!(v.langs, vec![("en".to_string(), 50259), ("de".to_string(), 50261), ("fr".to_string(), 50265)]);
        assert_eq!(v.suppress, vec![1, 2, 3]);
        assert_eq!(v.vocab_size, 51865);
    }

    #[test]
    fn prompt_assembly() {
        let v = Vocab::default();
        let (p, sot) = build_prompt(&v, &[], 50259, false);
        assert_eq!(p, vec![50258, 50259, 50359]);
        assert_eq!(sot, 0);
        let (p, sot) = build_prompt(&v, &[100, 101, 102], 50261, true);
        assert_eq!(p, vec![50361, 100, 101, 102, 50258, 50261, 50358]);
        assert_eq!(sot, 4);
        // A long prefix keeps only its tail.
        let long: Vec<u32> = (1000..2000).collect();
        let (p, sot) = build_prompt(&v, &long, 50259, false);
        assert_eq!(sot, 1 + 223);
        assert_eq!(p[1], 2000 - 223);
        assert_eq!(p.len(), 1 + 223 + 3);
    }

    fn logits_with(v: &Vocab, hot: &[(u32, f32)]) -> Vec<f32> {
        let mut l = vec![-10.0f32; v.vocab_size];
        for &(t, x) in hot {
            l[t as usize] = x;
        }
        l
    }

    #[test]
    fn rules_first_token_is_an_early_timestamp() {
        let v = Vocab::default();
        // The best raw token is a word; the rules must force a timestamp ≤ <|1.00|>.
        let mut l = logits_with(&v, &[(1000, 5.0), (v.timestamp_begin + 10, 1.0), (v.timestamp_begin + 200, 4.0), (v.eot, 3.0)]);
        apply_rules(&mut l, &v, &[]);
        assert_eq!(argmax(&l) as u32, v.timestamp_begin + 10);
        assert!(l[1000].is_infinite() && l[v.eot as usize].is_infinite());
        assert!(l[v.no_timestamps as usize].is_infinite());
    }

    #[test]
    fn rules_after_text_a_timestamp_must_close_and_pairs_force_text() {
        let v = Vocab::default();
        // ... <|0.00|> "hi" <|0.50|>  → next must be a timestamp (≥ 0.50) or eot.
        let sampled = [v.timestamp_begin, 1000, v.timestamp_begin + 25];
        let mut l = logits_with(&v, &[(1001, 9.0), (v.timestamp_begin + 20, 8.0), (v.timestamp_begin + 25, 1.0), (v.timestamp_begin + 30, 2.0)]);
        apply_rules(&mut l, &v, &sampled);
        assert!(l[1001].is_infinite(), "text is not allowed after a lone closing timestamp");
        assert!(l[(v.timestamp_begin + 20) as usize].is_infinite(), "timestamps must not go backwards");
        assert_eq!(argmax(&l) as u32, v.timestamp_begin + 30);
        // <|0.50|> <|0.50|> → next must be text.
        let sampled = [v.timestamp_begin, 1000, v.timestamp_begin + 25, v.timestamp_begin + 25];
        let mut l = logits_with(&v, &[(1001, 1.0), (v.timestamp_begin + 40, 9.0)]);
        apply_rules(&mut l, &v, &sampled);
        assert_eq!(argmax(&l), 1001);
        // Suppressed tokens never win.
        let mut l = logits_with(&v, &[(1001, 1.0), (v.sot, 9.0)]);
        apply_rules(&mut l, &v, &sampled);
        assert_eq!(argmax(&l), 1001);
    }

    #[test]
    fn rules_timestamp_mass_beats_text() {
        let v = Vocab::default();
        // Many mildly likely timestamps outweigh one text token.
        let mut hot: Vec<(u32, f32)> = (0..200).map(|i| (v.timestamp_begin + 100 + i, 2.0)).collect();
        hot.push((1000, 4.0));
        let mut l = logits_with(&v, &hot);
        apply_rules(&mut l, &v, &[v.timestamp_begin, 999]);
        assert!(l[1000].is_infinite());
        assert!(v.is_timestamp(argmax(&l) as u32));
    }

    #[test]
    fn sampling_is_greedy_at_zero_and_valid_otherwise() {
        let l = vec![0.0f32, 1.0, 5.0, -1.0];
        let mut rng = Rng::new(7);
        let (t, lp) = sample(&l, 0.0, &mut rng);
        assert_eq!(t, 2);
        assert!(lp < 0.0 && lp > -0.1);
        let mut counts = [0usize; 4];
        for _ in 0..500 {
            counts[sample(&l, 1.0, &mut rng).0 as usize] += 1;
        }
        assert!(counts[2] > 400, "{counts:?}");
        assert!(counts[3] < 20, "{counts:?}");
    }

    #[test]
    fn timestamp_tokens_split_segments() {
        let v = Vocab::default();
        let tb = v.timestamp_begin;
        // <|0.00|> a b <|1.00|> <|1.00|> c <|2.50|> <|2.50|> d  (window ends mid-segment)
        let toks = [tb, 10, 11, tb + 50, tb + 50, 12, tb + 125, tb + 125, 13];
        let lps = vec![-0.1; toks.len()];
        let (segs, adv) = split_segments(&v, &toks, &lps, 30.0);
        // The unfinished tail ("d") is not a segment: the next window starts
        // at 2.5 s and decodes it again in full.
        assert_eq!(segs.len(), 2);
        assert_eq!((segs[0].start, segs[0].end), (0.0, 1.0));
        assert_eq!(segs[0].tokens, vec![10, 11]);
        assert_eq!((segs[1].start, segs[1].end), (1.0, 2.5));
        assert_eq!(segs[1].tokens, vec![12]);
        assert_eq!(adv, 2.5, "the next window starts at the last complete segment");
        // Single timestamp ending: <|0.00|> a <|1.20|>
        let toks = [tb, 10, tb + 60];
        let (segs, adv) = split_segments(&v, &toks, &[-0.2; 3], 12.0);
        assert_eq!(segs.len(), 1);
        assert_eq!((segs[0].start, segs[0].end), (0.0, 1.2));
        assert_eq!(adv, 12.0);
        // No timestamps at all: one segment over the window.
        let (segs, adv) = split_segments(&v, &[10, 11], &[-0.2; 2], 7.5);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].end, 7.5);
        assert_eq!(adv, 7.5);
        // Only specials: nothing.
        let (segs, _) = split_segments(&v, &[tb, tb + 3], &[-0.2; 2], 7.5);
        assert!(segs.is_empty());
    }

    #[test]
    fn words_interpolate_and_group_tokens() {
        // A toy tokenizer: 1 = " Hello", 2 = " wor", 3 = "ld", 4 = ",", 5 = " ok".
        let decode = |ids: &[u32]| -> String {
            ids.iter().map(|&i| match i { 1 => " Hello", 2 => " wor", 3 => "ld", 4 => ",", 5 => " ok", _ => "\u{FFFD}" }).collect()
        };
        let lps = [-0.1f32, -0.2, -0.3, -0.05, -0.5];
        let w = words_by_interpolation(&[1, 2, 3, 4, 5], &lps, 10.0, 12.0, &decode);
        let texts: Vec<&str> = w.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["Hello", "world,", "ok"]);
        assert!((w[0].start - 10.0).abs() < 1e-6);
        assert!((w[2].end - 12.0).abs() < 1e-5);
        assert!(w[0].end <= w[1].start + 1e-6 && w[1].end <= w[2].start + 1e-6);
        // 5 + 6 + 2 = 13 characters over 2 s.
        assert!((w[0].end - w[0].start - 2.0 * 5.0 / 13.0).abs() < 1e-5);
        assert!(w.iter().all(|w| w.prob > 0.0 && w.prob <= 1.0));
        // Tokens that do not decode alone are joined with the next one.
        let decode2 = |ids: &[u32]| -> String { if ids == [7, 8] { " é".into() } else if ids == [9] { " x".into() } else { "\u{FFFD}".into() } };
        let w = words_by_interpolation(&[7, 8, 9], &[-0.1; 3], 0.0, 1.0, &decode2);
        assert_eq!(w.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), vec!["é", "x"]);
    }

    #[test]
    fn compression_ratio_flags_repetition() {
        let prose = "The quick brown fox jumps over the lazy dog while the cat sleeps by the fire.";
        let looped = "thank you thank you thank you thank you thank you thank you thank you thank you";
        assert!(compression_ratio(prose) < 2.0, "{}", compression_ratio(prose));
        assert!(compression_ratio(looped) > 2.4, "{}", compression_ratio(looped));
        assert_eq!(compression_ratio(""), 1.0);
    }

    #[test]
    fn ranges_are_merged_and_clamped() {
        assert_eq!(normalise_ranges(None, 12.0), vec![(0.0, 12.0)]);
        let r = normalise_ranges(Some(&[(5.0, 7.0), (0.0, 2.0), (2.2, 3.0), (-1.0, 0.01), (11.0, 40.0)]), 12.0);
        assert_eq!(r, vec![(0.0, 3.0), (5.0, 7.0), (11.0, 12.0)]);
    }

    #[test]
    fn fallback_mel_has_whisper_shape_and_tracks_frequency() {
        let sr = WHISPER_RATE as f32;
        let tone = |hz: f32| -> Vec<f32> { (0..sr as usize * 2).map(|i| (i as f32 / sr * hz * std::f32::consts::TAU).sin() * 0.5).collect() };
        let low = fallback_mel(&tone(300.0), 80);
        assert_eq!(low.len(), N_FRAMES);
        assert!(low.iter().all(|r| r.len() == 80));
        assert!(low.iter().flatten().all(|v| v.is_finite()));
        let peak = |m: &Vec<Vec<f32>>| -> usize { argmax(&m[50]) };
        let high = fallback_mel(&tone(3000.0), 80);
        assert!(peak(&low) < peak(&high), "{} {}", peak(&low), peak(&high));
        // Padding is silence: the frames past 2 s sit at the floor.
        let floor = low[2999].iter().copied().fold(f32::INFINITY, f32::min);
        assert!(low[2999].iter().all(|&v| (v - floor).abs() < 1e-4));
        assert!(low[50].iter().copied().fold(f32::NEG_INFINITY, f32::max) > floor + 0.5);
        // Mel filters cover the band and are normalised.
        let f = mel_filters(80);
        assert_eq!(f.len(), 80);
        assert!(f.iter().all(|w| w.len() == 201 && w.iter().sum::<f32>() > 0.0));
    }

    /// Needs whisper-base (and the AudioSet model to find speech) in the
    /// default model folder. Scans the DDO client sounds for three voice
    /// lines, transcribes them and prints text and seconds per 30 s window.
    #[test]
    #[ignore]
    #[cfg(feature = "ml")]
    fn transcribe_real_voice_lines() {
        use crate::models::ModelId;
        let mm = ModelManager::default();
        if mm.path(ModelId::WhisperBase).is_none() {
            eprintln!("whisper-base missing; skipping");
            return;
        }
        let root = std::path::Path::new(r"C:\Users\Adam Bennett\Development\artifacts\client-assets\audio\client_sound");
        if !root.is_dir() {
            eprintln!("client_sound folder missing; skipping");
            return;
        }
        let t0 = std::time::Instant::now();
        let whisper = WhisperOnnx::load(&mm, ModelId::WhisperBase).unwrap();
        println!("loaded whisper-base in {:?}", t0.elapsed());
        // Find candidate clips: longer files first.
        let mut files: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| crate::audio::is_supported(&e.path()))
            .map(|e| (e.metadata().map(|m| m.len()).unwrap_or(0), e.path()))
            .collect();
        files.sort_by_key(|f| std::cmp::Reverse(f.0));
        let mut voice: Vec<(std::path::PathBuf, Clip)> = Vec::new();
        let mut looked = 0;
        for (_, p) in files.iter().skip(200) {
            if voice.len() >= 3 || looked >= 400 {
                break;
            }
            looked += 1;
            let Ok(clip) = crate::audio::decode(p) else { continue };
            let d = clip.duration_secs();
            if !(1.5..=60.0).contains(&d) {
                continue;
            }
            println!("{}: {:.1} s", p.display(), d);
            voice.push((p.clone(), clip));
        }
        assert!(!voice.is_empty(), "no voice files found among {looked} candidates");
        for (p, clip) in &voice {
            let opts = TranscribeOptions { word_timestamps: true, ..Default::default() };
            let mut reports = 0;
            let t = std::time::Instant::now();
            let tr = whisper.transcribe(clip, &opts, &mut |_| reports += 1).unwrap();
            let secs = t.elapsed().as_secs_f32();
            let windows = (clip.duration_secs() / 30.0).ceil().max(1.0);
            println!(
                "{} [{}] {:.2}s for {:.1}s audio = {:.2}s per 30 s window, {} segments, {} progress reports\n  {}",
                p.file_name().unwrap().to_string_lossy(),
                tr.language,
                secs,
                clip.duration_secs(),
                secs / windows,
                tr.segments.len(),
                reports,
                tr.text()
            );
            for s in &tr.segments {
                println!("  [{:6.2} – {:6.2}] nsp={:.2} {}", s.start, s.end, s.no_speech_prob, s.text.trim());
                if !s.words.is_empty() {
                    println!("    words: {}", s.words.iter().map(|w| format!("{}@{:.2}", w.text, w.start)).collect::<Vec<_>>().join(" "));
                }
            }
            assert!(reports >= 2);
            assert!(!tr.language.is_empty());
        }
    }

    /// Synthetic end-to-end with the model: silence gives no segments and a
    /// forced language is reported back.
    #[test]
    #[ignore]
    #[cfg(feature = "ml")]
    fn whisper_on_silence_and_tone() {
        use crate::models::ModelId;
        let mm = ModelManager::default();
        if mm.path(ModelId::WhisperBase).is_none() {
            eprintln!("whisper-base missing; skipping");
            return;
        }
        let whisper = WhisperOnnx::load(&mm, ModelId::WhisperBase).unwrap();
        let silence = Clip { samples: vec![0.0; 16_000 * 3], sample_rate: 16_000, source_channels: 1 };
        let tr = whisper.transcribe(&silence, &TranscribeOptions { language: Some("en".into()), ..Default::default() }, &mut |_| {}).unwrap();
        assert!(tr.is_empty(), "{:?}", tr.segments);
        assert_eq!(tr.language, "en");
        let sr = 44_100u32;
        let tone: Vec<f32> = (0..sr * 4).map(|i| (i as f32 / sr as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.3).collect();
        let t = std::time::Instant::now();
        let tr = whisper.transcribe(&Clip { samples: tone, sample_rate: sr, source_channels: 1 }, &TranscribeOptions::default(), &mut |_| {}).unwrap();
        println!("tone ({:?}): lang={} text={:?}", t.elapsed(), tr.language, tr.text());
    }
}

#[cfg(test)]
mod hallucination_tests {
    use super::is_hallucination;
    #[test]
    fn stock_phrases_on_silence_are_dropped_but_speech_is_kept() {
        assert!(is_hallucination(" Thank you.", 0.5));
        assert!(is_hallucination("Thanks for watching!", 0.2));
        assert!(!is_hallucination("Thank you.", 0.05));
        assert!(!is_hallucination("You should proceed with caution.", 0.5));
        assert!(!is_hallucination("You hear a mechanism sliding open in the distance.", 0.9));
    }
}
