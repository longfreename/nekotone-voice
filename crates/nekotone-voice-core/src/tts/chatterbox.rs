//! Text to speech in a cloned voice: Resemble AI's Chatterbox-Turbo (MIT)
//! through ONNX Runtime (owner: the voice-clone package).
//!
//! ```text
//! reference speech (5–15 s) ─ 24 kHz mono, trimmed, −27 LUFS ─ speech_encoder ─▶ VoicePrint (saved once)
//!      audio_features [1, T, 1024] (LM prefix) · audio_tokens [1, P] (decoder prompt)
//!      speaker_embeddings [1, 192] (x-vector) · speaker_features [1, F, 80] (prompt mel)
//! text ─ normalise (numbers, money…) ─ Chatterbox punctuation clean-up ─ sentences (long ones cut at clauses)
//!      ─ GPT-2 BPE (tokenizer.json; the template appends two <|endoftext|>) ─ embed_tokens
//!      ─ [audio_features ‖ text embeds] ─ language_model (GPT-2 medium, 24 layers, KV cache)
//!      ─ sample speech tokens (temperature 0.8, top-k 1000, top-p 0.95, repetition penalty 1.2) until 6562
//!      ─ [audio_tokens ‖ speech tokens ‖ 3 × silence 4299] ─ conditional_decoder ─ 24 kHz mono
//! ```
//!
//! * Export: `ResembleAI/chatterbox-turbo-ONNX`, pinned to a commit. Four
//!   graphs, each in fp32 / fp16 / q8 ("quantized") / q4 / q4f16; the files
//!   are listed in [`model_files`] (see [`GRAPHS`] for the chosen precision
//!   of each graph; [`ALL_GRAPHS`] pins every precision so a measurement
//!   can switch without re-hashing).
//! * All graph boundaries are float32 except the KV cache of the fp16 /
//!   q4f16 language models (float16); the cache values are passed back
//!   opaque, so the element type does not matter here.
//! * Speech tokens run at 25 per second; the decoder output excludes the
//!   prompt. Upstream (PyTorch) needs more than 5 s of reference and uses
//!   the first 10 s (decoder) / 15 s (prompt tokens) of it.
//! * Not done here: Resemble's PerTh watermark (a separate Python package;
//!   the ONNX README makes it optional).

use super::Utterance;
use crate::models::ModelFile;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Licence of the weights, the export and the tokenizer.
pub const MODEL_LICENSE: &str = "MIT (Chatterbox-Turbo by Resemble AI; ONNX export by Resemble AI)";

/// Output rate of the decoder.
pub const SAMPLE_RATE: u32 = 24_000;
/// First speech token fed to the language model (implicitly, by the export).
pub const START_SPEECH_TOKEN: i64 = 6561;
/// The language model's end of speech.
pub const STOP_SPEECH_TOKEN: i64 = 6562;
/// Speech token for silence (three are appended before decoding).
pub const SILENCE_TOKEN: i64 = 4299;
/// Speech tokens per second of audio.
pub const TOKENS_PER_SEC: f32 = 25.0;
/// Shortest usable reference (upstream refuses 5 s or less).
pub const MIN_REFERENCE_SECS: f32 = 5.0;
/// Longest reference used (the prompt tokenizer's window).
pub const MAX_REFERENCE_SECS: f32 = 15.0;
/// Loudness the reference is brought to (upstream's `norm_loudness`).
pub const REFERENCE_LUFS: f32 = -27.0;

const REPO: &str = "https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/resolve/d21799bd0354adb85e348b8a0442a8405110a2cf";

/// File names in the model folder.
pub const TOKENIZER: &str = "tokenizer.json";

/// One graph of the export in one precision: the `.onnx` file and its
/// external weights (`<file>_data`, the name the graph refers to).
#[derive(Debug, Clone, Copy)]
pub struct GraphFile {
    pub graph: &'static str,
    pub onnx_bytes: u64,
    pub onnx_sha256: &'static str,
    pub data_bytes: u64,
    pub data_sha256: &'static str,
}

impl GraphFile {
    pub fn data_name(&self) -> String {
        format!("{}_data", self.graph)
    }
}

const fn g(graph: &'static str, onnx_bytes: u64, onnx_sha256: &'static str, data_bytes: u64, data_sha256: &'static str) -> GraphFile {
    GraphFile { graph, onnx_bytes, onnx_sha256, data_bytes, data_sha256 }
}

/// Every graph file of the pinned export (sizes and SHA-256 measured on the
/// downloaded files; they equal the repository's LFS object ids).
pub static ALL_GRAPHS: &[GraphFile] = &[
    g("speech_encoder.onnx", 1_172_072, "4d66128037517dd51d370edc9b89ce36d42c75dcbd96e7216c7fb45dfae36045", 1_044_712_832, "c9915ff6c529e7bb80983b525255e6744d6c39c7e35b12720925ba99ed0d0a2f"),
    g("speech_encoder_fp16.onnx", 1_189_888, "5544f87c3e1615a31f5dc3ccdf9ba52a9f5fb8f6018e72796235fc1cfd5ebc9f", 522_307_136, "d24b817ff61489373ab13033cbdc28a5ba8347341248a00f9a63f10888e81d73"),
    g("speech_encoder_q4.onnx", 1_200_346, "37956c20b67bed85a0da4bc83509d67b5969a1b257d1c546516a5236a17ad71e", 229_560_112, "58956db217c6443e49c91bdd54d7cf76b4a243f225c748b7bf746459fc27bc7d"),
    g("speech_encoder_quantized.onnx", 1_205_728, "5b6f15870a43cf97892df86fc550a0ef4763522d527cde72b2a4316f80a34de4", 354_676_576, "d59861fb55e806fbeee731da9d4f8ff819fb5735de5d15e262d902594ee4dbb6"),
    g("embed_tokens.onnx", 2_058, "27796e8252f36b463b0421cafdcc35b5f1e670ab0d96c9182f37ac6571c2f4bc", 232_812_544, "a1c37edc6ec6adb655351f02e958da297221b50211c2c01b69312cb6f008a293"),
    g("embed_tokens_fp16.onnx", 1_754, "f7a7a83e91337e10add2fad054544421f9080e27b72af1aef27f339a1a66776c", 116_406_272, "cdda886a9e58ad39059fe8f6cf5a3a628994a1a2dfc1a8b58502c252fec70327"),
    g("embed_tokens_q4.onnx", 2_844, "fd6ba1d22902e8f539d3dd6d7c1c44b98ebb4c84ebbb5e47fcb826ddcf667561", 37_286_384, "f54a51e234b509b64c3a03bb79e1149fba7e2eba6c2d9c222f18883379e1f5d8"),
    g("embed_tokens_quantized.onnx", 2_887, "0efe1bc01c2c48a98425a74444fd9887924d887f922c2722a6ec961ebb9e1db6", 67_297_376, "9025d04c124899823124b1d7bb7069b1f535fb8a6c2d88f97520eb6fecced986"),
    g("language_model.onnx", 207_266, "c12e31df78c74f9589b165c8d51e65171f5028b77b7fedb41900f55f7f410dc8", 1_269_724_812, "67db106868f5354b2e425651f1791aef36ae3e6f00ac5e1d91e32c985cad6b39"),
    g("language_model_fp16.onnx", 209_456, "396b35570dd1dbc5f537ffc932c3e69130ab8a09bf1390469d92338c8eae7da4", 634_862_406, "d730d6437126d3e232747b26e2a7adece68844f22133277ca5012298ffe36f93"),
    g("language_model_q4.onnx", 274_572, "b39d03d3f8b943b9e60c6fce3fb41191dbc1df4589f913291db1e214eef669b1", 204_456_572, "2c029dc0acf48752473d8c74c72b5ceaaad76b9886fe106eaf2022142d5b5d5e"),
    g("language_model_quantized.onnx", 279_670, "0b40581277e30b7034331ec8c3ad47ed71d321f015b387a95221e54e2fcbfde8", 367_962_860, "ec9945df36cb5d131d46688f2609fd715fbfcb0b8ee9681af5c84118de2d55a2"),
    g("conditional_decoder.onnx", 1_889_468, "8c43f3a1d0ddb1a86e226a244d7cda5396c67f5c6412789c23900c646e3ffc50", 768_593_792, "05f162a519f3e9abaf0b7337ae037f4af8b2b30c4455d39b2c61ed3a9b2b5476"),
    g("conditional_decoder_fp16.onnx", 2_104_016, "cbdc0281548eb90a02fa2430647237df63fa0cb695e42b91c10b860b9b6d2230", 384_019_328, "c6c3e79e6ff86bc41f77381a3d67b8edaa16b13e43a2700ca8159d0894bd594a"),
    g("conditional_decoder_q4.onnx", 2_179_022, "dccb7a6cea3472dc7f7d070eeb70ade18e6327fb4ec61a3d62cf211bfed90ea2", 246_397_384, "b5c5317e0b79a1a19dd3d5e2b2091ea06b15716716ab801a54eaeb906c6971ec"),
    g("conditional_decoder_quantized.onnx", 2_202_035, "2af3b150196d9d559cd3c91e03da80eb27a466032369dc2b57ea729cddad3ebb", 326_548_688, "4918ca09e05e41d2b4aa1ace6201d1cd911ffc58a42801002bab177d495cfe0a"),
];

/// The four roles, in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    SpeechEncoder,
    EmbedTokens,
    LanguageModel,
    Decoder,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::SpeechEncoder, Role::EmbedTokens, Role::LanguageModel, Role::Decoder];
    fn stem(self) -> &'static str {
        match self {
            Role::SpeechEncoder => "speech_encoder",
            Role::EmbedTokens => "embed_tokens",
            Role::LanguageModel => "language_model",
            Role::Decoder => "conditional_decoder",
        }
    }
    fn short(self) -> &'static str {
        match self {
            Role::SpeechEncoder => "enc",
            Role::EmbedTokens => "embed",
            Role::LanguageModel => "lm",
            Role::Decoder => "dec",
        }
    }
}

/// The precision shipped for each role. PROVISIONAL (not yet measured in
/// this build; see HANDOFF §5h): the language model in q4 (MatMulNBits,
/// fast on the CPU, 205 MB; the step that sets the pace), the decoder in
/// fp32 (Kokoro's fp16 vocoder returned NaN on the ORT this build links, so
/// the vocoder stays fp32 until fp16 is checked), the one-off speech encoder
/// in q4 and the embedding table in fp16 (a lookup). Total 1 325 956 517 B.
/// `speed_per_engine` with `NEKOTONE_CHATTERBOX_GRAPHS` compares the rest.
pub const GRAPHS: [(Role, &str); 4] = [
    (Role::SpeechEncoder, "speech_encoder_q4.onnx"),
    (Role::EmbedTokens, "embed_tokens_fp16.onnx"),
    (Role::LanguageModel, "language_model_q4.onnx"),
    (Role::Decoder, "conditional_decoder.onnx"),
];

fn graph_info(name: &str) -> Option<&'static GraphFile> {
    ALL_GRAPHS.iter().find(|g| g.graph == name)
}

/// The catalogue files for `ModelId::TtsChatterbox` (pinned URLs, sizes, SHA-256).
pub fn model_files() -> Vec<ModelFile> {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let mut files = vec![
        ModelFile { name: TOKENIZER, url: leak(format!("{REPO}/tokenizer.json")), size_bytes: 3_562_272, sha256: "3f04e34bea22f9144d1a19151154095bc9ce0430bf421304f5797e716288a906" },
        ModelFile { name: "CHATTERBOX-README.md", url: leak(format!("{REPO}/README.md")), size_bytes: 11_357, sha256: README_SHA256 },
    ];
    for (_, name) in GRAPHS {
        let gf = graph_info(name).expect("GRAPHS names a pinned file");
        files.push(ModelFile { name: gf.graph, url: leak(format!("{REPO}/onnx/{}", gf.graph)), size_bytes: gf.onnx_bytes, sha256: gf.onnx_sha256 });
        files.push(ModelFile { name: leak(gf.data_name()), url: leak(format!("{REPO}/onnx/{}", gf.data_name())), size_bytes: gf.data_bytes, sha256: gf.data_sha256 });
    }
    files
}

/// SHA-256 of the pinned README (the model card with the licence).
const README_SHA256: &str = "1e5f4f708ad60475377131e1cf5081108873e3cb424539c4ea6401e8d80303db";

/// The graph file used for `role` in `dir`: `NEKOTONE_CHATTERBOX_GRAPHS`
/// (`lm=q4,dec=fp32,…`; precisions fp32/fp16/q4/q4f16/quantized) for
/// measurements, else the shipped choice, else any precision present.
pub fn graph_for(dir: &Path, role: Role) -> Option<std::path::PathBuf> {
    let file = |prec: &str| -> String {
        match prec {
            "fp32" | "" => format!("{}.onnx", role.stem()),
            "q8" => format!("{}_quantized.onnx", role.stem()),
            p => format!("{}_{p}.onnx", role.stem()),
        }
    };
    if let Ok(spec) = std::env::var("NEKOTONE_CHATTERBOX_GRAPHS") {
        for part in spec.split(',') {
            if let Some((k, v)) = part.split_once('=') {
                if k.trim() == role.short() {
                    let p = dir.join(file(v.trim()));
                    return p.is_file().then_some(p);
                }
            }
        }
    }
    let shipped = GRAPHS.iter().find(|(r, _)| *r == role).map(|(_, n)| dir.join(n));
    if let Some(p) = shipped.filter(|p| p.is_file()) {
        return Some(p);
    }
    ["fp16", "q4", "fp32", "quantized", "q4f16"].iter().map(|p| dir.join(file(p))).find(|p| p.is_file())
}

// ───────────────────────── text ─────────────────────────

/// Chatterbox's own punctuation clean-up (`punc_norm` upstream): capital
/// first letter, single spaces, uncommon punctuation replaced, and a full
/// stop added when the text ends without one.
pub fn punc_norm(text: &str) -> String {
    let mut t: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return t;
    }
    for (a, b) in [("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'")] {
        t = t.replace(a, b);
    }
    // (upstream leaves the double space "…" → ", " can make; collapse it)
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut cs = t.chars();
    let mut out = match cs.next() {
        Some(f) if f.is_lowercase() => f.to_uppercase().collect::<String>() + cs.as_str(),
        _ => t.clone(),
    };
    if !out.ends_with(['.', '!', '?', '-', ',']) {
        out.push('.');
    }
    out
}

/// Characters allowed in the first piece of a line (~4 s of speech): the
/// voice starts sooner; the rest is generated while it plays. Not smaller:
/// the model makes speech at about real time, so the first piece must play
/// long enough to cover making the next one. Measured on a 115-character
/// line: 90 → first sound after 3.3 s, no gap; 45 → after 1.4 s but a
/// ~3.6 s silence after the first four words.
pub const FIRST_PIECE_CHARS: usize = 90;
/// Longest piece (~20 s of speech, ~500 speech tokens).
pub const MAX_PIECE_CHARS: usize = 260;

/// Text → pieces to generate one at a time: numbers and abbreviations
/// spelled out (`tts::normalize`), sentences, long sentences cut at clause
/// punctuation (the first piece of the text at `first_max` characters).
pub fn plan(text: &str, first_max: usize) -> Vec<Utterance> {
    let norm = super::normalize::normalize(text, false);
    let mut out = Vec::new();
    for sentence in super::split_sentences(&norm) {
        let first = if out.is_empty() { first_max } else { MAX_PIECE_CHARS };
        let pieces = super::chunk_phonemes(&sentence, MAX_PIECE_CHARS, first);
        let n = pieces.len();
        for (k, p) in pieces.into_iter().enumerate() {
            let last = k + 1 == n;
            let end = p.trim_end().chars().last().unwrap_or(' ');
            let pause_after = match end {
                '.' | '!' | '?' if last => 0.20,
                ',' | ';' | ':' => 0.08,
                _ if last => 0.16,
                _ => 0.04,
            };
            out.push(Utterance { text: if n == 1 { sentence.clone() } else { format!("{sentence} ({}/{n})", k + 1) }, phonemes: punc_norm(&p), pause_after });
        }
    }
    out.retain(|u| u.phonemes.chars().any(|c| c.is_alphanumeric()));
    out
}

/// Most speech tokens to generate for `text` (runaway guard): about four
/// times what the text needs at a slow 10 characters per second.
pub fn max_tokens_for(text: &str) -> usize {
    let chars = text.chars().count() as f32;
    ((chars / 10.0 * TOKENS_PER_SEC * 2.0) as usize + 75).min(1000)
}

// ───────────────────────── sampling ─────────────────────────

/// How the next speech token is chosen (upstream defaults).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    /// 0 = greedy (always the most likely token).
    pub temperature: f32,
    /// Keep the k most likely tokens (0 = all).
    pub top_k: usize,
    /// Keep the smallest set whose probability reaches p (1 = all).
    pub top_p: f32,
    /// Divide (or multiply, when negative) the logit of every token already generated.
    pub repetition_penalty: f32,
    /// Seed of the random choice (the same seed and text give the same audio).
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.8, top_k: 1000, top_p: 0.95, repetition_penalty: 1.2, seed: 0x6e656b6f }
    }
}

/// splitmix64: small, fast, good enough for sampling.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The repetition penalty (HF `RepetitionPenaltyLogitsProcessor`): every
/// token in `history` (once, however often it occurs) has its logit divided
/// by `penalty` when positive and multiplied when negative.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[i64], penalty: f32) {
    if penalty == 1.0 || penalty <= 0.0 {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &t in history {
        let i = t as usize;
        if t < 0 || i >= logits.len() || seen[i] {
            continue;
        }
        seen[i] = true;
        let v = logits[i];
        logits[i] = if v < 0.0 { v * penalty } else { v / penalty };
    }
}

/// Choose the next token from `logits` (modified in place) given the tokens
/// generated so far, in upstream's order: temperature → top-k → top-p →
/// repetition penalty → softmax → draw. Temperature 0 is greedy (penalty,
/// then the largest logit).
pub fn sample_next(logits: &mut [f32], history: &[i64], s: &Sampling, rng: &mut Rng) -> i64 {
    let n = logits.len();
    if n == 0 {
        return STOP_SPEECH_TOKEN;
    }
    for v in logits.iter_mut() {
        if !v.is_finite() && *v != f32::NEG_INFINITY {
            *v = f32::NEG_INFINITY; // NaN / +inf from a broken graph: never chosen
        }
    }
    if s.temperature <= 0.0 {
        apply_repetition_penalty(logits, history, s.repetition_penalty);
        return argmax(logits) as i64;
    }
    if s.temperature != 1.0 {
        for v in logits.iter_mut() {
            *v /= s.temperature;
        }
    }
    // candidates by descending logit
    let mut idx: Vec<usize> = (0..n).filter(|&i| logits[i] > f32::NEG_INFINITY).collect();
    if idx.is_empty() {
        return STOP_SPEECH_TOKEN;
    }
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    if s.top_k > 0 && idx.len() > s.top_k {
        // HF keeps every token tied with the k-th
        let kth = logits[idx[s.top_k - 1]];
        let keep = idx.iter().take_while(|&&i| logits[i] >= kth).count();
        idx.truncate(keep);
    }
    if s.top_p < 1.0 {
        let m = logits[idx[0]];
        let ws: Vec<f64> = idx.iter().map(|&i| ((logits[i] - m) as f64).exp()).collect();
        let total: f64 = ws.iter().sum();
        // HF removes tokens whose cumulative probability (ascending) is ≤ 1 − p,
        // i.e. keeps the most likely tokens until p is reached (at least one)
        let mut cum = 0.0;
        let mut keep = idx.len();
        for (k, w) in ws.iter().enumerate() {
            cum += w / total;
            if cum >= s.top_p as f64 {
                keep = k + 1;
                break;
            }
        }
        idx.truncate(keep.max(1));
    }
    // the penalty applies to the survivors (the others are already −inf)
    let mut kept: Vec<(usize, f32)> = idx.iter().map(|&i| (i, logits[i])).collect();
    if s.repetition_penalty != 1.0 && s.repetition_penalty > 0.0 {
        let hist: std::collections::HashSet<i64> = history.iter().copied().collect();
        for (i, v) in kept.iter_mut() {
            if hist.contains(&(*i as i64)) {
                *v = if *v < 0.0 { *v * s.repetition_penalty } else { *v / s.repetition_penalty };
            }
        }
    }
    let m = kept.iter().fold(f32::NEG_INFINITY, |a, (_, v)| a.max(*v));
    let ws: Vec<f64> = kept.iter().map(|(_, v)| ((v - m) as f64).exp()).collect();
    let total: f64 = ws.iter().sum();
    let mut r = rng.next_f32() as f64 * total;
    for ((i, _), w) in kept.iter().zip(ws.iter()) {
        if r < *w {
            return *i as i64;
        }
        r -= w;
    }
    kept.last().map(|(i, _)| *i as i64).unwrap_or(STOP_SPEECH_TOKEN)
}

fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best
}

/// Speech tokens for the decoder: the reference's prompt tokens, the
/// generated tokens without the start/stop markers (anything ≥ 6561), then
/// three silences.
pub fn decoder_tokens(prompt: &[i64], generated: &[i64]) -> Vec<i64> {
    let mut v = Vec::with_capacity(prompt.len() + generated.len() + 3);
    v.extend_from_slice(prompt);
    v.extend(generated.iter().copied().filter(|&t| (0..START_SPEECH_TOKEN).contains(&t)));
    v.extend([SILENCE_TOKEN; 3]);
    v
}

// ───────────────────────── reference and voice print ─────────────────────────

/// A reference recording made ready for the speech encoder: mono 24 kHz,
/// leading/trailing silence trimmed, at most [`MAX_REFERENCE_SECS`],
/// loudness [`REFERENCE_LUFS`]. Errors are sentences for the person.
pub fn prepare_reference(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == 0 || samples.is_empty() {
        return Err(Error::Model("the voice sample is empty: record a few sentences in your normal voice".into()));
    }
    let clip = crate::audio::Clip { samples: samples.to_vec(), sample_rate: rate, source_channels: 1 };
    let mut x = if rate == SAMPLE_RATE { clip.samples } else { crate::audio::resample(&clip, SAMPLE_RATE)?.samples };
    x.iter_mut().for_each(|v| {
        if !v.is_finite() {
            *v = 0.0
        }
    });
    super::trim_silence(&mut x, SAMPLE_RATE, 0.15);
    let secs = x.len() as f32 / SAMPLE_RATE as f32;
    if secs < MIN_REFERENCE_SECS {
        return Err(Error::Model(format!(
            "the voice sample has only {secs:.1} s of sound; Chatterbox needs more than {MIN_REFERENCE_SECS:.0} s: record 10 seconds of you reading or talking"
        )));
    }
    x.truncate((MAX_REFERENCE_SECS * SAMPLE_RATE as f32) as usize);
    if let Some(l) = crate::process::integrated_lufs(&x, 1, SAMPLE_RATE) {
        if l.is_finite() && l > -70.0 {
            let g = 10f32.powf((REFERENCE_LUFS - l) / 20.0);
            if g.is_finite() && g > 0.0 {
                x.iter_mut().for_each(|v| *v = (*v * g).clamp(-1.0, 1.0));
            }
        }
    }
    Ok(x)
}

/// A cloned voice: the speech encoder's outputs for a reference recording.
/// Encoded once (≈ a second) and saved, so speaking never re-reads the
/// recording.
#[derive(Debug, Clone, PartialEq)]
pub struct VoicePrint {
    /// Language-model prefix `[1, T, D]` (row-major).
    pub audio_features: Vec<f32>,
    pub features_shape: [usize; 3],
    /// Decoder prompt tokens `[1, P]`.
    pub audio_tokens: Vec<i64>,
    /// x-vector `[1, 192]`.
    pub speaker_embeddings: Vec<f32>,
    /// Prompt mel `[1, F, 80]`.
    pub speaker_features: Vec<f32>,
    pub speaker_features_shape: [usize; 3],
    /// Metadata shown in the app.
    pub info: VoicePrintInfo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct VoicePrintInfo {
    /// Display name ("My voice").
    pub name: String,
    /// Seconds of reference used.
    pub reference_secs: f32,
    /// Unix seconds.
    pub created: u64,
    /// Where the reference came from (a file name or "recorded sample").
    pub source: String,
    /// The export it was encoded with (prints are model-specific).
    pub model: String,
}

const PRINT_MAGIC: &[u8; 8] = b"NKVPRNT1";
/// Model tag stored in prints.
pub const PRINT_MODEL: &str = "chatterbox-turbo-onnx@d21799b";

impl VoicePrint {
    /// Cosine similarity of two prints' x-vectors (1 = the same speaker embedding).
    pub fn similarity(&self, other: &VoicePrint) -> f32 {
        cosine(&self.speaker_embeddings, &other.speaker_embeddings)
    }

    /// Write to `path` (atomically: a temporary file, then a rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        #[derive(Serialize)]
        struct Header<'a> {
            info: &'a VoicePrintInfo,
            features_shape: [usize; 3],
            tokens: usize,
            embedding: usize,
            speaker_features_shape: [usize; 3],
        }
        let header = serde_json::to_vec(&Header {
            info: &self.info,
            features_shape: self.features_shape,
            tokens: self.audio_tokens.len(),
            embedding: self.speaker_embeddings.len(),
            speaker_features_shape: self.speaker_features_shape,
        })
        .map_err(|e| Error::Model(format!("could not write the voice print ({e})")))?;
        let mut b = Vec::with_capacity(16 + header.len() + 4 * (self.audio_features.len() + self.speaker_features.len() + self.speaker_embeddings.len()) + 8 * self.audio_tokens.len());
        b.extend_from_slice(PRINT_MAGIC);
        b.extend_from_slice(&(header.len() as u32).to_le_bytes());
        b.extend_from_slice(&header);
        for v in &self.audio_features {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.audio_tokens {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.speaker_embeddings {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in &self.speaker_features {
            b.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &b)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read a print written by [`VoicePrint::save`].
    pub fn load(path: &Path) -> Result<VoicePrint> {
        let bad = |why: &str| Error::Model(format!("the voice print {} is damaged ({why}); make the voice again", path.display()));
        let b = std::fs::read(path).map_err(|e| Error::Model(format!("could not read the voice print {} ({e})", path.display())))?;
        if b.len() < 12 || &b[..8] != PRINT_MAGIC {
            return Err(bad("not a voice print"));
        }
        let hl = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
        let body = b.get(12 + hl..).ok_or_else(|| bad("short header"))?;
        #[derive(Deserialize)]
        struct Header {
            info: VoicePrintInfo,
            features_shape: [usize; 3],
            tokens: usize,
            embedding: usize,
            speaker_features_shape: [usize; 3],
        }
        let h: Header = serde_json::from_slice(&b[12..12 + hl]).map_err(|_| bad("header"))?;
        // sizes in u128 so a damaged header cannot overflow
        let nf128 = h.features_shape.iter().map(|&d| d as u128).product::<u128>();
        let ns128 = h.speaker_features_shape.iter().map(|&d| d as u128).product::<u128>();
        let need = 4 * nf128 + 8 * h.tokens as u128 + 4 * h.embedding as u128 + 4 * ns128;
        if body.len() as u128 != need || nf128 == 0 || h.embedding == 0 {
            return Err(bad("size"));
        }
        let (nf, ns) = (nf128 as usize, ns128 as usize);
        let f32s = |s: &[u8]| s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
        let mut o = 0;
        let audio_features = f32s(&body[o..o + 4 * nf]);
        o += 4 * nf;
        let audio_tokens = body[o..o + 8 * h.tokens].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("8 bytes"))).collect();
        o += 8 * h.tokens;
        let speaker_embeddings = f32s(&body[o..o + 4 * h.embedding]);
        o += 4 * h.embedding;
        let speaker_features = f32s(&body[o..o + 4 * ns]);
        Ok(VoicePrint { audio_features, features_shape: h.features_shape, audio_tokens, speaker_embeddings, speaker_features, speaker_features_shape: h.speaker_features_shape, info: h.info })
    }
}

/// Cosine similarity (0 for empty or zero vectors).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (d / (na.sqrt() * nb.sqrt())) as f32
    }
}

// ───────────────────────── the engine ─────────────────────────

/// Timings of the last [`Chatterbox::synthesize_piece`].
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PieceStats {
    pub text_tokens: usize,
    pub speech_tokens: usize,
    /// Prefill (first language-model call) in ms.
    pub prefill_ms: f32,
    /// Mean per-token language-model step (embed + LM + sampling) in ms.
    pub step_ms: f32,
    pub decode_ms: f32,
    pub total_ms: f32,
    pub audio_secs: f32,
    /// Stopped at the token limit instead of the stop token.
    pub hit_limit: bool,
}

/// Where each graph runs.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Engines {
    pub speech_encoder: String,
    pub embed_tokens: String,
    pub language_model: String,
    pub decoder: String,
}

#[cfg(feature = "ml")]
pub use engine::Chatterbox;

#[cfg(feature = "ml")]
mod engine {
    use super::*;
    use crate::models::{Accelerator, Backend};
    use ort::memory::Allocator;
    use ort::session::{Session, SessionInputValue};
    use ort::value::{DynTensor, DynValue, Tensor, TensorElementType, ValueType};
    use parking_lot::Mutex;
    use std::borrow::Cow;
    use std::time::Instant;

    fn ml(what: &str, e: impl std::fmt::Display) -> Error {
        Error::Model(format!("Chatterbox {what}: {e}"))
    }

    /// One past/present pair of the KV cache.
    struct Kv {
        input: String,
        output: String,
        ty: TensorElementType,
        heads: usize,
        head_dim: usize,
    }

    struct Lm {
        session: Session,
        kv: Vec<Kv>,
    }

    /// Chatterbox-Turbo: voice prints from reference speech, speech from text
    /// in a print's voice. Thread-safe; calls are serialised per graph.
    pub struct Chatterbox {
        tokenizer: tokenizers::Tokenizer,
        encoder_path: std::path::PathBuf,
        encoder: Mutex<Option<(Session, Backend)>>,
        encoder_accel: Accelerator,
        embed: Mutex<Session>,
        lm: Mutex<Lm>,
        /// The decoder, and whether it is on the CPU already (a GPU failure at
        /// run time reopens it there once).
        decoder: Mutex<(Session, bool)>,
        decoder_path: std::path::PathBuf,
        engines: Engines,
        last: Mutex<PieceStats>,
    }

    fn backend_name(b: Backend) -> String {
        match b {
            Backend::Cpu => "cpu",
            Backend::DirectMl => "directml",
            Backend::TensorRtRtx => "tensorrt-rtx",
            Backend::Cuda => "cuda",
        }
        .into()
    }

    impl Chatterbox {
        /// Load from the model manager (`Error::ModelMissing` when not
        /// downloaded); graphs run where [`crate::models::accelerator`] says.
        pub fn load(models: &crate::models::ModelManager) -> Result<Chatterbox> {
            let id = crate::models::ModelId::TtsChatterbox;
            let dir = models.path(id).ok_or_else(|| Error::ModelMissing { model: id.name().to_string() })?;
            Chatterbox::load_dir(&dir, crate::models::accelerator())
        }

        /// Load from a folder holding the graph files. The speech encoder
        /// opens on first use (it runs once per voice).
        pub fn load_dir(dir: &Path, accel: Accelerator) -> Result<Chatterbox> {
            let need = |role: Role| graph_for(dir, role).ok_or_else(|| Error::Model(format!("the Chatterbox {} graph is missing in {}; download the voice-clone model again", role.stem(), dir.display())));
            let tok_path = dir.join(TOKENIZER);
            let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| Error::Model(format!("could not read {} ({e}); download the voice-clone model again", tok_path.display())))?;
            let accel = if crate::models::gpu_supported() { accel } else { Accelerator::Cpu };
            // The CPU, whatever the setting: on DirectML the language model
            // runs but computes wrong tokens (q4 and fp16 alike: Whisper heard
            // unrelated sentences, every line ran to the token limit) and is
            // 6-10x slower (99-163 ms/token vs 17 on the CPU). Measured on an
            // RTX 4070 Ti with `make_voice_from_the_recorded_sample`.
            // NEKOTONE_CHATTERBOX_LM=gpu tries the setting's engine anyway
            // (=cpu forces the CPU). CUDA builds (a compute server) run it on
            // the GPU: CUDA has none of DirectML's faults here.
            let lm_accel = match std::env::var("NEKOTONE_CHATTERBOX_LM").ok().as_deref() {
                Some("gpu") => accel,
                Some("cpu") => Accelerator::Cpu,
                _ if cfg!(feature = "cuda") => accel,
                _ => Accelerator::Cpu,
            };
            let (embed, eb) = crate::models::onnx_session_backend(&need(Role::EmbedTokens)?, Accelerator::Cpu)?;
            let (lm, lb) = crate::models::onnx_session_backend(&need(Role::LanguageModel)?, lm_accel)?;
            let (decoder, db) = crate::models::onnx_session_backend(&need(Role::Decoder)?, accel)?;
            let kv = kv_pairs(&lm)?;
            Ok(Chatterbox {
                tokenizer,
                encoder_path: need(Role::SpeechEncoder)?,
                encoder: Mutex::new(None),
                // The CPU on DirectML builds: DirectML fails at run time in the
                // encoder's MultiHeadAttention ("The parameter is incorrect"),
                // after the session opened fine. CUDA builds use the GPU (the
                // encoder runs on every Change my voice phrase there).
                encoder_accel: if cfg!(feature = "cuda") { accel } else { Accelerator::Cpu },
                embed: Mutex::new(embed),
                lm: Mutex::new(Lm { session: lm, kv }),
                decoder: Mutex::new((decoder, db == Backend::Cpu)),
                decoder_path: need(Role::Decoder)?,
                engines: Engines { speech_encoder: String::new(), embed_tokens: backend_name(eb), language_model: backend_name(lb), decoder: backend_name(db) },
                last: Mutex::new(PieceStats::default()),
            })
        }

        /// Where each graph runs (the encoder's is empty until first used).
        pub fn engines(&self) -> Engines {
            let mut e = self.engines.clone();
            if let Some((_, b)) = self.encoder.lock().as_ref() {
                e.speech_encoder = backend_name(*b);
            }
            e
        }

        /// "cpu", "directml"… of the language model (the part that sets the pace).
        pub fn device(&self) -> String {
            self.engines.language_model.clone()
        }

        /// Timings of the last piece.
        pub fn last_stats(&self) -> PieceStats {
            *self.last.lock()
        }

        /// Encode a reference recording (any rate, mono) into a voice print.
        pub fn encode_voice(&self, samples: &[f32], rate: u32, name: &str, source: &str) -> Result<VoicePrint> {
            let x = prepare_reference(samples, rate)?;
            let secs = x.len() as f32 / SAMPLE_RATE as f32;
            let mut guard = self.encoder.lock();
            if guard.is_none() {
                *guard = Some(crate::models::onnx_session_backend(&self.encoder_path, self.encoder_accel)?);
            }
            let (sess, _) = guard.as_mut().expect("opened above");
            let n = x.len();
            let t = Tensor::from_array(([1usize, n], x)).map_err(|e| ml("encoder input", e))?;
            let out = sess.run(ort::inputs!["audio_values" => t]).map_err(|e| ml("speech encoder", e))?;
            let (fs, feats) = out["audio_features"].try_extract_tensor::<f32>().map_err(|e| ml("audio_features", e))?;
            let (_, toks) = out["audio_tokens"].try_extract_tensor::<i64>().map_err(|e| ml("audio_tokens", e))?;
            let (_, emb) = out["speaker_embeddings"].try_extract_tensor::<f32>().map_err(|e| ml("speaker_embeddings", e))?;
            let (ss, sf) = out["speaker_features"].try_extract_tensor::<f32>().map_err(|e| ml("speaker_features", e))?;
            let dims3 = |s: &[i64]| -> Result<[usize; 3]> {
                if s.len() == 3 && s.iter().all(|&d| d >= 0) {
                    Ok([s[0] as usize, s[1] as usize, s[2] as usize])
                } else {
                    Err(Error::Model(format!("Chatterbox speech encoder returned shape {s:?}; download the voice-clone model again")))
                }
            };
            let print = VoicePrint {
                features_shape: dims3(fs)?,
                audio_features: feats.to_vec(),
                audio_tokens: toks.to_vec(),
                speaker_embeddings: emb.to_vec(),
                speaker_features_shape: dims3(ss)?,
                speaker_features: sf.to_vec(),
                info: VoicePrintInfo {
                    name: name.to_string(),
                    reference_secs: secs,
                    created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
                    source: source.to_string(),
                    model: PRINT_MODEL.to_string(),
                },
            };
            if print.audio_features.iter().chain(print.speaker_embeddings.iter()).any(|v| !v.is_finite()) {
                return Err(Error::Model("the speech encoder returned invalid numbers for this recording; try another recording".into()));
            }
            Ok(print)
        }

        /// The speech tokens (25 per second) and the speaker x-vector of a
        /// recording, from the speech encoder (up to 15 s per call).
        fn analyse(&self, x: Vec<f32>) -> Result<(Vec<i64>, Vec<f32>)> {
            let mut guard = self.encoder.lock();
            if guard.is_none() {
                *guard = Some(crate::models::onnx_session_backend(&self.encoder_path, self.encoder_accel)?);
            }
            let (sess, _) = guard.as_mut().expect("opened above");
            let n = x.len();
            let t = Tensor::from_array(([1usize, n], x)).map_err(|e| ml("encoder input", e))?;
            let out = sess.run(ort::inputs!["audio_values" => t]).map_err(|e| ml("speech encoder", e))?;
            let (_, toks) = out["audio_tokens"].try_extract_tensor::<i64>().map_err(|e| ml("audio_tokens", e))?;
            let (_, emb) = out["speaker_embeddings"].try_extract_tensor::<f32>().map_err(|e| ml("speaker_embeddings", e))?;
            Ok((toks.to_vec(), emb.to_vec()))
        }

        /// Speaker x-vector of a recording (for likeness checks).
        pub fn speaker_vector(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
            let x = prepare_reference(samples, rate)?;
            Ok(self.analyse(x)?.1)
        }

        /// Voice conversion: say what `samples` says, with its timing and
        /// delivery, in the print's voice. The recording's speech tokens (the
        /// words and their timing, no voice) are decoded with the print's
        /// voice; long recordings go in pieces of up to 12 s. 24 kHz out.
        pub fn convert(&self, samples: &[f32], rate: u32, print: &VoicePrint) -> Result<Vec<f32>> {
            let clip = crate::audio::Clip { samples: samples.to_vec(), sample_rate: rate, source_channels: 1 };
            let x = if rate == SAMPLE_RATE { clip.samples } else { crate::audio::resample(&clip, SAMPLE_RATE)?.samples };
            if x.len() < SAMPLE_RATE as usize / 10 {
                return Ok(Vec::new());
            }
            // the encoder expects speech at a normal level
            let lufs = crate::process::integrated_lufs(&x, 1, SAMPLE_RATE);
            let gain = match lufs {
                Some(l) if l.is_finite() && l > -70.0 => 10f32.powf((REFERENCE_LUFS - l) / 20.0).min(30.0),
                _ => 1.0,
            };
            let x: Vec<f32> = x.iter().map(|v| (v * gain).clamp(-1.0, 1.0)).collect();
            // a single short word (under 0.5 s; under 0.25 s it was dropped)
            // gets 0.4 s of silence either side, cut off again after
            // (conversion keeps the timing). Longer phrases convert as well
            // without it: convert_short_phrases, 0.4-1 s phrases of the
            // owner, 3 of 7 understood padded vs 4 of 7 not (run-to-run noise)
            let mut pad = if x.len() < SAMPLE_RATE as usize / 2 { (0.4 * SAMPLE_RATE as f32) as usize } else { 0 };
            if cfg!(test) && std::env::var_os("NEKOTONE_CONVERT_NO_PAD").is_some() {
                pad = 0; // the measurement below compares with and without
            }
            let x: Vec<f32> = if pad > 0 { std::iter::repeat_n(0.0, pad).chain(x.iter().copied()).chain(std::iter::repeat_n(0.0, pad)).collect() } else { x };
            // even pieces of at most 12 s (a short last piece was skipped)
            let pieces = x.len().div_ceil(12 * SAMPLE_RATE as usize).max(1);
            let piece = x.len().div_ceil(pieces);
            let mut out = Vec::new();
            for chunk in x.chunks(piece) {
                let (toks, _) = self.analyse(chunk.to_vec())?;
                if toks.is_empty() {
                    continue;
                }
                out.extend(self.decode(print, &toks)?);
            }
            if pad > 0 && out.len() > 2 * pad {
                out.truncate(out.len() - pad);
                out.drain(..pad);
            }
            Ok(out)
        }

        /// Text token ids (the tokenizer's template appends two `<|endoftext|>`).
        pub fn text_ids(&self, text: &str) -> Result<Vec<i64>> {
            let enc = self.tokenizer.encode(text, true).map_err(|e| ml("tokenizer", e))?;
            Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
        }

        fn embed_ids(&self, ids: &[i64]) -> Result<(Vec<f32>, usize)> {
            let n = ids.len();
            let t = Tensor::from_array(([1usize, n], ids.to_vec())).map_err(|e| ml("embed input", e))?;
            let mut s = self.embed.lock();
            let out = s.run(ort::inputs!["input_ids" => t]).map_err(|e| ml("embed_tokens", e))?;
            let (shape, v) = out["inputs_embeds"].try_extract_tensor::<f32>().map_err(|e| ml("inputs_embeds", e))?;
            let d = *shape.last().unwrap_or(&0) as usize;
            Ok((v.to_vec(), d))
        }

        /// Generate the speech tokens for one piece of text (already cleaned).
        pub fn generate_tokens(&self, text: &str, print: &VoicePrint, sampling: &Sampling, max_tokens: usize, cancel: &dyn Fn() -> bool) -> Result<(Vec<i64>, PieceStats)> {
            let t0 = Instant::now();
            let ids = self.text_ids(text)?;
            let mut st = PieceStats { text_tokens: ids.len(), ..Default::default() };
            let (text_emb, d) = self.embed_ids(&ids)?;
            let [_, t_cond, d_cond] = print.features_shape;
            if d != d_cond || d == 0 {
                return Err(Error::Model(format!("this voice print was made with another model (width {d_cond}, model {d}); make the voice again")));
            }
            let mut prefix = Vec::with_capacity(print.audio_features.len() + text_emb.len());
            prefix.extend_from_slice(&print.audio_features);
            prefix.extend_from_slice(&text_emb);
            let seq = t_cond + ids.len();
            let mut lm = self.lm.lock();
            let lm = &mut *lm;
            // empty cache
            let alloc = Allocator::default();
            let mut past: Vec<DynValue> = Vec::with_capacity(lm.kv.len());
            for kv in &lm.kv {
                let v = DynTensor::new(&alloc, kv.ty, [1usize, kv.heads, 0, kv.head_dim]).map_err(|e| ml("cache", e))?;
                past.push(v.into_dyn());
            }
            let mut rng = Rng::new(sampling.seed ^ fnv(text.as_bytes()));
            let mut history: Vec<i64> = vec![START_SPEECH_TOKEN];
            let mut embeds = prefix;
            let mut embeds_len = seq;
            let mut total = seq;
            let mut pos_start = 0i64;
            let mut t_steps = 0f32;
            let mut steps = 0usize;
            loop {
                let ts = Instant::now();
                let e = Tensor::from_array(([1usize, embeds_len, d], std::mem::take(&mut embeds))).map_err(|e| ml("lm input", e))?;
                let mask = Tensor::from_array(([1usize, total], vec![1i64; total])).map_err(|e| ml("lm input", e))?;
                let pos = Tensor::from_array(([1usize, embeds_len], (0..embeds_len as i64).map(|k| pos_start + k).collect::<Vec<_>>())).map_err(|e| ml("lm input", e))?;
                let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = Vec::with_capacity(3 + past.len());
                inputs.push(("inputs_embeds".into(), e.into()));
                inputs.push(("attention_mask".into(), mask.into()));
                inputs.push(("position_ids".into(), pos.into()));
                for (kv, v) in lm.kv.iter().zip(past.drain(..)) {
                    inputs.push((Cow::Owned(kv.input.clone()), v.into()));
                }
                let mut out = lm.session.run(inputs).map_err(|e| ml("language model", e))?;
                let (shape, logits) = out["logits"].try_extract_tensor::<f32>().map_err(|e| ml("logits", e))?;
                let vocab = *shape.last().unwrap_or(&0) as usize;
                if vocab == 0 || logits.len() < vocab {
                    return Err(Error::Model("the Chatterbox language model returned no logits; download the voice-clone model again".into()));
                }
                let mut last = logits[logits.len() - vocab..].to_vec();
                for kv in &lm.kv {
                    past.push(out.remove(kv.output.as_str()).ok_or_else(|| Error::Model(format!("Chatterbox language model: no output {}", kv.output)))?);
                }
                drop(out);
                let next = sample_next(&mut last, &history, sampling, &mut rng);
                history.push(next);
                if steps == 0 {
                    st.prefill_ms = ts.elapsed().as_secs_f32() * 1000.0;
                } else {
                    t_steps += ts.elapsed().as_secs_f32() * 1000.0;
                }
                steps += 1;
                if next == STOP_SPEECH_TOKEN {
                    break;
                }
                if history.len() > max_tokens {
                    st.hit_limit = true;
                    break;
                }
                if steps % 8 == 0 && cancel() {
                    break;
                }
                // the new token's embedding
                let ts2 = Instant::now();
                let (emb, _) = self.embed_ids(&[next])?;
                t_steps += ts2.elapsed().as_secs_f32() * 1000.0;
                embeds = emb;
                embeds_len = 1;
                // position = tokens already in the cache
                pos_start = total as i64;
                total += 1;
            }
            let generated: Vec<i64> = history[1..].iter().copied().filter(|&t| t != STOP_SPEECH_TOKEN).collect();
            st.speech_tokens = generated.len();
            st.step_ms = if steps > 1 { t_steps / (steps - 1) as f32 } else { 0.0 };
            st.total_ms = t0.elapsed().as_secs_f32() * 1000.0;
            Ok((generated, st))
        }

        /// Speech tokens → 24 kHz audio in the print's voice.
        pub fn decode(&self, print: &VoicePrint, generated: &[i64]) -> Result<Vec<f32>> {
            let toks = decoder_tokens(&print.audio_tokens, generated);
            let n = toks.len();
            let t_tok = Tensor::from_array(([1usize, n], toks)).map_err(|e| ml("decoder input", e))?;
            let ne = print.speaker_embeddings.len();
            let t_emb = Tensor::from_array(([1usize, ne], print.speaker_embeddings.clone())).map_err(|e| ml("decoder input", e))?;
            let [a, b, c] = print.speaker_features_shape;
            let t_feat = Tensor::from_array(([a, b, c], print.speaker_features.clone())).map_err(|e| ml("decoder input", e))?;
            let mut g = self.decoder.lock();
            let run = |s: &mut Session| -> Result<Vec<f32>> {
                let out = s
                    .run(ort::inputs!["speech_tokens" => t_tok.clone(), "speaker_embeddings" => t_emb.clone(), "speaker_features" => t_feat.clone()])
                    .map_err(|e| ml("decoder", e))?;
                let (_, w) = out["waveform"].try_extract_tensor::<f32>().map_err(|e| ml("waveform", e))?;
                Ok(w.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect())
            };
            match run(&mut g.0) {
                Ok(w) => Ok(w),
                // a graph can open on the GPU and still fail when it runs
                // (the speech encoder did on DirectML): move to the CPU once
                Err(e) if !g.1 => {
                    log::warn!("Chatterbox decoder failed on the GPU ({e}); moving it to the CPU");
                    g.0 = crate::models::onnx_session_with(&self.decoder_path, Accelerator::Cpu)?;
                    g.1 = true;
                    run(&mut g.0)
                }
                Err(e) => Err(e),
            }
        }

        /// One piece of text (see [`plan`]) → 24 kHz audio, silence trimmed to ~30 ms.
        pub fn synthesize_piece(&self, text: &str, print: &VoicePrint, sampling: &Sampling, cancel: &dyn Fn() -> bool) -> Result<Vec<f32>> {
            let t0 = Instant::now();
            let (toks, mut st) = self.generate_tokens(text, print, sampling, max_tokens_for(text), cancel)?;
            let td = Instant::now();
            let mut audio = if toks.is_empty() { Vec::new() } else { self.decode(print, &toks)? };
            st.decode_ms = td.elapsed().as_secs_f32() * 1000.0;
            st.total_ms = t0.elapsed().as_secs_f32() * 1000.0;
            super::super::trim_silence(&mut audio, SAMPLE_RATE, 0.03);
            st.audio_secs = audio.len() as f32 / SAMPLE_RATE as f32;
            if std::env::var_os("NEKOTONE_TTS_DEBUG").is_some() {
                eprintln!("chatterbox: {st:?} for {text:?}");
            }
            *self.last.lock() = st;
            Ok(audio)
        }

        /// Speak `text` (any length) in the print's voice, with pauses.
        pub fn synthesize(&self, text: &str, print: &VoicePrint, sampling: &Sampling) -> Result<Vec<f32>> {
            let mut all = Vec::new();
            for u in plan(text, MAX_PIECE_CHARS) {
                let a = self.synthesize_piece(&u.phonemes, print, sampling, &|| false)?;
                all.extend_from_slice(&a);
                all.extend(std::iter::repeat_n(0.0, (u.pause_after * SAMPLE_RATE as f32) as usize));
            }
            Ok(all)
        }
    }

    /// Pair `past_key_values.N.key/value` inputs with `present.N.key/value` outputs.
    fn kv_pairs(s: &Session) -> Result<Vec<Kv>> {
        let outs: Vec<String> = s.outputs().iter().map(|o| o.name().to_string()).collect();
        let mut kv = Vec::new();
        for i in s.inputs() {
            let name = i.name();
            let Some(rest) = name.strip_prefix("past_key_values.") else { continue };
            let out = format!("present.{rest}");
            if !outs.contains(&out) {
                return Err(Error::Model(format!("the Chatterbox language model has {name} but no {out}; download the voice-clone model again")));
            }
            let (ty, heads, head_dim) = match i.dtype() {
                ValueType::Tensor { ty, shape, .. } => {
                    let h = shape.get(1).copied().filter(|&v| v > 0).unwrap_or(16) as usize;
                    let d = shape.get(3).copied().filter(|&v| v > 0).unwrap_or(64) as usize;
                    (*ty, h, d)
                }
                other => return Err(Error::Model(format!("the Chatterbox cache input {name} is {other:?}, not a tensor"))),
            };
            kv.push(Kv { input: name.to_string(), output: out, ty, heads, head_dim });
        }
        if kv.is_empty() {
            return Err(Error::Model("the Chatterbox language model has no cache inputs; download the voice-clone model again".into()));
        }
        Ok(kv)
    }

    fn fnv(b: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for x in b {
            h ^= *x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// ───────────────────────── voices on disk ─────────────────────────

/// File extension of saved voice prints.
pub const PRINT_EXT: &str = "nkvoice";

/// Voice id of a cloned voice in Speak for me: `clone:<name>`.
pub const CLONE_PREFIX: &str = "clone:";

/// The print file for voice `name` in `dir` (name = letters, digits, '-', '_').
pub fn print_path(dir: &Path, name: &str) -> std::path::PathBuf {
    let safe: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    dir.join(format!("{safe}.{PRINT_EXT}"))
}

#[cfg(test)]
#[path = "chatterbox_tests.rs"]
pub(crate) mod tests;
