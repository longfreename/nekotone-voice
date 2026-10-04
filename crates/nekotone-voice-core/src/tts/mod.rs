//! Text to speech: Kokoro-82M (hexgrad, Apache-2.0) through ONNX Runtime,
//! with an English front end that needs no GPL code (owner: the speak
//! package; `g2p`, `normalize` and `lts` by the G2P package).
//!
//! ```text
//! text ─ normalize (numbers, dates, money, abbreviations…) ─ sentences ─ G2P (misaki lexicons → morphology → letter-to-sound)
//!      ─ phoneme chunks ≤ 510 tokens, cut at clause boundaries ─ Kokoro (tokens + 256-d voice style + speed) ─ 24 kHz mono
//! ```
//!
//! * Model: the `onnx-community/Kokoro-82M-v1.0-ONNX` fp32 graph
//!   (`model.onnx`, 326 MB), pinned to a commit. Measured alternatives (HANDOFF
//!   §5c): the fp16 graph returns NaN for every sentence longer than a few
//!   words on the ONNX Runtime this build links (fine in ORT 1.30 from
//!   Python), at every optimisation level; `model_quantized` (8-bit) runs at
//!   only 1.1× real time on the CPU (dynamic quantisation of the convolutions);
//!   `uint8f16` (114 MB) runs but changes durations and halves the level.
//! * Voices: one `<id>.bin` per voice, `float32 [510, 1, 256]`: the style
//!   vector row is picked by the number of phoneme tokens.
//! * Lexicons: misaki (hexgrad, Apache-2.0) `us_gold/us_silver` and
//!   `gb_gold/gb_silver`; the British lexicon is loaded on first use of a
//!   British voice.
//! * Inputs `input_ids int64 [1, n+2]` (0 = pad at both ends), `style f32
//!   [1, 256]`, `speed f32 [1]`; output `waveform f32 [1, samples]` at 24 kHz.

pub mod chatterbox;
pub mod accent;
pub mod g2p;
pub mod lts;
pub mod normalize;

#[cfg(test)]
pub(crate) mod tests;

use crate::models::ModelFile;
use crate::{Error, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Kokoro's output rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// Most phoneme tokens one Kokoro call takes (context 512 minus the two pads).
pub const MAX_TOKENS: usize = 510;
/// Licence of the model and of the lexicons.
pub const MODEL_LICENSE: &str = "Apache-2.0 (Kokoro-82M by hexgrad; misaki lexicons by hexgrad)";

/// File names inside the model folder.
pub const KOKORO_MODEL: &str = "kokoro-v1.0.onnx";
pub const KOKORO_TOKENIZER: &str = "tokenizer.json";

const REPO: &str = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231";
const MISAKI: &str = "https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3";

/// A voice Kokoro can speak with.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct VoiceInfo {
    /// File stem, e.g. "af_heart".
    pub id: &'static str,
    /// Display name, e.g. "Heart".
    pub name: &'static str,
    /// "female" or "male".
    pub gender: &'static str,
    /// "american" or "british".
    pub accent: &'static str,
    /// Upstream quality grade (A best … F), from Kokoro's VOICES.md.
    pub grade: &'static str,
    /// Shown first in the gallery (the best-graded voices).
    pub featured: bool,
    /// One line for the gallery.
    pub character: &'static str,
    #[serde(skip)]
    sha256: &'static str,
}

const fn v(id: &'static str, name: &'static str, grade: &'static str, featured: bool, character: &'static str, sha256: &'static str) -> VoiceInfo {
    let b = id.as_bytes();
    let gender = if b[1] == b'f' { "female" } else { "male" };
    let accent = if b[0] == b'a' { "american" } else { "british" };
    VoiceInfo { id, name, gender, accent, grade, featured, character, sha256 }
}

/// Size of every voice file (`float32 [510, 1, 256]`).
const VOICE_BYTES: u64 = 522_240;

static VOICES: &[VoiceInfo] = &[
    v("af_heart", "Heart", "A", true, "warm, natural; the best voice", "d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b"),
    v("af_bella", "Bella", "A-", true, "bright and expressive", "f69d836209b78eb8c66e75e3cda491e26ea838a3674257e9d4e5703cbaf55c8b"),
    v("af_nicole", "Nicole", "B-", true, "soft, close to the microphone", "cd2191ab31b914ed7b318416b0e4440fdf392ddad9106a060819aa600a64f59a"),
    v("af_aoede", "Aoede", "C+", false, "calm", "4a004c33430762e2461eedb2013fad808ef4ab3121f5300f554476caf58d8361"),
    v("af_kore", "Kore", "C+", false, "clear, even", "9be5221b6a941c04b561959b8ff0b06e809444dcc4ab7e75a7b23606f691819e"),
    v("af_sarah", "Sarah", "C+", false, "friendly", "4409fbc125afabacc615d94db5398d847006a737b0247d6892b7a9a0007a2f0a"),
    v("af_alloy", "Alloy", "C", false, "neutral", "c4a6b876047fd7fb472edf4ebd63cfac7c3b958a7cae7c106e8f038ca6308c45"),
    v("af_nova", "Nova", "C", false, "crisp", "18778272caa0d0eebaea251c35fd635f038434f9eee5e691d02a174bd328414f"),
    v("af_sky", "Sky", "C-", false, "light", "4435255c9744f3f31659e0d714ab7689bf65d9e77ec1cce060f083912614f0b9"),
    v("af_jessica", "Jessica", "D", false, "casual", "a240a5e3c15b43563d6e923bdca8ef5613a23471d9b77653694012435df23bd8"),
    v("af_river", "River", "D", false, "relaxed", "00a2bcf82b1d86e8f19902ede58c65ccf6c0e43b44b7d74fad54e5d8933c9c30"),
    v("am_michael", "Michael", "C+", true, "steady, mid-range", "1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1"),
    v("am_fenrir", "Fenrir", "C+", true, "deep and strong", "c27989f741f7ee34d273a39d8a595cc0837d35f5ced9a29b7cc162614616df43"),
    v("am_puck", "Puck", "C+", true, "lively, playful", "fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0"),
    v("am_echo", "Echo", "D", false, "plain", "3968b92c3c4cd1c4416dbded36c13eaa388a90d5788d02a13e4d781f5f8cf3c3"),
    v("am_eric", "Eric", "D", false, "brisk", "e8b5be17edd1e3636901ce7598baafe2dc8dd8ff707a0c23bf9e461add7e2832"),
    v("am_liam", "Liam", "D", false, "young", "52403be32fd047c6a44517cb0bcd6b134f2a18baa73e70ef41651e0eab921ade"),
    v("am_onyx", "Onyx", "D", false, "low", "da5d135b424164916d75a68ffb4c2abce3d7d5ccc82dd1ee6cf447ce286145e6"),
    v("am_santa", "Santa", "D-", false, "jolly", "61150cf726ab6c5ed7a99f90a304f91f5a72c00c592e89ec94e5df11c319227a"),
    v("am_adam", "Adam", "F+", false, "rough", "162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716"),
    v("bf_emma", "Emma", "B-", true, "British, polished", "669fe0647f9dd04fcab92f1439a40eeb4c8b4ab1f82e4996fe3d918ce4a63b73"),
    v("bf_isabella", "Isabella", "C", false, "British, warm", "3754352c4aaa46d17f27654ab7518d65b62ad6163a0f55a5f4330c2da2c4e94f"),
    v("bf_alice", "Alice", "D", false, "British, light", "08afa6ba24da61ea5e8efa139e5aadc938d83f0a6da5a900adaf763ac1da5573"),
    v("bf_lily", "Lily", "D", false, "British, gentle", "5e0ee32ebe64a467124976b14e69590746f1c4ce41a12b587a50c862edfea335"),
    v("bm_george", "George", "C", true, "British, measured", "c4b235a4c1f2cd3b939fed08b899ce9385638b763f7b73a59616c4fc9bd6c9bc"),
    v("bm_fable", "Fable", "C", false, "British, storyteller", "f889083196807b4adb15e9204252165f503b8d33d3982e681c52443c49d798f1"),
    v("bm_lewis", "Lewis", "D+", false, "British, low", "b8f671cef828c30e66fdf0b0756a76bba58f6bb3398cbbf27058642acbcedb97"),
    v("bm_daniel", "Daniel", "D", false, "British, dry", "6b3194bbceffb746733cbc22c8f593dd44e401a71d53895a2dca891bc595a1e8"),
];

/// Every voice this build ships (no model needed to list them).
pub fn voices() -> Vec<VoiceInfo> {
    VOICES.to_vec()
}

/// The voice with this id.
pub fn voice(id: &str) -> Option<&'static VoiceInfo> {
    VOICES.iter().find(|v| v.id == id)
}

/// The voice used when none is chosen.
pub const DEFAULT_VOICE: &str = "af_heart";

/// The catalogue files for `ModelId::TtsKokoro` (pinned URLs, sizes, SHA-256).
pub fn model_files() -> Vec<ModelFile> {
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let mut files = vec![
        ModelFile {
            name: KOKORO_MODEL,
            url: leak(format!("{REPO}/onnx/model.onnx")),
            size_bytes: 325_532_232,
            sha256: "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb",
        },
        ModelFile { name: KOKORO_TOKENIZER, url: leak(format!("{REPO}/tokenizer.json")), size_bytes: 3_497, sha256: "77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34" },
        ModelFile { name: "KOKORO-README.md", url: leak(format!("{REPO}/README.md")), size_bytes: 10_627, sha256: "b8fd888b4782f4d4ae85e2f1587cafd727037f4f138f72429ef1bd5908c0e9d8" },
        ModelFile { name: "us_gold.json", url: leak(format!("{MISAKI}/misaki/data/us_gold.json")), size_bytes: 3_000_469, sha256: "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f" },
        ModelFile { name: "us_silver.json", url: leak(format!("{MISAKI}/misaki/data/us_silver.json")), size_bytes: 3_099_517, sha256: "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475" },
        ModelFile { name: "gb_gold.json", url: leak(format!("{MISAKI}/misaki/data/gb_gold.json")), size_bytes: 2_838_552, sha256: "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71" },
        ModelFile { name: "gb_silver.json", url: leak(format!("{MISAKI}/misaki/data/gb_silver.json")), size_bytes: 3_663_898, sha256: "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce" },
        ModelFile { name: "MISAKI-LICENSE", url: leak(format!("{MISAKI}/LICENSE")), size_bytes: 11_357, sha256: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4" },
    ];
    for vi in VOICES {
        files.push(ModelFile {
            name: leak(format!("{}.bin", vi.id)),
            url: leak(format!("{REPO}/voices/{}.bin", vi.id)),
            size_bytes: VOICE_BYTES,
            sha256: vi.sha256,
        });
    }
    files
}

// ───────────────────────── text → phoneme chunks ─────────────────────────

/// One piece of speech: a sentence (or part of a long one).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Utterance {
    /// The (normalised) text it came from, for captions.
    pub text: String,
    pub phonemes: String,
    /// Silence to leave after it (s): longer after a full stop than a comma.
    pub pause_after: f32,
}

/// Split normalised text into sentences: after `.`, `!`, `?` or `…`
/// (plus closing quotes/brackets) followed by white space, and at line
/// breaks. Sentence punctuation stays with its sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' || c == '\r' {
            push_trimmed(&mut out, &mut cur);
            i += 1;
            continue;
        }
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | '…') {
            // swallow more terminators and closing quotes/brackets
            while i + 1 < chars.len() && matches!(chars[i + 1], '.' | '!' | '?' | '…' | '"' | '\'' | '”' | '’' | ')' | ']') {
                i += 1;
                cur.push(chars[i]);
            }
            if i + 1 >= chars.len() || chars[i + 1].is_whitespace() {
                push_trimmed(&mut out, &mut cur);
            }
        }
        i += 1;
    }
    push_trimmed(&mut out, &mut cur);
    out
}

fn push_trimmed(out: &mut Vec<String>, cur: &mut String) {
    let t = cur.trim();
    if t.chars().any(|c| c.is_alphanumeric()) {
        out.push(t.to_string());
    }
    cur.clear();
}

/// How long to pause after a chunk ending in `last` (a phoneme string's last char).
fn pause_for(phonemes: &str, sentence_end: bool) -> f32 {
    let last = phonemes.trim_end().chars().last().unwrap_or(' ');
    match last {
        '.' | '!' | '?' | '…' if sentence_end => 0.22,
        '.' | '!' | '?' | '…' => 0.18,
        ',' | ';' | ':' | '—' => 0.10,
        _ if sentence_end => 0.18,
        _ => 0.05,
    }
}

/// Cut a phoneme string into pieces of at most `max` characters, preferring
/// clause punctuation, then word boundaries. Every piece keeps its
/// punctuation. `first_max` (≤ `max`) limits only the first piece, so
/// streaming can start speaking a long sentence sooner.
pub fn chunk_phonemes(ps: &str, max: usize, first_max: usize) -> Vec<String> {
    let chars: Vec<char> = ps.trim().chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let limit = if out.is_empty() { first_max.min(max) } else { max }.max(8);
        let rest = chars.len() - start;
        if rest <= limit {
            out.push(chars[start..].iter().collect::<String>().trim().to_string());
            break;
        }
        let window = &chars[start..start + limit];
        // last clause punctuation followed by a space, not too early
        let min_cut = limit / 3;
        let cut = window
            .iter()
            .enumerate()
            .rev()
            .find(|&(i, c)| i >= min_cut && matches!(c, ',' | ';' | ':' | '—' | '.' | '!' | '?' | '…') && window.get(i + 1).map(|n| *n == ' ').unwrap_or(true))
            .map(|(i, _)| i + 1)
            .or_else(|| window.iter().rposition(|c| *c == ' ').filter(|&i| i > 0))
            .unwrap_or(limit);
        out.push(chars[start..start + cut].iter().collect::<String>().trim().to_string());
        start += cut;
        while start < chars.len() && chars[start] == ' ' {
            start += 1;
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

// ───────────────────────── the engine ─────────────────────────

/// Kokoro with its front end. Thread-safe: calls are serialised on the
/// ONNX session.
#[cfg(feature = "ml")]
pub struct Tts {
    dir: PathBuf,
    session: parking_lot::Mutex<ort::session::Session>,
    vocab: HashMap<char, i64>,
    us: g2p::G2p,
    gb: std::sync::OnceLock<std::result::Result<g2p::G2p, String>>,
    styles: parking_lot::Mutex<HashMap<String, std::sync::Arc<Vec<f32>>>>,
    device: &'static str,
}

/// Parse `tokenizer.json` → phoneme character → id.
pub fn read_vocab(path: &Path) -> Result<HashMap<char, i64>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Model(format!("could not read {} ({e}); download the voice model again", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::Model(format!("{} is not valid JSON ({e}); download the voice model again", path.display())))?;
    let map = v.pointer("/model/vocab").and_then(|m| m.as_object()).ok_or_else(|| Error::Model(format!("{} has no model.vocab", path.display())))?;
    let mut out = HashMap::new();
    for (k, id) in map {
        let mut cs = k.chars();
        if let (Some(c), None, Some(id)) = (cs.next(), cs.next(), id.as_i64()) {
            out.insert(c, id);
        }
    }
    if out.len() < 100 {
        return Err(Error::Model(format!("{} lists only {} phonemes; download the voice model again", path.display(), out.len())));
    }
    Ok(out)
}

/// Token ids of a phoneme string (characters outside the vocabulary are skipped).
pub fn tokenize(vocab: &HashMap<char, i64>, phonemes: &str) -> Vec<i64> {
    phonemes.chars().filter_map(|c| vocab.get(&c).copied()).collect()
}

#[cfg(feature = "ml")]
impl Tts {
    /// Load from the model manager; `Error::ModelMissing` when Kokoro is not
    /// downloaded. Always on the CPU: DirectML rejects the graph's
    /// `ConvTranspose` (F0 upsampling) at the first sentence, and the CPU
    /// runs it at ~4× real time (HANDOFF §5c).
    pub fn load(models: &crate::models::ModelManager) -> Result<Tts> {
        let dir = models.path(crate::models::ModelId::TtsKokoro).ok_or_else(|| Error::ModelMissing { model: crate::models::ModelId::TtsKokoro.name().to_string() })?;
        Tts::load_dir(&dir, crate::models::Accelerator::Cpu)
    }

    /// Load from a folder holding the catalogue files. `accel` = `Cpu`
    /// forces the CPU; `Auto` tries DirectML first.
    pub fn load_dir(dir: &Path, accel: crate::models::Accelerator) -> Result<Tts> {
        let vocab = read_vocab(&dir.join(KOKORO_TOKENIZER))?;
        let us = g2p::G2p::new(g2p::Lexicon::load(dir, false)?);
        let path = dir.join(KOKORO_MODEL);
        let (session, device) = open_session(&path, accel)?;
        Ok(Tts {
            dir: dir.to_path_buf(),
            session: parking_lot::Mutex::new(session),
            vocab,
            us,
            gb: std::sync::OnceLock::new(),
            styles: parking_lot::Mutex::new(HashMap::new()),
            device,
        })
    }

    /// "cpu" or "directml".
    pub fn device(&self) -> &'static str {
        self.device
    }

    /// The voices whose files are present.
    pub fn voices(&self) -> Vec<VoiceInfo> {
        VOICES.iter().filter(|v| self.dir.join(format!("{}.bin", v.id)).is_file()).cloned().collect()
    }

    fn g2p_for(&self, british: bool) -> Result<&g2p::G2p> {
        if !british {
            return Ok(&self.us);
        }
        match self.gb.get_or_init(|| g2p::Lexicon::load(&self.dir, true).map(g2p::G2p::new).map_err(|e| e.to_string())) {
            Ok(g) => Ok(g),
            Err(e) => Err(Error::Model(e.clone())),
        }
    }

    /// Text → speakable chunks for `voice` (its accent picks the lexicon).
    /// `first_max` limits the first chunk's phoneme count (streaming:
    /// smaller = sooner first sound); pass [`MAX_TOKENS`] for no limit.
    pub fn plan(&self, text: &str, voice: &str, first_max: usize) -> Result<Vec<Utterance>> {
        self.plan_accent(text, voice, first_max, accent::Accent::Voice)
    }

    /// [`Tts::plan`] in an `accent` (any voice can speak any accent; see [`accent`]).
    pub fn plan_accent(&self, text: &str, voice: &str, first_max: usize, accent: accent::Accent) -> Result<Vec<Utterance>> {
        let voice_british = self::voice(voice).map(|v| v.accent == "british").unwrap_or(voice.starts_with('b'));
        let british = accent.british_base(voice_british);
        let g = self.g2p_for(british)?;
        let norm = normalize::normalize(text, british);
        let mut out = Vec::new();
        for sentence in split_sentences(&norm) {
            let ps = accent::apply(&g.phonemize(&sentence), accent);
            let ps = ps.trim();
            if tokenize(&self.vocab, ps).iter().all(|&t| t == 16 || t < 17) {
                continue; // nothing speakable (only punctuation / spaces)
            }
            let first = if out.is_empty() { first_max } else { MAX_TOKENS };
            let pieces = chunk_phonemes(ps, MAX_TOKENS, first);
            let n = pieces.len();
            for (k, p) in pieces.into_iter().enumerate() {
                let last = k + 1 == n;
                out.push(Utterance {
                    text: if n == 1 { sentence.clone() } else { format!("{sentence} ({}/{n})", k + 1) },
                    pause_after: pause_for(&p, last),
                    phonemes: p,
                });
            }
        }
        Ok(out)
    }

    fn style(&self, voice: &str) -> Result<std::sync::Arc<Vec<f32>>> {
        if let Some(s) = self.styles.lock().get(voice) {
            return Ok(s.clone());
        }
        let info = self::voice(voice).ok_or_else(|| Error::Model(format!("there is no voice called \"{voice}\"")))?;
        let path = self.dir.join(format!("{}.bin", info.id));
        let bytes = std::fs::read(&path).map_err(|e| Error::Model(format!("the voice file {} could not be read ({e}); download the voice model again", path.display())))?;
        if bytes.len() % (256 * 4) != 0 || bytes.is_empty() {
            return Err(Error::Model(format!("the voice file {} is damaged; download the voice model again", path.display())));
        }
        let v: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let v = std::sync::Arc::new(v);
        self.styles.lock().insert(voice.to_string(), v.clone());
        Ok(v)
    }

    /// Speak a phoneme string (≤ [`MAX_TOKENS`] tokens after filtering) at
    /// `speed` (0.5–2.0; 1 = natural). Returns 24 kHz mono with leading and
    /// trailing silence trimmed to ~30 ms.
    pub fn synthesize_phonemes(&self, phonemes: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        let mut ids = tokenize(&self.vocab, phonemes);
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        ids.truncate(MAX_TOKENS);
        let style = self.style(voice)?;
        let rows = style.len() / 256;
        let row = ids.len().min(rows - 1);
        let s = style[row * 256..(row + 1) * 256].to_vec();
        let n = ids.len() + 2;
        let mut input = Vec::with_capacity(n);
        input.push(0i64);
        input.extend_from_slice(&ids);
        input.push(0);
        let ml = |what: &str, e: ort::Error| Error::Model(format!("Kokoro {what}: {e}"));
        let t_ids = Tensor::from_array(([1usize, n], input)).map_err(|e| ml("input", e))?;
        let t_style = Tensor::from_array(([1usize, 256], s)).map_err(|e| ml("input", e))?;
        let t_speed = Tensor::from_array(([1usize], vec![speed.clamp(0.5, 2.0)])).map_err(|e| ml("input", e))?;
        let mut sess = self.session.lock();
        let out = sess.run(ort::inputs!["input_ids" => t_ids, "style" => t_style, "speed" => t_speed]).map_err(|e| ml("inference", e))?;
        let view = out["waveform"].try_extract_array::<f32>().map_err(|e| ml("output", e))?;
        if std::env::var_os("NEKOTONE_TTS_DEBUG").is_some() {
            let nan = view.iter().filter(|v| !v.is_finite()).count();
            let peak = view.iter().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
            eprintln!("kokoro: {} tokens → {} samples, {nan} not finite, peak {peak}", ids.len(), view.len());
        }
        let mut audio: Vec<f32> = view.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
        drop(out);
        drop(sess);
        trim_silence(&mut audio, SAMPLE_RATE, 0.03);
        Ok(audio)
    }

    /// Speak `text` in `voice`; one call per chunk, pauses between chunks.
    pub fn synthesize(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        self.synthesize_accent(text, voice, speed, accent::Accent::Voice)
    }

    /// [`Tts::synthesize`] in an `accent`.
    pub fn synthesize_accent(&self, text: &str, voice: &str, speed: f32, accent: accent::Accent) -> Result<Vec<f32>> {
        let mut all = Vec::new();
        for u in self.plan_accent(text, voice, 160, accent)? {
            let a = self.synthesize_phonemes(&u.phonemes, voice, speed)?;
            all.extend_from_slice(&a);
            all.extend(std::iter::repeat_n(0.0, (u.pause_after / speed.clamp(0.5, 2.0) * SAMPLE_RATE as f32) as usize));
        }
        Ok(all)
    }

    /// Streaming: `on_audio(chunk, samples)` is called as soon as each
    /// sentence (or clause of a long one) is ready, with its pause already
    /// appended; return false to stop early.
    pub fn synthesize_stream(&self, text: &str, voice: &str, speed: f32, on_audio: &mut dyn FnMut(&Utterance, &[f32]) -> bool) -> Result<()> {
        for u in self.plan(text, voice, 160)? {
            let mut a = self.synthesize_phonemes(&u.phonemes, voice, speed)?;
            a.extend(std::iter::repeat_n(0.0, (u.pause_after / speed.clamp(0.5, 2.0) * SAMPLE_RATE as f32) as usize));
            if !on_audio(&u, &a) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "ml")]
fn open_session(path: &Path, accel: crate::models::Accelerator) -> Result<(ort::session::Session, &'static str)> {
    use crate::models::Accelerator;
    let (s, gpu) = crate::models::onnx_session_where(path, if crate::models::gpu_supported() { accel } else { Accelerator::Cpu })?;
    Ok((s, if gpu { "directml" } else { "cpu" }))
}

/// Trim leading/trailing samples below −50 dBFS, keeping `keep_secs`.
pub fn trim_silence(audio: &mut Vec<f32>, rate: u32, keep_secs: f32) {
    let thr = 0.003f32;
    let keep = (keep_secs * rate as f32) as usize;
    let Some(first) = audio.iter().position(|v| v.abs() > thr) else {
        audio.clear();
        return;
    };
    let last = audio.iter().rposition(|v| v.abs() > thr).unwrap_or(first);
    let end = (last + 1 + keep).min(audio.len());
    audio.truncate(end);
    let start = first.saturating_sub(keep);
    audio.drain(..start);
    // 5 ms fades so a trimmed edge never clicks
    let fade = ((0.005 * rate as f32) as usize).min(audio.len() / 2);
    for i in 0..fade {
        let g = i as f32 / fade as f32;
        audio[i] *= g;
        let j = audio.len() - 1 - i;
        audio[j] *= g;
    }
}
