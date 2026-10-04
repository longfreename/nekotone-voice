//! The ONNX models Nekotone can use, where they live and how they are
//! fetched. Nothing is bundled: the first feature that needs a model asks
//! the manager, which downloads it (with progress) and verifies its size
//! and SHA-256. Models live in `data_dir()/models/<id>/`.
//!
//! Other modules only call `path()` / `ensure()`. The entries for the text
//! embedder, the two speaker models and the stem separator come from
//! `embed::model_files`, `diarize::{segmentation,embedding}_files` and
//! `stems::model_files`. [`onnx_session`] opens every graph on the best
//! engine available: NVIDIA TensorRT for RTX (an RTX card and the optional
//! runtime, see [`crate::nvidia`]), else DirectML (any DirectX 12 GPU),
//! else the CPU. Each step falls back to the next on any failure.
//!
//! ## What is in the catalogue
//!
//! * **Whisper tiny/base/small/medium** – the `onnx-community` exports on
//!   Hugging Face (Optimum, opset 14), 8-bit dynamically quantised
//!   (`*_quantized.onnx`). Two graphs per model: `encoder_model_quantized.onnx`
//!   (input `input_features` `[1, 80, 3000]` log-mel, output
//!   `last_hidden_state` `[1, 1500, d_model]`) and
//!   `decoder_model_merged_quantized.onnx` (inputs `input_ids` int64 `[1, n]`,
//!   `encoder_hidden_states`, `past_key_values.{l}.{decoder,encoder}.{key,value}`
//!   `[1, heads, t, 64]` and `use_cache_branch` `[1]` bool; outputs `logits`
//!   `[1, n, vocab]` and `present.{l}.{decoder,encoder}.{key,value}`). On the
//!   first step pass zero-length past tensors (`t = 0`) and
//!   `use_cache_branch = false`; afterwards feed the previous `present.*.decoder.*`
//!   back in, keep the `present.*.encoder.*` of the first step, and set the
//!   flag. All four use 80 mel bins. Plus `tokenizer.json`, `config.json`,
//!   `generation_config.json` and `preprocessor_config.json`. URLs are pinned
//!   to a commit so a hash mismatch means a broken download, never a silent
//!   upstream change.
//! * **AudioSet** – Google's YAMNet, converted straight from the TF-Hub
//!   SavedModel with tf2onnx (opset 15) by `audiomagic/yamnet-onnx`; the
//!   mel front end is inside the graph. Input `waveform` `float32 [n]`, mono
//!   16 kHz in −1..1 (at least 15 600 samples); `output_0` is `[frames, 521]`
//!   class scores (0.96 s patches every 0.48 s), `output_1` `[frames, 1024]`
//!   embeddings, `output_2` `[frames, 64]` the log-mel it computed. The 521
//!   label names live in [`crate::audioset::LABELS`].
//! * **basic-pitch** – Spotify's `nmp.onnx` from the basic-pitch repository
//!   (tag v0.4.0). Input `serving_default_input_2:0` `[batch, 43844, 1]`
//!   float32: 22 050 Hz mono, 2 s windows (43 844 samples) that overlap by
//!   30 frames × 256 samples = 7 680 samples; outputs (names as the graph
//!   declares them, matching `basic_pitch/inference.py`)
//!   `StatefulPartitionedCall:1` = `note` `[batch, 172, 88]`,
//!   `StatefulPartitionedCall:2` = `onset` `[batch, 172, 88]` and
//!   `StatefulPartitionedCall:0` = `contour` `[batch, 172, 264]` (frame hop
//!   256 samples ≈ 11.6 ms; the first 15 frames of every window are discarded
//!   to undo the overlap). Use [`onnx_session`] to open any of them with the
//!   shared thread settings.

use crate::{Error, OnProgress, Progress, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelId {
    WhisperTiny,
    WhisperBase,
    WhisperSmall,
    WhisperMedium,
    /// Whisper large-v3-turbo: large-v3's encoder with a 4-layer decoder;
    /// near large-v3 accuracy, 128 mel bins, best with a GPU.
    WhisperLargeTurbo,
    /// Text to speech (Kokoro-82M + misaki lexicons): Speak for me.
    TtsKokoro,
    /// Text to speech in a cloned voice (Chatterbox): Speak for me as you.
    TtsChatterbox,
    /// NVIDIA TensorRT for RTX runtime (not a model): the fastest way to run
    /// the models on an NVIDIA RTX card. Optional; without it DirectML or
    /// the CPU runs them.
    GpuNvidia,
}

impl ModelId {
    pub const ALL: &'static [ModelId] = &[
        ModelId::WhisperTiny,
        ModelId::WhisperBase,
        ModelId::WhisperSmall,
        ModelId::WhisperMedium,
        ModelId::WhisperLargeTurbo,
        ModelId::TtsKokoro,
        ModelId::TtsChatterbox,
        ModelId::GpuNvidia,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ModelId::WhisperTiny => "whisper-tiny",
            ModelId::WhisperBase => "whisper-base",
            ModelId::WhisperSmall => "whisper-small",
            ModelId::WhisperMedium => "whisper-medium",
            ModelId::WhisperLargeTurbo => "whisper-large-v3-turbo",
            ModelId::TtsKokoro => "tts-kokoro",
            ModelId::TtsChatterbox => "tts-chatterbox",
            ModelId::GpuNvidia => "gpu-nvidia",
        }
    }
    pub fn from_name(s: &str) -> Option<ModelId> {
        ModelId::ALL.iter().copied().find(|m| m.name() == s)
    }
    pub fn is_whisper(self) -> bool {
        matches!(self, ModelId::WhisperTiny | ModelId::WhisperBase | ModelId::WhisperSmall | ModelId::WhisperMedium | ModelId::WhisperLargeTurbo)
    }
}

/// One downloadable file of a model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelFile {
    /// File name inside the model folder, e.g. `encoder_model.onnx`.
    pub name: &'static str,
    pub url: &'static str,
    pub size_bytes: u64,
    /// Lower-case hex SHA-256; empty = not verified (development only).
    pub sha256: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: ModelId,
    /// What it is for, in one line, for the Settings page.
    pub purpose: &'static str,
    /// Rough quality/speed note, e.g. "fast, English-leaning".
    pub note: &'static str,
    pub files: Vec<ModelFile>,
    pub license: &'static str,
}

impl ModelInfo {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

/// File names inside a Whisper model folder (shared with `stt`).
pub const WHISPER_ENCODER: &str = "encoder_model_quantized.onnx";
pub const WHISPER_DECODER: &str = "decoder_model_merged_quantized.onnx";
pub const WHISPER_TOKENIZER: &str = "tokenizer.json";
pub const WHISPER_CONFIG: &str = "config.json";
pub const WHISPER_GENERATION_CONFIG: &str = "generation_config.json";
pub const WHISPER_PREPROCESSOR_CONFIG: &str = "preprocessor_config.json";
/// The AudioSet (YAMNet) graph and the basic-pitch graph.
pub const AUDIOSET_MODEL: &str = "yamnet.onnx";
pub const BASIC_PITCH_MODEL: &str = "nmp.onnx";

/// Number of mel bins the Whisper encoder of `id` expects (80 for every
/// model in the catalogue; large-v3 would be 128).
pub fn whisper_mel_bins(id: ModelId) -> usize {
    // the real value is read from the model's config.json; this is the fallback
    if id == ModelId::WhisperLargeTurbo {
        128
    } else {
        80
    }
}

/// NVIDIA TensorRT for RTX 1.4 (the version ONNX Runtime 1.28's provider is built against).
const NVIDIA_WHEEL: &str = "tensorrt_rtx_cu13_libs-1.4.0.76-py3-none-win_amd64.whl";
/// Shown with the download: NVIDIA's own licence, fetched from NVIDIA's own package.
pub const NVIDIA_RUNTIME_LICENSE: &str = "NVIDIA proprietary (TensorRT for RTX licence, downloaded from NVIDIA's package on PyPI)";

const fn file(name: &'static str, url: &'static str, size_bytes: u64, sha256: &'static str) -> ModelFile {
    ModelFile { name, url, size_bytes, sha256 }
}

/// The six files of a Whisper export at a pinned Hugging Face revision.
fn whisper_files(repo: &'static str, rev: &'static str, entries: [(u64, &'static str); 6]) -> Vec<ModelFile> {
    // `entries` order: encoder, decoder, tokenizer, config, generation_config, preprocessor_config.
    let names = [
        (WHISPER_ENCODER, "onnx/encoder_model_quantized.onnx"),
        (WHISPER_DECODER, "onnx/decoder_model_merged_quantized.onnx"),
        (WHISPER_TOKENIZER, "tokenizer.json"),
        (WHISPER_CONFIG, "config.json"),
        (WHISPER_GENERATION_CONFIG, "generation_config.json"),
        (WHISPER_PREPROCESSOR_CONFIG, "preprocessor_config.json"),
    ];
    names
        .iter()
        .zip(entries)
        .map(|(&(name, path), (size, sha))| {
            // URLs are `'static`: built once (the catalogue is cached) and leaked.
            let url: &'static str = Box::leak(format!("https://huggingface.co/{repo}/resolve/{rev}/{path}").into_boxed_str());
            file(name, url, size, sha)
        })
        .collect()
}

/// The catalogue: every model with its files, sizes and hashes.
pub fn catalogue() -> Vec<ModelInfo> {
    catalogue_ref().to_vec()
}

/// The catalogue without the copy (built once per process).
fn catalogue_ref() -> &'static [ModelInfo] {
    static CATALOGUE: std::sync::OnceLock<Vec<ModelInfo>> = std::sync::OnceLock::new();
    CATALOGUE.get_or_init(build_catalogue)
}

/// The catalogue entry of one model.
pub fn info(id: ModelId) -> &'static ModelInfo {
    catalogue_ref().iter().find(|m| m.id == id).expect("every ModelId is catalogued")
}

/// How many threads ONNX Runtime may use inside one operator: the
/// `NEKOTONE_THREADS` environment variable when set to a positive number,
/// otherwise half the logical CPUs (at least one, at most eight), which
/// keeps the app and the other indexer workers responsive.
pub fn onnx_threads() -> usize {
    if let Some(n) = std::env::var("NEKOTONE_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
        if n > 0 {
            return n;
        }
    }
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    cpus.div_ceil(2).clamp(1, 8)
}

/// Where the ONNX models run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Accelerator {
    /// The best engine there is: TensorRT for RTX on an NVIDIA RTX card
    /// with the runtime installed, else DirectML (any DirectX 12 GPU), else
    /// the CPU.
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
    /// DirectML even on an NVIDIA card (skips TensorRT-RTX), else the CPU.
    #[serde(rename = "directml")]
    DirectMl,
}

static ACCELERATOR: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Choose where models opened from now on run (the app's setting).
pub fn set_accelerator(a: Accelerator) {
    let v = match a {
        Accelerator::Auto => 0,
        Accelerator::Cpu => 1,
        Accelerator::DirectMl => 2,
    };
    ACCELERATOR.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The current choice. `NEKOTONE_GPU=0` forces the CPU and
/// `NEKOTONE_GPU=directml` skips TensorRT-RTX, whatever the setting says.
pub fn accelerator() -> Accelerator {
    match std::env::var("NEKOTONE_GPU").map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Ok("0") => return Accelerator::Cpu,
        Ok("directml") | Ok("dml") => return Accelerator::DirectMl,
        _ => {}
    }
    match ACCELERATOR.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Accelerator::Cpu,
        2 => Accelerator::DirectMl,
        _ => Accelerator::Auto,
    }
}

/// The engine a session runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Backend {
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "directml")]
    DirectMl,
    #[serde(rename = "tensorrt-rtx")]
    TensorRtRtx,
    /// NVIDIA CUDA (Linux builds with the `cuda` feature: a compute server).
    #[serde(rename = "cuda")]
    Cuda,
}

impl Backend {
    /// For people: "NVIDIA TensorRT for RTX", "DirectML", "CPU".
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::DirectMl => "DirectML",
            Backend::TensorRtRtx => "NVIDIA TensorRT for RTX",
            Backend::Cuda => "NVIDIA CUDA",
        }
    }
}

/// The engine of the most recent session opened with a GPU allowed.
static LAST_BACKEND: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// What the last model opened with a GPU allowed runs on (None before one).
pub fn last_backend() -> Option<Backend> {
    match LAST_BACKEND.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Some(Backend::Cpu),
        2 => Some(Backend::DirectMl),
        3 => Some(Backend::TensorRtRtx),
        4 => Some(Backend::Cuda),
        _ => None,
    }
}

/// True when this build can use the GPU at all (the `gpu` feature).
pub fn gpu_supported() -> bool {
    cfg!(any(all(feature = "gpu", windows), feature = "cuda"))
}

/// Open an ONNX model with the shared settings (threads from
/// [`onnx_threads`], full graph optimisation) on the GPU when
/// [`accelerator`] allows it and one is usable, otherwise on the CPU.
/// Errors are `Error::Model` sentences naming the file.
#[cfg(feature = "ml")]
pub fn onnx_session(path: &Path) -> Result<ort::session::Session> {
    onnx_session_with(path, accelerator())
}

/// [`onnx_session`] with an explicit accelerator (a model known to run
/// badly on DirectML passes `Accelerator::Cpu`).
#[cfg(feature = "ml")]
pub fn onnx_session_with(path: &Path, accel: Accelerator) -> Result<ort::session::Session> {
    onnx_session_where(path, accel).map(|(s, _)| s)
}

/// [`onnx_session_with`] that also says whether the session runs on the GPU.
#[cfg(feature = "ml")]
pub fn onnx_session_where(path: &Path, accel: Accelerator) -> Result<(ort::session::Session, bool)> {
    onnx_session_backend(path, accel).map(|(s, b)| (s, b != Backend::Cpu))
}

/// [`onnx_session_with`] that also says which engine the session runs on.
/// `Auto` tries TensorRT-RTX, then DirectML, then the CPU; `DirectMl`
/// skips the first. Every failure falls through to the next engine.
#[cfg(feature = "ml")]
pub fn onnx_session_backend(path: &Path, accel: Accelerator) -> Result<(ort::session::Session, Backend)> {
    onnx_session_fixed(path, accel, &[])
}

/// [`onnx_session_backend`] for a graph whose variable-size inputs are
/// always fed one known shape (`[("input", &[1, 4, 3072, 256])]`): TensorRT
/// for RTX then builds one engine for exactly that shape, which it can only
/// do when told. Inputs not listed must have fixed sizes in the graph.
#[cfg(feature = "ml")]
pub fn onnx_session_fixed(path: &Path, accel: Accelerator, shapes: &[(&str, &[usize])]) -> Result<(ort::session::Session, Backend)> {
    let r = pick_session(path, accel, shapes);
    if let Ok((_, b)) = &r {
        if accel != Accelerator::Cpu {
            let v = match b {
                Backend::Cpu => 1,
                Backend::DirectMl => 2,
                Backend::TensorRtRtx => 3,
                Backend::Cuda => 4,
            };
            LAST_BACKEND.store(v, std::sync::atomic::Ordering::Relaxed);
        }
    }
    r
}

#[cfg(feature = "ml")]
fn pick_session(path: &Path, accel: Accelerator, shapes: &[(&str, &[usize])]) -> Result<(ort::session::Session, Backend)> {
    let _ = shapes;
    #[cfg(all(feature = "gpu", windows))]
    if accel != Accelerator::Cpu && path.is_file() {
        if accel == Accelerator::Auto {
            match crate::nvidia::ready() {
                Ok(()) => match crate::nvidia::model_skip_reason(path, !shapes.is_empty()) {
                    Some(why) => log::info!("{}: TensorRT-RTX skipped ({why}); trying DirectML", path.display()),
                    None => match nvrtx_session(path, shapes) {
                        Ok(s) => return Ok((s, Backend::TensorRtRtx)),
                        Err(e) => {
                            log::info!("{}: TensorRT-RTX not used ({e}); trying DirectML", path.display());
                            crate::nvidia::remember_model_failure(path, &e);
                        }
                    },
                },
                Err(e) => log::debug!("TensorRT-RTX not available: {e}"),
            }
        }
        match gpu_session(path) {
            Ok(s) => return Ok((s, Backend::DirectMl)),
            Err(e) => log::info!("{}: GPU not used ({e}); running on the CPU", path.display()),
        }
    }
    #[cfg(feature = "cuda")]
    if accel != Accelerator::Cpu && path.is_file() {
        match cuda_session(path) {
            Ok(s) => return Ok((s, Backend::Cuda)),
            Err(e) => log::info!("{}: CUDA not used ({e}); running on the CPU", path.display()),
        }
    }
    let _ = accel;
    cpu_session(path).map(|s| (s, Backend::Cpu))
}

/// CUDA session (the `cuda` feature). Fails, so the caller falls back to
/// the CPU, when ONNX Runtime has no CUDA provider or no GPU is usable.
#[cfg(feature = "cuda")]
fn cuda_session(path: &Path) -> std::result::Result<ort::session::Session, String> {
    use ort::session::builder::GraphOptimizationLevel;
    let ep = ort::ep::CUDA::default().build().error_on_failure();
    ort::session::Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_intra_threads(onnx_threads())
        .map_err(|e| e.to_string())?
        .with_execution_providers([ep])
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string())
}

/// TensorRT for RTX session. The engines it builds for this GPU are kept in
/// the cache folder (`trt-rtx/`), so a model is optimised once, not every
/// start. A failure of the provider itself (driver too old, card not
/// supported) turns TensorRT-RTX off for the rest of the process.
#[cfg(all(feature = "ml", feature = "gpu", windows))]
fn nvrtx_session(path: &Path, shapes: &[(&str, &[usize])]) -> std::result::Result<ort::session::Session, String> {
    use ort::session::builder::GraphOptimizationLevel;
    let cache = crate::data_dir().join("cache").join("trt-rtx");
    let _ = std::fs::create_dir_all(&cache);
    let mut nv = ort::ep::NVRTX::default().with_runtime_cache_path(cache.display());
    if !shapes.is_empty() {
        // one profile, min = opt = max: an engine for exactly these shapes
        let profile = shapes
            .iter()
            .map(|(name, dims)| format!("{name}:{}", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")))
            .collect::<Vec<_>>()
            .join(",");
        nv = nv.with_profile_min_shapes(&profile).with_profile_opt_shapes(&profile).with_profile_max_shapes(&profile);
    }
    let ep = nv.build().error_on_failure();
    let t0 = std::time::Instant::now();
    let r = ort::session::Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_intra_threads(onnx_threads())
        .map_err(|e| e.to_string())?
        .with_execution_providers([ep])
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string());
    // TensorRT builds its engine for the input sizes it first sees; a later
    // call with another size then fails at run time (AudioSet: a 60 s
    // waveform after a 2 s one), which no session-time fallback can catch,
    // and changing sizes rebuild the engine (noise removal: 3× slower than
    // the CPU). So only graphs with fixed input shapes run on it.
    let r = r.and_then(|s| {
        let dynamic = s
            .inputs()
            .iter()
            .filter(|i| !shapes.iter().any(|(n, _)| *n == i.name()))
            .filter_map(|i| i.dtype().tensor_shape().map(|sh| (i.name().to_string(), sh.iter().any(|&d| d < 0))))
            .find(|(_, d)| *d);
        match dynamic {
            Some((name, _)) => Err(format!("input {name} has a variable size; TensorRT for RTX is used only for fixed-size graphs")),
            None => Ok(s),
        }
    });
    match &r {
        Ok(_) => log::info!("{}: TensorRT-RTX session in {:.1} s", path.display(), t0.elapsed().as_secs_f32()),
        Err(e) if e.contains("NvTensorRtRtx") || e.contains("nv_tensorrt_rtx") || e.contains("execution provider") => {
            crate::nvidia::disable(e.clone())
        }
        Err(_) => {}
    }
    r
}

/// DirectML session. DirectML wants sequential execution and no memory
/// pattern; a registration failure (no DirectX 12 GPU, old driver) is an
/// error here so the caller falls back to the CPU explicitly.
#[cfg(all(feature = "ml", feature = "gpu", windows))]
fn gpu_session(path: &Path) -> std::result::Result<ort::session::Session, String> {
    use ort::session::builder::GraphOptimizationLevel;
    let ep = ort::ep::DirectML::default().build().error_on_failure();
    ort::session::Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_parallel_execution(false)
        .map_err(|e| e.to_string())?
        .with_memory_pattern(false)
        .map_err(|e| e.to_string())?
        .with_execution_providers([ep])
        .map_err(|e| e.to_string())?
        .commit_from_file(path)
        .map_err(|e| e.to_string())
}

#[cfg(feature = "ml")]
fn cpu_session(path: &Path) -> Result<ort::session::Session> {
    use ort::session::builder::GraphOptimizationLevel;
    let describe = |e: String| {
        Error::Model(format!(
            "could not load the model file {} ({e}); if it is damaged, remove the model in Settings → Models and download it again",
            path.display()
        ))
    };
    if !path.is_file() {
        return Err(Error::Model(format!("the model file {} is missing; download the model again", path.display())));
    }
    ort::session::Session::builder()
        .map_err(|e| describe(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| describe(e.to_string()))?
        .with_intra_threads(onnx_threads())
        .map_err(|e| describe(e.to_string()))?
        .with_inter_threads(1)
        .map_err(|e| describe(e.to_string()))?
        .commit_from_file(path)
        .map_err(|e| describe(e.to_string()))
}

fn build_catalogue() -> Vec<ModelInfo> {
    ModelId::ALL
        .iter()
        .map(|&id| ModelInfo {
            id,
            purpose: match id {
                ModelId::TtsKokoro => "Natural text-to-speech voices for Speak for me",
                ModelId::TtsChatterbox => "Speak for me in your own voice from a short voice print",
                ModelId::GpuNvidia => "Runs the models faster on NVIDIA RTX graphics cards (not a model)",
                _ => "Speech to text, 99 languages",
            },
            note: match id {
                ModelId::WhisperTiny => "fastest, rough",
                ModelId::WhisperBase => "fast, good for clear speech (recommended to start)",
                ModelId::WhisperSmall => "slower, noticeably better",
                ModelId::WhisperMedium => "slow on CPU, very good",
                ModelId::WhisperLargeTurbo => "the most accurate; fast with a GPU, slow on the CPU",
                ModelId::TtsKokoro => "28 English voices (American and British); ~4x real time on the CPU",
                ModelId::TtsChatterbox => "best with a GPU",
                ModelId::GpuNvidia => "NVIDIA TensorRT for RTX 1.4; RTX cards on Windows only",
            },
            files: files_of(id),
            license: match id {
                ModelId::TtsKokoro => crate::tts::MODEL_LICENSE,
                ModelId::TtsChatterbox => crate::tts::chatterbox::MODEL_LICENSE,
                ModelId::GpuNvidia => NVIDIA_RUNTIME_LICENSE,
                _ => "MIT",
            },
        })
        .collect()
}

fn files_of(id: ModelId) -> Vec<ModelFile> {
    match id {
        ModelId::WhisperTiny => whisper_files(
            "onnx-community/whisper-tiny",
            "ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
            [
                (10_124_990, "2af4a414ca47aa30f61246017e5fe82b0a8d229281d1255ba666a2a7f6b84d19"),
                (30_719_241, "25e807a962b6349356d0ea5d0dfe530b7e5bf0e2a484aeca0359d03143faddd3"),
                (2_480_466, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"),
                (2_243, "46aeea0a406afbeb563fc8e59ca10609203df4299af6a83f73752fef369efd2d"),
                (3_772, "f5c67e5a4f7102f8cb4d058bc95da276bbc19eeec997267c3bb0f25ef68facd1"),
                (339, "a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d"),
            ],
        ),
        ModelId::WhisperBase => whisper_files(
            "onnx-community/whisper-base",
            "1846881b6b3a3024392c1eea3ad983695bc23925",
            [
                (23_201_314, "5862993336bf33acd23736071aae2b32261d3b1b2f37780194460d4ef974dd46"),
                (53_693_315, "fa3ef9902734ce5ae6f9ef2bdb2ba9a6c4b5785b09f4f420ce036573dc9d090b"),
                (2_480_466, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"),
                (2_243, "f4d0608f7d918166da7edb3e188de5ef1bfe70d9802e785d271fd88111e9cf4b"),
                (3_832, "61070cf8de25b1e9256e8e102ded49d8d24a8369ed36ef84fdf21549e68125a0"),
                (339, "a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d"),
            ],
        ),
        ModelId::WhisperSmall => whisper_files(
            "onnx-community/whisper-small",
            "36050c46d777d46dc4b5f43f6d90574fc38f8732",
            [
                (92_326_160, "a43a83f3c5361cd591cfa7c36f14b43cf7cb22f47a415cc14a8d557be800fa92"),
                (156_750_845, "ec07c3cbb64172c39791e26ee870a65ac22b458c36722bfe2776b3dbf741e0c9"),
                (2_480_466, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566"),
                (2_227, "457854d452f17661e197d74aee12b8e74fb75ba30ebfaa7426d0d61ea1e08a18"),
                (3_893, "f538b28220c6a6d6f1af1458d4141cacb4ef4963df3de98a19490440c412ddf0"),
                (339, "a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d"),
            ],
        ),
        ModelId::WhisperMedium => whisper_files(
            "onnx-community/whisper-medium-ONNX",
            "d3978248a6b5de6df7ec29ddfbde3993845fa806",
            [
                (313_154_567, "2507d090bf9f1d468f4c1cef274aa34eda025894118f5476e24a0d80308a5315"),
                (672_556_229, "a524f9f1759143e1a8ec3ca3559594ced7282e2061e23ef2fd602bf1813bbde1"),
                (3_930_494, "7b469ff15eb7816315aa45eec391f5943d639b9d73d110f5c003df5192fd54e3"),
                (1_389, "6f05735b8279a8eaadf51b276909151687cb94dc6d1eb2a32e6e9bfd36ea1c49"),
                (3_780, "060a0ba092ee90ee7124d00c00376ded564e6e4edf3f2c4a19a1a35076a1b91c"),
                (339, "a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d"),
            ],
        ),
        ModelId::WhisperLargeTurbo => whisper_files(
            "onnx-community/whisper-large-v3-turbo",
            "360ebcde2559d60bb474678be3c1de9ef347d01a",
            [
                (644_822_195, "d2f853dc3254fdc0079f55dd4433ea716ac98ec5574d3b475f288f2a77cebba9"),
                (439_936_716, "61481bd3be3a445d5a4b9070e8f8b2c6cc4fbbbbdc9f0e7ed048a132b8b84e0d"),
                (2_480_617, "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca"),
                (1_332, "35cd83669f75bc2867f3b3a4461850392d5e308cd6ea951c3700539883c28df1"),
                (3_897, "16f95291d2f47c944d3c2b19390bba7965666555c1ea2a0bdc850d1fab45612f"),
                (340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
            ],
        ),
        ModelId::TtsKokoro => crate::tts::model_files(),
        ModelId::TtsChatterbox => crate::tts::chatterbox::model_files(),
        // The runtime ships as NVIDIA's own Python wheel (a zip) on PyPI,
        // downloaded from there as it is; the two DLLs are unpacked from it
        // (see `nvidia_runtime_dir`).
        ModelId::GpuNvidia => vec![file(
            NVIDIA_WHEEL,
            "https://files.pythonhosted.org/packages/08/1d/2515cc7889897e0d61f07f81ba44f0d25b682067e3d78d213081aca1b15d/tensorrt_rtx_cu13_libs-1.4.0.76-py3-none-win_amd64.whl",
            82_417_649,
            "064561b951d1c7b2683e2f8e1fec2a8d4d204d338bb25e90db106acf952ddd52",
        )],
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub info: ModelInfo,
    pub installed: bool,
    pub dir: PathBuf,
}

/// Finds, downloads and verifies models.
#[derive(Debug, Clone)]
pub struct ModelManager {
    dir: PathBuf,
}

/// The models folder the program chose (`--models-dir` or the app's), for
/// code paths that are not handed a manager (export options).
static ACTIVE_ROOT: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// A manager for the models folder in use: the last one created with
/// [`ModelManager::new`], else the default folder.
pub fn active_manager() -> ModelManager {
    match ACTIVE_ROOT.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        Some(d) => ModelManager { dir: d },
        None => ModelManager::default(),
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        ModelManager { dir: crate::data_dir().join("models") }
    }
}

impl ModelManager {
    pub fn new(dir: PathBuf) -> Self {
        crate::nvidia::add_models_root(&dir);
        *ACTIVE_ROOT.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.clone());
        ModelManager { dir }
    }
    pub fn root(&self) -> &std::path::Path {
        &self.dir
    }
    pub fn dir_of(&self, id: ModelId) -> PathBuf {
        self.dir.join(id.name())
    }
    /// The model folder when every file of the model is present with the
    /// catalogued size (hashes are checked at download time only).
    pub fn path(&self, id: ModelId) -> Option<PathBuf> {
        let info = info(id);
        let dir = self.dir_of(id);
        if id == ModelId::GpuNvidia {
            // the download (NVIDIA's wheel) is unpacked and deleted; the
            // installer puts the unpacked files here directly
            return crate::nvidia::installed_in(&dir).then_some(dir);
        }
        let complete = !info.files.is_empty()
            && info.files.iter().all(|f| {
                std::fs::metadata(dir.join(f.name)).map(|m| m.is_file() && m.len() == f.size_bytes).unwrap_or(false)
            });
        complete.then_some(dir)
    }
    pub fn status(&self) -> Vec<ModelStatus> {
        catalogue()
            .into_iter()
            .map(|info| ModelStatus { installed: self.path(info.id).is_some(), dir: self.dir_of(info.id), info })
            .collect()
    }
    /// Download the model if needed; returns its folder. Files that are
    /// already complete are kept; interrupted downloads (`.part` files)
    /// are resumed with an HTTP `Range` request. Every file is verified
    /// against its SHA-256 before it is renamed into place.
    pub fn ensure(&self, id: ModelId, progress: OnProgress) -> Result<PathBuf> {
        self.ensure_cancellable(id, progress, &std::sync::atomic::AtomicBool::new(false))
    }
    /// [`ensure`](Self::ensure) that stops with `Error::Cancelled` soon after
    /// `cancel` is set. The partial file is kept, so the next attempt resumes.
    pub fn ensure_cancellable(&self, id: ModelId, progress: OnProgress, cancel: &std::sync::atomic::AtomicBool) -> Result<PathBuf> {
        if let Some(p) = self.path(id) {
            return Ok(p);
        }
        let info = info(id);
        let dir = self.dir_of(id);
        std::fs::create_dir_all(&dir).map_err(|e| {
            Error::Model(format!("could not create the model folder {} ({e}); check that the drive is writable", dir.display()))
        })?;
        let total = info.total_bytes();
        let mut done_before: u64 = 0;
        for f in &info.files {
            let dest = dir.join(f.name);
            let already = std::fs::metadata(&dest).map(|m| m.is_file() && m.len() == f.size_bytes).unwrap_or(false);
            if !already {
                let prefix = format!("Downloading {} ({})", id.name(), f.name);
                download_file(f, &dest, cancel, &mut |got| {
                    let overall = (done_before + got) as f32 / total.max(1) as f32;
                    progress(Progress::new(
                        format!("{prefix} {:.1} / {:.1} MB", (done_before + got) as f64 / 1e6, total as f64 / 1e6),
                        Some(overall.min(1.0)),
                    ));
                })?;
            }
            done_before += f.size_bytes;
        }
        if id == ModelId::GpuNvidia {
            progress(Progress::new("Unpacking the NVIDIA runtime".to_string(), None));
            crate::nvidia::unpack(&dir.join(info.files[0].name), &dir)?;
        }
        progress(Progress::new(format!("{} ready", id.name()), Some(1.0)));
        self.path(id).ok_or_else(|| Error::Model(format!("{} downloaded but its files do not add up; try `nekotone models remove {}` and download again", id.name(), id.name())))
    }
    pub fn remove(&self, id: ModelId) -> Result<()> {
        let d = self.dir_of(id);
        if d.is_dir() {
            std::fs::remove_dir_all(&d).map_err(|e| {
                if id == ModelId::GpuNvidia && e.kind() == std::io::ErrorKind::PermissionDenied {
                    Error::Model("the NVIDIA runtime is in use; close Neko Player (and any nekotone command) and remove it again".into())
                } else {
                    Error::Io(e)
                }
            })?;
        }
        Ok(())
    }
}

/// SHA-256 of a file, lower-case hex.
#[cfg(feature = "ml")]
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::Digest;
    let mut f = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(hex::encode(h.finalize()))
}

#[cfg(feature = "ml")]
fn http_agent() -> ureq::Agent {
    // Proxies come from HTTPS_PROXY / HTTP_PROXY / ALL_PROXY when set.
    let cfg = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
        .timeout_recv_body(None)
        .max_redirects(10)
        .user_agent(concat!("nekotone/", env!("CARGO_PKG_VERSION")))
        .proxy(ureq::Proxy::try_from_env())
        .build();
    ureq::Agent::new_with_config(cfg)
}

#[cfg(feature = "ml")]
fn network_error(url: &str, e: ureq::Error) -> Error {
    let hint = match &e {
        ureq::Error::HostNotFound => "the host name could not be resolved; check the internet connection or the proxy settings (HTTPS_PROXY)",
        ureq::Error::ConnectionFailed => "the connection was refused or dropped; check the internet connection, firewall or proxy",
        ureq::Error::Timeout(_) => "the server did not answer in time; try again",
        ureq::Error::ConnectProxyFailed(_) | ureq::Error::InvalidProxyUrl => "the proxy refused the connection; check HTTPS_PROXY / HTTP_PROXY",
        ureq::Error::Tls(_) | ureq::Error::Rustls(_) => "the secure connection failed (a corporate proxy that inspects TLS?)",
        ureq::Error::Io(_) => "the transfer broke off; it will resume where it stopped when you try again",
        _ => "the download failed; try again",
    };
    Error::Model(format!("{url}: {hint} ({e})"))
}

/// Download `f.url` into `dest`, resuming a `dest.part` left by an earlier
/// attempt, and verify the size and SHA-256. `got(bytes_so_far)` is called
/// as data arrives.
#[cfg(not(feature = "ml"))]
fn download_file(f: &ModelFile, dest: &Path, cancel: &std::sync::atomic::AtomicBool, got: &mut dyn FnMut(u64)) -> Result<()> {
    let _ = (dest, got, cancel);
    Err(Error::Model(format!("{}: this build of Nekotone was made without the `ml` feature, so it cannot download models", f.name)))
}

#[cfg(feature = "ml")]
fn download_file(f: &ModelFile, dest: &Path, cancel: &std::sync::atomic::AtomicBool, got: &mut dyn FnMut(u64)) -> Result<()> {
    use std::io::{Read, Seek, Write};
    let part = dest.with_extension(match dest.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{ext}.part"),
        None => "part".to_string(),
    });
    let agent = http_agent();
    let mut attempt = 0;
    loop {
        attempt += 1;
        let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > f.size_bytes {
            std::fs::remove_file(&part)?;
            have = 0;
        }
        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&part).map_err(|e| {
            Error::Model(format!("could not write {} ({e}); check that the folder is writable", part.display()))
        })?;
        if have < f.size_bytes {
            let mut req = agent.get(f.url);
            if have > 0 {
                req = req.header("Range", format!("bytes={have}-"));
            }
            let resp = req.call().map_err(|e| network_error(f.url, e))?;
            let status = resp.status().as_u16();
            match status {
                206 => {}
                200 => {
                    // The server ignored the range: start over.
                    if have > 0 {
                        out.set_len(0)?;
                        out.seek(std::io::SeekFrom::Start(0))?;
                        have = 0;
                    }
                }
                404 | 410 => {
                    return Err(Error::Model(format!("{}: the file is no longer at {} (HTTP {status}); a newer Nekotone may have an updated catalogue", f.name, f.url)))
                }
                401 | 403 => return Err(Error::Model(format!("{}: access denied by the server (HTTP {status}) for {}", f.name, f.url))),
                407 => return Err(Error::Model(format!("{}: the proxy wants authentication (HTTP 407); set HTTPS_PROXY with credentials", f.name))),
                416 => {
                    // Range not satisfiable: our .part is bad. Start over.
                    drop(out);
                    std::fs::remove_file(&part)?;
                    if attempt < 3 {
                        continue;
                    }
                    return Err(Error::Model(format!("{}: the server would not resume the download; delete {} and try again", f.name, part.display())));
                }
                s if s >= 500 => {
                    if attempt < 3 {
                        std::thread::sleep(std::time::Duration::from_secs(2 * attempt as u64));
                        continue;
                    }
                    return Err(Error::Model(format!("{}: the server is having trouble (HTTP {s}); try again later", f.name)));
                }
                s => return Err(Error::Model(format!("{}: unexpected HTTP status {s} from {}", f.name, f.url))),
            }
            let mut body = resp.into_body().into_reader();
            let mut buf = vec![0u8; 1 << 16];
            let mut last_report = std::time::Instant::now();
            got(have);
            let result: std::result::Result<(), std::io::Error> = (|| loop {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break Err(std::io::Error::new(std::io::ErrorKind::Interrupted, "cancelled"));
                }
                let n = body.read(&mut buf)?;
                if n == 0 {
                    break Ok(());
                }
                out.write_all(&buf[..n])?;
                have += n as u64;
                if last_report.elapsed().as_millis() >= 100 {
                    got(have);
                    last_report = std::time::Instant::now();
                }
            })();
            out.flush()?;
            drop(out);
            if let Err(e) = result {
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(Error::Cancelled);
                }
                if attempt < 3 {
                    log::warn!("{}: transfer interrupted ({e}); resuming", f.name);
                    continue;
                }
                return Err(Error::Model(format!("{}: the transfer broke off ({e}); run the download again to resume it", f.name)));
            }
            got(have);
        } else {
            drop(out);
        }
        if have != f.size_bytes {
            if have < f.size_bytes && attempt < 3 {
                continue; // short body: resume
            }
            let _ = std::fs::remove_file(&part);
            return Err(Error::Model(format!("{}: got {have} bytes, expected {}; the file on the server has changed or the download was corrupted", f.name, f.size_bytes)));
        }
        if !f.sha256.is_empty() {
            let sum = sha256_file(&part)?;
            if sum != f.sha256 {
                let _ = std::fs::remove_file(&part);
                return Err(Error::Model(format!("{}: checksum mismatch (got {sum}, expected {}); the download was corrupted or the file on the server has changed", f.name, f.sha256)));
            }
        }
        if dest.exists() {
            std::fs::remove_file(dest)?;
        }
        std::fs::rename(&part, dest)?;
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_round_trip() {
        for id in ModelId::ALL {
            assert_eq!(ModelId::from_name(id.name()).unwrap(), *id);
        }
    }

    #[test]
    fn catalogue_entries_are_well_formed() {
        for m in catalogue() {
            assert!(!m.files.is_empty(), "{}", m.id.name());
            for f in &m.files {
                assert!(f.url.starts_with("https://"), "{}", f.url);
                assert_eq!(f.sha256.len(), 64, "{}/{}", m.id.name(), f.name);
                assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
                assert!(f.size_bytes > 0);
                assert!(!f.name.contains('/'));
            }
        }
    }

    #[test]
    fn path_requires_every_file_with_the_right_size() {
        let tmp = tempfile::tempdir().unwrap();
        let mm = ModelManager::new(tmp.path().to_path_buf());
        assert!(mm.path(ModelId::TtsKokoro).is_none());
        let dir = mm.dir_of(ModelId::TtsKokoro);
        std::fs::create_dir_all(&dir).unwrap();
        let info = catalogue().into_iter().find(|m| m.id == ModelId::TtsKokoro).unwrap();
        let f = &info.files[0];
        std::fs::write(dir.join(f.name), b"short").unwrap();
        assert!(mm.path(ModelId::TtsKokoro).is_none(), "wrong size must not count as installed");
        std::fs::write(dir.join(f.name), vec![0u8; f.size_bytes as usize]).unwrap();
        for extra in info.files.iter().skip(1) {
            std::fs::write(dir.join(extra.name), vec![0u8; extra.size_bytes as usize]).unwrap();
        }
        assert_eq!(mm.path(ModelId::TtsKokoro), Some(dir.clone()));
        assert!(mm.status().iter().any(|s| s.info.id == ModelId::TtsKokoro && s.installed));
        mm.remove(ModelId::TtsKokoro).unwrap();
        assert!(mm.path(ModelId::TtsKokoro).is_none());
    }

    #[test]
    #[cfg(feature = "ml")]
    fn sha256_of_known_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("abc");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
