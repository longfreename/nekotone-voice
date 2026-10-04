//! Nekotone Voice core: shared voice DSP, TTS, cloning, STT, and compute-server plumbing.

pub mod audio;
pub mod error;
pub mod features;
pub mod models;
pub mod nvidia;
pub mod process;
pub mod record;
#[cfg(feature = "ml")]
pub mod remote;
pub mod stt;
pub mod threads;
pub mod tts;
pub mod voice;

pub use error::{Error, Result};

/// Where Nekotone Voice keeps its own files (models, settings, local data):
/// `%LOCALAPPDATA%\NekotoneVoice`.
pub fn data_dir() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("NekotoneVoice")
}

/// A progress report from a long job, for progress bars and logs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Progress {
    /// What is happening right now, e.g. "Transcribing 0A000013.wav".
    pub message: String,
    /// 0..=1 when known.
    pub fraction: Option<f32>,
}

impl Progress {
    pub fn new(message: impl Into<String>, fraction: Option<f32>) -> Self {
        Progress { message: message.into(), fraction }
    }
}

/// Callback for progress; a no-op is fine.
pub type OnProgress<'a> = &'a mut dyn FnMut(Progress);
