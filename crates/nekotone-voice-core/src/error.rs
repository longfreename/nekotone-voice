//! One error type for the crate. Messages are written for people: they say
//! what failed and what to do, because the app shows them as they are.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}: file not found")]
    NotFound(PathBuf),
    #[error("{path}: not an audio file Nekotone can read ({detail})")]
    Unsupported { path: PathBuf, detail: String },
    #[error("{path}: could not decode ({detail})")]
    Decode { path: PathBuf, detail: String },
    #[error("the model {model} is not downloaded; open Settings → Models, or run `nekotone models get {model}`")]
    ModelMissing { model: String },
    #[error("model error: {0}")]
    Model(String),
    #[error("index error: {0}")]
    Index(String),
    #[error("audio output error: {0}")]
    Output(String),
    #[error("{0} is not implemented yet")]
    NotImplemented(&'static str),
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
