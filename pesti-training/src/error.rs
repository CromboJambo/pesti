//! Error types for pesti-training.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias.
pub type Result<T> = std::result::Result<T, TrainingError>;

#[derive(Error, Debug)]
pub enum TrainingError {
    #[error("Dataset error: {0}")]
    Dataset(#[from] DatasetError),

    #[error("LoRA adapter error: {0}")]
    Lora(String),

    #[error("Training loop error: {0}")]
    Train(String),

    #[error("Checkpoint error at {path}: {source}")]
    Checkpoint {
        path: PathBuf,
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Serialization error: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("Failed to read file at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Invalid JSON at line {line} in {path}: {source}")]
    Json {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },

    #[error("Tokenization failed: {0}")]
    Tokenize(String),
}
