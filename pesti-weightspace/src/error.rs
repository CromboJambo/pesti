//! Error types for weight-space operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result type for weight-space operations.
pub type Result<T> = std::result::Result<T, WeightSpaceError>;

/// Errors that can occur during weight-space operations.
#[derive(Debug, Error)]
pub enum WeightSpaceError {
    /// Model file not found or unreadable.
    #[error("model file error: {path}: {source}")]
    ModelFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Tensor not found in the model.
    #[error("tensor '{0}' not found in model")]
    TensorNotFound(String),

    /// Index out of bounds for tensor shape.
    #[error("index {index:?} out of bounds for tensor '{tensor}' with shape {shape:?}")]
    IndexOutOfBounds {
        tensor: String,
        index: Vec<usize>,
        shape: Vec<usize>,
    },

    /// Invalid edit (e.g., value would overflow target precision).
    #[error("invalid edit on '{param}': {reason}")]
    InvalidEdit { param: String, reason: String },

    /// Atomic patch failed — all edits rolled back.
    #[error("atomic patch failed: {source}")]
    PatchFailed {
        #[source]
        source: Box<WeightSpaceError>,
    },

    /// No snapshot available for rollback.
    #[error("no snapshot available for rollback")]
    NoSnapshot,

    /// GGUF parsing or I/O error from underlying loader.
    #[error("GGUF error: {0}")]
    Gguf(String),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serde(String),

    /// Unsupported operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// State persistence operation failed.
    #[error("state error: {0}")]
    State(String),
}

impl From<pesti_runner::error::RunnerError> for WeightSpaceError {
    fn from(err: pesti_runner::error::RunnerError) -> Self {
        WeightSpaceError::Gguf(err.to_string())
    }
}
