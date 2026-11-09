//! Error types for pesti-state operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result type for pesti-state operations.
pub type Result<T> = std::result::Result<T, StateError>;

/// Errors that can occur in the PESTI state layer.
#[derive(Debug, Error)]
pub enum StateError {
    /// SQLite database error.
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// I/O error (opening/creating the state file).
    #[error("I/O error: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Model not found in registry.
    #[error("model '{0}' not found in registry")]
    ModelNotFound(String),

    /// Adapter not found in registry.
    #[error("adapter '{0}' not found in registry")]
    AdapterNotFound(String),

    /// Schema version mismatch — migration required but not implemented.
    #[error("unsupported schema version {found}, expected {expected}")]
    SchemaVersionMismatch { found: u32, expected: u32 },
}

impl From<std::io::Error> for StateError {
    fn from(err: std::io::Error) -> Self {
        StateError::Io {
            path: PathBuf::from("unknown"),
            source: err,
        }
    }
}
