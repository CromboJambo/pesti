use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlugInError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("weight manifest not found: {0}")]
    NotFound(String),

    #[error("runner config invalid: {0}")]
    ConfigInvalid(String),

    #[error("inference protocol mismatch: {0}")]
    ProtocolMismatch(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, PlugInError>;
