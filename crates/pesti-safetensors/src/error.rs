use thiserror::Error;

#[derive(Debug, Error)]
pub enum SafetensorsSchemaError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("schema initialization failed: {0}")]
    SchemaError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("internal safetensors error: {0}")]
    Internal(String),
}

#[derive(Debug, Error)]
pub enum SafetensorsError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("schema error: {0}")]
    Schema(#[from] SafetensorsSchemaError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("safetensors load failed: {0}")]
    Load(String),

    #[error("safetensors save failed: {0}")]
    Save(String),

    #[error("checksum mismatch: {0}")]
    ChecksumMismatch(String),

    #[error("path not found: {0}")]
    NotFound(String),

    #[error("internal safetensors error: {0}")]
    Internal(String),
}
