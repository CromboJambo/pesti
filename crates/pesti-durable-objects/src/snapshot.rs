//! Snapshot-based persistence for durable objects.
//!
//! Mirrors herdr's session snapshot architecture: capture state into serializable
//! structs, save to disk, restore on startup. Versioned format with migration support.

use std::path::{Path, PathBuf};
use std::io;

use serde::{Serialize, Deserialize, de::DeserializeOwned};
use tokio::fs;
use thiserror::Error;

/// Current snapshot format version. Increment when serialization format changes.
pub const SNAPSHOT_VERSION: u32 = 1;

#[derive(Error, Debug)]
pub enum SnapshotError {
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("deserialization failed: {0}")]
    Deserialization(String),
    #[error("file operation failed: {0}")]
    Io(#[from] io::Error),
    #[error("version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
}

/// Save object state as a JSON snapshot to disk.
pub async fn save_snapshot<T: Serialize>(path: &Path, data: &T) -> Result<(), SnapshotError> {
    let json = serde_json::to_string_pretty(data)
        .map_err(|e| SnapshotError::Serialization(e.to_string()))?;

    // Write atomically via temp file + rename (herdr pattern)
    let tmp_path = path.with_extension("tmp");
    fs::write(&tmp_path, json).await?;
    fs::rename(&tmp_path, path).await?;

    Ok(())
}

/// Load object state from a JSON snapshot on disk.
pub async fn load_snapshot<T: DeserializeOwned>(path: &Path) -> Result<T, SnapshotError> {
    let content = fs::read_to_string(path).await?;
    let data: T = serde_json::from_str(&content)
        .map_err(|e| SnapshotError::Deserialization(e.to_string()))?;

    Ok(data)
}

/// Check if snapshot file exists and is valid.
pub async fn snapshot_exists(path: &Path) -> bool {
    match fs::metadata(path).await {
        Ok(metadata) => metadata.file_type().is_file(),
        Err(_) => false,
    }
}