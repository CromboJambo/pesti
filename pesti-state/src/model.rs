//! Model registry records.

use serde::{Deserialize, Serialize};

/// A registered model in the PESTI state database.
///
/// Identified by SHA-256 of its GGUF header bytes (structure, not content).
/// Two files with different headers are different models even at the same path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    /// SHA-256 hex digest of the GGUF header bytes.
    pub id: String,
    /// Filesystem path at time of registration.
    pub path: String,
    /// Architecture name (e.g., "llama", "tinyllama").
    pub architecture: String,
    /// Quantization format (e.g., "Q4_K_M", "F16").
    pub quantization: String,
    /// Approximate parameter count.
    pub param_count: Option<u64>,
    /// Number of tensors in the model.
    pub tensor_count: usize,
    /// ISO 8601 timestamp of first registration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_seen: Option<String>,
    /// ISO 8601 timestamp of last hash re-verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_verified: Option<String>,
}

impl ModelRecord {
    /// Create a new model record with the current timestamp as first_seen.
    pub fn new(
        id: String,
        path: String,
        architecture: String,
        quantization: String,
        param_count: Option<u64>,
        tensor_count: usize,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id,
            path,
            architecture,
            quantization,
            param_count,
            tensor_count,
            first_seen: Some(now.clone()),
            last_verified: Some(now),
        }
    }
}
