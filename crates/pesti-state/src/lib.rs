//! # pesti-state
//!
//! Persistence and provenance layer for PESTI weight-space operations.
//!
//! Tracks three things:
//! - **Models**: what's been loaded, identified by GGUF header hash
//! - **Operations**: every action performed on a model (audit trail)
//! - **Adapters**: saved edits with lifecycle states and provenance
//!
//! ## Design principles
//!
//! 1. **Content-addressed**: Models identified by structure, not path or name
//! 2. **Append-only operations**: Nothing is ever deleted from the audit log
//! 3. **Provenance over trust**: Every adapter carries its assumptions and validation state
//! 4. **No silent inheritance**: Revalidation is explicit, never automatic
//!
//! ## Usage
//!
//! ```no_run
//! use pesti_state::{PestiState, ModelRecord};
//!
//! let state = PestiState::open("/tmp/pesti-test.db")?;
//!
//! // Register a model
//! let record = ModelRecord::new(
//!     "abc123...".to_string(),  // GGUF header hash
//!     "/path/to/model.gguf".to_string(),
//!     "llama".to_string(),
//!     "Q4_K_M".to_string(),
//!     Some(90_000_000),
//!     150,
//! );
//! let model_id = state.register_model(&record)?;
//!
//! // Query models
//! let models = state.list_models()?;
//! # Ok::<(), pesti_state::StateError>(())
//! ```

pub mod adapter;
pub mod error;
pub mod model;
pub mod operation;
pub mod state;

pub use adapter::{AdapterRecord, AdapterState};
pub use error::{Result, StateError};
pub use model::ModelRecord;
pub use operation::{OperationDetails, OperationLog, OperationType};
pub use state::PestiState;

/// Compute SHA-256 hash of a byte slice as a lowercase hex string.
///
/// Used for GGUF header hashing to create content-addressed model IDs.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute the content hash for a GGUF header.
///
/// Serializes the header to bytes and hashes with SHA-256. The resulting
/// hash identifies the model's structure (architecture, tensor layout,
/// metadata) regardless of weight values or file location.
pub fn gguf_header_hash(header: &pesti_gguf::GgufHeader) -> String {
    use std::io::Write;

    let mut buf = Vec::new();
    // Write version
    let _ = buf.write_all(&header.version.to_le_bytes());
    // Write KV pairs (key, value_type, data)
    for pair in &header.kv_pairs {
        let key_bytes = pair.key.as_bytes();
        let _ = buf.write_all(&(key_bytes.len() as u64).to_le_bytes());
        let _ = buf.write_all(key_bytes);
        // Value type tag + Debug representation of value
        let value_repr = format!("{:?}: {:?}", pair.value_type, pair.value);
        let value_bytes = value_repr.as_bytes();
        let _ = buf.write_all(&(value_bytes.len() as u64).to_le_bytes());
        let _ = buf.write_all(value_bytes);
    }
    // Write tensor names and metadata
    for tensor in &header.tensors {
        let name_bytes = tensor.name.as_bytes();
        let _ = buf.write_all(&(name_bytes.len() as u64).to_le_bytes());
        let _ = buf.write_all(name_bytes);
        let _ = buf.write_all(&(tensor.shape.len() as u32).to_le_bytes());
        for dim in &tensor.shape {
            let _ = buf.write_all(&dim.to_le_bytes());
        }
        let _ = buf.write_all(&tensor.dtype.to_le_bytes());
    }

    sha256_hex(&buf)
}
