use serde::{Deserialize, Serialize};

/// Weight manifest for external LLM runner consumption.
///
/// JSON schema that external runner can consume to load tensors.
/// aligns with safetensors lazy_loading=true concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightManifest {
    pub weight_id: String,
    pub model_name: String,
    pub repo_id: String,
    pub file_path: String,
    pub tensor_count: i32,
    pub dtype: String,
    pub device: String,
    pub size_bytes: i64,
    pub checksum: String,
    pub tensors: Vec<TensorMetadataRow>,
    pub metadata: serde_json::Value,
    pub lazy_loading: bool,
}

/// A single tensor metadata row for manifest output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadataRow {
    pub id: String,
    pub weight_id: String,
    pub tensor_name: String,
    pub shape: String,
    pub dtype: String,
    pub size_bytes: i64,
    pub checksum: String,
}

/// A single model weight row for manifest queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightRow {
    pub id: String,
    pub model_name: String,
    pub repo_id: String,
    pub file_path: String,
    pub tensor_count: i32,
    pub dtype: String,
    pub device: String,
    pub size_bytes: i64,
    pub checksum: String,
    pub metadata: serde_json::Value,
    pub loaded_at: i64,
    pub created_at: i64,
    pub active: i32,
}

use crate::PlugInError;

use tracing::debug;

/// Generate a weight manifest from safetensors DB for external runner consumption.
pub fn generate_weight_manifest(
    conn: &rusqlite::Connection,
    model_name: &str,
) -> Result<WeightManifest, PlugInError> {
    let mut stmt = conn.prepare(
        "SELECT id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active FROM model_weights
         WHERE model_name = ?1 AND active = 1
         ORDER BY loaded_at DESC LIMIT 1",
    )?;

    let row = stmt
        .query_row(rusqlite::params![model_name], |row| {
            let metadata_str: String = row.get(9)?;
            let metadata: serde_json::Value =
                serde_json::from_str(&metadata_str).unwrap_or_default();

            Ok(ModelWeightRow {
                id: row.get(0)?,
                model_name: row.get(1)?,
                repo_id: row.get(2)?,
                file_path: row.get(3)?,
                tensor_count: row.get(4)?,
                dtype: row.get(5)?,
                device: row.get(6)?,
                size_bytes: row.get(7)?,
                checksum: row.get(8)?,
                metadata,
                loaded_at: row.get(10)?,
                created_at: row.get(11)?,
                active: row.get(12)?,
            })
        })
        .map_err(PlugInError::Sqlite)?;

    let mut tensor_stmt = conn.prepare(
        "SELECT id, weight_id, tensor_name, shape, dtype, size_bytes, checksum FROM tensor_metadata
         WHERE weight_id = ?1",
    )?;

    let tensors: Vec<TensorMetadataRow> = tensor_stmt
        .query_map(rusqlite::params![row.id], |row| {
            Ok(TensorMetadataRow {
                id: row.get(0)?,
                weight_id: row.get(1)?,
                tensor_name: row.get(2)?,
                shape: row.get(3)?,
                dtype: row.get(4)?,
                size_bytes: row.get(5)?,
                checksum: row.get(6)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let manifest = WeightManifest {
        weight_id: row.id,
        model_name: row.model_name,
        repo_id: row.repo_id,
        file_path: row.file_path,
        tensor_count: row.tensor_count,
        dtype: row.dtype,
        device: row.device,
        size_bytes: row.size_bytes,
        checksum: row.checksum,
        tensors,
        metadata: row.metadata,
        lazy_loading: true,
    };

    debug!(
        model_name = %model_name,
        tensor_count = manifest.tensor_count,
        "Weight manifest generated for external runner"
    );

    Ok(manifest)
}
