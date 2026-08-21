use pesti_gguf::GgufDtype;
use pesti_gguf::GgufHeader;
use pesti_plug_in::manifest::WeightManifest;

use crate::error::RunnerError;
use pesti_safetensors::error::SafetensorsSchemaError;
use std::path::Path;
use tracing::debug;

/// Extension trait for GgufHeader to provide convenient metadata access.
/// This bridges the gap between the external pesti-gguf crate and our needs.
pub trait GgufHeaderExt {
    fn attention_head_count_kv(&self) -> Option<u32>;
    fn rope_dimension_count(&self) -> Option<i32>;
    fn normalization_epsilon(&self) -> Option<f32>;
    fn vocab_size(&self) -> u32;
    fn get_kv_f32(&self, key: &str) -> Option<f32>;
    fn get_kv_bool(&self, key: &str) -> Option<bool>;
    fn get_kv_array_str(&self, key: &str) -> Option<Vec<String>>;
    fn get_kv_array_i32(&self, key: &str) -> Option<Vec<i32>>;
}

impl GgufHeaderExt for GgufHeader {
    fn attention_head_count_kv(&self) -> Option<u32> {
        self.get_kv_u32("attention.head_count_kv")
            .or_else(|| self.get_kv_u32("attention.head_count"))
    }

    fn rope_dimension_count(&self) -> Option<i32> {
        self.get_kv_u32("rope.dimension_count").map(|v| v as i32)
    }

    fn normalization_epsilon(&self) -> Option<f32> {
        self.kv_pairs
            .iter()
            .find(|kv| kv.key == "attention.layer_norm_epsilon")
            .and_then(|kv| kv.value.as_f32())
            .or_else(|| {
                self.kv_pairs
                    .iter()
                    .find(|kv| kv.key == "norm_eps")
                    .and_then(|kv| kv.value.as_f32())
            })
    }

    fn vocab_size(&self) -> u32 {
        self.get_kv_u32("tokenizer.ggml.vocab_size")
            .or_else(|| self.get_kv_u32("tokenizer.ggml.tokens"))
            .or_else(|| {
                // Standard llama.cpp stores `tokenizer.ggml.tokens` as an ARRAY of
                // token strings — its length IS the vocab size. `get_kv_u32` returns
                // None for an array, so read the length directly.
                self.kv_pairs
                    .iter()
                    .find(|kv| kv.key == "tokenizer.ggml.tokens")
                    .and_then(|kv| match &kv.value {
                        pesti_gguf::types::GgufKvValue::Array(a) => Some(a.len() as u32),
                        _ => None,
                    })
            })
            .unwrap_or(32000)
    }

    fn get_kv_f32(&self, key: &str) -> Option<f32> {
        self.kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| kv.value.as_f32())
    }

    fn get_kv_bool(&self, key: &str) -> Option<bool> {
        self.kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| kv.value.as_bool())
    }

    fn get_kv_array_str(&self, key: &str) -> Option<Vec<String>> {
        self.kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| match &kv.value {
                pesti_gguf::GgufKvValue::Array(arr) => Some(arr),
                _ => None,
            })
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
    }

    fn get_kv_array_i32(&self, key: &str) -> Option<Vec<i32>> {
        self.kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| match &kv.value {
                pesti_gguf::GgufKvValue::Array(arr) => Some(arr),
                _ => None,
            })
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| match v {
                        pesti_gguf::GgufKvValue::Int32(x) => Some(*x),
                        _ => None,
                    })
                    .collect()
            })
    }
}

/// Model loader that consumes WeightManifest from safetensors DB.
///
/// loads tensors for inference engine consumption.
pub struct ModelLoader {
    pub conn: rusqlite::Connection,
}

impl ModelLoader {
    pub fn new(conn: rusqlite::Connection) -> Self {
        Self { conn }
    }

    /// Load weight manifest for a model from safetensors DB.
    pub fn load_manifest(&self, model_name: &str) -> Result<WeightManifest, RunnerError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active FROM model_weights
             WHERE model_name = ?1 AND active = 1
             ORDER BY loaded_at DESC LIMIT 1",
        )?;

        let row = stmt
            .query_row(rusqlite::params![model_name], |row| {
                let metadata_str: String = row.get(9)?;
                let metadata: serde_json::Value =
                    serde_json::from_str(&metadata_str).unwrap_or_default();

                Ok(pesti_plug_in::manifest::ModelWeightRow {
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
            .map_err(RunnerError::Sqlite)?;

        let mut tensor_stmt = self.conn.prepare(
            "SELECT id, weight_id, tensor_name, shape, dtype, size_bytes, checksum FROM tensor_metadata
             WHERE weight_id = ?1",
        )?;

        let tensors: Vec<pesti_plug_in::manifest::TensorMetadataRow> = tensor_stmt
            .query_map(rusqlite::params![row.id], |row| {
                Ok(pesti_plug_in::manifest::TensorMetadataRow {
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
            "Model loader: manifest loaded from safetensors DB"
        );

        Ok(manifest)
    }

    /// Verify weight checksum integrity.
    pub fn verify_checksum(&self, weight_id: &str, expected: &str) -> Result<bool, RunnerError> {
        pesti_safetensors::schema::verify_weight_checksum(&self.conn, weight_id, expected).map_err(
            |_: SafetensorsSchemaError| RunnerError::Sqlite(rusqlite::Error::QueryReturnedNoRows),
        )
    }

    /// List active weights for model selection.
    pub fn list_active(
        &self,
        limit: usize,
    ) -> Result<Vec<pesti_safetensors::schema::ModelWeightRow>, RunnerError> {
        pesti_safetensors::schema::list_active_weights(&self.conn, limit).map_err(
            |e: SafetensorsSchemaError| {
                RunnerError::Sqlite(match e {
                    SafetensorsSchemaError::Sqlite(r) => r,
                    _ => rusqlite::Error::QueryReturnedNoRows,
                })
            },
        )
    }

    /// Parse a GGUF file and return its header (metadata only, no tensor data).
    pub fn load_gguf_header(path: &Path) -> Result<GgufHeader, RunnerError> {
        pesti_gguf::parser::parse_gguf(path).map_err(RunnerError::Gguf)
    }

    /// Detect the quantization type from a GGUF header.
    pub fn detect_quantization(header: &GgufHeader) -> Option<GgufDtype> {
        header
            .get_kv_str("general.file_type")
            .and_then(|s| s.parse::<u32>().ok())
            .map(GgufDtype::from_u32)
            .or_else(|| {
                header.get_kv_str("general.file_type").map(|s| match s {
                    "F16" => GgufDtype::F16,
                    "F32" => GgufDtype::F32,
                    "Q4_0" => GgufDtype::Q4_0,
                    "Q4_1" => GgufDtype::Q4_1,
                    "Q5_0" => GgufDtype::Q5_0,
                    "Q5_1" => GgufDtype::Q5_1,
                    "Q8_0" => GgufDtype::Q8_0,
                    "Q8_1" => GgufDtype::Q8_1,
                    "Q2_K" => GgufDtype::Q2_K,
                    "Q3_K" => GgufDtype::Q3_K,
                    "Q4_K" => GgufDtype::Q4_K,
                    "Q5_K" => GgufDtype::Q5_K,
                    "Q6_K" => GgufDtype::Q6_K,
                    "Q8_K" => GgufDtype::Q8_K,
                    "BF16" => GgufDtype::BF16,
                    _ => GgufDtype::Unknown(0),
                })
            })
    }

    /// Extract raw tensor bytes from a GGUF file by tensor name.
    pub fn extract_gguf_tensor(path: &Path, tensor_name: &str) -> Result<Vec<u8>, RunnerError> {
        let header = Self::load_gguf_header(path)?;
        let tensor = header
            .tensors
            .iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| {
                RunnerError::Gguf(pesti_gguf::GgufError::InvalidTensor(format!(
                    "tensor '{tensor_name}' not found"
                )))
            })?;

        let size = tensor.stored_size()? as usize;
        pesti_gguf::parser::extract_tensor_bytes_from_path(path, tensor.offset, size)
            .map_err(RunnerError::Gguf)
    }

    /// Get architecture from a GGUF header.
    pub fn gguf_architecture(header: &GgufHeader) -> Option<&str> {
        header.architecture()
    }

    /// Get context length from a GGUF header.
    pub fn gguf_context_length(header: &GgufHeader) -> Option<u32> {
        header.context_length()
    }

    /// Get embedding dimension from a GGUF header.
    pub fn gguf_embedding_length(header: &GgufHeader) -> Option<u32> {
        header.embedding_length()
    }

    /// Get block count (number of layers) from a GGUF header.
    pub fn gguf_block_count(header: &GgufHeader) -> Option<u32> {
        header.block_count()
    }

    /// Get attention head count from a GGUF header.
    pub fn gguf_attention_head_count(header: &GgufHeader) -> Option<u32> {
        header.get_kv_u32("attention.head_count")
    }

    /// Get attention head count KV from a GGUF header.
    pub fn gguf_attention_head_count_kv(header: &GgufHeader) -> Option<u32> {
        header.get_kv_u32("attention.head_count_kv")
    }

    /// Get rope dimension count from a GGUF header.
    pub fn gguf_rope_dimension_count(header: &GgufHeader) -> Option<i32> {
        header.get_kv_u32("rope.dimension_count").map(|v| v as i32)
    }

    /// Get normalization epsilon from a GGUF header.
    pub fn gguf_normalization_epsilon(header: &GgufHeader) -> Option<f32> {
        header.kv_pairs.iter().find_map(|kv| {
            if kv.key == "attention.layer_norm_epsilon" || kv.key == "norm_eps" {
                kv.value.as_f32()
            } else {
                None
            }
        })
    }
}
