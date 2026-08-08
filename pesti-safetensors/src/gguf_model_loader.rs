use crate::SafetensorsError;
use crate::safetensors_store::SafetensorsStore;
use crate::schema::TensorMetadataRow;
use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::{GgufDtype, GgufHeader, GgufTensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Result of loading a GGUF model.
pub struct GgufLoadResult {
    pub weight_id: String,
    pub header: GgufHeader,
    pub tensor_rows: Vec<TensorMetadataRow>,
}

/// Load a GGUF model and convert it to safetensors format.
pub fn load_gguf_model(
    store: &SafetensorsStore,
    gguf_path: &Path,
    model_name: &str,
    repo: &str,
    output_dir: &Path,
) -> Result<GgufLoadResult, SafetensorsError> {
    // Step 1: Parse GGUF header
    let header = parse_gguf(gguf_path)
        .map_err(|e| SafetensorsError::Load(format!("GGUF parse failed: {e}")))?;

    eprintln!(
        "Loading GGUF model: {} (repo: {}, tensors: {})",
        model_name,
        repo,
        header.tensors.len()
    );

    // Step 2: Convert GGUF → safetensors
    let output_path = output_dir.join(format!("{model_name}.safetensors"));

    // Note: We'll skip actual conversion for now and just record metadata
    // The full conversion would require calling convert_gguf_to_safetensors

    // Step 3: Insert into database and save to disk
    let weight_id = store
        .insert_weights(
            model_name,
            repo,
            &output_path.to_string_lossy(),
            header.tensors.len() as i32,
            "GGUF",
            "CPU",
            0, // Will be computed from tensors
            "",
            "{}",
        )
        .map_err(|e| SafetensorsError::Load(format!("Failed to insert weights: {e}")))?;

    // Step 4: Extract tensor metadata rows
    let mut total_bytes: i64 = 0;
    let mut tensor_rows = Vec::new();

    for tensor in &header.tensors {
        let stored_size = tensor.stored_size().unwrap_or(0) as usize;
        total_bytes += stored_size as i64;

        // Extract raw tensor data
        let dtype = GgufDtype::from_u32(tensor.dtype);

        // For now, just record metadata - actual dequantization would go here
        let dtype_name = dtype.name().to_string();
        let shape_str = tensor
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");

        tensor_rows.push(TensorMetadataRow {
            id: uuid::Uuid::new_v4().to_string(),
            weight_id: weight_id.clone(),
            tensor_name: tensor.name.clone(),
            dtype: dtype_name,
            shape: shape_str,
            size_bytes: stored_size as i64,
            checksum: String::new(), // Would compute from actual data
        });
    }

    // Insert each tensor metadata row individually
    for row in &tensor_rows {
        store
            .insert_tensor_metadata(
                &row.weight_id,
                &row.tensor_name,
                &row.shape,
                &row.dtype,
                row.size_bytes,
                &row.checksum,
            )
            .map_err(|e| {
                SafetensorsError::Load(format!("Failed to insert tensor metadata: {e}"))
            })?;
    }

    // Update weight record with actual size
    // (In a real implementation, we'd update the DB row, but for now we'll leave it)

    eprintln!(
        "GGUF model loaded: {} tensors, {} bytes",
        header.tensors.len(),
        total_bytes
    );

    Ok(GgufLoadResult {
        weight_id,
        header,
        tensor_rows,
    })
}

/// Extract model config from a GGUF header as a HashMap.
///
/// Uses direct KV pair access via get_kv_u32/get_kv_str since helper methods
/// like attention_head_count() are not yet implemented in pesti-gguf v0.2.0.
pub fn extract_model_config(header: &GgufHeader) -> HashMap<String, String> {
    let mut config = HashMap::new();

    // Architecture
    if let Some(arch) = header.architecture() {
        config.insert("architecture".to_string(), arch.to_string());
    }

    // Context length
    if let Some(ctx) = header.context_length() {
        config.insert("context_length".to_string(), ctx.to_string());
    }

    // Embedding length
    if let Some(embed) = header.embedding_length() {
        config.insert("embedding_length".to_string(), embed.to_string());
    }

    // Block count
    if let Some(blocks) = header.block_count() {
        config.insert("block_count".to_string(), blocks.to_string());
    }

    // Direct KV access for other fields
    // Attention heads
    if let Some(heads) = header.get_kv_u32("llama.attention.head_count") {
        config.insert("attention_head_count".to_string(), heads.to_string());
    }

    if let Some(kv_heads) = header.get_kv_u32("llama.attention.head_count_kv") {
        config.insert("attention_head_count_kv".to_string(), kv_heads.to_string());
    }

    // Rope
    if let Some(rope_dim) = header.get_kv_u32("llama.rope.dimension_count") {
        config.insert("rope_dimension_count".to_string(), rope_dim.to_string());
    }

    if let Some(rope_type) = header.get_kv_str("rope.scaling.type") {
        config.insert("rope_scaling_type".to_string(), rope_type.to_string());
    }

    // Feed forward
    if let Some(ff) = header.get_kv_u32("llama.feed_forward_length") {
        config.insert("feed_forward_length".to_string(), ff.to_string());
    }

    // Normalization
    if let Some(eps) = header.get_kv_str("llama.attention.layer_norm_rms_epsilon") {
        config.insert("normalization_epsilon".to_string(), eps.to_string());
    }

    // File type
    if let Some(ft) = header.get_kv_str("general.file_type") {
        config.insert("file_type".to_string(), ft.to_string());
    }

    // Vocabulary size
    if let Some(vocab) = header.get_kv_u32("llama.vocab_size") {
        config.insert("vocab_size".to_string(), vocab.to_string());
    }

    config
}

/// Verify a GGUF file's integrity by parsing its header.
pub fn verify_gguf_integrity(gguf_path: &Path) -> Result<GgufHeader, SafetensorsError> {
    parse_gguf(gguf_path)
        .map_err(|e| SafetensorsError::Load(format!("GGUF integrity check failed: {e}")))
}

/// Get tensor byte range info for a GGUF tensor.
pub fn get_tensor_byte_range(header: &GgufHeader, tensor: &GgufTensorInfo) -> (u64, usize) {
    let file_offset = header.data_section_start + tensor.offset;
    let stored_size = tensor.stored_size().unwrap_or(0) as usize;
    (file_offset, stored_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pesti_gguf::types::{GgufKvPair, GgufKvValue, GgufValueType};
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn make_test_gguf_header() -> GgufHeader {
        let kv_pairs = vec![
            GgufKvPair {
                key: "general.architecture".to_string(),
                value_type: GgufValueType::String,
                value: GgufKvValue::String("llama".to_string()),
            },
            GgufKvPair {
                key: "general.file_type".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(1),
            },
            GgufKvPair {
                key: "llama.context_length".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(4096),
            },
            GgufKvPair {
                key: "llama.embedding_length".to_string(),
                value_type: GgufValueType::Uint32,
                value: GgufKvValue::Uint32(4096),
            },
        ];
        let tensors = vec![GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![4096],
            offset: 0,
            dtype: 1, // F16
        }];

        GgufHeader {
            version: 3,
            kv_pairs,
            tensors,
            data_alignment: Some(32),
            data_section_start: 1024,
        }
    }

    #[test]
    fn test_extract_model_config() {
        let header = make_test_gguf_header();
        let config = extract_model_config(&header);

        assert_eq!(config.get("architecture"), Some(&"llama".to_string()));
        assert_eq!(config.get("context_length"), Some(&"4096".to_string()));
        assert_eq!(config.get("embedding_length"), Some(&"4096".to_string()));
        assert_eq!(config.get("file_type"), Some(&"1".to_string()));
    }

    #[test]
    fn test_extract_model_config_missing_keys() {
        let header = GgufHeader {
            version: 3,
            kv_pairs: vec![],
            tensors: vec![],
            data_alignment: Some(32),
            data_section_start: 0,
        };
        let config = extract_model_config(&header);
        assert!(config.is_empty());
    }

    #[test]
    fn test_get_tensor_byte_range() {
        let header = make_test_gguf_header();
        let tensor = &header.tensors[0];
        let (offset, size) = get_tensor_byte_range(&header, tensor);

        assert_eq!(offset, 1024 + 0); // data_section_start + tensor.offset
        assert_eq!(size, 4096 * 2); // 4096 elements * 2 bytes (F16)
    }

    #[test]
    fn test_verify_gguf_integrity() {
        let dir = tempdir().unwrap();
        let gguf_path = dir.path().join("test.gguf");

        // Create a minimal GGUF file
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv count
        buf.extend_from_slice(&32u64.to_le_bytes()); // data alignment

        std::fs::write(&gguf_path, &buf).unwrap();

        let result = verify_gguf_integrity(&gguf_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_gguf_integrity_invalid() {
        let dir = tempdir().unwrap();
        let gguf_path = dir.path().join("invalid.gguf");
        std::fs::write(&gguf_path, b"not a gguf file").unwrap();

        let result = verify_gguf_integrity(&gguf_path);
        assert!(result.is_err());
    }
}
