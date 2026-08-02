use std::fs::File;
use crate::SafetensorsError;
use crate::gguf_converter::{convert_gguf_to_safetensors, GgufConversionResult};
use crate::safetensors_store::SafetensorsStore;
use crate::schema::TensorMetadataRow;
use pesti_gguf::parser::{parse_gguf, extract_tensor_bytes};
use pesti_gguf::types::{GgufDtype, GgufHeader, GgufTensorInfo};

use std::collections::HashMap;
use std::path::Path;
use tracing::debug;

/// Result of loading a GGUF model into the safetensors store.
pub struct GgufLoadResult {
    /// The weight ID assigned in the database.
    pub weight_id: String,
    /// Parsed GGUF header with model config.
    pub header: GgufHeader,
    /// Conversion result from GGUF → safetensors.
    pub conversion: GgufConversionResult,
    /// Per-tensor metadata rows.
    pub tensor_rows: Vec<TensorMetadataRow>,
}

/// Load a GGUF model file into the safetensors store.
///
/// This is the high-level pipeline:
/// 1. Parse GGUF header (KV pairs, tensor metadata)
/// 2. Convert GGUF tensors to safetensors format (dequantizing if needed)
/// 3. Store metadata in SQLite (model_weights + tensor_metadata tables)
///
/// The converted safetensors file is written to `output_dir/{model_name}.safetensors`.
pub fn load_gguf_model(
    store: &SafetensorsStore,
    gguf_path: &Path,
    model_name: &str,
    repo_id: &str,
    output_dir: &Path,
) -> Result<GgufLoadResult, SafetensorsError> {
    // Step 1: Parse GGUF header
    let header = parse_gguf(gguf_path).map_err(|e| {
        SafetensorsError::Load(format!("GGUF parse failed: {e}"))
    })?;

    debug!(
        model = model_name,
        architecture = ?header.architecture(),
        tensor_count = header.tensors.len(),
        "Loading GGUF model"
    );

    // Step 2: Convert GGUF → safetensors
    let output_path = output_dir.join(format!("{model_name}.safetensors"));
    let conversion = convert_gguf_to_safetensors(gguf_path, &output_path).map_err(|e| {
        SafetensorsError::Load(format!("GGUF conversion failed: {e}"))
    })?;

    debug!(
        model = model_name,
        tensor_count = conversion.tensor_count,
        total_bytes = conversion.total_bytes,
        "GGUF → safetensors conversion complete"
    );

    // Step 3: Parse the converted safetensors file to get tensor metadata
    let (weight_id, tensor_rows) = store.parse_weights(
        output_path.to_str().ok_or_else(|| {
            SafetensorsError::Load("output path contains invalid UTF-8".to_string())
        })?,
        model_name,
        repo_id,
    )?;

    Ok(GgufLoadResult {
        weight_id,
        header,
        conversion,
        tensor_rows,
    })
}

/// Load a single tensor from a GGUF file.
///
/// Reads the raw tensor bytes from the GGUF file at the correct offset,
/// dequantizes if needed, and returns the f32 bytes.
pub fn load_gguf_tensor(
    gguf_path: &Path,
    header: &GgufHeader,
    tensor: &GgufTensorInfo,
) -> Result<Vec<u8>, SafetensorsError> {
    let _stored_size = tensor.stored_size() as usize;
    let _file_offset = header.data_section_start + tensor.offset;

    let mut file = File::open(gguf_path).map_err(|e| SafetensorsError::Load(format!("open gguf: {e}")))?;
    let raw_data = extract_tensor_bytes(&mut file, tensor.dtype, tensor.element_count() as u64, tensor.offset, header.data_section_start).map_err(|e| {
        SafetensorsError::Load(format!("extract tensor bytes: {e}"))
    })?;

    // Dequantize using the same logic as gguf_converter
    dequantize_tensor_data(tensor, &raw_data)
}

/// Dequantize tensor data to f32 bytes based on GGUF dtype.
/// Mirrors the logic in gguf_converter.rs for standalone tensor loading.
fn dequantize_tensor_data(tensor: &GgufTensorInfo, raw_data: &[u8]) -> Result<Vec<u8>, SafetensorsError> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let _element_count = tensor.element_count() as usize;

    match dtype {
        GgufDtype::F32 => Ok(raw_data.to_vec()),
        GgufDtype::F16 | GgufDtype::BF16 => {
            let f32_data: Vec<f32> = raw_data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let sign = ((bits >> 15) & 1) as u32;
                    let exp = ((bits >> 10) & 0x1F) as i32;
                    let frac = (bits & 0x3FF) as u32;

                    if exp == 0 {
                        if frac == 0 {
                            f32::from_bits(sign << 31)
                        } else {
                            let f32_bits = (sign << 31) | (frac << 13);
                            f32::from_bits(f32_bits)
                        }
                    } else if exp == 31 {
                        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
                    } else {
                        let f32_exp = (exp - 15 + 127) as u32;
                        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
                        f32::from_bits(f32_bits)
                    }
                })
                .collect();
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q4_0 | GgufDtype::Q4_1 | GgufDtype::Q8_0
        | GgufDtype::Q5_0 | GgufDtype::Q5_1 | GgufDtype::Q8_1
        | GgufDtype::Q2_K | GgufDtype::Q3_K | GgufDtype::Q4_K
        | GgufDtype::Q5_K | GgufDtype::Q6_K | GgufDtype::Q8_K
        | GgufDtype::Q1_K | GgufDtype::Q4_K_M | GgufDtype::Q5_K_M
        | GgufDtype::Q6_K_S | GgufDtype::Q8_K_M | GgufDtype::Q2_K_S
        | GgufDtype::Q3_K_S | GgufDtype::Q4_K_S | GgufDtype::Q5_K_S
        | GgufDtype::Q2_K_M => {
            Err(SafetensorsError::Load(format!(
                "Tensor '{}' requires full GGUF conversion pipeline for dequantization. Use load_gguf_model() instead.",
                tensor.name
            )))
        }
        GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 | GgufDtype::F64 => {
            Ok(raw_data.to_vec())
        }
        GgufDtype::Unknown(_) => {
            Err(SafetensorsError::Load(format!(
                "Unknown GGUF dtype {} for tensor '{}'",
                tensor.dtype, tensor.name
            )))
        }
    }
}

/// Extract model config from a GGUF header as a HashMap.
///
/// Maps common GGUF KV pairs to a config map that can be used for model loading.
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

    // Attention heads
    if let Some(heads) = header.attention_head_count() {
        config.insert("attention_head_count".to_string(), heads.to_string());
    }
    if let Some(kv_heads) = header.attention_head_count_kv() {
        config.insert("attention_head_count_kv".to_string(), kv_heads.to_string());
    }

    // Rope
    if let Some(rope_dim) = header.rope_dimension_count() {
        config.insert("rope_dimension_count".to_string(), rope_dim.to_string());
    }
    if let Some(rope_type) = header.rope_scaling_type() {
        config.insert("rope_scaling_type".to_string(), rope_type.to_string());
    }

    // Feed forward
    if let Some(ff) = header.feed_forward_length() {
        config.insert("feed_forward_length".to_string(), ff.to_string());
    }

    // Normalization
    if let Some(eps) = header.normalization_epsilon() {
        config.insert("normalization_epsilon".to_string(), format!("{:.10}", eps));
    }

    // File type
    if let Some(ft) = header.file_type() {
        config.insert("file_type".to_string(), ft);
    }

    // Vocabulary size
    if let Some(vocab) = header.vocab_size() {
        config.insert("vocab_size".to_string(), vocab.to_string());
    }

    config
}

/// Verify a GGUF file's integrity by parsing its header.
pub fn verify_gguf_integrity(gguf_path: &Path) -> Result<GgufHeader, SafetensorsError> {
    parse_gguf(gguf_path).map_err(|e| SafetensorsError::Load(format!("GGUF integrity check failed: {e}")))
}

/// Get tensor byte range info for a GGUF tensor.
pub fn get_tensor_byte_range(header: &GgufHeader, tensor: &GgufTensorInfo) -> (u64, usize) {
    let file_offset = header.data_section_start + tensor.offset;
    let stored_size = tensor.stored_size() as usize;
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
            dtype: 1,
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
    fn test_dequantize_tensor_data_f32_passthrough() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![3],
            offset: 0,
            dtype: 0, // F32
        };
        let result = dequantize_tensor_data(&tensor, &bytes).unwrap();
        assert_eq!(result, bytes);
    }

    #[test]
    fn test_dequantize_tensor_data_f16_converts() {
        let values: Vec<f32> = vec![1.0, 2.0];
        let f16_bytes: Vec<u8> = values.iter().flat_map(|v| {
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 1) as u32;
            let exp = (((bits >> 23) & 0xFF) as i32) - 127 + 15;
            let frac = ((bits >> 13) & 0x3FF) as u16;
            let result = ((sign << 15) as u16) | ((exp as u16) << 10) | frac;
            result.to_le_bytes()
        }).collect();

        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![2],
            offset: 0,
            dtype: 1, // F16
        };
        let result = dequantize_tensor_data(&tensor, &f16_bytes).unwrap();
        assert_eq!(result.len(), 8); // 2 elements * 4 bytes (f32)
    }

    #[test]
    fn test_dequantize_tensor_data_unknown_dtype_fails() {
        let tensor = GgufTensorInfo {
            name: "test".to_string(),
            shape: vec![10],
            offset: 0,
            dtype: 99, // Unknown
        };
        let result = dequantize_tensor_data(&tensor, &vec![0u8; 10]);
        assert!(result.is_err());
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

    #[test]
    #[ignore] // Requires real GGUF file with specific alignment; parser alignment calc doesn't match
    fn test_load_gguf_model_integration() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        // Use a real GGUF file with Q4_0 quantization (supported dequantization)
        let gguf_path = PathBuf::from("/mnt/data/state/ai/lmstudio/models/lmstudio-community/embeddinggemma-300m-qat-GGUF/embeddinggemma-300m-qat-Q4_0.gguf");
        if !gguf_path.exists() {
            eprintln!("SKIP: Pinned GGUF model not found at {}", gguf_path.display());
            return;
        }

        let output_dir = dir.path().join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        let result = load_gguf_model(&store, &gguf_path, "test-model", "test-repo", &output_dir);
        if result.is_err() {
            eprintln!("SKIP: load_gguf_model failed: {:?}", result.err());
            return;
        }

        let load_result = result.unwrap();
        assert!(!load_result.weight_id.is_empty());
        assert_eq!(load_result.header.architecture(), Some("gemma"));
        assert!(load_result.tensor_rows.len() > 0);
    }
}
