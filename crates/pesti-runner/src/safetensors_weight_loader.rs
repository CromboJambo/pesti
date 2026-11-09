//! SafeTensors weight loading.
//!
//! Loads tensors from a safetensors file into memory, converting f16/bf16 to f32.
//! Unlike GGUF, safetensors stores weights in their native (dequantized) format,
//! so no dequantization is needed.
//!
//! ## Supported dtypes
//!
//! - F32 — passthrough
//! - F16 — convert to f32
//! - BF16 — convert to f32
//! - U8 / I8 / I16 / I32 / I64 — passthrough
//!
//! ## Usage
//!
//! ```rust,ignore
//! use pesti_runner::safetensors_weight_loader::{load_safetensors_weights, SafetensorsWeights};
//!
//! let weights = load_safetensors_weights("/path/to/model.safetensors")?;
//! let model = LlamaModel::from_safetensors_weights(weights)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use tracing::debug;

use crate::error::{Result, RunnerError};

/// A loaded safetensors model's tensors in memory.
///
/// Each tensor is stored as f32 bytes (f16/bf16 converted, others passthrough).
/// The header provides model config (architecture, context length, etc.).
#[derive(Debug, Clone)]
pub struct SafetensorsWeights {
    /// Path to the loaded safetensors file.
    pub path: std::path::PathBuf,
    /// Tensor metadata: name → (shape, dtype, size_bytes).
    pub metadata: HashMap<String, (Vec<usize>, String, usize)>,
    /// Tensor data: name → loaded f32 bytes.
    pub tensors: HashMap<String, Vec<u8>>,
}

impl SafetensorsWeights {
    /// Get the shape of a tensor as `(out_features, in_features)`.
    ///
    /// Safetensors stores weight tensors as 2D `[out_features, in_features]`.
    /// Returns `(0, 0)` if the tensor is not found or has wrong ndims.
    pub fn tensor_shape(&self, name: &str) -> (usize, usize) {
        self.metadata
            .get(name)
            .and_then(|(shape, _, _)| {
                if shape.len() >= 2 {
                    Some((shape[0], shape[1]))
                } else {
                    None
                }
            })
            .unwrap_or((0, 0))
    }
}

/// Load all tensors from a safetensors file into memory.
///
/// Converts F16/BF16 to f32. F32 tensors are passed through.
/// U8/I8/I16/I32/I64 tensors are passed through as raw bytes.
pub fn load_safetensors_weights(safetensors_path: &Path) -> Result<SafetensorsWeights> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    let tensor_count = handle.tensors().len();
    debug!(
        path = %safetensors_path.display(),
        tensor_count,
        "Loading safetensors weights"
    );

    let mut tensors = HashMap::with_capacity(tensor_count);
    let mut metadata = HashMap::with_capacity(tensor_count);

    for (tensor_name, tensor_view) in handle.tensors() {
        let dtype = tensor_view.dtype().to_string();
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let size_bytes = data.len();

        metadata.insert(
            tensor_name.clone(),
            (shape.to_vec(), dtype.clone(), size_bytes),
        );

        // Convert to f32 bytes (dequantize)
        let loaded = convert_dtype(data, &dtype)?;
        tensors.insert(tensor_name.clone(), loaded);
    }

    Ok(SafetensorsWeights {
        path: safetensors_path.to_path_buf(),
        metadata,
        tensors,
    })
}

/// Load a single tensor from a safetensors file.
///
/// Converts F16/BF16 to f32. F32 tensors are passed through.
pub fn load_safetensors_tensor(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<(String, Vec<u8>)> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    let tensor_view = handle
        .tensor(tensor_name)
        .map_err(|e| RunnerError::ModelLoad(format!("Tensor '{tensor_name}' not found: {e}")))?;

    let dtype = tensor_view.dtype().to_string();
    let data = tensor_view.data();

    let loaded = convert_dtype(data, &dtype)?;

    Ok((dtype, loaded))
}

/// Convert raw tensor bytes to f32 bytes based on dtype.
///
/// - F32 — passthrough
/// - F16 — convert to f32
/// - BF16 — convert to f32
/// - U8/I8/I16/I32/I64 — passthrough as raw bytes
fn convert_dtype(raw: &[u8], dtype: &str) -> Result<Vec<u8>> {
    match dtype {
        "F32" | "FLOAT_32" => Ok(raw.to_vec()),
        "F16" | "FLOAT_16" => {
            let f32_data = half_f32(raw);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        "BF16" | "BFLOAT_16" => {
            let f32_data = bf16_f32(raw);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        "U8" | "UINT_8" | "I8" | "INT_8" | "I16" | "INT_16" | "I32" | "INT_32" | "I64"
        | "INT_64" | "U32" | "UINT_32" | "U64" | "UINT_64" => Ok(raw.to_vec()),
        _ => Err(RunnerError::ModelLoad(format!(
            "Unsupported safetensors dtype: {dtype}"
        ))),
    }
}

/// Convert F16 (half-float) bytes to f32.
fn half_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
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
        .collect()
}

/// Convert BF16 (bfloat16) bytes to f32.
///
/// BF16 has the same exponent width as F32 (8 bits) but fewer mantissa bits (7 vs 23).
/// Conversion is a simple bit extension (pad mantissa with zeros).
fn bf16_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            // BF16 → F32: extend mantissa with zeros (no rounding needed for exact conversion)
            let f32_bits = (bits as u32) << 16;
            f32::from_bits(f32_bits)
        })
        .collect()
}

/// Extract model config from safetensors metadata.
///
/// Safetensors files store metadata as a JSON object in the file header.
/// Common keys: `general.architecture`, `llama.context_length`, etc.
///
/// The file format is: u64 LE num_tensors + JSON header (null-terminated) + tensor data.
pub fn extract_safetensors_config(
    safetensors_path: &Path,
) -> Result<std::collections::HashMap<String, String>> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    // Use read_metadata to correctly parse the num_tensors-prefixed JSON header.
    let (_header_size, metadata) = safetensors::SafeTensors::read_metadata(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors metadata: {e}")))?;

    let mut config = std::collections::HashMap::new();

    if let Some(meta_map) = metadata.metadata() {
        for (key, value) in meta_map {
            config.insert(key.clone(), value.clone());
        }
    }

    Ok(config)
}

/// Get tensor count from a safetensors file.
pub fn get_safetensors_tensor_count(safetensors_path: &Path) -> Result<usize> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    Ok(handle.tensors().len())
}

/// Get total size of all tensors in a safetensors file.
pub fn get_safetensors_total_size(safetensors_path: &Path) -> Result<usize> {
    let file_data = std::fs::read(safetensors_path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read safetensors file: {e}")))?;

    let handle = safetensors::SafeTensors::deserialize(&file_data)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to deserialize safetensors: {e}")))?;

    Ok(handle.tensors().iter().map(|(_, tv)| tv.data().len()).sum())
}
