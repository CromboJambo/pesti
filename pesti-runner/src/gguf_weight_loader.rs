//! GGUF weight loading with dequantization.
//!
//! Loads tensors from a GGUF file into memory, dequantizing Q4_0 and converting
//! F16/BF16 to f32. Returns a `GgufWeights` struct that can be fed directly into
//! the inference engine.
//!
//! ## Supported dtypes
//!
//! - F32 — passthrough
//! - F16 / BF16 — convert to f32
//! - Q4_0 — dequantize to f32 (32 elements per block)
//! - Q4_1 — dequantize to f32
//! - Q8_0 — dequantize to f32
//! - I8 / I16 / I32 / I64 — passthrough
//!

use std::collections::HashMap;
use std::path::Path;

use pesti_gguf::parser::{extract_tensor_bytes_from_path, parse_gguf};
use pesti_gguf::types::{GgufDtype, GgufHeader, GgufTensorInfo};

use crate::dequantize::{
    dequantize_q4_0_ggml, dequantize_q4_1_ggml, dequantize_q5_0, dequantize_q8_0_ggml,
};
use crate::error::{Result, RunnerError};
use half::f16;

/// A loaded GGUF model's tensors in memory.
#[derive(Debug, Clone)]
pub struct GgufWeights {
    /// Parsed GGUF header with model config.
    pub header: GgufHeader,
    /// Tensor data: name → dequantized f32 bytes.
    pub tensors: HashMap<String, Vec<u8>>,
    /// Raw quantized tensor bytes (before dequantization).
    /// Used by QuantizedLinear for tile-by-tile dequantization.
    pub raw_tensors: HashMap<String, Vec<u8>>,
}

impl GgufWeights {
    /// Get the dtype name of a tensor (e.g., "Q4K_M", "F32").
    pub fn tensor_dtype(&self, name: &str) -> Option<String> {
        self.header.has_tensor(name).then(|| {
            let dtype = pesti_gguf::types::GgufDtype::from_u32(
                self.header
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .map(|t| t.dtype)
                    .unwrap_or(0),
            );
            format!("{:?}", dtype)
        })
    }

    /// Get the shape of a tensor as `(in_features, out_features)`.
    ///
    /// GGUF stores weight tensors as `[in_features, out_features]`.
    /// Returns `(0, 0)` if the tensor is not found or has wrong ndims.
    pub fn tensor_shape(&self, name: &str) -> (usize, usize) {
        self.header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| {
                if t.shape.len() >= 2 {
                    Some((t.shape[0] as usize, t.shape[1] as usize))
                } else {
                    None
                }
            })
            .unwrap_or((0, 0))
    }
}

/// Load all tensors from a GGUF file into memory.
pub fn load_gguf_weights(gguf_path: &Path) -> Result<GgufWeights> {
    let header = parse_gguf(gguf_path)?;
    let mut tensors = HashMap::with_capacity(header.tensors.len());
    let mut raw_tensors = HashMap::with_capacity(header.tensors.len());

    for tensor in &header.tensors {
        let stored_size = tensor.stored_size()? as usize;
        let file_offset = header.data_section_start + tensor.offset;
        let raw_data = extract_tensor_bytes_from_path(gguf_path, file_offset, stored_size)?;
        // Store raw quantized bytes for QuantizedLinear
        raw_tensors.insert(tensor.name.clone(), raw_data.clone());
        let dequantized = dequantize_tensor(tensor, &raw_data)?;
        tensors.insert(tensor.name.clone(), dequantized);
    }

    Ok(GgufWeights {
        header,
        tensors,
        raw_tensors,
    })
}

/// Load a single tensor from a GGUF file.
pub fn load_gguf_tensor(gguf_path: &Path, tensor_name: &str) -> Result<(GgufHeader, Vec<u8>)> {
    let header = parse_gguf(gguf_path)?;
    let tensor = header
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .ok_or_else(|| {
            RunnerError::Gguf(pesti_gguf::GgufError::InvalidTensor(format!(
                "tensor '{tensor_name}' not found in file"
            )))
        })?;

    let stored_size = tensor.stored_size()? as usize;
    let file_offset = header.data_section_start + tensor.offset;
    let raw_data = extract_tensor_bytes_from_path(gguf_path, file_offset, stored_size)?;
    let dequantized = dequantize_tensor(tensor, &raw_data)?;

    Ok((header, dequantized))
}

/// Dequantize tensor data to f32 bytes based on GGUF dtype.
fn dequantize_tensor(tensor: &GgufTensorInfo, raw_data: &[u8]) -> Result<Vec<u8>> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let claimed_element_count = tensor.element_count() as usize;

    // Detect dtype mismatch: reverse inference for K-family quant types
    let (inferred_dtype, inferred_element_count) = if matches!(
        dtype,
        GgufDtype::Q4K
            | GgufDtype::Q4K_M
            | GgufDtype::Q4K_S
            | GgufDtype::Q5K
            | GgufDtype::Q5K_M
            | GgufDtype::Q5K_S
            | GgufDtype::Q6K
            | GgufDtype::Q6K_S
            | GgufDtype::Q8K
            | GgufDtype::Q8K_M
            | GgufDtype::Q2K
            | GgufDtype::Q2K_S
            | GgufDtype::Q2K_M
            | GgufDtype::Q3K
            | GgufDtype::Q3K_S
            | GgufDtype::Q1K
    ) {
        // For K-family, use the claimed dtype but recalculate element count from data size
        let (block_size, elements_per_block) = match dtype {
            GgufDtype::Q4K | GgufDtype::Q4K_M | GgufDtype::Q4K_S => (28, 16),
            GgufDtype::Q5K | GgufDtype::Q5K_M | GgufDtype::Q5K_S => (36, 16),
            GgufDtype::Q6K | GgufDtype::Q6K_S => (42, 16),
            GgufDtype::Q8K | GgufDtype::Q8K_M => (40, 16),
            GgufDtype::Q2K | GgufDtype::Q2K_S | GgufDtype::Q2K_M => (32, 16),
            GgufDtype::Q3K | GgufDtype::Q3K_S => (24, 16),
            GgufDtype::Q1K => (20, 16),
            // Non-K-family quant types - shouldn't reach here but just in case
            _ => return dequantize_tensor(tensor, raw_data),
        };

        let num_blocks = raw_data.len() / block_size;
        let inferred_count = num_blocks * elements_per_block;

        (dtype, inferred_count)
    } else {
        // Non-K-family types - use claimed values
        (dtype, claimed_element_count)
    };

    tracing::debug!(
        tensor = %tensor.name,
        dtype = ?dtype,
        elements = claimed_element_count,
        bytes = raw_data.len(),
        "Dequantizing tensor"
    );
    if inferred_dtype != dtype || inferred_element_count != claimed_element_count {
        tracing::warn!(
            inferred_dtype = ?inferred_dtype,
            inferred_elements = inferred_element_count,
            claimed_dtype = ?dtype,
            claimed_elements = claimed_element_count,
            "Dtype/element count mismatch — using inferred values from data size"
        );
    }

    match inferred_dtype {
        GgufDtype::F32 => Ok(raw_data.to_vec()),
        GgufDtype::F16 => {
            let f32_data = half_f32(raw_data);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::BF16 => {
            let f32_data = bf16_f32(raw_data);
            Ok(f32_data.into_iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        GgufDtype::Q4_0 => {
            let dequantized = dequantize_q4_0_ggml(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_1 => {
            let dequantized = dequantize_q4_1_ggml(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_0 => {
            let dequantized = dequantize_q5_0(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_1 => {
            let dequantized = dequantize_q5_0(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_0 => {
            let dequantized = dequantize_q8_0_ggml(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2K => {
            let dequantized = dequantize_q2_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3K => {
            let dequantized = dequantize_q3_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4K | GgufDtype::Q4K_M => {
            let dequantized = dequantize_q4_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5K | GgufDtype::Q5K_M | GgufDtype::Q5K_S => {
            let dequantized = dequantize_q5_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q6K | GgufDtype::Q6K_S => {
            match dequantize_q6_k(raw_data, inferred_element_count) {
                Ok(result) => Ok(result.into_iter().flat_map(|v| v.to_le_bytes()).collect()),
                Err(_) => Ok(vec![]),
            }
        }
        GgufDtype::Q8K | GgufDtype::Q8K_M => {
            let dequantized = dequantize_q8_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q1K => {
            let dequantized = dequantize_q1_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2K_S => {
            let dequantized = dequantize_q2_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3K_S => {
            let dequantized = dequantize_q3_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4K_S => {
            let dequantized = dequantize_q4_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2K_M => {
            let dequantized = dequantize_q2_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::I8 | GgufDtype::I16 | GgufDtype::I32 | GgufDtype::I64 => Ok(raw_data.to_vec()),
        GgufDtype::Unknown(_) => Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Unknown GGUF dtype {} for tensor '{}'",
            tensor.dtype, tensor.name
        )))),
        _ => Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Unsupported GGUF dtype {} for tensor '{}'",
            tensor.dtype, tensor.name
        )))),
    }
}

// ── K-family dequantization implementations ─────────────────────────

fn dequantize_q1_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 20
        + if remaining > 0 {
            2 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q1K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let _delta = [
            f16_to_f32(&data[offset + 6..offset + 8]),
            f16_to_f32(&data[offset + 8..offset + 10]),
            f16_to_f32(&data[offset + 10..offset + 12]),
            f16_to_f32(&data[offset + 12..offset + 14]),
        ];
        let h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..16usize {
            let q1_val = (((q1 >> i) & 0x01) as u16) << 2;
            let q = q1_val as i32 - 4;
            let scale = if q1_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
        offset += 20;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u16::from_le_bytes([data[offset + 4], data[offset + 5]]);
        let h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..remaining {
            let q1_val = (((q1 >> i) & 0x01) as u16) << 2;
            let q = q1_val as i32 - 4;
            let scale = if q1_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
    }

    Ok(result)
}

fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = (num_full_blocks * 17 + if remaining > 0 { 2 } else { 0 }) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q2K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u8::from_le_bytes([data[offset + 4]]);
        let q2 = u32::from_le_bytes([
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
        ]);
        let h = [
            f16_to_f32(&data[offset + 9..offset + 11]),
            f16_to_f32(&data[offset + 11..offset + 13]),
            f16_to_f32(&data[offset + 13..offset + 15]),
            f16_to_f32(&data[offset + 15..offset + 17]),
        ];

        for i in 0..16usize {
            let q2_val = ((q2 >> (2 * i)) & 0x03) as u16;
            let q1_val = (((q1 >> i) & 0x01) as u16) << 2;
            let q = (q1_val | q2_val) as i32 - 4;
            let scale = if q2_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
        offset += 17;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let _d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q1 = u8::from_le_bytes([data[offset + 4]]);
        let q2 = u32::from_le_bytes([
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
        ]);

        for i in 0..remaining {
            let q2_val = ((q2 >> (2 * i)) & 0x03) as u16;
            let q1_val = (((q1 >> i) & 0x01) as u16) << 2;
            let q = (q1_val | q2_val) as i32 - 4;
            result.push(d * (q as f32));
        }
    }

    Ok(result)
}

fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 21 + if remaining > 0 { 3 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q3K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let _delta = data[offset + 4];
        let k_scale = [
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
        ];
        let _mask = data[offset + 9];
        let q3 = u32::from_le_bytes([
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
        ]);
        let _h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..16usize {
            let q3_val = ((q3 >> (2 * i)) & 0x03) as u16;
            let q = (q3_val as i32 - 4) as f32 * k_scale[i / 4] as f32;
            result.push(d * q + d_min);
        }
        offset += 21;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let _delta = data[offset + 4];
        let k_scale = [
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
        ];
        let _mask = data[offset + 9];
        let q3 = u32::from_le_bytes([
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
        ]);
        let _h = [
            f16_to_f32(&data[offset + 14..offset + 16]),
            f16_to_f32(&data[offset + 16..offset + 18]),
            f16_to_f32(&data[offset + 18..offset + 20]),
            f16_to_f32(&data[offset + 20..offset + 22]),
        ];

        for i in 0..remaining {
            let q3_val = ((q3 >> (2 * i)) & 0x03) as u16;
            let q = (q3_val as i32 - 4) as f32 * k_scale[i / 4] as f32;
            result.push(d * q + d_min);
        }
    }

    Ok(result)
}

fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 28 + if remaining > 0 { 4 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q4K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        // Q4K format: qs stores 16 nibbles (8 bytes) + h stores 2 scales (4 bytes)
        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        // First 8 elements use qs_low
        for i in 0..8usize {
            let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
            let v = h[0] * ((q as f32) - 4.0);
            result.push(d + delta * v);
        }

        // Second 8 elements use qs_high
        for i in 0..8usize {
            let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
            let v = h[1] * ((q as f32) - 4.0);
            result.push(d + delta * v);
        }

        offset += 28;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        // Q4K format: qs stores 16 nibbles (8 bytes) + h stores 2 scales (4 bytes)
        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        for i in 0..remaining {
            let q = if i < 8 {
                ((qs_low >> (i * 4)) & 0x0F) as u8
            } else {
                ((qs_high >> ((i - 8) * 4)) & 0x0F) as u8
            };
            let v = if i < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }
    }

    Ok(result)
}
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 36 + if remaining > 0 { 4 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q5K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        // Q5K format: 36 bytes/block
        // - 2B: scale (f16)
        // - 2B: delta (f16)
        // - 8B: qs (two u32s storing 16 nibbles: first 4 bits for values 0-7, last 4 bits for values 8-15)
        // - 4B: h (two f16 scales)

        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        // First 8 elements (values 0-7 use h[0])
        for i in 0..8usize {
            let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }

        // Next 8 elements (values 8-15 use qs_high)
        for i in 0..8usize {
            let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }
        offset += 36;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        for i in 0..remaining {
            let q = if i < 8 {
                ((qs_low >> (i * 4)) & 0x0F) as u8
            } else {
                ((qs_high >> ((i - 8) * 4)) & 0x0F) as u8
            };
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }
    }

    Ok(result)
}

fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    // Q6K block size: 42 bytes per 16 elements (total tensor = 256 elements = 105 * 42 bytes)
    // Per-block layout: d(2) + scales(8) + qs_low(8) + h_extra/qs_high_flags(4) + padding(20) = 42 bytes
    let expected_size = num_full_blocks * 42 + if remaining > 0 { 5 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q6K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        // d (scale): f16 at offset 0-1
        let d = f16_to_f32(&data[offset..offset + 2]);

        // scales: 4 × f16 (8 bytes) at offsets 2-9, one per group of 4 elements
        let scales = [
            f16_to_f32(&data[offset + 2..offset + 4]),
            f16_to_f32(&data[offset + 4..offset + 6]),
            f16_to_f32(&data[offset + 6..offset + 8]),
            f16_to_f32(&data[offset + 8..offset + 10]),
        ];

        // qs_low: 8 bytes at offset 10-17, stores lower 2 bits for all 16 elements
        // Each byte holds 4 values (2 bits each)
        let qs_low_start = offset + 10;

        // h_extra/qs_high_flags: 4 bytes at offset 18-21, bit-packed flags for upper nibbles
        // These determine which of the 4 h_extra scales to use for each element
        let qs_high_flags_start = offset + 18;

        // Dequantize 16 elements per block
        for i in 0..16usize {
            // Extract lower 2 bits from qs_low (stored as 2-bit values, 4 per byte)
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;

            // Extract upper bits from qs_high_flags (2 bits per element, bit-packed)
            let flag_byte_idx = i / 4;
            let flag_bit = (i % 4) * 2;
            let flag = ((data[qs_high_flags_start + flag_byte_idx] >> flag_bit) & 0x03) as u8;

            // In Q6K, the upper nibble is derived from flags and scales
            // q = q_low + 4 * q_high where q_high comes from h_extra based on flag
            let q_high = if flag == 0 { 0 } else { (flag - 1) as usize };

            // Combine: full quantized value is a 6-bit integer
            let q = (q_low as i32) + 4 * (q_high as i32);

            // Select scale based on which group of 4 this element belongs to
            let scale = scales[i / 4];

            // Dequantize: value = d * (q - 32) * scale
            // The -32 is the zero-point offset for Q6K
            let v = (q as f32 - 32.0) * scale;
            result.push(d * v);
        }

        offset += 42;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let scales = [
            f16_to_f32(&data[offset + 2..offset + 4]),
            f16_to_f32(&data[offset + 4..offset + 6]),
            f16_to_f32(&data[offset + 6..offset + 8]),
            f16_to_f32(&data[offset + 8..offset + 10]),
        ];

        let qs_low_start = offset + 10;

        // h_extra/qs_high_flags: 4 bytes at offset 18-21, bit-packed flags for upper nibbles
        let qs_high_flags_start = offset + 18;

        for i in 0..remaining {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;

            // Extract upper bits from qs_high_flags (2 bits per element, bit-packed)
            let flag_byte_idx = i / 4;
            let flag_bit = (i % 4) * 2;
            let flag = ((data[qs_high_flags_start + flag_byte_idx] >> flag_bit) & 0x03) as u8;

            // In Q6K, the upper nibble is derived from flags and scales
            let q_high = if flag == 0 { 0 } else { (flag - 1) as usize };

            // Combine: full quantized value is a 6-bit integer
            let q = (q_low as i32) + 4 * (q_high as i32);

            // Select scale based on which group of 4 this element belongs to
            let scale = scales[i / 4];

            // Dequantize: value = d * (q - 32) * scale
            let v = (q as f32 - 32.0) * scale;
            result.push(d * v);
        }
    }

    Ok(result)
}

fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 40 + if remaining > 0 { 4 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q8K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        // Q8K format: 40 bytes/block
        // - 2B: scale (f16)
        // - 2B: delta (f16)
        // - 8B: qs (two u32s storing 16 nibbles)
        // - 4B: h (two f16 scales)

        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        // First 8 elements (values 0-7 use h[0])
        for i in 0..8usize {
            let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }

        // Next 8 elements (values 8-15 use qs_high)
        for i in 0..8usize {
            let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }
        offset += 40;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let delta = f16_to_f32(&data[offset + 2..offset + 4]);

        let qs_low = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let h = [
            f16_to_f32(&data[offset + 12..offset + 14]),
            f16_to_f32(&data[offset + 14..offset + 16]),
        ];

        for i in 0..remaining {
            let q = if i < 8 {
                ((qs_low >> (i * 4)) & 0x0F) as u8
            } else {
                ((qs_high >> ((i - 8) * 4)) & 0x0F) as u8
            };
            let v = if q < 8 {
                h[0] * ((q as f32) - 4.0)
            } else {
                h[1] * ((q as f32) - 4.0)
            };
            result.push(d + delta * v);
        }
    }

    Ok(result)
}

// Helper conversion functions
fn half_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f16::from_be_bytes([b[0], b[1]]).to_f32())
        .collect()
}

fn bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = (b[0] as u32) << 16 | (b[1] as u32);
            f32::from_bits(bits)
        })
        .collect()
}

fn f16_to_f32(bytes: &[u8]) -> f32 {
    f16::from_be_bytes([bytes[0], bytes[1]]).to_f32()
}
