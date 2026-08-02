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

use crate::dequantize::{dequantize_q4_0_ggml, dequantize_q4_1_ggml, dequantize_q5_0, dequantize_q8_0_ggml};
use crate::error::{Result, RunnerError};

/// A loaded GGUF model's tensors in memory.
///
/// Each tensor is stored as f32 bytes (dequantized if needed).
/// The header provides model config (architecture, context length, etc.).
#[derive(Debug, Clone)]
pub struct GgufWeights {
    /// Parsed GGUF header with model config.
    pub header: GgufHeader,
    /// Tensor data: name → dequantized f32 bytes.
    pub tensors: HashMap<String, Vec<u8>>,
}

/// Load all tensors from a GGUF file into memory.
///
/// Dequantizes Q4_0, converts F16/BF16 to f32. F32 tensors are passed through.
/// Returns the header + all tensor data.
pub fn load_gguf_weights(gguf_path: &Path) -> Result<GgufWeights> {
    let header = parse_gguf(gguf_path)?;
    let file_size = std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0);
    if !header.tensors.is_empty() {
        std::fs::write("/tmp/llm-debug.log", format!(
            "load_gguf_weights: data_section_start={}, tensor_count={}, file_size={}\\n\\\n             first tensor: {} offset={} stored_size={}\\n\\\n             will read at absolute offset={}\\n",
            header.data_section_start, header.tensors.len(), file_size,
            header.tensors[0].name, header.tensors[0].offset, header.tensors[0].stored_size(),
            header.data_section_start + header.tensors[0].offset,
        )).ok();
    }

    let mut tensors = HashMap::with_capacity(header.tensors.len());

    for tensor in &header.tensors {
        let stored_size = tensor.stored_size() as usize;
        let file_offset = header.data_section_start + tensor.offset;
        eprintln!("  extract: {} offset={} stored_size={} file_total={}", 
            tensor.name, file_offset, stored_size, std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0));

        let raw_data = extract_tensor_bytes_from_path(gguf_path, file_offset, stored_size).map_err(|e| {
            RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
                "extract {} at {} size {}: {e}", tensor.name, file_offset, stored_size
            )))
        })?;

        let dequantized = dequantize_tensor(tensor, &raw_data)?;

        tensors.insert(tensor.name.clone(), dequantized);
    }

    Ok(GgufWeights { header, tensors })
}

/// Load a single tensor from a GGUF file.
///
/// Dequantizes Q4_0, converts F16/BF16 to f32. F32 tensors are passed through.
pub fn load_gguf_tensor(gguf_path: &Path, tensor_name: &str) -> Result<(GgufHeader, Vec<u8>)> {
    let header = parse_gguf(gguf_path)?;

    let tensor = header.get_tensor(tensor_name).ok_or_else(|| {
        RunnerError::Gguf(pesti_gguf::GgufError::InvalidTensor(format!(
            "tensor '{tensor_name}' not found in file"
        )))
    })?;

    let stored_size = tensor.stored_size() as usize;
    let file_offset = header.data_section_start + tensor.offset;

    let raw_data = extract_tensor_bytes_from_path(gguf_path, file_offset, stored_size)?;

    let dequantized = dequantize_tensor(tensor, &raw_data)?;

    Ok((header, dequantized))
}

/// Dequantize tensor data to f32 bytes based on GGUF dtype.
fn dequantize_tensor(tensor: &GgufTensorInfo, raw_data: &[u8]) -> Result<Vec<u8>> {
    let dtype = GgufDtype::from_u32(tensor.dtype);
    let element_count = tensor.element_count() as usize;

    match dtype {
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
            let dequantized = dequantize_q4_0_ggml(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_1 => {
            let dequantized = dequantize_q4_1_ggml(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_0 => {
            let dequantized = dequantize_q5_0(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_0 => {
            let dequantized = dequantize_q8_0_ggml(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K => {
            let dequantized = dequantize_q2_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3_K => {
            let dequantized = dequantize_q3_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_K | GgufDtype::Q4_K_M => {
            let dequantized = dequantize_q4_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => {
            let dequantized = dequantize_q5_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q6_K | GgufDtype::Q6_K_S => {
            let dequantized = dequantize_q6_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_K | GgufDtype::Q8_K_M => {
            let dequantized = dequantize_q8_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q1_K => {
            let dequantized = dequantize_q1_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K_S => {
            let dequantized = dequantize_q2_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3_K_S => {
            let dequantized = dequantize_q3_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_K_S => {
            let dequantized = dequantize_q4_k(raw_data, element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K_M => {
            let dequantized = dequantize_q2_k(raw_data, element_count)
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
            "Unsupported GGUF dtype {} for tensor '{}'. Use load_gguf_model() for full conversion pipeline.",
            tensor.dtype, tensor.name
        )))),
    }
}

// ── K-family dequantization implementations ─────────────────────────

/// Dequantize Q1_K data to f32.
///
/// Q1_K block: 16 elements, 20 bytes per block.
fn dequantize_q1_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 20 + if remaining > 0 { 2 + remaining.div_ceil(2) } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q1_K data too small: got {} bytes, need {}",
            data.len(), expected_size
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

/// Dequantize Q2_K data to f32.
///
/// Q2_K block: 16 elements, 16 bytes per block.
fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q2_K layout: d(f16,2B)+d_min(f16,2B)+q1(u8,1B)+q2(u32,4B)+h(f16x4,8B)=17B per block
    let expected_size = (num_full_blocks * 17 + if remaining > 0 { 2 } else { 0 }) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q2_K data too small: got {} bytes, need {}",
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
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
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
            result.push(d * (q as f32)); // simplified for partial block
        }
    }

    Ok(result)
}

/// Dequantize Q3_K data to f32.
fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q3_K layout: d(f16,2B)+d_min(f16,2B)+delta(i8,1B)+k_scale(u8x4,4B)+mask(u8,1B)+q3(u8x3,3B)+h(f16x4,8B)=21B
    let expected_size = num_full_blocks * 21 + if remaining > 0 { 3 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q3_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let delta = data[offset + 4] as i8 as f32;
        let k_scale = [
            data[offset + 5] as f32,
            data[offset + 6] as f32,
            data[offset + 7] as f32,
            data[offset + 8] as f32,
        ];
        let mask = data[offset + 9];
        let q3 = [
            data[offset + 10],
            data[offset + 11],
        ];

        for i in 0..16usize {
            let q3_val = ((q3[i / 2] >> (4 * (i % 2))) & 0x07) as u8;
            let mask_bit = (mask >> i) & 1;
            let q = q3_val as i32 - (((mask_bit as i32) << 2) | ((mask_bit as i32) << 1));
            let scale = d * k_scale[i / 4] + d_min;
            result.push(scale);
        }
        offset += 21;
    }

    Ok(result)
}

/// Dequantize Q4_K data to f32.
fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;

    // Q4_K layout: d(f16,2B)+min(f16,2B)+scales(10B)+qs(16B)=30B per block
    // Actually: n/4 + n*6/32 + 48 bytes total = 14B per block
    let expected_size = num_full_blocks * 14 + if remaining > 0 { 5 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 14;

        // Parse scale (f16)
        let scale = f16_to_f32(&data[base..base + 2]);

        // Parse min (f16)
        let min = f16_to_f32(&data[base + 2..base + 4]);

        // Parse scales (10 bytes for 32 elements = 4 scale groups of 8)
        let scales = &data[base + 4..base + 14];

        // Extract nibbles and dequantize
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }

            // Get low 4 bits from nibbles (stored separately after scales)
            let q_idx = base + 14 + i / 2;
            let lo = (data[q_idx] >> (4 * (i % 2))) & 0x0F;

            // Q4_K: value = scale * (lo - min) / 16.0
            let q = lo as f32 - min;
            result.push(scale * (q / 16.0));
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 14;

        let scale = f16_to_f32(&data[base..base + 2]);
        let min = f16_to_f32(&data[base + 2..base + 4]);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let q_idx = base + 14 + i / 2;
            let lo = (data[q_idx] >> (4 * (i % 2))) & 0x0F;
            let q = lo as f32 - min;
            result.push(scale * (q / 16.0));
        }
    }

    Ok(result)
}

/// Dequantize Q5_K data to f32.
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q5_K layout: d(f16,2B)+min(f16,2B)+scale(u8,1B)+q4_0(12B)+h(u8x2,2B)=19B per block
    let expected_size = num_full_blocks * 19 + if remaining > 0 { 3 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale = data[offset + 4] as f32;

        // q5_lo: 8 bytes (low nibbles for all 16 elements)
        let q5_lo = [
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
        ];

        // q5_h: 2 bytes (high bits for elements 0-15)
        let q5_h = [data[offset + 9], data[offset + 10]];

        for i in 0..16usize {
            let lo = (q5_lo[i / 4] >> (4 * (i % 4))) & 0x0F;
            let hi = ((q5_h[i / 8] >> (i % 8)) & 1) as i32;

            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - min) / scale));
        }
        offset += 19;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let min = f16_to_f32(&data[offset + 2..offset + 4]);
        let scale = data[offset + 4] as f32;

        // q5_lo: 8 bytes (low nibbles for all 16 elements) — need at least remaining.min(8) bytes
        let q5_lo = &data[offset + 5..offset + 5 + remaining.min(8)];
        let q5_h = data[offset + 9];

        for i in 0..remaining {
            let lo = (q5_lo[i / 4] >> (4 * (i % 4))) & 0x0F;
            let hi = ((q5_h >> i) & 1) as i32;

            let q = lo as i32 + hi * 16;
            result.push(d * ((q as f32 - min) / scale));
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32.
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q6_K layout: d(f16,2B)+q6(12B)=14B per block (not 16B!)
    // Actually: n/2 + n/4 + 256 bytes total
    let expected_size = num_full_blocks * 12 + if remaining > 0 { 3 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let q6 = &data[offset + 2..offset + 14];

        for i in 0..16usize {
            // Q6_K: 2 bits per element, with high bit from mask
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as u8;
            let q = q6_val as i32 - 32;
            result.push(d * (q as f32 / 64.0));
        }
        offset += 12;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let q6 = &data[offset + 2..offset + 2 + remaining.min(8)];

        for i in 0..remaining {
            let q6_val = ((q6[i / 4] >> (2 * (i % 4))) & 0x03) as u8;
            let q = q6_val as i32 - 32;
            result.push(d * (q as f32 / 64.0));
        }
    }

    Ok(result)
}

/// Dequantize Q8_K data to f32.
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q8_K layout: d(f16,2B)+q8(16B)=18B per block
    let expected_size = (num_full_blocks * 18 + if remaining > 0 { 2 } else { 0 }) as usize;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);

        for i in 0..16usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[offset + 2 + i] as i8 as f32 / 128.0;
            result.push(d * q);
        }
        offset += 18;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);

        for i in 0..remaining {
            let q = data[offset + 2 + i] as i8 as f32 / 128.0;
            result.push(d * q);
        }
    }

    Ok(result)
}

// ── Half-float helpers ───────────────────────────────────────────────

/// Convert half-float (f16) bytes to f32.
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormal: treat as 1.0 * 2^-14 * (frac / 1024)
            let f32_bits = (sign << 31) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal: bias conversion from 15 to 127
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

/// Convert half-float bytes to f32 slice.
fn half_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some(f16_to_f32(chunk))
            } else {
                None
            }
        })
        .collect()
}

/// Convert brain-float (bf16) bytes to f32.
fn bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                // BF16: upper 8 bits are exponent+sign, lower 8 bits are mantissa
                // To convert to F32: shift left by 16 and keep top 8 bits of mantissa
                let f32_bits = ((bits as u32) << 16);
                Some(f32::from_bits(f32_bits))
            } else {
                None
            }
        })
        .collect()
}
