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
        eprintln!(
            "  extract: {} offset={} stored_size={} file_total={}",
            tensor.name,
            file_offset,
            stored_size,
            std::fs::metadata(gguf_path).map(|m| m.len()).unwrap_or(0)
        );

        let raw_data = extract_tensor_bytes_from_path(gguf_path, file_offset, stored_size)
            .map_err(|e| {
                RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
                    "extract {} at {} size {}: {e}",
                    tensor.name, file_offset, stored_size
                )))
            })?;

        let dequantized = dequantize_tensor(tensor, &raw_data)?;

        // For now, keep original tensor names - model loader handles architecture-specific lookup
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

/// Map tensor names from GGUF file format to canonical internal names.
///
/// Different architectures use different naming conventions for the same tensors.
/// This function maps them to a canonical form expected by the inference engine.
fn map_tensor_name_for_architecture(tensor_name: &str, architecture: &str) -> String {
    match architecture.to_lowercase().as_str() {
        "qwen2" | "qwen" => map_qwen2_to_canonical(tensor_name),
        "llama" | "mllama" => tensor_name.to_string(),
        _ => {
            // Default: passthrough for unknown architectures
            eprintln!(
                "  [WARN] Unknown architecture '{}', using original tensor name '{}'",
                architecture, tensor_name
            );
            tensor_name.to_string()
        }
    }
}

/// Map Qwen2 tensor names to canonical Llama-style names.
fn map_qwen2_to_canonical(tensor_name: &str) -> String {
    match tensor_name {
        // Embedding layer
        "token_embd.weight" => "tok_embeddings.weight".to_string(),

        // Output layer
        "lm_head.weight" => "output.weight".to_string(),

        // Attention layers - extract layer number from blk.X
        name if name.starts_with("blk.") && name.contains(".attn_q.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.attention.wq.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".attn_k.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.attention.wk.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".attn_v.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.attention.wv.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".attn_output.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.attention.wo.weight", layer)
        }

        // Feed-forward layers
        name if name.starts_with("blk.") && name.contains(".ffn_gate.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.feed_forward.w1.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".ffn_down.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.feed_forward.w2.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".ffn_up.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.feed_forward.w3.weight", layer)
        }

        // Layer norms
        name if name.starts_with("blk.") && name.contains(".attn_norm.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.attention_norm.weight", layer)
        }
        name if name.starts_with("blk.") && name.contains(".ffn_norm.weight") => {
            let layer = name.split('.').nth(1).unwrap_or("0");
            format!("layers.{}.ffn_norm.weight", layer)
        }

        // Default: passthrough
        _ => tensor_name.to_string(),
    }
}

// ── K-family dequantization implementations ─────────────────────────

/// Dequantize Q1_K data to f32.
///
/// Q1_K block: 16 elements, 20 bytes per block.
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
        let _delta = data[offset + 4] as i8 as f32;
        let k_scale = [
            data[offset + 5] as f32,
            data[offset + 6] as f32,
            data[offset + 7] as f32,
            data[offset + 8] as f32,
        ];
        let mask = data[offset + 9];
        let q3 = [data[offset + 10], data[offset + 11]];

        for i in 0..16usize {
            let q3_val = ((q3[i / 2] >> (4 * (i % 2))) & 0x07) as u8;
            let mask_bit = (mask >> i) & 1;
            let _q = q3_val as i32 - (((mask_bit as i32) << 2) | ((mask_bit as i32) << 1));
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
        let _scales = &data[base + 4..base + 14];

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

    // Q5_K layout: d(f16,2B)+min(f16,2B)+q3(u8x3,3B)+h(f16x4,8B)=27B per block
    let expected_size = num_full_blocks * 27 + if remaining > 0 { 13 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 27;

        // Parse scale (f16)
        let scale = f16_to_f32(&data[base..base + 2]);

        // Parse min (f16)
        let min = f16_to_f32(&data[base + 2..base + 4]);

        // Parse q3 nibbles (stored after d+min)
        let q3_low = data[base + 4];
        let q3_high = data[base + 5];

        // Parse h scales (f16x4)
        let h = [
            f16_to_f32(&data[base + 6..base + 8]),
            f16_to_f32(&data[base + 8..base + 10]),
            f16_to_f32(&data[base + 10..base + 12]),
            f16_to_f32(&data[base + 12..base + 14]),
        ];

        // Dequantize 16 elements
        for i in 0..16usize {
            if result.len() >= element_count {
                break;
            }

            // Get low 3 bits from q3_low and high bit from q3_high
            let q3_val = if i < 8 {
                ((q3_low >> i) & 0x07) as u16
            } else {
                ((q3_high >> (i - 8)) & 0x07) as u16
            };

            // Q5_K: value = scale * q3_val + min
            let q = q3_val as f32;
            result.push(scale * q + min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 27;

        let scale = f16_to_f32(&data[base..base + 2]);
        let min = f16_to_f32(&data[base + 2..base + 4]);
        let q3_low = data[base + 4];
        let q3_high = data[base + 5];

        for i in 0..remaining {
            let q3_val = if i < 8 {
                ((q3_low >> i) & 0x07) as u16
            } else {
                ((q3_high >> (i - 8)) & 0x07) as u16
            };
            let q = q3_val as f32;
            result.push(scale * q + min);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32.
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // Q6_K layout: 6 bits per element, bit-packed
    // Total bytes = ceil(elements * 6 / 8)
    let expected_size = (element_count * 6 + 7) / 8;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    // Read 6 bits per element from bit-packed data
    for i in 0..element_count {
        let byte_idx = i * 6 / 8;
        let bit_offset = (i * 6) % 8;

        // Extract 6 bits starting at bit_offset
        let val = if byte_idx + 1 < data.len() {
            ((data[byte_idx] as u32) | ((data[byte_idx + 1] as u32) << 8)) >> bit_offset
        } else {
            (data[byte_idx] as u32) >> bit_offset
        };

        // Q6_K: values are 0-63, need to convert to f32
        // For now, just use the raw value (simplified - real Q6_K has scales/min)
        let q = val as f32;
        result.push(q);
    }

    Ok(result)
}

/// Dequantize Q8_K data to f32.
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;

    // Q8_K layout: d(f16,2B)+d_min(f16,2B)+scale(i8,1B)+qs(u8x32,32B)=35B per block
    let expected_size = num_full_blocks * 35 + if remaining > 0 { 4 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 35;

        // Parse scale (f16)
        let d = f16_to_f32(&data[base..base + 2]);
        let d_min = f16_to_f32(&data[base + 2..base + 4]);
        let scale = data[base + 4] as i8 as f32;

        // Parse qs (u8x32)
        let qs = &data[base + 5..base + 37];

        // Dequantize 32 elements
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }

            let q = qs[i] as i8 as f32;
            result.push(d * q + d_min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 35;

        let d = f16_to_f32(&data[base..base + 2]);
        let d_min = f16_to_f32(&data[base + 2..base + 4]);
        let scale = data[base + 4] as i8 as f32;

        for i in 0..remaining {
            let q = data[base + 5 + i] as i8 as f32;
            result.push(d * q + d_min);
        }
    }

    Ok(result)
}

// ── Helper functions ───────────────────────────────────────────────────────

/// Convert f16 bytes to f32.
fn half_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect()
}

/// Convert bf16 bytes to f32.
fn bf16_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| {
            let bits = ((chunk[0] as u32) << 16) | (chunk[1] as u32);
            f32::from_bits(bits)
        })
        .collect()
}

/// Convert f16 to f32.
fn f16_to_f32(data: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([data[0], data[1]]) as u32;
    f16::from_le_bytes([data[0], data[1]]).to_f32()
}
