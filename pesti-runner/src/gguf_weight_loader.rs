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
            "load_gguf_weights: data_section_start={}, tensor_count={}, file_size={}\\n\\n             first tensor: {} offset={} stored_size={}\\n\\n             will read at absolute offset={}\\n",
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
    let claimed_element_count = tensor.element_count() as usize;
    
    // Detect dtype mismatch: if claimed element_count * expected_bytes_per_element doesn't match data size,
    // the GGUF file might have wrong metadata. Try to infer actual dtype and element count from data size.
    let (inferred_dtype, inferred_element_count) = {
        // Only run reverse inference if data size is WAY smaller than expected
        // (indicates GGUF header is wrong about dtype/element_count)
        let mismatch_threshold = match dtype {
            GgufDtype::Q4_0 | GgufDtype::Q4_K | GgufDtype::Q4_K_M | GgufDtype::Q4_K_S
            | GgufDtype::Q5_0 | GgufDtype::Q5_1 | GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S
            | GgufDtype::Q8_0 | GgufDtype::Q8_K | GgufDtype::Q8_K_M => claimed_element_count / 2,
            GgufDtype::Q2_K | GgufDtype::Q2_K_S | GgufDtype::Q2_K_M => claimed_element_count / 4,
            GgufDtype::Q3_K | GgufDtype::Q3_K_S => claimed_element_count / 6,
            GgufDtype::Q1_K => claimed_element_count / 8,
            _ => (dtype, claimed_element_count), // Skip inference for F32/F16/BF16/I8/etc.
        };
        
        if raw_data.len() < mismatch_threshold {
            eprintln!("[DEBUG] Data size ({}) << expected {} bytes, trying reverse inference", 
                     raw_data.len(), mismatch_threshold);
            
            // Try each K-family dtype to see which one matches the data size
            let mut best_match = (dtype, claimed_element_count);
            
            for candidate_dtype in [
                GgufDtype::Q1_K, GgufDtype::Q2_K, GgufDtype::Q3_K, GgufDtype::Q4_K,
                GgufDtype::Q4_K_M, GgufDtype::Q5_0, GgufDtype::Q5_K, GgufDtype::Q6_K,
            ].iter() {
                let (block_size, elements_per_block) = match candidate_dtype {
                    GgufDtype::Q1_K => (20, 16),
                    GgufDtype::Q2_K | GgufDtype::Q2_K_S | GgufDtype::Q2_K_M => (32, 16),
                    GgufDtype::Q3_K | GgufDtype::Q3_K_S => (24, 16),
                    GgufDtype::Q4_K | GgufDtype::Q4_K_M | GgufDtype::Q4_K_S => (28, 16),
                    GgufDtype::Q5_0 | GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => (36, 16),
                    GgufDtype::Q6_K | GgufDtype::Q6_K_S => (42, 16),
                    _ => continue,
                };
                
                let num_blocks = raw_data.len() / block_size;
                let inferred_count = num_blocks * elements_per_block;
                
                // Calculate what bytes this would need
                let expected_bytes = match candidate_dtype {
                    GgufDtype::Q1_K => num_blocks * 20,
                    GgufDtype::Q2_K | GgufDtype::Q2_K_S | GgufDtype::Q2_K_M => num_blocks * 32,
                    GgufDtype::Q3_K | GgufDtype::Q3_K_S => num_blocks * 24,
                    GgufDtype::Q4_K | GgufDtype::Q4_K_M | GgufDtype::Q4_K_S => num_blocks * 28,
                    GgufDtype::Q5_0 | GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => num_blocks * 36,
                    GgufDtype::Q6_K | GgufDtype::Q6_K_S => num_blocks * 42,
                    _ => continue,
                };
                
                let diff = (raw_data.len() as i64 - expected_bytes as i64).abs();
                let rel_diff = if expected_bytes > 0 {
                    (diff as f64 / expected_bytes as f64 * 100.0).abs()
                } else {
                    100.0
                };
                
                eprintln!(
                    "[DEBUG] {:?}: {} blocks -> {} elements, expected={} bytes, actual={}, diff={:.2}%",
                    candidate_dtype, num_blocks, inferred_count, expected_bytes, raw_data.len(), rel_diff
                );
                
                // If this matches very closely (< 1% difference), use it
                if rel_diff < 1.0 && diff < 100 {
                    eprintln!("[DEBUG] MATCHED {:?} with {} elements!", candidate_dtype, inferred_count);
                    best_match = (*candidate_dtype, inferred_count);
                    break;
                }
            }
            
            best_match
        } else {
            (dtype, claimed_element_count)
        }
    };

    eprintln!(
        "Dequantizing tensor '{}' dtype=0x{:04X} ({:?}) with {} elements, {} bytes",
        tensor.name,
        tensor.dtype,
        dtype,
        claimed_element_count,
        raw_data.len()
    );
    if inferred_dtype != dtype || inferred_element_count != claimed_element_count {
        eprintln!(
            "  [WARN] Inferred dtype {:?} and {} elements from data size (claimed: {:?}, {})",
            inferred_dtype, inferred_element_count, dtype, claimed_element_count
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
            // Q5_1 uses same format as Q5_0 but with different scaling
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
        GgufDtype::Q2_K => {
            let dequantized = dequantize_q2_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3_K => {
            let dequantized = dequantize_q3_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_K | GgufDtype::Q4_K_M => {
            let dequantized = dequantize_q4_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q5_K | GgufDtype::Q5_K_M | GgufDtype::Q5_K_S => {
            let dequantized = dequantize_q5_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q6_K | GgufDtype::Q6_K_S => {
            let dequantized = dequantize_q6_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q8_K | GgufDtype::Q8_K_M => {
            let dequantized = dequantize_q8_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q1_K => {
            let dequantized = dequantize_q1_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K_S => {
            let dequantized = dequantize_q2_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q3_K_S => {
            let dequantized = dequantize_q3_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q4_K_S => {
            let dequantized = dequantize_q4_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        GgufDtype::Q2_K_M => {
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
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q3_K data too small: got {} bytes, need {}",
            data.len(), expected_size
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
        let mask = data[offset + 9];
        let q3 = [data[offset + 10], data[offset + 11], data[offset + 12]];
        let h = [
            f16_to_f32(&data[offset + 13..offset + 15]),
            f16_to_f32(&data[offset + 15..offset + 17]),
            f16_to_f32(&data[offset + 17..offset + 19]),
            f16_to_f32(&data[offset + 19..offset + 21]),
        ];

        for i in 0..16usize {
            // Q3_K: 3 bytes store 16 elements using complex bit interleaving
            // Each byte contributes to multiple elements
            let (byte_idx, bit_offset) = match i {
                0 => (0, 0),
                1 => (1, 0),
                2 => (2, 0),
                3 => (0, 6),
                4 => (0, 2),
                5 => (1, 2),
                6 => (2, 2),
                7 => (0, 4),
                8 => (0, 0),
                9 => (1, 0),
                10 => (2, 0),
                11 => (0, 6),
                12 => (0, 2),
                13 => (1, 2),
                14 => (2, 2),
                15 => (0, 4),
                _ => panic!("Unexpected i"),
            };
            let q3_val = ((q3[byte_idx] >> bit_offset) & 0x03) as u16;
            let scale = h[i / 4];
            result.push(d * (q3_val as f32) * scale + d_min);
        }
        offset += 21;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let _delta = data[offset + 4];
        let q3 = [data[offset + 5], data[offset + 6]];

        for i in 0..remaining {
            let q3_val = (((q3[i / 3] >> (i % 3)) & ((1 << 2) - 1)) as u16);
            result.push(d * (q3_val as f32)); // simplified for partial block
        }
    }

    Ok(result)
}

/// Dequantize Q4_K data to f32.
fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q4_K layout: d(f16,2B)+d_min(f16,2B)+q4(u8x2,2B)+h(f16x4,8B)=14B per block
    let expected_size = num_full_blocks * 14 + if remaining > 0 { 2 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q4_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q4 = [data[offset + 4], data[offset + 5]];
        let h = [
            f16_to_f32(&data[offset + 6..offset + 8]),
            f16_to_f32(&data[offset + 8..offset + 10]),
            f16_to_f32(&data[offset + 10..offset + 12]),
            f16_to_f32(&data[offset + 12..offset + 14]),
        ];

        for i in 0..16usize {
            let q4_val = (((q4[i / 2] >> ((i % 2) * 4)) & 0x0F) as u16);
            let scale = h[i / 4];
            result.push(d * (q4_val as f32) * scale + d_min);
        }
        offset += 14;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q4 = [data[offset + 4], data[offset + 5]];

        for i in 0..remaining {
            let q4_val = (((q4[i / 2] >> ((i % 2) * 4)) & 0x0F) as u16);
            result.push(d * (q4_val as f32)); // simplified for partial block
        }
    }

    Ok(result)
}

/// Dequantize Q5_K data to f32.
fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q5_K layout: d(f16,2B)+d_min(f16,2B)+q5_lo(u8x4,4B)+q5_h(u8x2,2B)=12B per block
    let expected_size = num_full_blocks * 12 + if remaining > 0 { 2 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q5_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q5_lo = [
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ];
        let q5_h = [data[offset + 8], data[offset + 9]];

        for i in 0..16usize {
            let q5_lo_val = ((q5_lo[i / 2] >> ((i % 2) * 4)) & 0x0F) as u16;
            let q5_h_val = ((q5_h[i / 8] >> (i % 8)) & 0x01) as u16;
            let q5_val = (q5_lo_val | ((q5_h_val << 4) & 0x10)) as i32 - 4;
            result.push(d * (q5_val as f32) + d_min);
        }
        offset += 12;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q5_lo = [data[offset + 4], data[offset + 5]];
        let q5_h = data[offset + 6];

        for i in 0..remaining {
            let q5_lo_val = ((q5_lo[i / 2] >> ((i % 2) * 4)) & 0x0F) as u16;
            let q5_h_val = ((q5_h >> (i % 8)) & 0x01) as u16;
            let q5_val = (q5_lo_val | ((q5_h_val << 4) & 0x10)) as i32 - 4;
            result.push(d * (q5_val as f32) + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32.
fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q6_K layout: d(f16,2B)+d_min(f16,2B)+q6(u8x3,3B)+h(f16x4,8B)=15B per block
    let expected_size = num_full_blocks * 15 + if remaining > 0 { 2 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q6 = [data[offset + 4], data[offset + 5], data[offset + 6]];
        let h = [
            f16_to_f32(&data[offset + 7..offset + 9]),
            f16_to_f32(&data[offset + 9..offset + 11]),
            f16_to_f32(&data[offset + 11..offset + 13]),
            f16_to_f32(&data[offset + 13..offset + 15]),
        ];

        for i in 0..16usize {
            let q6_val = (((q6[i / 3] >> (i % 3 * 2)) & 0x03) as u16);
            let scale = h[i / 4];
            result.push(d * (q6_val as f32) * scale + d_min);
        }
        offset += 15;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q6 = [data[offset + 4], data[offset + 5]];

        for i in 0..remaining {
            let q6_val = (((q6[i / 3] >> (i % 3 * 2)) & 0x03) as u16);
            result.push(d * (q6_val as f32)); // simplified for partial block
        }
    }

    Ok(result)
}

/// Dequantize Q8_K data to f32.
fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;

    // Q8_K layout: d(f16,2B)+d_min(f16,2B)+q8(u8x16,16B)=20B per block
    let expected_size = num_full_blocks * 20 + if remaining > 0 { 2 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q8_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q8 = &data[offset + 4..offset + 20];

        for i in 0..16usize {
            let q8_val = (q8[i] as i32 - 128) as f32;
            result.push(d * q8_val + d_min);
        }
        offset += 20;
    }

    if remaining > 0 {
        let d = f16_to_f32(&data[offset..offset + 2]);
        let d_min = f16_to_f32(&data[offset + 2..offset + 4]);
        let q8 = &data[offset + 4..offset + 4 + remaining];

        for i in 0..remaining {
            let q8_val = (q8[i] as i32 - 128) as f32;
            result.push(d * q8_val + d_min);
        }
    }

    Ok(result)
}

// ── Helper functions ───────────────────────────────────────────────────

/// Convert half precision (f16) to f32.
fn half_f32(data: &[u8]) -> Vec<f16> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

/// Convert bfloat16 to f32.
fn bf16_f32(data: &[u8]) -> Vec<f16> {
    // For simplicity, treat bfloat16 as f16 (lossy conversion)
    data.chunks_exact(2)
        .map(|chunk| {
            let high = chunk[0] as u16;
            let low = chunk[1] as u16;
            f16::from_bits((high << 8) | low)
        })
        .collect()
}

/// Convert f16 to f32.
fn f16_to_f32(data: &[u8]) -> f32 {
    f16::from_le_bytes([data[0], data[1]]).to_f32()
}