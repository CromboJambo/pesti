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
        // For K-family, use the claimed dtype but recalculate element count from data size.
        // Canonical ggml K-quant block sizes (bytes per 256-element block), from ggml-common.h
        // static_asserts: Q2_K=84, Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210, Q8_K=292.
        // Every K-quant block holds exactly QK_K = 256 elements.
        let (block_size, elements_per_block) = match dtype {
            // All Q4_K variants share the block_q4_K layout (144 B / 256 elem)
            GgufDtype::Q4K | GgufDtype::Q4K_M | GgufDtype::Q4K_S => (144, 256),
            GgufDtype::Q5K | GgufDtype::Q5K_M | GgufDtype::Q5K_S => (176, 256),
            GgufDtype::Q6K | GgufDtype::Q6K_S => (210, 256),
            GgufDtype::Q8K | GgufDtype::Q8K_M => (292, 256),
            GgufDtype::Q2K | GgufDtype::Q2K_S | GgufDtype::Q2K_M => (84, 256),
            GgufDtype::Q3K | GgufDtype::Q3K_S => (110, 256),
            // Q1K is a pesti-gguf phantom (ID 20 = IQ4_NL in official ggml.h); there is no
            // canonical Q1_K layout, so this keeps the legacy best-effort path.
            GgufDtype::Q1K => (20, 16),
            // Non-K-family quant types - shouldn't reach here but just in case
            _ => return dequantize_tensor(tensor, raw_data),
        };

        let num_blocks = raw_data.len() / block_size;
        let inferred_count = num_blocks * elements_per_block;
        
        // Use inferred count from actual data size for K-family types
        // This handles GGUF files with incorrect tensor shape claims
        tracing::debug!(
            tensor = %tensor.name,
            inferred = inferred_count,
            claimed = claimed_element_count,
            "Q4_K — using inferred element count: {} blocks × {} elem/block",
            num_blocks, elements_per_block
        );
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
            let dequantized = dequantize_q6_k(raw_data, inferred_element_count)
                .map_err(|e| RunnerError::Dequant(tensor.name.clone(), e.to_string()))?;
            Ok(dequantized
                .into_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
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
            let q1_val = ((q1 >> i) & 0x01) << 2;
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
            let q1_val = ((q1 >> i) & 0x01) << 2;
            let q = q1_val as i32 - 4;
            let scale = if q1_val > 0 { h[i / 4] } else { 1.0 };
            result.push(d * (q as f32) * scale + d_min);
        }
    }

    Ok(result)
}

// Canonical ggml K-quant dequantizers (256-element super-blocks).
// Ported from llama.cpp ggml/src/ggml-quants.c (dequantize_row_qX_K) and
// ggml/src/ggml-common.h block layouts. Every K-quant block holds QK_K = 256
// elements. Block byte sizes: Q2_K=84, Q3_K=110, Q4_K=144, Q5_K=176, Q6_K=210,
// Q8_K=292.
//
// NOTE on offsets: the C reference advances pointers (q, ql, qh, sc) each
// iteration. Here we fold those advances into explicit byte offsets so the
// Rust slices stay fixed. `chunk` indexes the 128- (or 64-) element sub-blocks
// within a 256-element super-block.

/// Extract a 6-bit scale and 6-bit min from the 12-byte K-scale field.
/// Mirrors ggml `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | (((scales[j - 4] >> 6) & 0x03) << 4),
            (scales[j + 4] >> 4) | (((scales[j] >> 6) & 0x03) << 4),
        )
    }
}

fn dequantize_q2_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q2_K (84 B / 256 elem):
    //   [0..16)   scales[16]  (4-bit scale + 4-bit min per 16-elem group)
    //   [16..80)  qs[64]      (2-bit quants)
    //   [82..84)  d (f16)
    //   [84..86)  dmin (f16)
    const BLOCK: usize = 84;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q2K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let dequant_block = |base: usize, limit: usize, out: &mut Vec<f32>| {
        let scales = &data[base..base + 16];
        let qs = &data[base + 16..base + 80];
        let d = f16_to_f32(&data[base + 82..base + 84]);
        let dmin = f16_to_f32(&data[base + 84..base + 86]);
        for chunk in 0..2 {
            let qs_base = chunk * 32;
            let sc_base = chunk * 8;
            let mut shift = 0usize;
            for j in 0..4 {
                let sc = scales[sc_base + 2 * j];
                let dl = d * (sc & 0x0F) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    if chunk * 128 + l < limit {
                        out.push(dl * ((qs[qs_base + l] >> shift) & 0x03) as f32 - ml);
                    }
                }
                let sc = scales[sc_base + 2 * j + 1];
                let dl = d * (sc & 0x0F) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 {
                    if chunk * 128 + 16 + l < limit {
                        out.push(dl * ((qs[qs_base + 16 + l] >> shift) & 0x03) as f32 - ml);
                    }
                }
                shift += 2;
            }
        }
    };

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        dequant_block(b * BLOCK, ELEM, &mut result);
    }
    if remaining > 0 {
        dequant_block(num_blocks * BLOCK, remaining, &mut result);
    }
    Ok(result)
}

fn dequantize_q3_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q3_K (110 B / 256 elem):
    //   [0..32)   hmask[32]   (high bit per element, global)
    //   [32..96)  qs[64]      (2-bit low quants)
    //   [96..108) scales[12]  (6-bit scales, packed)
    //   [108..110) d (f16)
    const BLOCK: usize = 110;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q3K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0F0F_0F0F;

    let dequant_block = |base: usize, limit: usize, out: &mut Vec<f32>| {
        let d_all = f16_to_f32(&data[base + 108..base + 110]);
        let hmask = &data[base..base + 32];
        let qs = &data[base + 32..base + 96];
        let scales_raw = &data[base + 96..base + 108];

        // Decode 16 six-bit scales from the 12-byte packed field.
        // The C reference does `memcpy(aux, scales, 12)` into a `uint32_t aux[4]`,
        // so only the first 12 bytes (aux[0..3]) are read; aux[3] is overwritten
        // before use. We mirror that: read 12 bytes, leave aux[3] = 0.
        let mut aux = [0u32; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes([
                scales_raw[4 * i],
                scales_raw[4 * i + 1],
                scales_raw[4 * i + 2],
                scales_raw[4 * i + 3],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let mut scales = [0i8; 16];
        for i in 0..4 {
            let bytes = aux[i].to_le_bytes();
            for k in 0..4 {
                scales[4 * i + k] = bytes[k] as i8;
            }
        }

        for chunk in 0..2 {
            let qs_base = chunk * 32;
            let sc_base = chunk * 8;
            let mut shift = 0usize;
            let mut m: u8 = 1 << (chunk * 4);
            for j in 0..4 {
                let dl = d_all * (scales[sc_base + 2 * j] as f32 - 32.0);
                for l in 0..16 {
                    if chunk * 128 + l < limit {
                        let q = ((qs[qs_base + l] >> shift) & 0x03) as i8;
                        let hb = if hmask[l] & m != 0 { 0 } else { 4 };
                        out.push(dl * (q - hb) as f32);
                    }
                }
                let dl = d_all * (scales[sc_base + 2 * j + 1] as f32 - 32.0);
                for l in 0..16 {
                    if chunk * 128 + 16 + l < limit {
                        let q = ((qs[qs_base + 16 + l] >> shift) & 0x03) as i8;
                        let hb = if hmask[16 + l] & m != 0 { 0 } else { 4 };
                        out.push(dl * (q - hb) as f32);
                    }
                }
                shift += 2;
                m <<= 1;
            }
        }
    };

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        dequant_block(b * BLOCK, ELEM, &mut result);
    }
    if remaining > 0 {
        dequant_block(num_blocks * BLOCK, remaining, &mut result);
    }
    Ok(result)
}

fn dequantize_q4_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q4_K (144 B / 256 elem):
    //   [0..2)    d (f16)
    //   [2..4)    dmin (f16)
    //   [4..16)   scales[12]   (6-bit scale + 6-bit min, packed)
    //   [16..144) qs[128]      (4-bit quants)
    const BLOCK: usize = 144;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q4K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let dequant_block = |base: usize, limit: usize, out: &mut Vec<f32>| {
        let d = f16_to_f32(&data[base..base + 2]);
        let dmin = f16_to_f32(&data[base + 2..base + 4]);
        let scales = &data[base + 4..base + 16];
        let qs = &data[base + 16..base + 144];
        for chunk in 0..4 {
            let qs_base = chunk * 32;
            let (sc1, m1) = get_scale_min_k4(chunk * 2, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(chunk * 2 + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                if chunk * 64 + l < limit {
                    out.push(d1 * (qs[qs_base + l] & 0x0F) as f32 - m1);
                }
            }
            for l in 0..32 {
                if chunk * 64 + 32 + l < limit {
                    out.push(d2 * (qs[qs_base + l] >> 4) as f32 - m2);
                }
            }
        }
    };

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        dequant_block(b * BLOCK, ELEM, &mut result);
    }
    if remaining > 0 {
        dequant_block(num_blocks * BLOCK, remaining, &mut result);
    }
    Ok(result)
}

fn dequantize_q5_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q5_K (176 B / 256 elem):
    //   [0..2)    d (f16)
    //   [2..4)    dmin (f16)
    //   [4..16)   scales[12]   (6-bit scale + 6-bit min, packed)
    //   [16..48)  qh[32]       (high bit per element)
    //   [48..176) qs[128]      (4-bit low quants)
    const BLOCK: usize = 176;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q5K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let dequant_block = |base: usize, limit: usize, out: &mut Vec<f32>| {
        let d = f16_to_f32(&data[base..base + 2]);
        let dmin = f16_to_f32(&data[base + 2..base + 4]);
        let scales = &data[base + 4..base + 16];
        let qh = &data[base + 16..base + 48];
        let qs = &data[base + 48..base + 176];
        for chunk in 0..4 {
            let qs_base = chunk * 32;
            let qh_base = chunk * 32;
            let (sc1, m1) = get_scale_min_k4(chunk * 2, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(chunk * 2 + 1, scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            let u1 = 1u8 << (chunk * 2);
            let u2 = 2u8 << (chunk * 2);
            for l in 0..32 {
                if chunk * 64 + l < limit {
                    let q1 = (qs[qs_base + l] & 0x0F) + if qh[qh_base + l] & u1 != 0 { 16 } else { 0 };
                    out.push(d1 * q1 as f32 - m1);
                }
            }
            for l in 0..32 {
                if chunk * 64 + 32 + l < limit {
                    let q2 = (qs[qs_base + l] >> 4) + if qh[qh_base + l] & u2 != 0 { 16 } else { 0 };
                    out.push(d2 * q2 as f32 - m2);
                }
            }
        }
    };

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        dequant_block(b * BLOCK, ELEM, &mut result);
    }
    if remaining > 0 {
        dequant_block(num_blocks * BLOCK, remaining, &mut result);
    }
    Ok(result)
}

fn dequantize_q6_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q6_K (210 B / 256 elem):
    //   [0..128)   ql[128]    (lower 4 bits)
    //   [128..192) qh[64]     (upper 2 bits)
    //   [192..208) scales[16] (int8, one per 16-elem group)
    //   [208..210) d (f16)
    const BLOCK: usize = 210;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q6K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let dequant_block = |base: usize, limit: usize, out: &mut Vec<f32>| {
        let d = f16_to_f32(&data[base + 208..base + 210]);
        let ql = &data[base..base + 128];
        let qh = &data[base + 128..base + 192];
        // scales are int8_t in ggml (block_q6_K.scales) — sign matters for
        // bytes >= 128. Reading them as u8 mis-decodes ~half the groups.
        let scales = &data[base + 192..base + 208];
        for chunk in 0..2 {
            let ql_base = chunk * 64;
            let qh_base = chunk * 32;
            let sc_base = chunk * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_base + l] & 0x0F) | (((qh[qh_base + l] >> 0) & 0x03) << 4)) as i8 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0x0F) | (((qh[qh_base + l] >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 0x03) << 4)) as i8 - 32;
                let s0 = scales[sc_base + is] as i8 as f32;
                let s2 = scales[sc_base + is + 2] as i8 as f32;
                let s4 = scales[sc_base + is + 4] as i8 as f32;
                let s6 = scales[sc_base + is + 6] as i8 as f32;
                if chunk * 128 + l < limit {
                    out.push(d * s0 * q1 as f32);
                }
                if chunk * 128 + 32 + l < limit {
                    out.push(d * s2 * q2 as f32);
                }
                if chunk * 128 + 64 + l < limit {
                    out.push(d * s4 * q3 as f32);
                }
                if chunk * 128 + 96 + l < limit {
                    out.push(d * s6 * q4 as f32);
                }
            }
        }
    };

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        dequant_block(b * BLOCK, ELEM, &mut result);
    }
    if remaining > 0 {
        dequant_block(num_blocks * BLOCK, remaining, &mut result);
    }
    Ok(result)
}

fn dequantize_q8_k(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    // block_q8_K (292 B / 256 elem):
    //   [0..4)    d (f32)
    //   [4..260)  qs[256]   (int8 quants)
    //   [260..292) bsums[16] (int16, unused by dequant)
    const BLOCK: usize = 292;
    const ELEM: usize = 256;
    let num_blocks = element_count / ELEM;
    let remaining = element_count % ELEM;
    let expected = num_blocks * BLOCK + if remaining > 0 { BLOCK } else { 0 };
    if data.len() < expected {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q8K data too small: got {} bytes, need {}",
            data.len(),
            expected
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    for b in 0..num_blocks {
        let base = b * BLOCK;
        let d = f32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]);
        for j in 0..ELEM {
            result.push(d * data[base + 4 + j] as i8 as f32);
        }
    }
    if remaining > 0 {
        let base = num_blocks * BLOCK;
        let d = f32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]);
        for j in 0..remaining {
            result.push(d * data[base + 4 + j] as i8 as f32);
        }
    }
    Ok(result)
}

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
    // GGML stores f16 scales little-endian (ggml_half).
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

// Q4_0 dequantization (simple 4-bit quantization without K-family scales)
// Format: 16 elements per block, 12 bytes per block
// - 2B: scale (f16)
// - 8B: quantized values (16 nibbles = 8 bytes)
fn dequantize_q4_0(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    let expected_size = num_full_blocks * 12
        + if remaining > 0 {
            2 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Gguf(pesti_gguf::GgufError::Io(format!(
            "Q4_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        ))));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        // Q4_0 format: 12 bytes per block
        let scale = f16_to_f32(&data[offset..offset + 2]);
        
        // Read 8 bytes of quantized values (16 nibbles)
        let qs = [
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
            data[offset + 8],
            data[offset + 9],
        ];

        // Dequantize: each byte contains two 4-bit values (0-15)
        for i in 0..16 {
            let q = if i % 2 == 0 {
                (qs[i / 2] >> 0) & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            
            // Q4_0: values are in range [0,15], centered at 7.5
            let v = scale * ((q as f32) - 7.5);
            result.push(v);
        }

        offset += 12;
    }

    // Handle remaining elements (partial block)
    if remaining > 0 {
        let scale = f16_to_f32(&data[offset..offset + 2]);
        
        let mut qs = [0u8; 4]; // Only need bytes for remaining nibbles
        for i in 0..remaining.div_ceil(2) {
            qs[i] = data[offset + 2 + i];
        }

        for i in 0..remaining {
            let q = if i % 2 == 0 {
                (qs[i / 2] >> 0) & 0x0F
            } else {
                (qs[i / 2] >> 4) & 0x0F
            };
            
            let v = scale * ((q as f32) - 7.5);
            result.push(v);
        }
    }

    Ok(result)
}

/// Transpose a 2D weight tensor from [in_features, out_features] to [out_features, in_features].
/// GGUF stores weights as [in_features, out_features], but Linear expects [out_features, in_features] row-major.
pub fn transpose_weight(weight: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; out_features * in_features];
    for i in 0..in_features {
        for j in 0..out_features {
            // GGUF: weight[i * out_features + j] (row-major [in, out])
            // Transposed: weight[j * in_features + i] (row-major [out, in])
            transposed[j * in_features + i] = weight[i * out_features + j];
        }
    }
    transposed
}
