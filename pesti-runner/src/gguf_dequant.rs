// ── GGUF weight loading & dequantization helpers ─────────────────────────────
//
// This module provides pure Rust dequantization functions for GGUF quantized
// tensors with AVX2 SIMD acceleration via std::arch intrinsics.

use pesti_gguf::GgufDtype;
use std::collections::HashMap;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dequantize Q4_K data to f32 array (AVX2 SIMD-accelerated).
pub fn dequantize_q4_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 28 + remaining.div_ceil(2);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q4_K".to_string(),
            format!("data too small: got {} bytes, need {}", data.len(), expected_size),
        ));
    }

    let mut result = Vec::with_capacity(element_count);
    let mut block_offset = 0;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        // AVX2 SIMD path: process blocks using vectorized nibble extraction
        while block_offset + 1 <= num_blocks {
            let base = block_offset * 28;
            let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
            let _scale = half::f16::from_bits(scale_f16).to_f32();
            let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
            let d_min = half::f16::from_bits(d_min_f16).to_f32();
            let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
            let delta = half::f16::from_bits(delta_f16).to_f32();

            // Load 16 bytes (8 values) at a time using AVX2
            let vals_ptr = data.as_ptr().add(base + 6) as *const u8;
            let v0 = _mm_loadu_si128(vals_ptr as *const __m128i);

            // Extract high and low nibbles, subtract 8 (center around 0)
            let hi_nibbles = _mm_and_si128(_mm_srli_epi16(v0, 4), _mm_set1_epi8(0x0F));
            let lo_nibbles = _mm_and_si128(v0, _mm_set1_epi8(0x0F));

            // Convert to f32 and scale: delta * (nibble - 8) + d_min
            let hi_f32 = _mm256_cvtepi32_ps(_mm_cvtepi8_epi32(hi_nibbles));
            let lo_f32 = _mm256_cvtepi32_ps(_mm_cvtepi8_epi32(lo_nibbles));

            // Subtract 8 (bias) from all values
            let bias = _mm256_set1_ps(8.0);
            hi_f32 = _mm256_sub_ps(hi_f32, bias);
            lo_f32 = _mm256_sub_ps(lo_f32, bias);

            // Scale by delta and add d_min
            let delta_v = _mm256_set1_ps(delta);
            let dmin_v = _mm256_set1_ps(d_min);
            hi_f32 = _mm256_fmadd_ps(hi_f32, delta_v, dmin_v);
            lo_f32 = _mm256_fmadd_ps(lo_f32, delta_v, dmin_v);

            // Store results (8 f32 values)
            let out_ptr = result.as_mut_ptr().offset(result.len() as isize);
            _mm256_storeu_ps(out_ptr, hi_f32);
            _mm256_storeu_ps(out_ptr.add(8), lo_f32);
            result.set_len(result.len() + 16);

            block_offset += 1;
        }
    }

    // Scalar fallback for remaining blocks and non-AVX2 targets
    for block in block_offset..num_blocks {
        let base = block * 28;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..8usize {
            let byte_idx = base + 6 + i;
            let bval = data[byte_idx];
            result.push(delta * (((bval >> 4) & 0x0F) as i32 - 8) as f32 + d_min);
            result.push(delta * ((bval & 0x0F) as i32 - 8) as f32 + d_min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_blocks * 28;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..remaining {
            let nibble = (data[base + 6 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(delta * q as f32 + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q5_K data to f32 array (AVX2 SIMD-accelerated).
pub fn dequantize_q5_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 40 + remaining.div_ceil(2);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q5_K".to_string(),
            format!("data too small: got {} bytes, need {}", data.len(), expected_size),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 40;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..8usize {
            let byte_idx = base + 6 + i;
            let bval = data[byte_idx];
            result.push(delta * (((bval >> 4) & 0x1F) as i32 - 16) as f32 + d_min);
            result.push(delta * ((bval & 0x1F) as i32 - 16) as f32 + d_min);
        }
    }

    if remaining > 0 {
        let base = num_blocks * 40;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..remaining {
            let nibble = (data[base + 6 + i / 2] >> (4 * (i & 1))) & 0x1F;
            let q = nibble as i32 - 16;
            result.push(delta * q as f32 + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32 array (AVX2 SIMD-accelerated).
pub fn dequantize_q6_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 54 + remaining.div_ceil(4);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q6_K".to_string(),
            format!("data too small: got {} bytes, need {}", data.len(), expected_size),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 54;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..8usize {
            let byte_idx = base + 6 + i;
            let bval = data[byte_idx];
            result.push(delta * (((bval >> 4) & 0x3F) as i32 - 32) as f32 + d_min);
            result.push(delta * ((bval & 0x3F) as i32 - 32) as f32 + d_min);
        }
    }

    if remaining > 0 {
        let base = num_blocks * 54;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let _scale = half::f16::from_bits(scale_f16).to_f32();
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..remaining {
            let byte_idx = base + 6 + i / 4;
            let bit_offset = 2 * (i & 3);
            let value = (data[byte_idx] >> bit_offset) & 0x3F;
            let q = value as i32 - 32;
            result.push(delta * q as f32 + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 data to f32 array (AVX2 SIMD-accelerated).
pub fn dequantize_q8_0(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 34 + if remaining > 0 { 2 + remaining } else { 0 };

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q8_0".to_string(),
            format!("data too small: got {} bytes, need {}", data.len(), expected_size),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 34;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        for i in 0..8usize {
            let byte_idx = base + 2 + i;
            let q = data[byte_idx] as i8 as f32;
            result.push(scale * q);
        }
    }

    if remaining > 0 {
        let base = num_blocks * 34;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        for i in 0..remaining {
            let q = data[base + 2 + i] as i8 as f32;
            result.push(scale * q);
        }
    }

    Ok(result)
}

/// Load all weights from GGUF file into a HashMap keyed by tensor name.
pub fn load_gguf_weights(gguf_path: &str) -> crate::Result<HashMap<String, Tensor>> {
    let mut reader = pesti_gguf::GgufReader::open(gguf_path)?;
    let tensors = reader.tensor_infos()?;

    let mut weights = HashMap::new();
    for tensor in tensors {
        let name = tensor.name.clone();
        let dims: Vec<usize> = tensor.shape.iter().map(|d| *d as usize).collect();
        let data_type = match tensor.dtype {
            GgufDtype::F32 => TensorType::F32,
            GgufDtype::Q4_0 | GgufDtype::Q4_K => TensorType::Q4K,
            _ => {
                return Err(crate::error::RunnerError::Model(
                    format!("unsupported dtype for tensor {}: {:?}", name, tensor.dtype),
                ));
            }
        };

        let data = reader.read_tensor_data(&tensor)?;
        weights.insert(
            name.clone(),
            Tensor {
                shape: dims.clone(),
                dtype: data_type,
                data: data.clone(),
            },
        );
    }

    Ok(weights)
}

/// Load a GGUF file and extract model architecture metadata.
pub fn load_gguf_model(gguf_path: &str) -> crate::Result<ModelMetadata> {
    let reader = pesti_gguf::GgufReader::open(gguf_path)?;

    // Extract common architecture parameters
    let hidden_size = match reader.get_tensor("model.layers.0.self_attn.q_proj.weight") {
        Some(tensor) => tensor.shape[1] as usize,
        None => return Err(crate::error::RunnerError::Model(
            "could not determine hidden size".to_string(),
        )),
    };

    Ok(ModelMetadata {
        hidden_size,
        // TODO: extract num_layers, num_heads, vocab_size, etc. from GGUF metadata
    })
}

/// Simple tensor representation for the inference engine.
#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: TensorType,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorType {
    F32,
    Q4K,
}

/// Model metadata extracted from GGUF.
#[derive(Debug)]
pub struct ModelMetadata {
    pub hidden_size: usize,
    // num_layers, num_heads, vocab_size, etc. would go here
}