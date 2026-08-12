// ── GGUF weight loading & dequantization helpers ─────────────────────────────
//
// This module provides pure Rust dequantization functions for GGUF quantized
// tensors. It uses pesti_gguf's parser and types, avoiding manual parsing.

use pesti_gguf::GgufDtype;
use std::collections::HashMap;

/// Dequantize Q4_K data to f32 array.
pub fn dequantize_q4_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    // Q4_K format: 32 elements per block, 28 bytes per block (Q4_0 + extra metadata)
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 28 + remaining.div_ceil(2);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q4_K".to_string(),
            format!(
                "data too small: got {} bytes, need {}",
                data.len(),
                expected_size
            ),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 28;

        // Parse scale (f16) - little-endian
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        // Parse d_min (f16) - little-endian
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();

        // Parse delta (f16) - little-endian
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        // Extract nibbles and dequantize
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 6 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8; // Centered around 0
            result.push(delta * q as f32 + d_min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_blocks * 28;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

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

/// Dequantize Q5_K data to f32 array.
pub fn dequantize_q5_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    // Q5_K format: 32 elements per block, 40 bytes per block
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 40 + remaining.div_ceil(2);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q5_K".to_string(),
            format!(
                "data too small: got {} bytes, need {}",
                data.len(),
                expected_size
            ),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 40;

        // Parse scale (f16) - little-endian
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        // Parse d_min (f16) - little-endian
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();

        // Parse delta (f16) - little-endian
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        // Extract nibbles and dequantize (Q5_K uses 5-bit values)
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 6 + i / 2] >> (4 * (i & 1))) & 0x1F; // 5 bits
            let q = nibble as i32;
            result.push(delta * q as f32 + d_min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_blocks * 40;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();

        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..remaining {
            let nibble = (data[base + 6 + i / 2] >> (4 * (i & 1))) & 0x1F;
            let q = nibble as i32;
            result.push(delta * q as f32 + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q6_K data to f32 array.
pub fn dequantize_q6_k(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    // Q6_K format: 32 elements per block, 54 bytes per block
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 54 + remaining.div_ceil(4);

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q6_K".to_string(),
            format!(
                "data too small: got {} bytes, need {}",
                data.len(),
                expected_size
            ),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 54;

        // Parse scale (f16) - little-endian
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        // Parse d_min (f16) - little-endian
        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();

        // Parse delta (f16) - little-endian
        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        // Extract nibbles and dequantize (Q6_K uses 6-bit values)
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let byte_idx = 6 + i / 4;
            let bit_offset = 2 * (i & 3);
            let value = (data[byte_idx] >> bit_offset) & 0x3F; // 6 bits
            let q = value as i32 - 32; // Centered around 0
            result.push(delta * q as f32 + d_min);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_blocks * 54;
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        let d_min_f16 = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let d_min = half::f16::from_bits(d_min_f16).to_f32();

        let delta_f16 = u16::from_le_bytes([data[base + 4], data[base + 5]]);
        let delta = half::f16::from_bits(delta_f16).to_f32();

        for i in 0..remaining {
            let byte_idx = 6 + i / 4;
            let bit_offset = 2 * (i & 3);
            let value = (data[byte_idx] >> bit_offset) & 0x3F;
            let q = value as i32 - 32;
            result.push(delta * q as f32 + d_min);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 data to f32 array (GGML format).
pub fn dequantize_q8_0(data: &[u8], element_count: usize) -> crate::Result<Vec<f32>> {
    // Q8_0 format: 32 elements per block, 34 bytes per block
    let num_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_blocks * 34 + if remaining > 0 { 2 + remaining } else { 0 };

    if data.len() < expected_size {
        return Err(crate::error::RunnerError::Dequant(
            "Q8_0".to_string(),
            format!(
                "data too small: got {} bytes, need {}",
                data.len(),
                expected_size
            ),
        ));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 34;

        // Parse scale (f16) - little-endian
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = half::f16::from_bits(scale_f16).to_f32();

        // Extract signed int8 values and dequantize
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32;
            result.push(scale * q);
        }
    }

    // Handle remaining elements
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

/// Helper function to convert u16 little-endian bytes to f32.
fn half_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .map(|f| f.to_f32())
        .collect()
}

/// Helper function to convert u16 big-endian bytes to f32.
fn half_f32_be(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_be_bytes([b[0], b[1]])))
        .map(|f| f.to_f32())
        .collect()
}

/// Convert u16 little-endian to f32.
fn f16_to_f32_le(val: u16) -> f32 {
    half::f16::from_bits(val).to_f32()
}
