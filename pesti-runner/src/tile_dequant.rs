//! Tile-by-tile dequantization for memory-efficient inference.
//!
//! Instead of loading entire quantized weight matrices into f32, this module
//! provides functions to dequantize small tiles on-demand during GEMM operations.
//! This reduces peak memory usage by 4-8x for quantized models.

use crate::error::{Result, RunnerError};
use half::f16;

/// Tile size for dequantization (elements per tile)
const TILE_SIZE: usize = 256;

/// Dequantize a single Q4_0 block (32 elements) to f32
#[inline]
pub fn dequantize_q4_0_block(data: &[u8]) -> [f32; 32] {
    let mut result = [0.0f32; 32];

    // Parse scale (f16)
    let scale_f16 = u16::from_le_bytes([data[0], data[1]]);
    let scale = f16::from_bits(scale_f16).to_f32();

    // Extract nibbles and dequantize
    for i in 0..32 {
        let nibble = (data[2 + i / 2] >> (4 * (i & 1))) & 0x0F;
        let q = nibble as i32 - 8;
        result[i] = scale * q as f32;
    }

    result
}

/// Dequantize a single Q4_1 block (32 elements) to f32
#[inline]
pub fn dequantize_q4_1_block(data: &[u8]) -> [f32; 32] {
    let mut result = [0.0f32; 32];

    // Parse scale and min
    let scale_f16 = u16::from_le_bytes([data[0], data[1]]);
    let scale = f16::from_bits(scale_f16).to_f32();

    let min_f16 = u16::from_le_bytes([data[2], data[3]]);
    let min = f16::from_bits(min_f16).to_f32();

    // Extract nibbles: dequantized = scale * q + min
    for i in 0..32 {
        let nibble = (data[4 + i / 2] >> (4 * (i & 1))) & 0x0F;
        let q = nibble as f32;
        result[i] = scale * q + min;
    }

    result
}

/// Dequantize a single Q8_0 block (32 elements) to f32
#[inline]
pub fn dequantize_q8_0_block(data: &[u8]) -> [f32; 32] {
    let mut result = [0.0f32; 32];

    // Parse scale
    let scale_f16 = u16::from_le_bytes([data[0], data[1]]);
    let scale = f16::from_bits(scale_f16).to_f32();

    // Extract int8 values: dequantized = scale * quantized_value
    for i in 0..32 {
        let q = data[2 + i] as i8 as f32;
        result[i] = scale * q;
    }

    result
}

/// Dequantize Q4_K block (16 elements) to f32
#[inline]
pub fn dequantize_q4_k_block(data: &[u8]) -> [f32; 16] {
    let mut result = [0.0f32; 16];

    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
    let delta = f16::from_le_bytes([data[2], data[3]]).to_f32();

    // Q4_K format: 16 nibbles (8 bytes) + 2 scales (4 bytes)
    let qs_low = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let qs_high = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let h = [
        f16::from_le_bytes([data[12], data[13]]).to_f32(),
        f16::from_le_bytes([data[14], data[15]]).to_f32(),
    ];

    // First 8 elements use qs_low
    for i in 0..8 {
        let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
        result[i] = delta * h[0] * (q as f32 - 4.0) + d;
    }

    // Next 8 elements use qs_high
    for i in 0..8 {
        let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
        result[i + 8] = delta * h[1] * (q as f32 - 4.0) + d;
    }

    result
}

/// Dequantize Q5_K block (16 elements) to f32
#[inline]
pub fn dequantize_q5_k_block(data: &[u8]) -> [f32; 16] {
    let mut result = [0.0f32; 16];

    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
    let delta = f16::from_le_bytes([data[2], data[3]]).to_f32();

    // Q5_K format: 16 nibbles (8 bytes) + h_low/h_high (4 bytes each)
    let qs_low = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let qs_high = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let h_low = f16::from_le_bytes([data[12], data[13]]).to_f32();
    let h_high = f16::from_le_bytes([data[14], data[15]]).to_f32();

    // First 8 elements use qs_low with h_low
    for i in 0..8 {
        let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
        result[i] = delta * h_low * (q as f32 - 4.0) + d;
    }

    // Next 8 elements use qs_high with h_high
    for i in 0..8 {
        let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
        result[i + 8] = delta * h_high * (q as f32 - 4.0) + d;
    }

    result
}

/// Dequantize Q6_K block (16 elements) to f32
///
/// Q6_K block layout (42 bytes per 16 elements):
///   offset 0-1:   d (f16 global scale)
///   offset 2-9:   scales (4 × f16, one per group of 4 elements)
///   offset 10-17: qs_low (8 bytes, 2-bit packed lower bits for all 16 elements)
///   offset 18-21: qs_high_flags (4 bytes, 2-bit flags per element selecting scale)
///   offset 22-41: reserved/padding
#[inline]
pub fn dequantize_q6_k_block(data: &[u8]) -> [f32; 16] {
    let mut result = [0.0f32; 16];

    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();

    // 4 scales, one per group of 4 elements
    let scales = [
        f16::from_le_bytes([data[2], data[3]]).to_f32(),
        f16::from_le_bytes([data[4], data[5]]).to_f32(),
        f16::from_le_bytes([data[6], data[7]]).to_f32(),
        f16::from_le_bytes([data[8], data[9]]).to_f32(),
    ];

    // qs_low: 2-bit packed lower bits (8 bytes, 4 elements per byte)
    let qs_low_start = 10;
    // qs_high_flags: 2-bit flags per element selecting which scale to use
    let qs_high_flags_start = 18;

    for i in 0..16 {
        // Lower 2 bits from qs_low
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;

        // Flag bits selecting scale group
        let flag = ((data[qs_high_flags_start + byte_idx] >> bit_offset) & 0x03) as u8;
        let q_high = if flag == 0 { 0 } else { (flag - 1) as usize };

        // Full 6-bit quantized value
        let q = (q_low as i32) + 4 * (q_high as i32);

        // Scale for this group of 4 elements
        let scale = scales[i / 4];

        // Dequantize: d * (q - 32) * scale
        result[i] = d * (q as f32 - 32.0) * scale;
    }

    result
}

/// Dequantize a tile of Q4_0 weights (up to TILE_SIZE elements)
pub fn dequantize_q4_0_tile(data: &[u8], start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_blocks = (tile_size + 31) / 32; // Round up to full blocks
    let expected_size = num_blocks * 18;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_0 tile too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(tile_size);
    let _elements_processed = start_idx;

    for block in 0..num_blocks {
        let base = block * 18;

        // Parse scale (f16)
        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = f16::from_bits(scale_f16).to_f32();

        // Extract nibbles and dequantize
        for i in 0..32 {
            if result.len() >= tile_size {
                break;
            }

            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32);
        }
    }

    // Trim to exact tile size if needed
    result.truncate(tile_size);

    Ok(result)
}

/// Dequantize a tile of Q4_K weights (up to TILE_SIZE elements)
pub fn dequantize_q4_k_tile(data: &[u8], _start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_full_blocks = tile_size / 16;
    let remaining = tile_size % 16;
    // Q4_K: full blocks are 28 bytes, partial block is only 4 bytes (d + delta)
    let expected_size = num_full_blocks * 28 + if remaining > 0 { 4 } else { 0 };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_K tile too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(tile_size);

    for block in 0..num_full_blocks {
        let base = block * 28;

        let d = f16::from_le_bytes([data[base], data[base + 1]]).to_f32();
        let delta = f16::from_le_bytes([data[base + 2], data[base + 3]]).to_f32();

        let qs_low = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[base + 8],
            data[base + 9],
            data[base + 10],
            data[base + 11],
        ]);
        let h = [
            f16::from_le_bytes([data[base + 12], data[base + 13]]).to_f32(),
            f16::from_le_bytes([data[base + 14], data[base + 15]]).to_f32(),
        ];

        // First 8 elements use qs_low
        for i in 0..8 {
            if result.len() >= tile_size {
                break;
            }
            let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
            result.push(delta * h[0] * (q as f32 - 4.0) + d);
        }

        // Next 8 elements use qs_high
        for i in 0..8 {
            if result.len() >= tile_size {
                break;
            }
            let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
            result.push(delta * h[1] * (q as f32 - 4.0) + d);
        }
    }

    // Handle remaining elements (< 16) with only d and delta
    if remaining > 0 {
        let base = num_full_blocks * 28;

        let d = f16::from_le_bytes([data[base], data[base + 1]]).to_f32();
        let _delta = f16::from_le_bytes([data[base + 2], data[base + 3]]).to_f32();

        // For partial blocks in Q4_K: only d and delta are present, no qs/h values
        // The dequantization formula is: result[i] = d (since there's no quantized value)
        for _i in 0..remaining {
            if result.len() >= tile_size {
                break;
            }
            // Just use d as the value for partial block elements
            result.push(d);
        }
    }

    Ok(result)
}

/// Dequantize a tile of Q8_0 weights (up to TILE_SIZE elements)
pub fn dequantize_q8_0_tile(data: &[u8], _start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_blocks = (tile_size + 31) / 32;
    let expected_size = num_blocks * 34;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_0 tile too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(tile_size);

    for block in 0..num_blocks {
        let base = block * 34;

        let scale_f16 = u16::from_le_bytes([data[base], data[base + 1]]);
        let scale = f16::from_bits(scale_f16).to_f32();

        for i in 0..32 {
            if result.len() >= tile_size {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32;
            result.push(scale * q);
        }
    }

    result.truncate(tile_size);

    Ok(result)
}

/// Dequantize a tile of Q6_K weights (up to TILE_SIZE elements)
///
/// Q6_K block layout: 42 bytes per 16 elements.
/// See `dequantize_q6_k_block` for the per-block format.
pub fn dequantize_q6_k_tile(data: &[u8], _start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_blocks = (tile_size + 15) / 16;
    let expected_size = num_blocks * 42;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q6_K tile too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(tile_size);

    for block in 0..num_blocks {
        let base = block * 42;

        let d = f16::from_le_bytes([data[base], data[base + 1]]).to_f32();

        let scales = [
            f16::from_le_bytes([data[base + 2], data[base + 3]]).to_f32(),
            f16::from_le_bytes([data[base + 4], data[base + 5]]).to_f32(),
            f16::from_le_bytes([data[base + 6], data[base + 7]]).to_f32(),
            f16::from_le_bytes([data[base + 8], data[base + 9]]).to_f32(),
        ];

        let qs_low_start = base + 10;
        let qs_high_flags_start = base + 18;

        for i in 0..16 {
            if result.len() >= tile_size {
                break;
            }
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;
            let flag = ((data[qs_high_flags_start + byte_idx] >> bit_offset) & 0x03) as u8;
            let q_high = if flag == 0 { 0 } else { (flag - 1) as usize };
            let q = (q_low as i32) + 4 * (q_high as i32);
            let scale = scales[i / 4];
            result.push(d * (q as f32 - 32.0) * scale);
        }
    }

    result.truncate(tile_size);
    Ok(result)
}

/// Get the number of blocks needed for a given element count
pub fn num_blocks(dtype: &QuantDtype, element_count: usize) -> usize {
    match dtype {
        QuantDtype::Q4_0 => (element_count + 31) / 32,
        QuantDtype::Q4_1 => (element_count + 31) / 32,
        QuantDtype::Q8_0 => (element_count + 31) / 32,
        QuantDtype::Q4_K => (element_count + 15) / 16,
        QuantDtype::Q5_K => (element_count + 15) / 16,
        QuantDtype::Q6_K => (element_count + 15) / 16,
    }
}

/// Quantization dtype enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    Q4_0,
    Q4_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl QuantDtype {
    /// Get bytes per element for this quantization type
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            Self::Q4_0 => 0.5625, // 18 bytes / 32 elems = 0.5625 B/elem
            Self::Q4_1 => 0.625,  // 32 elems / 20 bytes = 1.5625 B/elem
            Self::Q8_0 => 1.0625, // 32 elems / 34 bytes = 1.0625 B/elem
            Self::Q4_K => 1.75,   // 16 elems / 28 bytes = 1.75 B/elem
            Self::Q5_K => 2.25,   // 16 elems / 36 bytes = 2.25 B/elem
            Self::Q6_K => 2.625,  // 16 elems / 42 bytes = 2.625 B/elem
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_block_dequantization() {
        // Simple sanity check: all zeros should dequantize to 0
        let data = vec![0u8; 18];
        let result = dequantize_q4_0_block(&data);

        // Scale will be 0, so all outputs should be 0
        for i in 0..32 {
            assert_eq!(result[i], 0.0, "Expected 0 at index {}", i);
        }
    }

    #[test]
    fn test_tile_dequantization_sizes() {
        // Test that tiles dequantize to correct sizes
        let q4_0_data = vec![0u8; 180]; // 10 blocks
        let result = dequantize_q4_0_tile(&q4_0_data, 0, 50).unwrap();
        assert_eq!(result.len(), 50);

        let q4_k_data = vec![0u8; 280]; // 10 blocks
        let result = dequantize_q4_k_tile(&q4_k_data, 0, 50).unwrap();
        assert_eq!(result.len(), 50);

        let q6_k_data = vec![0u8; 420]; // 10 blocks × 42 bytes
        let result = dequantize_q6_k_tile(&q6_k_data, 0, 50).unwrap();
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_q6_k_block_dequantization() {
        // Q6_K block: 42 bytes, 16 elements
        // All zeros → d=0, scales=0 → all outputs should be 0
        let data = vec![0u8; 42];
        let result = dequantize_q6_k_block(&data);
        for i in 0..16 {
            assert_eq!(result[i], 0.0, "Expected 0 at index {}", i);
        }

        // Test with known values: set d=1.0 (f16), first scale=1.0, q_low=0, flag=0
        // q = 0 + 4*0 = 0, result = 1.0 * (0 - 32) * 1.0 = -32.0
        let mut data = vec![0u8; 42];
        // d = 1.0 in f16 = 0x3C00
        data[0] = 0x00;
        data[1] = 0x3C;
        // scales[0] = 1.0 in f16 = 0x3C00
        data[2] = 0x00;
        data[3] = 0x3C;
        // qs_low and flags are all zeros
        let result = dequantize_q6_k_block(&data);
        // Element 0: q=0, scale=scales[0]=1.0, result = 1.0 * (0-32) * 1.0 = -32.0
        assert!((result[0] - (-32.0)).abs() < 0.01, "Got {}", result[0]);
    }

    #[test]
    fn test_q4_k_tile_partial_block() {
        // Test Q4_K tile with partial block (e.g., 20 elements = 1 full block + 4 partial)
        let mut data = vec![0u8; 32]; // 1 full block (28 bytes) + partial (4 bytes)

        // Set d=1.0 in first block (offsets 0-1)
        data[0] = 0x00;
        data[1] = 0x3C; // f16(1.0)

        // Set delta=0.5 in first block (offsets 2-3)
        data[2] = 0x00;
        data[3] = 0x3E; // f16(0.5)

        // Set d=2.0 in partial block (offsets 28-29, after the full block)
        data[28] = 0x00;
        data[29] = 0x40; // f16(2.0)

        // Full block dequantizes to 16 elements
        let result = dequantize_q4_k_tile(&data, 0, 20).unwrap();

        assert_eq!(
            result.len(),
            20,
            "Expected 20 elements, got {}",
            result.len()
        );

        // First 16 elements from full block (all zeros → q=4 → delta*0*(q-4)+d = d = 1.0)
        for i in 0..16 {
            assert!(
                (result[i] - 1.0).abs() < 0.01,
                "Element {} should be ~1.0, got {}",
                i,
                result[i]
            );
        }

        // Last 4 elements from partial block (just d = 2.0)
        for i in 16..20 {
            assert!(
                (result[i] - 2.0).abs() < 0.01,
                "Element {} should be ~2.0, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_q4_k_tile_300_elements() {
        // Test with exactly 300 elements (like the diagnostic example)
        let num_full_blocks = 300 / 16; // 18
        let remaining = 300 % 16; // 12
        let expected_size = num_full_blocks * 28 + if remaining > 0 { 4 } else { 0 }; // 508

        let mut data = vec![0u8; expected_size];

        // Set d=1.0 in each block
        for block in 0..num_full_blocks {
            let base = block * 28;
            data[base] = 0x00;
            data[base + 1] = 0x3C; // f16(1.0)
        }
        // Set d=1.0 in partial block
        let partial_base = num_full_blocks * 28;
        data[partial_base] = 0x00;
        data[partial_base + 1] = 0x3C; // f16(1.0)

        let result = dequantize_q4_k_tile(&data, 0, 300).unwrap();

        assert_eq!(
            result.len(),
            300,
            "Expected 300 elements, got {}",
            result.len()
        );

        // All elements should be ~1.0 (since q=0 in zeroed data)
        for i in 0..300 {
            assert!(
                (result[i] - 1.0).abs() < 0.01,
                "Element {} should be ~1.0, got {}",
                i,
                result[i]
            );
        }
    }
}
