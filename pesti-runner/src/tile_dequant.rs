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
#[inline]
pub fn dequantize_q6_k_block(data: &[u8]) -> [f32; 16] {
    let mut result = [0.0f32; 16];
    
    let d = f16::from_le_bytes([data[0], data[1]]).to_f32();
    let delta = f16::from_le_bytes([data[2], data[3]]).to_f32();
    
    // Q6_K format: 16 nibbles (8 bytes) + h_low/h_mid/h_high (4 bytes each)
    let qs_low = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let qs_high = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let h_low = f16::from_le_bytes([data[12], data[13]]).to_f32();
    let h_mid = f16::from_le_bytes([data[14], data[15]]).to_f32();
    let h_high = f16::from_le_bytes([data[16], data[17]]).to_f32();
    
    // First 8 elements use qs_low with h_low
    for i in 0..8 {
        let q = ((qs_low >> (i * 4)) & 0x0F) as u8;
        result[i] = delta * h_low * (q as f32 - 4.0) + d;
    }
    
    // Next 4 elements use qs_high with h_mid
    for i in 0..4 {
        let q = ((qs_high >> (i * 4)) & 0x0F) as u8;
        result[i + 8] = delta * h_mid * (q as f32 - 4.0) + d;
    }
    
    // Last 4 elements use qs_high with h_high
    for i in 0..4 {
        let q = ((qs_high >> ((i + 4) * 4)) & 0x0F) as u8;
        result[i + 12] = delta * h_high * (q as f32 - 4.0) + d;
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
            data.len(), expected_size
        )));
    }
    
    let mut result = Vec::with_capacity(tile_size);
    let elements_processed = start_idx;
    
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
pub fn dequantize_q4_k_tile(data: &[u8], start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_blocks = (tile_size + 15) / 16; // Round up to full blocks
    let expected_size = num_blocks * 28;
    
    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_K tile too small: got {} bytes, need {}",
            data.len(), expected_size
        )));
    }
    
    let mut result = Vec::with_capacity(tile_size);
    
    for block in 0..num_blocks {
        let base = block * 28;
        
        let d = f16::from_le_bytes([data[base], data[base + 1]]).to_f32();
        let delta = f16::from_le_bytes([data[base + 2], data[base + 3]]).to_f32();
        
        let qs_low = u32::from_le_bytes([
            data[base + 4], data[base + 5], data[base + 6], data[base + 7],
        ]);
        let qs_high = u32::from_le_bytes([
            data[base + 8], data[base + 9], data[base + 10], data[base + 11],
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
    
    result.truncate(tile_size);
    
    Ok(result)
}

/// Dequantize a tile of Q8_0 weights (up to TILE_SIZE elements)
pub fn dequantize_q8_0_tile(data: &[u8], start_idx: usize, tile_size: usize) -> Result<Vec<f32>> {
    let num_blocks = (tile_size + 31) / 32;
    let expected_size = num_blocks * 34;
    
    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_0 tile too small: got {} bytes, need {}",
            data.len(), expected_size
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
            Self::Q4_0 => 0.5,   // 32 elems / 18 bytes = 1.778 B/elem
            Self::Q4_1 => 0.625, // 32 elems / 20 bytes = 1.5625 B/elem
            Self::Q8_0 => 1.0625, // 32 elems / 34 bytes = 1.0625 B/elem
            Self::Q4_K => 1.75,  // 16 elems / 28 bytes = 1.75 B/elem
            Self::Q5_K => 2.25,  // 16 elems / 36 bytes = 2.25 B/elem
            Self::Q6_K => 2.625, // 16 elems / 42 bytes = 2.625 B/elem
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
    }
}
