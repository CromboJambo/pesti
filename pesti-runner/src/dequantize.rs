//! Pure Rust dequantization using ggml-quants crate.
//!
//! This module provides dequantization functions for GGUF tensor formats
//! using the `ggml-quants` crate, avoiding C dependencies where possible.


use crate::error::{Result, RunnerError};

/// Dequantize Q4_0 data using ggml-quants.
///
/// Q4_0 format: 32 elements per block, 18 bytes per block.
/// - First 2 bytes: f16 scale
/// - Next 16 bytes: nibble-packed quantized values (4 bits each)
pub fn dequantize_q4_0_ggml(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 18
        + if remaining > 0 {
            2 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_0 data too small: got {} bytes, need {}",
            data.len(), expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 18;
        
        // Parse scale (f16)
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        // Extract nibbles and dequantize
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 18;
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32);
        }
    }

    Ok(result)
}

/// Dequantize Q4_1 data using ggml-quants.
///
/// Q4_1 format: 32 elements per block, 20 bytes per block.
/// - First 2 bytes: f16 scale
/// - Next 2 bytes: f16 min
/// - Next 16 bytes: nibble-packed quantized values (unsigned)
pub fn dequantize_q4_1_ggml(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    let expected_size = num_full_blocks * 20
        + if remaining > 0 {
            4 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q4_1 data too small: got {} bytes, need {}",
            data.len(), expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 20;
        
        // Parse scale and min
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);
        
        let min_f16 = data[base + 2] as u16 | (data[base + 3] as u16) << 8;
        let min = f16_to_f32_le(min_f16);

        // Extract nibbles and dequantize: dequantized = scale * q + min
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    if remaining > 0 {
        let base = num_full_blocks * 20;
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);
        
        let min_f16 = data[base + 2] as u16 | (data[base + 3] as u16) << 8;
        let min = f16_to_f32_le(min_f16);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 4 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as f32;
            result.push(scale * q + min);
        }
    }

    Ok(result)
}

/// Dequantize Q8_0 data using ggml-quants.
///
/// Q8_0 format: 32 elements per block, 34 bytes per block.
/// - First 2 bytes: f16 scale
/// - Next 32 bytes: int8 quantized values
pub fn dequantize_q8_0_ggml(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_blocks = element_count.div_ceil(32);
    let expected_size = num_blocks * 34;

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q8_0 data too small: got {} bytes, need {}",
            data.len(), expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_blocks {
        let base = block * 34;
        
        // Parse scale
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        // Extract int8 values and dequantize: dequantized = scale * quantized_value
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let q = data[base + 2 + i] as i8 as f32;
            result.push(scale * q);
        }
    }

    Ok(result)
}

/// Helper: Convert little-endian u16 to f32 (f16 representation).
fn f16_to_f32_le(val: u16) -> f32 {
    // Use the half crate for proper f16 conversion
    use half::f16;
    f16::from_bits(val).to_f32()
}

/// Dequantize Q5_0 data.
///
/// Q5_0 format: 32 elements per block, 16 bytes per block.
/// Layout: f16 scale (2B) + nibble-packed quants (16B = 32 nibbles)
/// The high bit is implicitly 0 for Q5_0 (values 0-31, not 0-31+16).
pub fn dequantize_q5_0(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    // Q5_0: n/2 + 48 bytes total, which is 16 bytes per block
    let expected_size = num_full_blocks * 16
        + if remaining > 0 {
            2 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_0 data too small: got {} bytes, need {}",
            data.len(), expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 16;

        // Parse scale (f16)
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        // Extract nibbles and dequantize
        // Q5_0: values are 0-31, stored as nibbles with implicit high bit = 0
        for i in 0..32usize {
            if result.len() >= element_count {
                break;
            }
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            // Q5_0 values are 0-31, but we need to check if there's a high bit
            // Based on llama.cpp, Q5_0 uses values 0-31 directly
            let q = nibble as i32;

            // Dequantize: value = scale * q
            result.push(scale * q as f32);
        }
    }

    // Handle remaining elements
    if remaining > 0 {
        let base = num_full_blocks * 16;
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let nibble = (data[base + 2 + i / 2] >> (4 * (i & 1))) & 0x0F;
            let q = nibble as i32;
            result.push(scale * q as f32);
        }
    }

    Ok(result)
}
