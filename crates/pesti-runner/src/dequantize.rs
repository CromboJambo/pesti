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
            data.len(),
            expected_size
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
            data.len(),
            expected_size
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
            data.len(),
            expected_size
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
/// Q5_0 format: 32 elements per block, 22 bytes per block.
/// Layout (llama.cpp `block_q5_0`, `sizeof == ggml_half + uint32_t + QK5_0/2`):
///   - 2 bytes : f16 scale `d`
///   - 4 bytes : `qh` — 32-bit mask, bit i = element i's 5th bit (in element order)
///   - 16 bytes: `qs` — low 4 bits, INTERLEAVED: byte j holds element j (low nibble)
///                and element j+16 (high nibble). NOT sequential nibble packing.
/// Dequantized value = d * ((nibble | qh_bit) - 16), where for element i:
///   - i < 16  : nibble = qs[i] & 0x0F
///   - i >= 16 : nibble = qs[i-16] >> 4
///   - always  : qh_bit = i
/// (mirrors ggml `dequantize_row_q5_0`)
pub fn dequantize_q5_0(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    // Q5_0: 22 bytes per full block (d:2 + qh:4 + qs:16); a partial block still
    // carries d(2) + qh(4) + qs(ceil(n/2)).
    let expected_size = num_full_blocks * 22
        + if remaining > 0 {
            2 + 4 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_0 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 22;

        // Parse scale (f16)
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);

        // 16 high bits (bit j = element j, bit j+12 = element j+16), little-endian u32
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);

        // 32 elements in OUTPUT order. Interleaved nibble layout: element i<16
        // comes from the low nibble of qs[i]; element i>=16 from the high nibble
        // of qs[i-16]. The 5th bit of element i is ALWAYS qh bit i (qh is a
        // 32-bit mask, one bit per element in order) — mirrors ggml
        // dequantize_row_q5_0 (xh_0 = (qh>>j)<<4&0x10, xh_1 = (qh>>(j+12))&0x10).
        for i in 0..32usize {
            let (qs_byte, shift) = if i < 16 {
                (data[base + 6 + i], 0u8)
            } else {
                (data[base + 6 + (i - 16)], 4u8)
            };
            let low = (qs_byte >> shift) & 0x0F;
            let high = (((qh >> i) & 1) << 4) as u8;
            let q = ((low | high) as i32) - 16;
            result.push(scale * q as f32);
        }
    }

    // Handle remaining elements (partial block)
    if remaining > 0 {
        let base = num_full_blocks * 22;
        let scale_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let scale = f16_to_f32_le(scale_f16);
        let qh = u32::from_le_bytes([
            data[base + 2],
            data[base + 3],
            data[base + 4],
            data[base + 5],
        ]);

        // Interleaved layout: element i < 16 from low nibble of qs[i],
        // element i >= 16 from high nibble of qs[i-16]. 5th bit of element i
        // is always qh bit i.
        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let (qs_byte, shift) = if i < 16 {
                (data[base + 6 + i], 0u8)
            } else {
                (data[base + 6 + (i - 16)], 4u8)
            };
            let low = (qs_byte >> shift) & 0x0F;
            let high = (((qh >> i) & 1) << 4) as u8;
            let q = ((low | high) as i32) - 16;
            result.push(scale * q as f32);
        }
    }

    Ok(result)
}

/// Dequantize Q5_1 data.
///
/// Q5_1 format: 32 elements per block, 24 bytes per block.
/// Layout (llama.cpp `block_q5_1`, `sizeof == 2*ggml_half + uint32_t + QK5_1/2`):
///   - 2 bytes : f16 scale `d`
///   - 2 bytes : f16 min `m`
///   - 4 bytes : `qh` — 32-bit mask, bit i = element i's 5th bit (in element order)
///   - 16 bytes: `qs` — low 4 bits, INTERLEAVED: byte j holds element j (low nibble)
///                and element j+16 (high nibble). NOT sequential nibble packing.
/// Dequantized value = d * ((nibble | qh_bit) - 16) + m, where for element i:
///   - i < 16  : nibble = qs[i] & 0x0F
///   - i >= 16 : nibble = qs[i-16] >> 4
///   - always  : qh_bit = i
/// (mirrors ggml `dequantize_row_q5_1`)
pub fn dequantize_q5_1(data: &[u8], element_count: usize) -> Result<Vec<f32>> {
    let num_full_blocks = element_count / 32;
    let remaining = element_count % 32;
    // Q5_1: 24 bytes per full block (d:2 + m:2 + qh:4 + qs:16); a partial block
    // still carries d(2) + m(2) + qh(4) + qs(ceil(n/2)).
    let expected_size = num_full_blocks * 24
        + if remaining > 0 {
            2 + 2 + 4 + remaining.div_ceil(2)
        } else {
            0
        };

    if data.len() < expected_size {
        return Err(RunnerError::Internal(format!(
            "Q5_1 data too small: got {} bytes, need {}",
            data.len(),
            expected_size
        )));
    }

    let mut result = Vec::with_capacity(element_count);

    for block in 0..num_full_blocks {
        let base = block * 24;

        // Parse scale (f16) and min (f16)
        let d_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let d = f16_to_f32_le(d_f16);
        let m_f16 = data[base + 2] as u16 | (data[base + 3] as u16) << 8;
        let m = f16_to_f32_le(m_f16);

        // 16 high bits, one per element (little-endian u32)
        let qh = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);

        // 32 elements in OUTPUT order. Interleaved nibble layout (mirrors ggml
        // dequantize_row_q5_1, identical to Q5_0 but with an f16 min added):
        // element i<16 = low nibble of qs[i]; element i>=16 = high nibble of
        // qs[i-16]. The 5th bit of element i is ALWAYS qh bit i (qh is a
        // 32-bit mask, one bit per element in order).
        for i in 0..32usize {
            let (qs_byte, shift) = if i < 16 {
                (data[base + 8 + i], 0u8)
            } else {
                (data[base + 8 + (i - 16)], 4u8)
            };
            let low = (qs_byte >> shift) & 0x0F;
            let high = (((qh >> i) & 1) << 4) as u8;
            let q = ((low | high) as i32) - 16;
            result.push(d * q as f32 + m);
        }
    }

    // Handle remaining elements (partial block)
    if remaining > 0 {
        let base = num_full_blocks * 24;
        let d_f16 = data[base] as u16 | (data[base + 1] as u16) << 8;
        let d = f16_to_f32_le(d_f16);
        let m_f16 = data[base + 2] as u16 | (data[base + 3] as u16) << 8;
        let m = f16_to_f32_le(m_f16);
        let qh = u32::from_le_bytes([
            data[base + 4],
            data[base + 5],
            data[base + 6],
            data[base + 7],
        ]);

        let elems_in_block = remaining.min(32);
        for i in 0..elems_in_block {
            let (qs_byte, shift) = if i < 16 {
                (data[base + 8 + i], 0u8)
            } else {
                (data[base + 8 + (i - 16)], 4u8)
            };
            let low = (qs_byte >> shift) & 0x0F;
            let high = (((qh >> i) & 1) << 4) as u8;
            let q = ((low | high) as i32) - 16;
            result.push(d * q as f32 + m);
        }
    }

    Ok(result)
}
