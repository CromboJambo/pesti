//! Q4_K quantization for KV cache (inverse of ggml block_q4_K dequantization).
//!
//! Takes f32 values and quantizes them to the ggml canonical block_q4_K format:
//! 144 bytes per 256 elements. Each block stores a global scale factor, a global
//! min offset, per-group (32-element) scale/min adjustments, and 4-bit nibbles.
//!
//! Layout of a Q4_K block (144 bytes):
//! - Bytes 0-1:   d (global scale, f16, little-endian)
//! - Bytes 2-3:   dmin (global min offset, f16, little-endian)
//! - Bytes 4-15:  Per-group scales and mins (8 groups × 2 values, packed as 6-bit pairs)
//! - Bytes 16-143: Quantized data (256 nibbles, packed 2 per byte)

use half::f16;

/// Q4_K block constants (ggml canonical format).
pub const Q4K_BLOCK_SIZE: usize = 144; // bytes per block
pub const Q4K_BLOCK_ELEMS: usize = 256; // elements per block
const Q4K_GROUPS: usize = 8; // 8 groups of 32 values each

/// Pack a 6-bit scale and 6-bit min into the K-scale field.
/// Inverse of get_scale_min_k4 from gguf_weight_loader.rs.
fn pack_scale_min_k4(j: usize, scales: &mut [u8], sc: u8, m: u8) {
    if j < 4 {
        scales[j] = sc & 0x3F;
        scales[j + 4] = m & 0x3F;
    } else {
        let idx = j - 4;
        // For j >= 4: sc uses scales[8+idx] (lower nibble) and scales[idx] (high bits)
        //             m uses scales[8+idx] (upper nibble) and scales[4+idx] (high bits)
        scales[8 + idx] = (sc & 0x0F) | (((m >> 4) & 0x0F) << 4);
        scales[idx] |= ((sc >> 4) & 0x03) << 6;
        scales[4 + idx] |= ((m >> 4) & 0x03) << 6;
    }
}

/// Quantize a block of f32 values to Q4_K format.
///
/// Inverse of the ggml block_q4_K dequantization formula:
/// out = d * sc * q - dmin * m
///
/// Therefore, quantization is:
/// q = round((x + dmin * m) / (d * sc))
pub fn quantize_q4k_block(input: &[f32], output: &mut [u8]) {
    assert_eq!(input.len(), Q4K_BLOCK_ELEMS, "Q4_K block requires exactly 256 elements");
    assert!(output.len() >= Q4K_BLOCK_SIZE, "Output buffer too small for Q4_K block");

    // Step 1: Compute global scale (d) and min offset (dmin).
    // d is chosen so that d * 63 * 15 can represent the maximum absolute value.
    let max_abs = input.iter().fold(0.0f32, |max, &x| {
        if x.abs() > max { x.abs() } else { max }
    });

    // Use a safety factor of 8 to avoid saturation in edge cases
    let d = if max_abs > 0.0 {
        max_abs / (8.0f32 * 63.0f32)
    } else {
        1e-6f32
    };

    // In ggml, dmin is typically equal to d for symmetric ranges
    let dmin = d;

    // Step 2: For each group of 32 values, compute per-group scale and min factors.
    let mut scales = [0u8; 12]; // 8 groups × 2 values (sc, m), packed as 6-bit pairs

    for g in 0..Q4K_GROUPS {
        let start = g * 32;
        let group = &input[start..start + 32];

        // Find min and max of this group
        let (mut group_min, mut group_max) = (group[0], group[0]);
        for &x in group {
            if x < group_min { group_min = x; }
            if x > group_max { group_max = x; }
        }

        // Compute scale factor (sc): maps value range to 4-bit precision.
        // We want d * sc * 15 to cover both the range and absolute values.
        let range = group_max - group_min;
        let sc = if range > 0.0 {
            let raw_sc = range / (d * 15.0f32);
            raw_sc.round().clamp(1.0, 63.0) as u8
        } else {
            1u8 // Constant group: use minimum scale
        };

        // Compute min offset (m): maps group minimum to quantized 0.
        // From dequant formula: -dmin * m ≈ group_min → m = -group_min / dmin
        let m = if group_min < 0.0 {
            let raw_m = -group_min / (dmin * sc as f32);
            raw_m.round().clamp(0.0, 63.0) as u8
        } else {
            0u8 // All values positive: no offset needed
        };

        pack_scale_min_k4(g, &mut scales, sc, m);
    }

    // Step 3: Quantize each value to a 4-bit nibble using the inverse formula.
    let mut qs = [0u8; 128]; // 256 nibbles packed into 128 bytes

    for g in 0..Q4K_GROUPS {
        let start = g * 32;
        let group = &input[start..start + 32];

        // Re-find min/max and recompute scale/min factors (same as above)
        let (mut group_min, mut group_max) = (group[0], group[0]);
        for &x in group {
            if x < group_min { group_min = x; }
            if x > group_max { group_max = x; }
        }

        let range = group_max - group_min;
        let sc = if range > 0.0 {
            (range / (d * 15.0f32)).round().clamp(1.0, 63.0) as u8
        } else {
            1u8
        };

        let m = if group_min < 0.0 {
            (-group_min / (dmin * sc as f32)).round().clamp(0.0, 63.0) as u8
        } else {
            0u8
        };

        for (i, &x) in group.iter().enumerate() {
            // Inverse of dequant: q = round((x + dmin * m) / (d * sc))
            let denom = d * sc as f32;
            let q = if denom > 0.0 {
                ((x + dmin * m as f32) / denom).round().clamp(0.0, 15.0) as u8
            } else {
                0u8
            };

            // Pack two nibbles per byte (lower and upper)
            if i < 32 {
                qs[i] = q;
            } else {
                qs[i - 32] |= q << 4;
            }
        }
    }

    // Step 4: Write block header (d, dmin, scales) and quantized data.
    let d_f16 = f16::from_f32(d);
    output[0] = d_f16.to_le_bytes()[0];
    output[1] = d_f16.to_le_bytes()[1];

    let dmin_f16 = f16::from_f32(dmin);
    output[2] = dmin_f16.to_le_bytes()[0];
    output[3] = dmin_f16.to_le_bytes()[1];

    // Write per-group scales (already packed in ggml canonical layout)
    for i in 0..12 {
        output[4 + i] = scales[i];
    }

    // Write quantized data
    for i in 0..128 {
        output[16 + i] = qs[i];
    }
}

/// Quantize f32 values to Q4_K format (handles multiple blocks).
pub fn quantize_q4k(input: &[f32]) -> Vec<u8> {
    let num_blocks = input.len().div_ceil(Q4K_BLOCK_ELEMS);
    let mut output = vec![0u8; num_blocks * Q4K_BLOCK_SIZE];

    for (b, block_out) in output.chunks_mut(Q4K_BLOCK_SIZE).enumerate() {
        let start = b * Q4K_BLOCK_ELEMS;
        let end = (start + Q4K_BLOCK_ELEMS).min(input.len());
        let block_input = &input[start..end];

        // Pad with zeros if last block is incomplete
        let mut padded = vec![0.0f32; Q4K_BLOCK_ELEMS];
        padded[..block_input.len()].copy_from_slice(block_input);

        quantize_q4k_block(&padded, block_out);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_weight_loader::dequantize_q4_k;

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Create test values with a known range
        let mut input = Vec::with_capacity(Q4K_BLOCK_ELEMS);
        for i in 0..Q4K_BLOCK_ELEMS {
            // Values in range [-1.0, 1.0]
            let x = (i as f32 / Q4K_BLOCK_ELEMS as f32 - 0.5) * 2.0;
            input.push(x);
        }

        // Quantize
        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        quantize_q4k_block(&input, &mut block);

        // Debug: print scale factors
        let d_f16_bytes = [block[0], block[1]];
        let d = f16::from_le_bytes(d_f16_bytes).to_f32();
        let dmin_f16_bytes = [block[2], block[3]];
        let dmin = f16::from_le_bytes(dmin_f16_bytes).to_f32();
        eprintln!("d = {}, dmin = {}", d, dmin);

        // Dequantize using the existing pesti-runner dequantizer
        let dequantized = dequantize_q4_k(&block, Q4K_BLOCK_ELEMS).unwrap();

        // Verify round-trip error is reasonable (should be < 0.1 for this range)
        let mut max_error = 0.0f32;
        for i in 0..Q4K_BLOCK_ELEMS {
            let error = (input[i] - dequantized[i]).abs();
            if error > max_error { max_error = error; }
        }

        eprintln!("max_error = {}", max_error);
        assert!(max_error < 0.1, "Round-trip error too large: {}", max_error);
    }

    #[test]
    fn test_quantize_constant_values() {
        // All same value should quantize without error
        let input = vec![0.5f32; Q4K_BLOCK_ELEMS];
        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        quantize_q4k_block(&input, &mut block);

        // Should not panic and should produce valid block structure
        let d_f16_bytes = [block[0], block[1]];
        let d = f16::from_le_bytes(d_f16_bytes).to_f32();
        assert!(d > 0.0, "Scale factor should be positive for constant input");
    }

    #[test]
    fn test_quantize_zero_values() {
        // All zeros
        let input = vec![0.0f32; Q4K_BLOCK_ELEMS];
        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        quantize_q4k_block(&input, &mut block);

        // Should handle gracefully (no division by zero)
        let d_f16_bytes = [block[0], block[1]];
        let d = f16::from_le_bytes(d_f16_bytes).to_f32();
        assert!(d > 0.0, "Scale factor should be positive for zero input");
    }

    #[test]
    fn test_quantize_large_values() {
        // Large values to test scale computation
        let input: Vec<f32> = (0..Q4K_BLOCK_ELEMS)
            .map(|i| (i as f32 - 128.0) * 10.0)
            .collect();

        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        quantize_q4k_block(&input, &mut block);

        let d_f16_bytes = [block[0], block[1]];
        let d = f16::from_le_bytes(d_f16_bytes).to_f32();
        assert!(d > 0.0, "Scale factor should be positive for large input");
    }

    #[test]
    fn test_quantize_multiple_blocks() {
        // Test quantization of more than one block
        let input: Vec<f32> = (0..512)
            .map(|i| (i as f32 - 256.0) * 0.01)
            .collect();

        let output = quantize_q4k(&input);
        assert_eq!(output.len(), 2 * Q4K_BLOCK_SIZE, "Should produce 2 blocks");
    }

    #[test]
    fn test_quantize_realistic_kv_cache() {
        // Simulate realistic KV cache values (small, mostly positive)
        let mut input = Vec::with_capacity(Q4K_BLOCK_ELEMS);
        for i in 0..Q4K_BLOCK_ELEMS {
            // Typical attention weights: small positive values with some variance
            let x = 0.01 + (i as f32 % 17.0) * 0.001;
            input.push(x);
        }

        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        quantize_q4k_block(&input, &mut block);

        let dequantized = dequantize_q4_k(&block, Q4K_BLOCK_ELEMS).unwrap();

        // Verify round-trip error is reasonable
        let mut max_error = 0.0f32;
        for i in 0..Q4K_BLOCK_ELEMS {
            let error = (input[i] - dequantized[i]).abs();
            if error > max_error { max_error = error; }
        }

        assert!(max_error < 0.01, "Round-trip error too large for realistic values: {}", max_error);
    }
}