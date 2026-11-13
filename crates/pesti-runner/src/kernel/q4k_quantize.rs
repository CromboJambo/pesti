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

/// Group parameters computed during quantization.
struct GroupParams {
    scale: u8,
    min_offset: u8,
}

/// Compute per-group scale and min offset for a group of values.
fn compute_group_params(group: &[f32], d: f32, dmin: f32) -> GroupParams {
    let (mut group_min, mut group_max) = (group[0], group[0]);
    for &x in group {
        if x < group_min {
            group_min = x;
        }
        if x > group_max {
            group_max = x;
        }
    }

    // Compute min offset first: m such that -dmin*m ≈ group_min (when group_min < 0)
    let min_offset = if group_min < 0.0 {
        (-group_min / dmin).round().clamp(0.0, 63.0) as u8
    } else {
        0u8
    };

    // Scale must ensure: d*sc*15 - dmin*m >= group_max (so q=15 covers the max value)
    // Therefore: sc >= (group_max + dmin*m) / (d*15)
    let effective_max = group_max + dmin * min_offset as f32;
    let scale = if effective_max > 0.0 {
        (effective_max / (d * 15.0f32)).ceil().clamp(1.0, 63.0) as u8
    } else {
        // All values are zero or negative and covered by offset
        1u8
    };

    GroupParams { scale, min_offset }
}

/// Quantize a block of f32 values to Q4_K format.
///
/// Inverse of the ggml block_q4_K dequantization formula:
/// out = d * sc * q - dmin * m
///
/// Therefore, quantization is:
/// q = round((x + dmin * m) / (d * sc))
pub fn quantize_q4k_block(input: &[f32], output: &mut [u8]) {
    assert_eq!(
        input.len(),
        Q4K_BLOCK_ELEMS,
        "Q4_K block requires exactly 256 elements"
    );
    assert!(
        output.len() >= Q4K_BLOCK_SIZE,
        "Output buffer too small for Q4_K block"
    );

    // Step 1: Compute global scale (d) and min offset (dmin).
    let max_abs = input
        .iter()
        .fold(0.0f32, |max, &x| if x.abs() > max { x.abs() } else { max });

    let global_min = input
        .iter()
        .fold(0.0f32, |min, &x| if x < min { x } else { min });

    // d must be large enough that sc = range/(15*d) ≤ 63 for any group range.
    // Using max_abs/63 ensures sc ≤ 63/15 = 4.2 for the worst case.
    let d = if max_abs > 0.0 {
        max_abs / 63.0f32
    } else {
        1e-6f32
    };

    // dmin must be large enough that m = -group_min/dmin ≤ 63 for the most negative group.
    let dmin = if global_min < 0.0 {
        -global_min / 63.0f32
    } else {
        d
    };

    // Step 2: Compute per-group parameters and pack scales.
    let mut scales = [0u8; 12];
    let mut group_params = Vec::with_capacity(Q4K_GROUPS);

    for g in 0..Q4K_GROUPS {
        let start = g * 32;
        let group = &input[start..start + 32];
        let params = compute_group_params(group, d, dmin);
        pack_scale_min_k4(g, &mut scales, params.scale, params.min_offset);
        group_params.push(params);
    }

    // Step 3: Quantize each value using pre-computed group parameters.
    // Nibble layout (matching dequantize_q4_k):
    //   chunk c covers elements [c*64, c*64+64) from qs[c*32 .. c*32+32]
    //     - lower nibbles: elements [c*64, c*64+32)  (even group g=2c)
    //     - upper nibbles: elements [c*64+32, c*64+64) (odd group g=2c+1)
    let mut qs = [0u8; 128];
    for g in 0..Q4K_GROUPS {
        let start = g * 32;
        let group = &input[start..start + 32];
        let params = &group_params[g];

        let denom = d * params.scale as f32;
        let qs_base = (g / 2) * 32;

        for (i, &x) in group.iter().enumerate() {
            let q = if denom > 0.0 {
                ((x + dmin * params.min_offset as f32) / denom)
                    .round()
                    .clamp(0.0, 15.0) as u8
            } else {
                0u8
            };

            if g % 2 == 0 {
                qs[qs_base + i] = q; // lower nibble
            } else {
                qs[qs_base + i] |= q << 4; // upper nibble
            }
        }
    }

    // Step 4: Write block header and quantized data.
    let d_f16 = f16::from_f32(d);
    output[0] = d_f16.to_le_bytes()[0];
    output[1] = d_f16.to_le_bytes()[1];

    let dmin_f16 = f16::from_f32(dmin);
    output[2] = dmin_f16.to_le_bytes()[0];
    output[3] = dmin_f16.to_le_bytes()[1];

    for i in 0..12 {
        output[4 + i] = scales[i];
    }

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

        // Dequantize using the existing pesti-runner dequantizer
        let dequantized = dequantize_q4_k(&block, Q4K_BLOCK_ELEMS).unwrap();

        // Verify round-trip error is reasonable (should be < 0.1 for this range)
        let mut max_error = 0.0f32;
        let mut worst_idx = 0;
        for i in 0..Q4K_BLOCK_ELEMS {
            let error = (input[i] - dequantized[i]).abs();
            if error > max_error {
                max_error = error;
                worst_idx = i;
            }
        }

        eprintln!(
            "max_error = {} at index {} (in={}, out={})",
            max_error, worst_idx, input[worst_idx], dequantized[worst_idx]
        );
        assert!(
            max_error < 0.1,
            "Q4_K roundtrip error too large: {} at index {}",
            max_error,
            worst_idx
        );
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
        assert!(
            d > 0.0,
            "Scale factor should be positive for constant input"
        );
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
        let input: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();

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
            if error > max_error {
                max_error = error;
            }
        }

        assert!(
            max_error < 0.01,
            "Round-trip error too large for realistic values: {}",
            max_error
        );
    }
}
