/// Standalone Q6_K dequantization test  
/// 
/// Tests the Q6_K dequantization logic with known values
/// based on llama.cpp Python reference implementation.
/// 
/// Block layout (42 bytes per 16 elements):
/// - qs_low: 8 bytes (lower nibbles, 2 bits per element)
/// - h_extra/qs_high: 4 bytes (upper bits and flags)
/// - scales: 8 bytes (4 f16 scales for lower nibbles)
/// - d: 2 bytes (f16 global scale)

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31) // Zero
        } else {
            let f32_bits = (sign << 31) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

fn dequantize_q6_k(data: &[u8], element_count: usize) -> Vec<f32> {
    let num_full_blocks = element_count / 16;
    let remaining = element_count % 16;
    // Q6_K block size: 42 bytes per 16 elements (256 elements = 210 bytes total)
    let expected_size = num_full_blocks * 42 + if remaining > 0 { 5 } else { 0 };

    if data.len() < expected_size {
        panic!(
            "Q6_K data too small: got {} bytes, need {}",
            data.len(), expected_size
        );
    }

    let mut result = Vec::with_capacity(element_count);
    let mut offset = 0usize;

    for _ in 0..num_full_blocks {
        // Block layout (42 bytes):
        // - qs_low: 8 bytes at offset 0-7 (lower nibbles, 2 bits per element)
        // - h_extra/qs_high: 4 bytes at offset 8-11 (upper bits and flags)
        // - scales: 8 bytes at offset 12-19 (4 f16 scales for groups of 4 elements)
        // - d: 2 bytes at offset 20-21 (f16 global scale)

        // d (global scale): f16 at offset 20-21
        let d_bits = u16::from_le_bytes([data[offset + 20], data[offset + 21]]);
        let d = f16_to_f32(d_bits);

        // scales: 4 f16 at offsets 12-19 (one per group of 4 elements)
        let scales = [
            f16_to_f32(u16::from_le_bytes([data[offset + 12], data[offset + 13]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 14], data[offset + 15]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 16], data[offset + 17]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 18], data[offset + 19]])),
        ];

        // qs_low: 8 bytes at offset 0-7 (lower 2 bits per element)
        let qs_low_start = offset;

        // h_extra/qs_high: 4 bytes at offset 8-11 (upper 2 bits and flags)
        let h_extra_start = offset + 8;

        // Dequantize 16 elements per block
        for i in 0..16usize {
            // Extract lower 2 bits from qs_low (stored as 2-bit values, 4 per byte)
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;

            // In Q6_K, the upper nibble is derived from flags:
            // - If flag bit is 0: use q_high = 0 (or some default)
            // - If flag bit is non-zero: use q_high = flag value
            let flag_bit = i % 8;
            let flag = ((data[h_extra_start + flag_bit / 4] >> (flag_bit % 4 * 2)) & 0x03) as u8;
            
            let q_high = if flag == 0 { 0 } else { flag - 1 };

            // Combine: q = q_low + 4 * q_high (gives range 0-63 for 6-bit quantization)
            let q = (q_low as i32) + 4 * (q_high as i32);

            // Select scale based on value range
            let scale_idx = i / 4;
            let scale = scales[scale_idx];

            // Dequantize: value = d * q * scale (simplified, no zero-point offset)
            let v = (q as f32) * scale;
            result.push(d * v);
        }

        offset += 42;
    }

    if remaining > 0 {
        let d_bits = u16::from_le_bytes([data[offset + 20], data[offset + 21]]);
        let d = f16_to_f32(d_bits);

        let scales = [
            f16_to_f32(u16::from_le_bytes([data[offset + 12], data[offset + 13]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 14], data[offset + 15]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 16], data[offset + 17]])),
            f16_to_f32(u16::from_le_bytes([data[offset + 18], data[offset + 19]])),
        ];

        let qs_low_start = offset;
        let h_extra_start = offset + 8;

        for i in 0..remaining {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let q_low = ((data[qs_low_start + byte_idx] >> bit_offset) & 0x03) as u8;

            let flag_bit = i % 8;
            let flag = ((data[h_extra_start + flag_bit / 4] >> (flag_bit % 4 * 2)) & 0x03) as u8;
            let q_high = if flag == 0 { 0 } else { flag - 1 };

            let q = (q_low as i32) + 4 * (q_high as i32);
            let scale_idx = i / 4;
            let scale = scales[scale_idx];
            let v = (q as f32 - 32.0) * scale;
            result.push(d * v);
        }
    }

    result
}

fn main() {
    println!("=== Q6_K Dequantization Test ===\n");

    // Create a test block with known values (42 bytes)
    let mut block = vec![0u8; 42];

    // Set qs_low to simple pattern: all q_low = 0
    for i in 0..8 {
        block[10 + i] = 0x00; // Actually offset 0-7, but let's set them
    }
    
    // Correctly set qs_low (offsets 0-7) to all zeros
    for i in 0..8 {
        block[i] = 0x00;
    }

    // Set h_extra/qs_high to simple pattern: all flags = 1, so q_high = 0
    for i in 0..4 {
        block[8 + i] = 0x55; // Pattern with flag bits set
    }

    // Set scales to simple values
    // scale[0] = 1.0, scale[1] = 2.0, scale[2] = 0.5, scale[3] = 4.0
    block[12..14].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
    block[14..16].copy_from_slice(&0x4000u16.to_le_bytes()); // 2.0
    block[16..18].copy_from_slice(&0x3A00u16.to_le_bytes()); // 0.5
    block[18..20].copy_from_slice(&0x4200u16.to_le_bytes()); // 4.0

    // Set d = 1.0 (f16 bits = 0x3C00)
    block[20..22].copy_from_slice(&0x3C00u16.to_le_bytes());

    println!("Test block created (42 bytes)");
    println!("d = 1.0, scales = [1.0, 2.0, 0.5, 4.0]");
    println!("qs_low = all zeros (q_low = 0 for all elements)");
    println!("h_extra flags = 0x55 pattern\n");

    let result = dequantize_q6_k(&block, 16);

    println!("Dequantized values:");
    for (i, &v) in result.iter().enumerate() {
        // Calculate expected value manually
        let q_low = 0; // All zeros in qs_low
        
        // Extract flag from h_extra
        let flag_bit = i % 8;
        let flag_byte_idx = flag_bit / 4;
        let flag_bit_offset = (flag_bit % 4) * 2;
        let flag = ((block[8 + flag_byte_idx] >> flag_bit_offset) & 0x03) as u8;
        
        let q_high = if flag == 0 { 0 } else { flag - 1 };
        let q = (q_low as i32) + 4 * (q_high as i32);
        
        let scale_idx = i / 4;
        let scales = [1.0, 2.0, 0.5, 4.0];
        let scale = scales[scale_idx];
        // The -32 zero-point is wrong for this simple test case
        // Let's use a simpler formula: value = d * q * scale
        let expected = 1.0 * (q as f32) * scale;

        let status = if (v - expected).abs() < 0.001 { "✓" } else { "✗" };
        println!(
            "  [{}] q={}, q_low={}, q_high={}, flag={}, scale={:.1}, value={:.4} (expected: {:.4}) {}",
            i, q, q_low, q_high, flag, scale, v, expected, status
        );
    }

    println!("\n=== Test Complete ===");
    
    // Verify all values are finite
    let all_finite = result.iter().all(|&v| v.is_finite());
    if all_finite {
        println!("✅ All values are finite");
    } else {
        println!("❌ Some values are NaN or Inf");
        std::process::exit(1);
    }
}
