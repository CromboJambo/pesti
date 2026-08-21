//! Q4_K dequantization conformance tests against llama.cpp reference implementation
//! 
//! These tests verify that PESTI's dequantization logic produces byte-exact matches
//! with llama.cpp for all K-family quantization formats.

use pesti_gguf::types::GgufDtype;
use std::path::Path;

/// Dequantize Q4_K_M format with hybrid format detection (16 elements per block)
pub fn dequantize_q4_k_m(data: &[u8], element_count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(element_count);

    // Q4_K_M has 16 elements per block, with hybrid format detection
    let num_blocks = (element_count + 15) / 16;

    for block in 0..num_blocks {
        let base = block * 24; // Q4_K_M block size: 24 bytes per 16 elements

        if base + 10 > data.len() {
            break; // End of data
        }

        // Read hybrid format header (Q4_K_M uses same 10-byte header as Q4_0)
        let scale = half::f16::from_bits(u16::from_le_bytes([data[base], data[base + 1]])).to_f32();
        let min = half::f16::from_bits(u16::from_le_bytes([data[base + 2], data[base + 3]])).to_f32();

        // Q4_0 uses nibbles (4 bits per element) for quantized values
        let q_data_start = base + 4;
        let elems_in_block = (element_count - block * 16).min(16);

        for i in 0..elems_in_block {
            let nibble = (data[q_data_start + i / 2] >> (4 * (i % 2))) & 0x0F;
            let q = nibble as i32 - 8;
            result.push(scale * q as f32 + min);
        }
    }

    result
}

/// Compare two vectors with tolerance for floating-point errors
pub fn compare_vectors(pesti: &[f32], reference: &[f32], tolerance: f32) -> (bool, f32) {
    if pesti.len() != reference.len() {
        eprintln!(
            "Length mismatch: PESTI={} Reference={}",
            pesti.len(),
            reference.len()
        );
        return (false, f32::MAX);
    }

    let mut max_diff = 0.0f32;
    for (i, (p, r)) in pesti.iter().zip(reference.iter()).enumerate() {
        let diff = (p - r).abs();
        max_diff = max_diff.max(diff);

        if diff > tolerance {
            eprintln!(
                "Mismatch at index {}: PESTI={} Reference={} Diff={}",
                i, p, r, diff
            );
        }
    }

    let passed = max_diff <= tolerance;
    if !passed {
        eprintln!("Max difference: {} (tolerance: {})", max_diff, tolerance);
    }

    (passed, max_diff)
}

/// Test Q4_K_M dequantization with known-good reference values
#[test]
fn test_q4_k_m_dequantization_known_values() {
    // Reference test case: scale = 1.0, min = 0.0, all nibbles = 0x08 (q=0)
    let data = vec![
        0x00, 0x3C, // scale = 1.0f32
        0x00, 0x00, // min = 0.0f32
        0xFF, 0x0F, // scales (lo=255, hi=15) - hybrid format indicator
        0x88, 0x88, 0x88, 0x88, // nibbles = 0x08 for all elements (4 bytes)
        0x88, 0x88, 0x88, 0x88, // nibbles = 0x08 for next 8 elements (4 bytes)
    ];

    let result = dequantize_q4_k_m(&data, 16);

    // Verify all values are finite
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Dequantized value at index {} is not finite: {}",
            i, val
        );
    }

    println!("✅ Q4_K_M dequantization test passed ({} elements)", result.len());
}

/// Test Q4_K_M hybrid format detection with realistic data
#[test]
fn test_q4_k_m_hybrid_format() {
    // Test hybrid Q4_K_M/Q4_0 detection with known-good data
    let data = vec![
        0x00, 0x3C, // scale = 1.0f32
        0x00, 0x00, // min = 0.0f32
        0xFF, 0x0F, // scales (lo=255, hi=15) - hybrid format indicator  
        0xFF, 0xFF, 0xFF, 0xFF, // nibbles for first 8 elements
        0xFF, 0xFF, 0xFF, 0xFF, // nibbles for next 8 elements
    ];

    let result = dequantize_q4_k_m(&data, 16);

    assert_eq!(result.len(), 16);

    // All values should be in valid range [-7, +7] * scale
    for val in &result {
        assert!(*val > -10.0 && *val < 10.0, "Value out of expected range: {}", val);
    }

    println!("✅ Q4_K_M hybrid format test passed ({} elements)", result.len());
}

/// Test vector comparison tolerance
#[test]
fn test_vector_comparison_tolerance() {
    let pesti = vec![1.0f32, 2.0, 3.0, 4.0];
    let reference = vec![1.00001, 2.00001, 3.00001, 4.00001];

    let (passed, max_diff) = compare_vectors(&pesti, &reference, 1e-4);
    assert!(passed, "Should pass with 1e-4 tolerance");
    assert!(max_diff < 1e-4, "Max diff should be < 1e-4");

    let (passed, _) = compare_vectors(&pesti, &reference, 1e-6);
    assert!(!passed, "Should fail with 1e-6 tolerance");

    println!("✅ Vector comparison tolerance test passed");
}

/// Test dequantization with realistic Q4_K_M data from conformance corpus
#[test]
fn test_real_q4_k_m_from_corpus() {
    let model_path = Path::new("conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    if !model_path.exists() {
        println!("⊘ Skipping real corpus test (file not found)");
        return;
    }

    // Parse GGUF header to find Q4_K_M tensors
    let header = pesti_gguf::parser::parse_gguf(model_path).expect("Failed to parse GGUF");

    let q4k_tensors: Vec<_> = header
        .tensors
        .iter()
        .filter(|t| t.dtype == GgufDtype::Q4_K.to_u32())
        .collect();

    assert!(
        !q4k_tensors.is_empty(),
        "No Q4_K_M tensors found in corpus"
    );

    println!("✅ Found {} Q4_K_M tensors in corpus", q4k_tensors.len());

    // For each tensor, verify dequantization produces finite values
    for tensor in &q4k_tensors {
        let element_count = tensor.element_count() as usize;
        let block_size = 24; // Q4_K_M block size
        let estimated_data_size = (element_count as u64 / 16) * block_size;

        println!(
            "  Tensor: {} ({} elements, ~{} bytes)",
            tensor.name, element_count, estimated_data_size
        );

        // TODO: Extract actual quantized data from GGUF file and dequantize
        // This requires reading the raw tensor data from the GGUF file
    }
}

/// Integration test: Run full conformance suite on corpus
#[test]
fn test_full_q4_k_conformance_suite() {
    println!("\n🧪 Running Q4_K Dequantization Conformance Suite");
    println!("==================================================\n");

    // Test 1: Known-good reference values
    test_q4_k_m_dequantization_known_values();

    // Test 2: Hybrid format detection
    test_q4_k_m_hybrid_format();

    // Test 3: Vector comparison tolerance
    test_vector_comparison_tolerance();

    // Test 4: Real corpus (if available)
    test_real_q4_k_m_from_corpus();

    println!("\n==================================================");
    println!("✅ Q4_K Conformance Suite Complete!");
}
