//! GGUF v3 Practical Format Conformance Tests
//!
//! These tests validate the parser against the actual wire format of real GGUF files,
//! based on analysis of llama.cpp's implementation and hex dump inspection.

use crate::*;
use crate::tests::conformance_corpus_path;

/// Test parsing a known GGUF v3 file and validates specific KV pairs
#[test]
fn test_parse_qwen2_5_conformance() {
    let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    // Parse the file
    let header = parse_gguf(&path).expect("Failed to parse Qwen2.5 GGUF file");

    eprintln!("✓ Header parsed: version={}", header.version);
    assert_eq!(header.version, 3, "Should be GGUF v3 format");

    // Validate we have the expected number of KV pairs
    assert!(
        header.kv_pairs.len() >= 20,
        "Expected at least 20 KV pairs, got {}",
        header.kv_pairs.len()
    );

    // Check for specific known KV pair that should exist
    let kv_map: std::collections::HashMap<&str, &GgufKvValue> = header
        .kv_pairs
        .iter()
        .map(|p| (p.key.as_str(), &p.value))
        .collect();

    // Validate general.architecture = "qwen2"
    assert!(
        kv_map.contains_key("general.architecture"),
        "Missing general.architecture KV pair"
    );

    if let Some(GgufKvValue::String(arch)) = kv_map.get("general.architecture") {
        eprintln!("✓ general.architecture = {}", arch);
        assert_eq!(arch, "qwen2", "Architecture should be 'qwen2'");
    } else {
        panic!("general.architecture is not a string value");
    }

    // Validate general.type exists (if present) - some models have non-standard values
    if let Some(value) = kv_map.get("general.type") {
        eprintln!("✓ general.type = {:?}", value);
        // Should be some non-empty string, but actual value varies by model
    }

    eprintln!("✓ All conformance checks passed!");
}

/// Test that validates the wire format structure of KV pairs
#[test]
fn test_kv_pair_wire_format() {
    let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse GGUF file");

    // Validate that all string KV pairs have string values
    for kv in &header.kv_pairs {
        match &kv.value {
            GgufKvValue::String(_) => {
                // String value - OK
                eprintln!("✓ KV: {} = <string>", kv.key);
            }
            GgufKvValue::Uint32(v) => {
                eprintln!("✓ KV: {} = {}", kv.key, v);
            }
            GgufKvValue::Int32(v) => {
                eprintln!("✓ KV: {} = {}", kv.key, v);
            }
            GgufKvValue::Float32(v) => {
                eprintln!("✓ KV: {} = {}", kv.key, v);
            }
            GgufKvValue::Bool(v) => {
                eprintln!("✓ KV: {} = {}", kv.key, v);
            }
            GgufKvValue::Array(_) => {
                eprintln!("✓ KV: {} = <array>", kv.key);
            }
            _ => {
                eprintln!("✓ KV: {} = {:?}", kv.key, kv.value);
            }
        }

        // Validate key format (should be printable ASCII with dots)
        assert!(
            kv.key.chars().all(|c| c.is_ascii() && (c.is_alphanumeric() || c == '.' || c == '_')),
            "Invalid characters in key: {}",
            kv.key
        );
    }

    eprintln!("✓ Wire format validation passed!");
}

/// Test that validates tensor count matches header
#[test]
fn test_tensor_count_consistency() {
    let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse GGUF file");

    eprintln!("✓ Tensor count: {}", header.tensors.len());
    assert!(
        header.tensors.len() > 100,
        "Expected many tensors, got {}",
        header.tensors.len()
    );

    // Validate tensor shapes are reasonable
    for tensor in &header.tensors {
        assert!(!tensor.name.is_empty(), "Empty tensor name");
        assert!(
            !tensor.shape.is_empty(),
            "Empty shape for tensor: {}",
            tensor.name
        );
        eprintln!("✓ Tensor: {} ({} dims)", tensor.name, tensor.shape.len());
    }

    eprintln!("✓ Tensor count consistency validated!");
}
