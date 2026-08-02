//! Comprehensive Conformance Tests for Larger Models
//!
//! These tests validate the parser against larger GGUF files with more complex structures,
//! ensuring correctness across a wider range of model architectures and tensor configurations.

use crate::*;
use crate::tests::conformance_corpus_path;
use std::collections::HashMap;

/// Test parsing a larger model (3B parameters) — verifies basic structure works
/// Requires conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf
#[test]
#[ignore = "Requires conformance corpus files"]
fn test_parse_qwen2_5_3b_conformance() {
    let path = conformance_corpus_path("qwen2.5-3b-instruct-q4_k_m.gguf");

    // Parse the file — if this succeeds, parser handles larger files correctly
    let header = parse_gguf(&path).expect("Failed to parse Qwen2.5 3B GGUF file");

    eprintln!("✓ Header parsed: version={}", header.version);
    assert_eq!(header.version, 3, "Should be GGUF v3 format");

    // Validate we have many KV pairs (larger models have more metadata)
    assert!(
        header.kv_pairs.len() >= 30,
        "Expected at least 30 KV pairs for 3B model, got {}",
        header.kv_pairs.len()
    );
    eprintln!("✓ Total KV pairs: {}", header.kv_pairs.len());

    // Check architecture key exists
    let kv_map: HashMap<&str, &GgufKvValue> = header
        .kv_pairs
        .iter()
        .map(|p| (p.key.as_str(), &p.value))
        .collect();
    
    assert!(
        kv_map.contains_key("general.architecture"),
        "Missing general.architecture"
    );

    eprintln!("✓ Large model parsing successful!");
}

/// Test that validates tensor count and structure for larger models
#[test]
#[ignore = "Requires conformance corpus files"]
fn test_large_model_tensor_structure() {
    let path = conformance_corpus_path("qwen2.5-3b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse GGUF file");

    // Larger models have many more tensors
    assert!(
        header.tensors.len() >= 300,
        "Expected at least 300 tensors for 3B model, got {}",
        header.tensors.len()
    );
    eprintln!("✓ Total tensors: {}", header.tensors.len());

    // Validate tensor names follow expected patterns (at least some exist)
    let has_embedding = header
        .tensors
        .iter()
        .any(|t| t.name.contains("token_embd") || t.name.contains("tok_embeddings"));
    let has_lm_head = header
        .tensors
        .iter()
        .any(|t| t.name.contains("output") || t.name.contains("lm_head"));
    let has_blocks = header.tensors.iter().any(|t| t.name.contains("blk."));
    
    assert!(has_embedding, "Missing token embedding tensor");
    assert!(has_lm_head, "Missing output/lm head tensor");
    assert!(has_blocks, "Missing transformer block tensors");
    
    eprintln!("✓ Found expected tensor groups: embedding={}, lm_head={}, blocks={}", 
              has_embedding, has_lm_head, has_blocks);

    // Validate no duplicate names
    let mut seen_names = std::collections::HashSet::new();
    for tensor in &header.tensors {
        assert!(seen_names.insert(&tensor.name), "Duplicate tensor name: {}", tensor.name);
    }
    eprintln!("✓ No duplicate tensor names");

    eprintln!("✓ Large model tensor structure validated!");
}

/// Test that validates data section alignment and offsets for larger models
#[test]
#[ignore = "Requires conformance corpus files"]
fn test_large_model_data_section() {
    let path = conformance_corpus_path("qwen2.5-3b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse GGUF file");

    // Validate data section start exists and is positive
    assert!(header.data_section_start > 0, "Data section start should be positive");
    eprintln!("✓ Data section starts at: {}", header.data_section_start);

    // Validate alignment if present
    if let Some(alignment) = header.data_alignment {
        eprintln!("✓ Data alignment: {}", alignment);
        assert!(alignment >= 32, "Alignment should be at least 32 for quantized models");
    }

    eprintln!("✓ Large model data section validated!");
}

/// Test that validates KV pair value types are consistent in larger files
#[test]
#[ignore = "Requires conformance corpus files"]
fn test_large_model_kv_type_consistency() {
    let path = conformance_corpus_path("qwen2.5-3b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse GGUF file");

    // Count different value types present
    let string_count: usize = header.kv_pairs.iter().filter(|p| matches!(p.value, GgufKvValue::String(_))).count();
    
    eprintln!("✓ Found {} string-valued KV pairs in large model", string_count);
    assert!(string_count >= 10, "Expected at least 10 string KV pairs");

    // Validate no NaN values (caught during parsing)
    for kv in &header.kv_pairs {
        if let GgufKvValue::Float32(f) = &kv.value {
            assert!(!f.is_nan(), "NaN value in KV pair: {}", kv.key);
        }
    }
    eprintln!("✓ No NaN values in large model");

    eprintln!("✓ Large model KV type consistency validated!");
}
