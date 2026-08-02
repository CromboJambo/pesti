//! Integration tests for real GGUF model files.
//!
//! These tests validate the parser against actual model files from the conformance corpus.

use pesti_gguf::parser::parse_gguf;
use std::path::Path;

/// Helper to get the path to a conformance corpus file relative to the workspace root.
fn conformance_corpus_path(filename: &str) -> std::path::PathBuf {
    let manifest_dir = std::env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("conformance-corpus").join(filename))
        .expect("Failed to compute conformance corpus path")
}

/// Test parsing a real Qwen2.5 0.5B GGUF file.
///
/// This validates the parser against an actual model file, not synthetic test data.
#[test]
fn test_parse_real_qwen2_5_0_5b() {
    let path = conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    // Should parse without error
    let header = parse_gguf(&path).expect("Failed to parse real Qwen2.5 GGUF file");

    eprintln!("Header version: {}", header.version);
    assert_eq!(header.version, 3, "Should be GGUF v3 format");

    // Should have KV pairs
    assert!(
        header.kv_pairs.len() > 0,
        "Should have KV pairs, got {}",
        header.kv_pairs.len()
    );
    eprintln!("KV pair count: {}", header.kv_pairs.len());

    // Check architecture key exists and is qwen2
    let has_architecture = header
        .kv_pairs
        .iter()
        .any(|p| p.key == "general.architecture");
    assert!(has_architecture, "Should have general.architecture KV pair");

    let arch_value: Option<&str> = header
        .kv_pairs
        .iter()
        .find(|p| p.key == "general.architecture")
        .and_then(|p| p.value.as_str());

    if let Some(arch) = arch_value {
        assert_eq!(arch, "qwen2", "Architecture should be 'qwen2'");
        eprintln!("Architecture: {}", arch);
    } else {
        panic!("Could not extract architecture value");
    }

    // Should have tensors
    assert!(header.tensors.len() > 0, "Should have tensors, got {}", header.tensors.len());
    eprintln!("Tensor count: {}", header.tensors.len());

    // Validate tensor shapes are reasonable
    for tensor in &header.tensors {
        assert!(!tensor.name.is_empty(), "Empty tensor name");
        assert!(
            !tensor.shape.is_empty(),
            "Empty shape for tensor: {}",
            tensor.name
        );
    }

    eprintln!("SUCCESS: Real Qwen2.5 0.5B GGUF file parsed correctly!");
}

/// Test parsing a larger Qwen2.5 3B model.
#[test]
fn test_parse_real_qwen2_5_3b() {
    let path = conformance_corpus_path("qwen2.5-3b-instruct-q4_k_m.gguf");

    let header = parse_gguf(&path).expect("Failed to parse real Qwen2.5 3B GGUF file");

    eprintln!("Header version: {}", header.version);
    assert_eq!(header.version, 3);

    // Larger models should have more KV pairs
    assert!(
        header.kv_pairs.len() >= 30,
        "Expected at least 30 KV pairs for 3B model, got {}",
        header.kv_pairs.len()
    );

    // Should have many tensors
    assert!(
        header.tensors.len() >= 300,
        "Expected at least 300 tensors for 3B model, got {}",
        header.tensors.len()
    );

    eprintln!("SUCCESS: Real Qwen2.5 3B GGUF file parsed correctly!");
}
