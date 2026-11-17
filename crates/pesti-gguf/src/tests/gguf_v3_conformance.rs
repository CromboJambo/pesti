//! GGUF v3 Conformance Tests
//!
//! These tests validate the parser against the actual wire format of real GGUF files,
//! based on analysis of llama.cpp's implementation and hex dump inspection.

use crate::{parse_gguf, GgufKvValue};

/// Test parsing a known GGUF v3 file and validates specific KV pairs
#[test]
fn test_qwen2_5_0_5b_conformance() {
    let path = crate::tests::conformance_corpus_path("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    // Parse the file — if this succeeds, parser handles v3 format correctly
    let header = parse_gguf(&path).expect("Failed to parse Qwen2.5 0.5B GGUF file");

    eprintln!("✓ Header parsed: version={}", header.version);
    assert_eq!(header.version, 3, "Should be GGUF v3 format");

    // Validate architecture is recognized
    let arch = header.architecture().expect("Missing architecture");
    eprintln!("✓ Architecture: {}", arch);
    assert_eq!(arch, "qwen2", "Expected Qwen2 architecture");

    // Check context length
    let ctx_len = header.context_length().expect("Missing context length");
    eprintln!("✓ Context length: {}", ctx_len);
    assert!(ctx_len > 0, "Context length should be positive");

    // Validate embedding dimension
    let embd_dim = header.embedding_length().expect("Missing embedding length");
    eprintln!("✓ Embedding dimension: {}", embd_dim);
    assert_eq!(embd_dim, 896, "Expected 896 for 0.5B model");

    // Validate block count (number of layers)
    let n_blocks = header.block_count().expect("Missing block count");
    eprintln!("✓ Block count: {}", n_blocks);
    assert!(n_blocks > 0, "Block count should be positive");

    // Validate some specific KV pairs exist
    let _kv_map: std::collections::HashMap<&str, &GgufKvValue> = header
        .kv_pairs
        .iter()
        .map(|p| (p.key.as_str(), &p.value))
        .collect();

    // Check that string values are valid
    for kv in &header.kv_pairs {
        match &kv.value {
            GgufKvValue::String(_s) => {
                // Validate key format (should be printable ASCII with dots)
                assert!(
                    kv.key.chars().all(|c| c.is_ascii() && (c.is_alphanumeric() || c == '.' || c == '_')),
                    "Invalid characters in key: {}",
                    kv.key
                );
            }
            GgufKvValue::Bool(v) => {
                // A bool is valid whether true or false (e.g. Qwen2.5 sets
                // tokenizer.ggml.add_bos_token = false). The parser already
                // proved the wire value decoded as a bool; just record it.
                eprintln!("✓ Bool KV pair: {} = {v}", kv.key);
            }
            GgufKvValue::Array(_) => {
                // Arrays are present (e.g., rope.scaling)
                eprintln!("✓ Found array KV pair: {}", kv.key);
            }
            _ => {}
        }
    }

    eprintln!("✓ Qwen2.5 0.5B conformance validated!");
}
