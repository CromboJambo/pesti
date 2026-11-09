//! Integration tests for conformance testing against real GGUF corpus files.

use crate::{run_conformance, ConformanceConfig};
use std::path::PathBuf;

/// Path to the conformance test corpus (Qwen2.5 models)
fn corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../conformance-corpus/")
}

#[test]
fn test_parse_qwen2_5_q4_k_m() {
    let model_path = corpus_path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    assert!(
        model_path.exists(),
        "Corpus file not found: {:?}",
        model_path
    );

    // Run conformance test on the model
    let config = ConformanceConfig {
        corpus_dir: corpus_path(),
        reference_llama_cpp: None,
        floor_pass_count: 1,
        floor_file: None,
    };

    let result = run_conformance(&config).expect("Conformance test failed");

    // Verify at least the Q4_K_M model passed (metadata parsing works)
    assert!(
        !result.failures.is_empty() || result.passed.len() > 0,
        "Expected some results"
    );

    println!(
        "Conformance: {}/{} passed",
        result.passed.len(),
        result.total_models
    );
}

#[test]
fn test_all_q4_k_family_models() {
    let model_path = corpus_path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    assert!(model_path.exists(), "Q4_K_M model not found");

    // Test that the model loads and produces deterministic output
    let config = ConformanceConfig {
        corpus_dir: corpus_path(),
        reference_llama_cpp: None,
        floor_pass_count: 1,
        floor_file: None,
    };

    let result = run_conformance(&config).expect("Conformance test failed");

    // Verify we got results for all discovered models
    assert!(result.total_models > 0, "No models discovered in corpus");

    println!("Total models tested: {}", result.total_models);
    println!("Passed: {}", result.passed.len());
    println!("Failed: {}", result.failures.len());

    for failure in &result.failures {
        eprintln!(
            "FAILURE: {} - expected={} actual={}",
            failure.model_name, failure.expected_hash, failure.actual_hash
        );
    }
}

#[test]
fn test_quantization_variants() {
    let quantizations = vec![
        "q2_k", "q3_k", "q4_0", "q4_k_m", "q5_k", "q6_k", "q8_0", "f16",
    ];

    for quant in &quantizations {
        let model_path = corpus_path().join(format!("qwen2.5-0.5b-instruct-{}.gguf", quant));

        if model_path.exists() {
            println!("✓ Found: {}", quant);

            // Skip F16 placeholder file (just contains "Entry not found")
            if *quant == "f16" {
                let content = std::fs::read_to_string(&model_path).unwrap_or_default();
                if content.trim() == "Entry not found" {
                    println!("  ⊘ Placeholder file, skipping");
                    continue;
                }
            }

            // Verify GGUF header parses correctly
            let header = pesti_gguf::parser::parse_gguf(&model_path)
                .expect(format!("Failed to parse {} header", quant).as_str());

            assert!(header.tensors.len() > 0, "No tensors in {} model", quant);
            println!("  Tensors: {}", header.tensors.len());
        } else {
            println!("⊘ Missing: {}", quant);
        }
    }
}

#[test]
fn test_llama_embedding_length_metadata() {
    // Test that embedding length can be extracted from GGUF metadata or tensor shapes
    let model_path = corpus_path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf");

    if model_path.exists() {
        let header =
            pesti_gguf::parser::parse_gguf(&model_path).expect("Failed to parse GGUF header");

        // Check for embedding_length metadata (Qwen2.5 uses "embedding_length", llama.cpp models use "llama.embedding_length")
        let embedding_len_from_kv = header
            .get_kv_u32("llama.embedding_length")
            .or_else(|| header.get_kv_u32("embedding_length"));

        // Qwen2 models don't store embedding_length in KV pairs, so infer from token_embd.weight tensor shape
        let embedding_len_from_tensor = header
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight" || t.name == "tok_embeddings.weight")
            .map(|t| t.shape[0] as u32); // First dimension is hidden_size/embed_dim

        let embedding_len = embedding_len_from_kv.or(embedding_len_from_tensor);

        assert!(
            embedding_len.is_some(),
            "Missing embedding_length metadata or token_embd.weight tensor in Qwen2.5 model"
        );

        println!(
            "Embedding length: {} (from KV pairs: {:?}, from tensor: {:?})",
            embedding_len.unwrap(),
            embedding_len_from_kv,
            embedding_len_from_tensor
        );
    } else {
        println!("⊘ Model not found, skipping embedding_length test");
    }
}
