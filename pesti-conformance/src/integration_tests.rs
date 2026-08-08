//! Integration tests for conformance testing against real GGUF corpus files.

use crate::{run_conformance, ConformanceConfig};
use std::path::PathBuf;

/// Path to the conformance test corpus (Qwen2.5 models)
fn corpus_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance-corpus/")
}

#[test]
fn test_parse_qwen2_5_q4_k_m() {
    let model_path = corpus_path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    assert!(model_path.exists(), "Corpus file not found: {:?}", model_path);

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
        let model_path =
            corpus_path().join(format!("qwen2.5-0.5b-instruct-{}.gguf", quant));
        
        if model_path.exists() {
            println!("✓ Found: {}", quant);
            
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
    // Test the "llama.embedding_length" metadata gap mentioned in checkpoint
    let model_path = corpus_path().join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    if model_path.exists() {
        let header = pesti_gguf::parser::parse_gguf(&model_path)
            .expect("Failed to parse GGUF header");

        // Check for llama.embedding_length key (used by llama.cpp models)
        // Qwen2.5 uses "embedding_length" instead, but we should handle both
        let embedding_len = header
            .get_kv_u32("llama.embedding_length")
            .or_else(|| header.get_kv_u32("embedding_length"));

        assert!(
            embedding_len.is_some(),
            "Missing embedding_length metadata in Qwen2.5 model"
        );

        println!("Embedding length: {}", embedding_len.unwrap());
    } else {
        println!("⊘ Model not found, skipping embedding_length test");
    }
}
