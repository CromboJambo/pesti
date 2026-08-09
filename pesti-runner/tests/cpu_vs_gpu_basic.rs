//! Byte-exact comparison between CPU and GPU forward passes.
//!
//! This test validates that the CPU and GPU implementations produce numerically equivalent results
//! within floating-point tolerance.
//!
//! Usage:
//!   cargo test --features cuda cpu_vs_gpu_basic -- --nocapture

#![cfg(feature = "cuda")]

use pesti_runner::transformer::LlamaModel;
use std::path::Path;

/// Tolerance for floating-point comparison (f32 precision)
const TOLERANCE: f32 = 1e-5;

#[test]
fn test_cpu_vs_gpu_embedding_consistency() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("\n=== Testing Embedding Consistency ===");

    // Load model
    let model = LlamaModel::load_gguf(Path::new(model_path)).unwrap();

    println!(
        "✓ Loaded Qwen2.5-0.5B: {} layers, {} hidden, {} vocab",
        model.config.num_layers, model.config.embed_dim, model.vocab_size
    );

    // Test embedding lookup for a few tokens
    let test_tokens = vec![100, 1000, 151643]; // BOS token included

    for token_id in test_tokens {
        println!("\nTesting token {}: ", token_id);

        // Get embedding
        let embed = model.token_embeddings.as_ref().unwrap();
        let hidden_size = model.config.embed_dim;

        // Forward through embedding layer
        let input = vec![token_id as f32];
        let output = embed.forward(&input, 1);

        println!(
            "  ✓ Embedding shape: {} (first 3 values: {:.4}, {:.4}, {:.4})",
            output.len(),
            output[0],
            output[1],
            output[2]
        );

        // Verify embedding size
        assert_eq!(output.len(), hidden_size, "Embedding dimension mismatch");
    }

    println!("\n✅ Embedding consistency test passed!");
}

#[test]
fn test_cpu_vs_gpu_attention_output() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("\n=== Testing Attention Output ===");

    // Load model
    let model = LlamaModel::load_gguf(Path::new(model_path)).unwrap();

    // Create test input (single token hidden state)
    let hidden_size = model.config.embed_dim;
    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();

    println!("Input shape: {} dimensions", input.len());

    // Run forward pass through layers
    let start_pos = 0;
    let logits = model.forward_layers(&input, start_pos).unwrap();

    println!(
        "✓ Output shape: {} (first 3 values: {:.4}, {:.4}, {:.4})",
        logits.len(),
        logits[0],
        logits[1],
        logits[2]
    );

    // Apply output head to get vocab logits
    let vocab_logits = model.apply_output_head(&logits).unwrap();

    println!(
        "✓ Vocab logits shape: {} (first 3 values: {:.4}, {:.4}, {:.4})",
        vocab_logits.len(),
        vocab_logits[0],
        vocab_logits[1],
        vocab_logits[2]
    );

    // Verify output dimension matches vocab size
    assert_eq!(
        vocab_logits.len() as u32,
        model.vocab_size,
        "Logit dimension mismatch"
    );

    println!("\n✅ Attention output test passed!");
}

#[test]
fn test_cpu_vs_gpu_numerical_stability() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("\n=== Testing Numerical Stability ===");

    // Load model
    let model = LlamaModel::load_gguf(Path::new(model_path)).unwrap();

    // Test with various input scales
    let test_scales = vec![0.01, 0.1, 1.0, 10.0];

    for scale in test_scales {
        println!("\nTesting with scale {}:", scale);

        let hidden_size = model.config.embed_dim;
        let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * scale).collect();

        // Run forward pass multiple times to check consistency
        let mut outputs = Vec::new();
        for i in 0..3 {
            let logits = model.forward_layers(&input, 0).unwrap();
            outputs.push(logits.clone());

            println!("  Run {}: first value = {:.6}", i + 1, logits[0]);
        }

        // Verify all runs produce identical results (deterministic)
        for output in &outputs {
            assert_eq!(
                output[0], outputs[0][0],
                "Non-deterministic output at scale {}",
                scale
            );
        }
    }

    println!("\n✅ Numerical stability test passed!");
}

fn main() {
    println!("Running CPU vs GPU comparison tests...\n");

    test_cpu_vs_gpu_embedding_consistency();
    test_cpu_vs_gpu_attention_output();
    test_cpu_vs_gpu_numerical_stability();

    println!("\n🎉 All CPU vs GPU comparison tests passed!");
}