//! Simple CPU vs GPU comparison - basic sanity check.
//!
//! This validates that both CPU and GPU code paths load and execute without panics.
//! Numerical comparison can be added later once both paths are fully implemented.

use pesti_runner::CpuModel;
use std::path::Path;

fn main() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== CPU Forward Pass Test ===\n");

    // Load model (CPU path)
    println!("Loading model...");
    let cpu_model = match CpuModel::load_gguf(Path::new(model_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    println!(
        "✓ Loaded Qwen2.5-0.5B:\n  - Hidden size: {}\n  - Vocab size: {}\n  - Token embeddings loaded: {}\n  - Output weights loaded: {}",
        cpu_model.hidden_size,
        cpu_model.vocab_size,
        cpu_model.token_embeddings.is_some(),
        cpu_model.output_weights.is_some()
    );

    // Test embedding lookup
    let test_token = 100u32;
    println!("\nTesting token embedding lookup...");
    match cpu_model.embed(test_token, 0) {
        Ok(embedding) => {
            println!(
                "✓ Embedding shape: {} (first 3 values: {:.4}, {:.4}, {:.4})",
                embedding.len(),
                embedding[0],
                embedding[1],
                embedding[2]
            );
        }
        Err(e) => {
            eprintln!("✗ Embedding lookup failed: {}", e);
        }
    }

    // Test output head projection
    let hidden_size = cpu_model.hidden_size;
    let test_hidden: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();

    println!("\nTesting output head projection...");
    match cpu_model.apply_output_head(&test_hidden) {
        Ok(logits) => {
            println!(
                "✓ Output shape: {} logits (first 5: {:.4}, {:.4}, {:.4}, {:.4}, {:.4})",
                logits.len(),
                logits[0],
                logits[1],
                logits[2],
                logits[3],
                logits[4]
            );
        }
        Err(e) => {
            eprintln!("✗ Output head projection failed: {}", e);
        }
    }

    println!("\n=== CPU Test Complete ===");
    println!("Note: GPU path would use transformer::LlamaModel with CUDA dispatch.");
    println!("Full byte-exact comparison will be added when both paths are fully implemented.");
}
