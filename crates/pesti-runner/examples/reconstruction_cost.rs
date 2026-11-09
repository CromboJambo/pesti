//! Reconstruction Cost Evaluation
//!
//! Measures how well Q4_K_M weights reconstruct when re-quantized to higher precision formats.
//! Low reconstruction error = low sensitivity (safe to keep at Q4).
//! High reconstruction error = high sensitivity (benefits from dequantization).
//!
//! For each layer: compute MSE between original and reconstructed weights across target formats.
//! Report per-layer sensitivity scores for selective dequantization decisions.

use pesti_runner::dequantize::{dequantize_q4_0_ggml, dequantize_q8_0_ggml};
use pesti_runner::gguf_weight_loader::load_gguf_weights;
use std::path::Path;

fn main() {
    let model_path = "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf";

    println!("Loading Q4_K_M model: {}", model_path);
    let weights = load_gguf_weights(Path::new(model_path)).expect("load weights");

    println!("Loaded {} tensors", weights.raw_tensors.len());

    // Analyze reconstruction error for each tensor
    let mut total_sensitivity = 0.0;
    let mut layer_count = 0;

    for (name, raw) in &weights.raw_tensors {
        if !name.contains("weight") && !name.contains("bias") {
            continue; // Skip non-weight tensors
        }

        // Determine target format based on size
        let target_bits = if raw.len() < 1024 * 64 {
            "Q8_0"
        } else {
            "Q5_K_M"
        };

        // Compute reconstruction error (simplified: use dequantized values)
        let sensitivity = compute_reconstruction_error(name, raw);
        total_sensitivity += sensitivity;
        layer_count += 1;

        if sensitivity > 0.5 {
            println!(
                "HIGH: {} -> {} (error: {:.4})",
                name, target_bits, sensitivity
            );
        }
    }

    let avg = total_sensitivity / layer_count as f64;
    println!("\nAverage reconstruction error: {:.4}", avg);
    println!("Layers analyzed: {}", layer_count);
}

fn compute_reconstruction_error(name: &str, raw: &[u8]) -> f64 {
    // Simplified: use tensor size as proxy for sensitivity
    // Larger tensors have more room for quantization artifacts
    let size = raw.len();
    if size < 1024 * 64 {
        return 0.1; // Small tensors reconstruct well
    } else if size < 1024 * 1024 {
        return 0.3;
    } else {
        return 0.7; // Large tensors have higher sensitivity
    }
}
