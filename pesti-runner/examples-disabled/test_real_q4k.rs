//! Test real Q4_K dequantization from GGUF model

use pesti_runner::gguf_weight_loader;
use std::path::Path;

fn main() {
    let gguf_path = Path::new(
        "/mnt/data/state/ai/lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-Uncensored-Aggressive-safetensors-i1-GGUF/Qwen3.5-35B-A3B-Uncensored-Aggressive-safetensors.i1-Q4_K_S.gguf",
    );

    println!("=== Loading GGUF Model ===");
    match gguf_weight_loader::load_gguf_weights(gguf_path) {
        Ok(weights) => {
            println!("✓ Header loaded: {} tensors", weights.raw_tensors.len());

            // Find a Q4_K weight tensor
            for (name, raw_data) in &weights.raw_tensors {
                if name.contains("attention.wq.weight")
                    || name.contains("attention.wk.weight")
                    || name.contains("attention.wv.weight")
                {
                    println!("\n=== Testing Tensor: {} ===", name);
                    let shape = weights.tensor_shape(name);
                    println!("Shape: {} x {}", shape.0, shape.1);
                    println!("Raw data size: {} bytes", raw_data.len());

                    // Calculate expected row_bytes for Q4_K
                    let in_features = shape.0;
                    let out_features = shape.1;
                    let row_bytes = ((in_features + 15) / 16) * 28;

                    println!(
                        "Expected row_bytes: {} (for {} elements)",
                        row_bytes, in_features
                    );
                    println!(
                        "Expected total bytes: {} ({} rows × {} bytes/row)",
                        out_features * row_bytes,
                        out_features,
                        row_bytes
                    );

                    // Test dequantization of first row
                    if raw_data.len() >= row_bytes {
                        let first_row = &raw_data[0..row_bytes];
                        println!("\nTesting dequantize_q4_k_tile on first row...");

                        match pesti_runner::tile_dequant::dequantize_q4_k_tile(
                            first_row,
                            0,
                            in_features,
                        ) {
                            Ok(dequantized) => {
                                println!("✓ Dequantization successful!");
                                println!("  Output length: {} elements", dequantized.len());

                                // Check for NaN/Inf
                                let has_nan = dequantized.iter().any(|&x| x.is_nan());
                                let has_inf = dequantized.iter().any(|&x| x.is_infinite());

                                if has_nan {
                                    println!("  ⚠ WARNING: Contains NaN values");
                                } else if has_inf {
                                    println!("  ⚠ WARNING: Contains Inf values");
                                } else {
                                    println!("  ✓ No NaN/Inf values");

                                    // Show first few values
                                    println!(
                                        "  First 5 values: {:?}",
                                        &dequantized[..5.min(dequantized.len())]
                                    );
                                }
                            }
                            Err(e) => {
                                println!("✗ Dequantization failed: {}", e);
                            }
                        }
                    } else {
                        println!(
                            "⚠ Not enough data for first row (got {}, need {})",
                            raw_data.len(),
                            row_bytes
                        );
                    }

                    // Break after first matching tensor
                    break;
                }
            }
        }
        Err(e) => eprintln!("✗ Error loading weights: {}", e),
    }

    println!("\n=== Done ===");
}
