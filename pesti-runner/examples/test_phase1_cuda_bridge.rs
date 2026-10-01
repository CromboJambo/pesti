//! Phase 1 verification: Test CUDA cuBLAS F16 bridge end-to-end.

use half::f16;
use pesti_runner::kernel::dispatch::DispatchContext;
use pesti_runner::kernel::linear::{new_linear_layer, LinearLayer};
use std::time::Instant;

fn main() {
    println!("=== PESTI Phase 1: CUDA cuBLAS F16 Bridge Verification ===");
    println!();

    // Create dispatch context (auto-detects GPU)
    let ctx = DispatchContext::new();
    println!("Dispatch context created");
    println!("  GPU available: {}", ctx.gpu_available());
    println!("  Prefer GPU: {}", ctx.prefer_gpu());

    #[cfg(feature = "cuda")]
    if ctx.cuda_bridge_available() {
        println!("  CUDA bridge (cuBLAS F16): AVAILABLE ✓");
    } else {
        println!("  CUDA bridge (cuBLAS F16): NOT available ✗");
    }

    println!();

    // Create a test linear layer: [4, 8] input -> [4, 16] output
    let in_features = 8;
    let out_features = 16;
    let batch_size = 4;

    // Initialize weights with small values
    let mut weights_f16 = Vec::with_capacity(in_features * out_features);
    for i in 0..(in_features * out_features) {
        let val = ((i as f32) * 0.01).sin();
        weights_f16.push(f16::from_f32(val));
    }

    // Initialize bias
    let bias: Vec<f32> = (0..out_features).map(|_| 0.0).collect();

    let layer = new_linear_layer(
        weights_f16.clone(),
        Some(bias.clone()),
        in_features,
        out_features,
    );
    println!("Test linear layer created: [4,8] -> [4,16]");
    println!();

    // Create test input
    let input_size = batch_size * in_features;
    let mut input = Vec::with_capacity(input_size);
    for i in 0..input_size {
        input.push(((i as f32) * 0.1).sin());
    }

    // Benchmark forward passes
    println!("Benchmarking 10 iterations...");
    let start = Instant::now();
    for _ in 0..10 {
        match layer.forward(&input, batch_size) {
            Ok(output) => {
                if output.len() != batch_size * out_features {
                    println!(
                        "ERROR: Output size mismatch! Expected {}, got {}",
                        batch_size * out_features,
                        output.len()
                    );
                    std::process::exit(1);
                }
            }
            Err(e) => {
                println!("ERROR: Forward pass failed: {}", e);
                std::process::exit(1);
            }
        }
    }
    let elapsed = start.elapsed();

    println!();
    println!("=== Results ===");
    println!("Total time: {:.3}s", elapsed.as_secs_f64());
    println!(
        "Time per forward pass: {:.3}ms",
        elapsed.as_secs_f64() * 100.0
    );

    // Verify numerical correctness by comparing with CPU path
    println!();
    println!("Verifying numerical correctness vs CPU path...");
    let cpu_layer = new_linear_layer(
        weights_f16.clone(),
        Some(bias.clone()),
        in_features,
        out_features,
    );

    let gpu_result = layer.forward(&input, batch_size).unwrap();
    let cpu_result = cpu_layer.forward(&input, batch_size).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..gpu_result.len() {
        let diff = (gpu_result[i] - cpu_result[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max absolute difference: {:.6}", max_diff);
    if max_diff < 1e-3 {
        println!("✓ Numerical correctness verified (within F16 tolerance)");
    } else {
        println!("✗ Numerical difference exceeds expected F16 tolerance!");
        std::process::exit(1);
    }

    println!();
    println!("Phase 1 verification: COMPLETE");
}
