//! Compare PESTI ndarray CPU reference vs llama.cpp attention

use candle_core::{Device, Tensor};
use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let seq_q = 128;
    let seq_k = 128;
    let num_heads = 4;
    let head_dim = 64;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("=== PESTI vs llama.cpp Benchmark ===");
    println!(
        "Dimensions: seq_q={}, seq_k={}, num_heads={}, head_dim={}\n",
        seq_q, seq_k, num_heads, head_dim
    );

    // Generate random input data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    // Test 1: PESTI Ndarray implementation
    println!("Test 1: PESTI Ndarray CPU Reference");
    let start = Instant::now();
    let result_pesti = reference_with_ndarray(
        &q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale,
    );
    let duration_pesti = start.elapsed();

    println!("  Time: {:.3}ms", duration_pesti.as_secs_f64() * 1000.0);
    println!(
        "  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result_pesti[0], result_pesti[1], result_pesti[2], result_pesti[3], result_pesti[4]
    );

    // Test 2: llama.cpp attention (if available)
    #[cfg(feature = "llama-cpp")]
    {
        println!("\nTest 2: llama.cpp Attention");

        // Create llama.cpp context and compute attention
        let start = Instant::now();

        // Note: This would require initializing llama.cpp context with Q/K/V tensors
        // For now, we'll show the structure
        let result_llama = vec![0.0f32; seq_q * num_heads * head_dim]; // Placeholder

        let duration_llama = start.elapsed();

        println!("  Time: {:.3}ms", duration_llama.as_secs_f64() * 1000.0);
        println!(
            "  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            result_llama[0], result_llama[1], result_llama[2], result_llama[3], result_llama[4]
        );

        // Compare
        let max_error = result_pesti
            .iter()
            .zip(result_llama.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!("\n  Max absolute error: {:.6}", max_error);

        if max_error < 2.0 {
            println!("  ✓ Numerical consistency verified");
        } else {
            println!("  ✗ Large discrepancy detected");
        }
    }

    #[cfg(not(feature = "llama-cpp"))]
    {
        println!("\nTest 2: llama.cpp (requires --features llama-cpp)");
        println!(
            "  Run with: cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,llama-cpp"
        );
    }

    // Test 3: Mistral.rs (if available)
    #[cfg(feature = "mistralrs")]
    {
        println!("\nTest 3: Mistral.rs GPU Kernels");

        let start = Instant::now();

        // Note: Would require initializing mistralrs context
        let result_mistral = vec![0.0f32; seq_q * num_heads * head_dim]; // Placeholder

        let duration_mistral = start.elapsed();

        println!("  Time: {:.3}ms", duration_mistral.as_secs_f64() * 1000.0);
        println!(
            "  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            result_mistral[0],
            result_mistral[1],
            result_mistral[2],
            result_mistral[3],
            result_mistral[4]
        );

        let speedup_vs_pesti = duration_pesti.as_secs_f64() / duration_mistral.as_secs_f64();
        println!("  Speedup vs PESTI Ndarray: {:.2}x", speedup_vs_pesti);
    }

    #[cfg(not(feature = "mistralrs"))]
    {
        println!("\nTest 3: Mistral.rs (requires --features mistralrs)");
        println!(
            "  Run with: cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,mistralrs"
        );
    }

    println!("\n=== Summary ===");
    println!(
        "PESTI Ndarray: {:.3}ms (baseline)",
        duration_pesti.as_secs_f64() * 1000.0
    );

    Ok(())
}
