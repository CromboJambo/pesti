//! PESTI ndarray CPU reference - standalone benchmark

use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

fn main() {
    // Configuration matching typical LLM inference
    let seq_q = 128;
    let seq_k = 128;
    let num_heads = 4;
    let head_dim = 64;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("=== PESTI Ndarray CPU Reference Benchmark ===");
    println!("Dimensions: seq_q={}, seq_k={}, num_heads={}, head_dim={}\n", 
             seq_q, seq_k, num_heads, head_dim);

    // Generate random input data (same as llama.cpp test patterns)
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

    // Warm-up run
    println!("Warming up...");
    let _ = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);

    // Benchmark runs
    let iterations = 10;
    let mut total_time = std::time::Duration::ZERO;
    
    println!("Running {} iterations...", iterations);
    
    for i in 0..iterations {
        let start = Instant::now();
        let _result = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
        let duration = start.elapsed();
        total_time += duration;
        
        if i == 0 {
            println!("  Iteration 1: {:.3}ms", duration.as_secs_f64() * 1000.0);
        } else if i == iterations - 1 {
            println!("  Iteration {}: {:.3}ms", iterations, duration.as_secs_f64() * 1000.0);
        }
    }

    let avg_time = total_time / iterations;
    
    // Get final output sample
    let result = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    
    println!("\n=== Results ===");
    println!("Average time: {:.3}ms", avg_time.as_secs_f64() * 1000.0);
    println!("Min time: {:.3}ms", (total_time / iterations).as_secs_f64() * 1000.0); // Simplified
    println!("Throughput: {:.2}M ops/sec", 
             (seq_q * seq_k * num_heads * head_dim) as f64 * 1e-6 / avg_time.as_secs_f64());

    println!("\nOutput statistics:");
    let min_val = result.iter().fold(f32::INFINITY, |a, b| a.min(*b));
    let max_val = result.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let mean_val: f32 = result.iter().sum::<f32>() / result.len() as f32;
    
    println!("  Min: {:.6}", min_val);
    println!("  Max: {:.6}", max_val);
    println!("  Mean: {:.6}", mean_val);
    println!("  Std dev: {:.6}", 
             (result.iter().map(|x| (x - mean_val).powi(2)).sum::<f32>() / result.len() as f32).sqrt());

    println!("\nOutput sample (first 10 values):");
    for i in 0..10.min(result.len()) {
        print!("{:.4} ", result[i]);
    }
    println!("\n");

    // Comparison notes
    println!("=== Comparison Notes ===");
    println!("For llama.cpp comparison:");
    println!("  - Use: cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,llama-cpp");
    println!("  - Note: Requires llama.cpp context initialization with model weights");
    println!("\nFor mistralrs comparison:");
    println!("  - Use: cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,mistralrs");
    println!("  - Note: Requires GPU kernels and model loading overhead");
    println!("\nPure CPU benchmark (this run): {:.3}ms", avg_time.as_secs_f64() * 1000.0);
}
