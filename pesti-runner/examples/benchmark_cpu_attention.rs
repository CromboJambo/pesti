//! Benchmark CPU vs optimized CPU attention implementations

use half::f16;
use pesti_runner::cpu_optimized::{reference_raw_scores, reference_raw_scores_optimized};
use std::time::Instant;

fn main() {
    println!("=== Fused Attention CPU Optimization Benchmark ===\n");
    
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Generate test data
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    // Warmup run
    let _ = reference_raw_scores(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    
    // Benchmark naive implementation
    println!("Benchmarking naive CPU implementation...");
    let start = Instant::now();
    let mut iterations = 0;
    while iterations < 100 {
        let _ = reference_raw_scores(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
        iterations += 1;
    }
    let naive_time = start.elapsed();
    println!("  Time: {:.2}ms ({} iterations)", naive_time.as_millis(), iterations);
    
    // Benchmark optimized implementation
    println!("\nBenchmarking optimized CPU implementation...");
    let start = Instant::now();
    iterations = 0;
    while iterations < 100 {
        let _ = reference_raw_scores_optimized(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
        iterations += 1;
    }
    let optimized_time = start.elapsed();
    println!("  Time: {:.2}ms ({} iterations)", optimized_time.as_millis(), iterations);
    
    // Calculate speedup
    let speedup = naive_time.as_secs_f64() / optimized_time.as_secs_f64();
    println!("\n⚡ Speedup: {}x", speedup);
    
    // Verify correctness
    let cpu_output = reference_raw_scores(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let optimized_output = reference_raw_scores_optimized(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    
    let max_diff: f32 = cpu_output
        .iter()
        .zip(optimized_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("\n✅ Numerical check: Max difference between implementations: {:.6e}", max_diff);
    
    if max_diff < 1e-4 {
        println!("✅ Outputs match within tolerance!");
    } else {
        println!("⚠️  Outputs differ significantly - check correctness");
    }
}
