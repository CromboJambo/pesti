//! Benchmark CPU reference implementations: gemm vs ndarray

use pesti_runner::cpu_optimized::reference_raw_scores_optimized;
use pesti_runner::cpu_optimized_ndarray::{reference_with_ndarray, reference_with_ndarray_manual};
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

fn main() {
    let seq_q = 128;
    let seq_k = 128;
    let num_heads = 4;
    let head_dim = 64;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Generate random input data using rand_distr
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    // Use simple approach: just generate values directly
    let q_h: Vec<half::f16> = (0..seq_q * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let k_h: Vec<half::f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let v_h: Vec<half::f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    println!("Benchmark: seq_q={}, seq_k={}, num_heads={}, head_dim={}", 
             seq_q, seq_k, num_heads, head_dim);
    println!();

    // Warm-up run
    let _ = reference_raw_scores_optimized(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    
    // Benchmark 1: Optimized CPU (gemm + rayon)
    println!("1. Optimized CPU (gemm + rayon):");
    let start = Instant::now();
    let result1 = reference_raw_scores_optimized(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration1 = start.elapsed();
    println!("   Time: {:.3}ms", duration1.as_secs_f64() * 1000.0);

    // Benchmark 2: Ndarray (structured arrays)
    println!("\n2. Ndarray-based implementation:");
    let start = Instant::now();
    let result2 = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration2 = start.elapsed();
    println!("   Time: {:.3}ms", duration2.as_secs_f64() * 1000.0);

    // Benchmark 3: Ndarray + manual dot products
    println!("\n3. Ndarray + manual dot products:");
    let start = Instant::now();
    let result3 = reference_with_ndarray_manual(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration3 = start.elapsed();
    println!("   Time: {:.3}ms", duration3.as_secs_f64() * 1000.0);

    // Calculate speedups
    println!("\n=== Speedup Analysis ===");
    let speedup_ndarray_vs_gemm = duration1.as_secs_f64() / duration2.as_secs_f64();
    let speedup_manual_vs_gemm = duration1.as_secs_f64() / duration3.as_secs_f64();

    println!(
        "Ndarray vs Gemm: {:.2}x {}",
        speedup_ndarray_vs_gemm,
        if speedup_ndarray_vs_gemm > 1.0 { "slower" } else { "faster" }
    );

    println!(
        "Manual dot vs Gemm: {:.2}x {}",
        speedup_manual_vs_gemm,
        if speedup_manual_vs_gemm > 1.0 { "slower" } else { "faster" }
    );

    // Verify numerical consistency
    let max_error = result1
        .iter()
        .zip(result2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!(
        "\nMax numerical difference (gemm vs ndarray): {:.6}",
        max_error
    );
}
