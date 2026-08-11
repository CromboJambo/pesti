//! Comprehensive OR/AND gate tests for CPU/GPU attention implementations

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

    // Generate random input data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
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

    println!("=== Comprehensive OR/AND Gate Tests ===");
    println!("Dimensions: seq_q={}, seq_k={}, num_heads={}, head_dim={}\n", 
             seq_q, seq_k, num_heads, head_dim);

    // Test 1: GEMM vs Ndarray
    println!("Test 1: GEMM vs Ndarray");
    let start = Instant::now();
    let result_gemm = reference_raw_scores_optimized(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration_gemm = start.elapsed();

    let start = Instant::now();
    let result_ndarray = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration_ndarray = start.elapsed();

    let max_error_1 = result_gemm.iter()
        .zip(result_ndarray.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  GEMM time: {:.3}ms", duration_gemm.as_secs_f64() * 1000.0);
    println!("  Ndarray time: {:.3}ms", duration_ndarray.as_secs_f64() * 1000.0);
    println!("  Max error: {:.6}", max_error_1);
    if max_error_1 < 2.0 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }

    // Test 2: GEMM vs Ndarray Manual
    println!("Test 2: GEMM vs Ndarray Manual (manual dot products)");
    let start = Instant::now();
    let result_manual = reference_with_ndarray_manual(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration_manual = start.elapsed();

    let max_error_2 = result_gemm.iter()
        .zip(result_manual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  Manual time: {:.3}ms", duration_manual.as_secs_f64() * 1000.0);
    println!("  Max error: {:.6}", max_error_2);
    if max_error_2 < 2.0 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }

    // Test 3: Ndarray vs Ndarray Manual
    println!("Test 3: Ndarray vs Ndarray Manual");
    let max_error_3 = result_ndarray.iter()
        .zip(result_manual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  Max error: {:.6}", max_error_3);
    if max_error_3 < 2.0 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }

    // Test 4: Performance comparison
    println!("=== Performance Summary ===");
    let speedup_ndarray = duration_gemm.as_secs_f64() / duration_ndarray.as_secs_f64();
    let speedup_manual = duration_gemm.as_secs_f64() / duration_manual.as_secs_f64();

    println!("  Ndarray vs GEMM: {:.2}x faster", speedup_ndarray);
    println!("  Manual vs GEMM: {:.2}x faster", speedup_manual);

    // Test 5: Edge cases
    println!("\n=== Edge Case Tests ===");
    
    // Small dimensions
    let small_seq_q = 4;
    let small_seq_k = 4;
    let small_num_heads = 1;
    let small_head_dim = 8;

    println!("Small dimensions: seq_q={}, seq_k={}, num_heads={}, head_dim={}", 
             small_seq_q, small_seq_k, small_num_heads, small_head_dim);

    let mut rng_small = rand::rngs::StdRng::seed_from_u64(123);
    let q_h_small: Vec<half::f16> = (0..small_seq_q * small_num_heads * small_head_dim)
        .map(|_| {
            let val = rng_small.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let k_h_small: Vec<half::f16> = (0..small_seq_k * small_num_heads * small_head_dim)
        .map(|_| {
            let val = rng_small.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let v_h_small: Vec<half::f16> = (0..small_seq_k * small_num_heads * small_head_dim)
        .map(|_| {
            let val = rng_small.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let result_gemm_small = reference_raw_scores_optimized(&q_h_small, &k_h_small, &v_h_small, 
                                                          small_seq_q, small_seq_k, small_num_heads, small_head_dim, rope_base, scale);
    let result_ndarray_small = reference_with_ndarray(&q_h_small, &k_h_small, &v_h_small, 
                                                      small_seq_q, small_seq_k, small_num_heads, small_head_dim, rope_base, scale);

    let max_error_small = result_gemm_small.iter()
        .zip(result_ndarray_small.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  Max error (small dims): {:.6}", max_error_small);
    if max_error_small < 2.0 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }

    // Large dimensions
    let large_seq_q = 256;
    let large_seq_k = 256;
    let large_num_heads = 8;
    let large_head_dim = 128;

    println!("Large dimensions: seq_q={}, seq_k={}, num_heads={}, head_dim={}", 
             large_seq_q, large_seq_k, large_num_heads, large_head_dim);

    let mut rng_large = rand::rngs::StdRng::seed_from_u64(456);
    let q_h_large: Vec<half::f16> = (0..large_seq_q * large_num_heads * large_head_dim)
        .map(|_| {
            let val = rng_large.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let k_h_large: Vec<half::f16> = (0..large_seq_k * large_num_heads * large_head_dim)
        .map(|_| {
            let val = rng_large.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let v_h_large: Vec<half::f16> = (0..large_seq_k * large_num_heads * large_head_dim)
        .map(|_| {
            let val = rng_large.random::<f32>() * 2.0 - 1.0;
            half::f16::from_f32(val)
        })
        .collect();

    let result_gemm_large = reference_raw_scores_optimized(&q_h_large, &k_h_large, &v_h_large, 
                                                          large_seq_q, large_seq_k, large_num_heads, large_head_dim, rope_base, scale);
    let result_ndarray_large = reference_with_ndarray(&q_h_large, &k_h_large, &v_h_large, 
                                                      large_seq_q, large_seq_k, large_num_heads, large_head_dim, rope_base, scale);

    let max_error_large = result_gemm_large.iter()
        .zip(result_ndarray_large.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("  Max error (large dims): {:.6}", max_error_large);
    if max_error_large < 2.0 {
        println!("  ✓ PASS\n");
    } else {
        println!("  ✗ FAIL\n");
    }

    println!("=== All Tests Complete ===");
}
