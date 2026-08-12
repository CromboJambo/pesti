//! Scaling benchmark for ndarray CPU reference across different configurations

use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct BenchmarkConfig {
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
}

fn run_benchmark(config: &BenchmarkConfig) -> (f64, f64) {
    let rope_base = 10_000.0;
    let scale = 1.0 / (config.head_dim as f32).sqrt();

    // Generate random input data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let q_h: Vec<f16> = (0..config.seq_q * config.num_heads * config.head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let k_h: Vec<f16> = (0..config.seq_k * config.num_heads * config.head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let v_h: Vec<f16> = (0..config.seq_k * config.num_heads * config.head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    // Warm-up
    let _ = reference_with_ndarray(
        &q_h,
        &k_h,
        &v_h,
        config.seq_q,
        config.seq_k,
        config.num_heads,
        config.head_dim,
        rope_base,
        scale,
    );

    // Benchmark
    let iterations = 5;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..iterations {
        let start = Instant::now();
        let _result = reference_with_ndarray(
            &q_h,
            &k_h,
            &v_h,
            config.seq_q,
            config.seq_k,
            config.num_heads,
            config.head_dim,
            rope_base,
            scale,
        );
        total_time += start.elapsed();
    }

    let avg_ms = (total_time / iterations).as_secs_f64() * 1000.0;

    // Calculate throughput in ops/sec (Q@K^T + softmax + V sum)
    let total_ops = config.seq_q as f64
        * config.seq_k as f64
        * config.num_heads as f64
        * config.head_dim as f64
        * 3.0; // Multiply by 3 for all operations
    let throughput_mops = total_ops / avg_ms / 1e6;

    (avg_ms, throughput_mops)
}

fn main() {
    println!("=== PESTI Ndarray Scaling Benchmark ===\n");

    // Test configurations
    let configs = vec![
        // Small scale
        BenchmarkConfig {
            seq_q: 32,
            seq_k: 32,
            num_heads: 4,
            head_dim: 64,
        },
        BenchmarkConfig {
            seq_q: 64,
            seq_k: 64,
            num_heads: 4,
            head_dim: 64,
        },
        // Medium scale (current baseline)
        BenchmarkConfig {
            seq_q: 128,
            seq_k: 128,
            num_heads: 4,
            head_dim: 64,
        },
        // Large scale
        BenchmarkConfig {
            seq_q: 256,
            seq_k: 256,
            num_heads: 8,
            head_dim: 128,
        },
        // Very large scale
        BenchmarkConfig {
            seq_q: 512,
            seq_k: 512,
            num_heads: 8,
            head_dim: 128,
        },
        // Extreme scale
        BenchmarkConfig {
            seq_q: 1024,
            seq_k: 1024,
            num_heads: 16,
            head_dim: 128,
        },
    ];

    println!(
        "{:<35} {:>12} {:>15} {:>12}",
        "Configuration", "Time (ms)", "Throughput (M/s)", "Ops/sec"
    );
    println!("{}", "-".repeat(80));

    for config in &configs {
        let (time_ms, throughput_mops) = run_benchmark(&config);

        let total_ops = config.seq_q as f64
            * config.seq_k as f64
            * config.num_heads as f64
            * config.head_dim as f64;
        let ops_per_sec = total_ops / time_ms / 1e-3;

        println!(
            "{:<35} {:>12.3} {:>15.2} {:>12.2e}",
            format!(
                "seq={}x{}, heads={}, dim={}",
                config.seq_q, config.seq_k, config.num_heads, config.head_dim
            ),
            time_ms,
            throughput_mops,
            ops_per_sec
        );
    }

    println!("\n=== Memory Bandwidth Analysis ===");

    // Calculate memory footprint for largest config
    let max_config = &configs[configs.len() - 1];
    let q_bytes = max_config.seq_q * max_config.num_heads * max_config.head_dim * 2; // f16 = 2 bytes
    let k_bytes = max_config.seq_k * max_config.num_heads * max_config.head_dim * 2;
    let v_bytes = max_config.seq_k * max_config.num_heads * max_config.head_dim * 2;
    let output_bytes = max_config.seq_q * max_config.num_heads * max_config.head_dim * 4; // f32 output

    let total_memory = q_bytes + k_bytes + v_bytes + output_bytes;
    let memory_mb = total_memory as f64 / 1e6;

    println!(
        "Largest config ({}x{}, {} heads, {} dim):",
        max_config.seq_q, max_config.seq_k, max_config.num_heads, max_config.head_dim
    );
    println!("  Q tensor: {:.2} MB", q_bytes as f64 / 1e6);
    println!("  K tensor: {:.2} MB", k_bytes as f64 / 1e6);
    println!("  V tensor: {:.2} MB", v_bytes as f64 / 1e6);
    println!("  Output: {:.2} MB", output_bytes as f64 / 1e6);
    println!("  Total: {:.2} MB", memory_mb);

    // Estimate bandwidth (assuming each tensor read once, output written once)
    let memory_access = total_memory * 3; // Read Q, K, V and write output

    // Calculate average time from last benchmark
    let (_, last_throughput_mops) = run_benchmark(max_config);
    let avg_time_ms = (max_config.seq_q as f64
        * max_config.seq_k as f64
        * max_config.num_heads as f64
        * max_config.head_dim as f64
        * 3.0
        / 1e6)
        / last_throughput_mops;

    let bandwidth_gb_s = memory_access as f64 / 1e9 / (avg_time_ms / 1000.0);

    println!("  Estimated bandwidth: {:.2} GB/s", bandwidth_gb_s);

    println!("\n=== Performance Summary ===");
    println!("All benchmarks completed successfully!");
    println!("Configuration space tested: {} variants", configs.len());
}
