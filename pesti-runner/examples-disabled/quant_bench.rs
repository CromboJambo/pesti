//! Benchmark: QuantizedLinear vs Linear forward pass.
//!
//! Compares tile-dequantized GEMM (quantized) vs pre-dequantized f32 GEMM.
//! Shows memory savings and any performance difference.
//!
//! Usage: cargo run --package pesti-runner --features cuda --example quant_bench

use pesti_runner::quantized_linear::QuantizedLinear;
use pesti_runner::tile_dequant::QuantDtype;
use pesti_runner::transformer::linear::Linear;
use std::time::Instant;

fn main() {
    println!("=== QuantizedLinear vs Linear Benchmark ===\n");

    // Simulate a typical attention layer: Q projection
    // Qwen2.5-0.5B: in=896, out=896 (Q projection for MHA)
    let in_features = 500;
    let out_features = 500;
    let batch_size = 1;

    // Create random-ish input
    let x: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.01).sin()).collect();

    // ── Test 1: Q4_0 ──
    println!("--- Q4_0 (in={}, out={}) ---", in_features, out_features);
    benchmark_quant(QuantDtype::Q4_0, in_features, out_features, &x, batch_size);

    // ── Test 2: Q8_0 ──
    println!("\n--- Q8_0 (in={}, out={}) ---", in_features, out_features);
    benchmark_quant(QuantDtype::Q8_0, in_features, out_features, &x, batch_size);

    // ── Test 3: Q4_K ──
    println!("\n--- Q4_K (in={}, out={}) ---", in_features, out_features);
    benchmark_quant(QuantDtype::Q4_K, in_features, out_features, &x, batch_size);

    // ── Test 4: Larger layer (FFN gate) ──
    let in_features = 896;
    let out_features = 4864;
    let x_ffn: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.01).sin()).collect();
    println!(
        "\n--- Q4_K (in={}, out={}) FFN gate ---",
        in_features, out_features
    );
    benchmark_quant(
        QuantDtype::Q4_K,
        in_features,
        out_features,
        &x_ffn,
        batch_size,
    );

    println!("\n=== Benchmark Complete ===");
}

fn benchmark_quant(
    dtype: QuantDtype,
    in_features: usize,
    out_features: usize,
    x: &[f32],
    batch_size: usize,
) {
    // Create fake quantized weight data
    // For Q4_0: 18 bytes per 32 elements → row_bytes = (in_features/32)*18
    // For Q8_0: 34 bytes per 32 elements → row_bytes = (in_features/32)*34
    // For Q4_K: 28 bytes per 16 elements → row_bytes = (in_features/16)*28
    let row_bytes = match dtype {
        QuantDtype::Q4_0 => ((in_features + 31) / 32) * 18,
        QuantDtype::Q4_1 => ((in_features + 31) / 32) * 20,
        QuantDtype::Q8_0 => ((in_features + 31) / 32) * 34,
        QuantDtype::Q4_K => ((in_features + 15) / 16) * 28,
        QuantDtype::Q5_K => ((in_features + 15) / 16) * 36,
        QuantDtype::Q6_K => ((in_features + 15) / 16) * 42,
    };
    let raw_bytes = out_features * row_bytes;

    // Fill with zeros (dequantizes to zeros)
    let raw_data = vec![0u8; raw_bytes];

    // Create QuantizedLinear
    let ql = QuantizedLinear::new(raw_data.clone(), dtype, in_features, out_features, None);

    // Create equivalent Linear with f32 zeros
    let f32_weights = vec![0.0f32; out_features * in_features];
    let linear = Linear::new(f32_weights, None, in_features, out_features);

    // Memory comparison
    let quant_mem = ql.memory_bytes();
    let f32_mem = ql.f32_memory_bytes();
    let savings = ql.memory_savings();

    println!(
        "  Memory: quantized={:.1} KB, f32={:.1} KB, savings={:.1}x",
        quant_mem as f64 / 1024.0,
        f32_mem as f64 / 1024.0,
        savings,
    );

    // Warmup
    let _ = linear.forward(x, batch_size);
    let _ = ql.forward(x, batch_size);

    // Benchmark Linear (f32)
    let iters = 10;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = linear.forward(x, batch_size);
    }
    let linear_time = start.elapsed() / iters;

    // Benchmark QuantizedLinear
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ql.forward(x, batch_size);
    }
    let quant_time = start.elapsed() / iters;

    println!(
        "  Performance: linear={:.2}ms, quantized={:.2}ms, ratio={:.1}x",
        linear_time.as_secs_f64() * 1000.0,
        quant_time.as_secs_f64() * 1000.0,
        linear_time.as_secs_f64() / quant_time.as_secs_f64(),
    );
}
