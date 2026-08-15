//! Benchmark single-token decode throughput with one-stage attention kernel
//!
//! Measures tokens/sec for autoregressive generation (seq_len = 1)

#![cfg(feature = "cuda")]

use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::kvcache::Kvcache;
use pesti_runner::kernel::one_stage_attention::{OneStageAttentionConfig, OneStageAttentionKernel};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== One-Stage Attention Kernel Benchmark ===\n");

    // Initialize CUDA
    let cuda_rt = CudaRuntime::new(0)?;
    let device_info = cuda_rt.device_info();
    println!("GPU: {}", device_info.name);
    println!("Memory: {} MiB total, {} MiB free", 
             device_info.total_memory / (1024 * 1024),
             device_info.free_memory / (1024 * 1024));
    println!();

    // Model configuration (Qwen2.5-0.5B)
    let num_heads = 32;
    let num_kv_heads = 8;
    let head_dim = 64;
    let max_seq = 2048;

    println!("Model: Qwen2.5-0.5B (simulated)");
    println!("  - num_heads: {}", num_heads);
    println!("  - num_kv_heads: {}", num_kv_heads);
    println!("  - head_dim: {}", head_dim);
    println!("  - max_seq: {}", max_seq);
    println!();

    // Initialize KV caches
    let key_cache = Kvcache::new(
        num_kv_heads,
        num_kv_heads,
        head_dim,
        max_seq,
        true, // on_device
    );
    let value_cache = Kvcache::new(
        num_kv_heads,
        num_kv_heads,
        head_dim,
        max_seq,
        true, // on_device
    );

    println!("KV caches initialized");
    println!();

    // Create one-stage attention kernel
    let config = OneStageAttentionConfig::new(num_heads, num_kv_heads, head_dim);
    let backend = Arc::new(pesti_runner::kernel::memory::CudaMemoryBackend::new());
    let kernel = OneStageAttentionKernel::new(config, backend);

    // Benchmark parameters
    let batch_size = 1;      // Single token decode
    let seq_len = 1;         // One-step autoregressive decode
    let num_iterations = 1000;
    let warmup_iterations = 100;

    println!("Benchmark: {} iterations ({} warmup)", num_iterations, warmup_iterations);
    println!();

    // Generate random input Q (simulating token embeddings)
    let embed_dim = num_heads * head_dim;
    let q: Vec<f32> = (0..embed_dim)
        .map(|i| (i as f32 - embed_dim as f32 / 2.0) * 0.1)
        .collect();

    // Warmup
    println!("Warmup...");
    for _ in 0..warmup_iterations {
        let _ = kernel.forward(&q, &key_cache, &value_cache, batch_size, seq_len, 0)?;
        cuda_rt.synchronize()?;
    }
    cuda_rt.synchronize()?;

    // Benchmark
    println!("Running benchmark...");
    let start = Instant::now();

    for _ in 0..num_iterations {
        let _ = kernel.forward(&q, &key_cache, &value_cache, batch_size, seq_len, 0)?;
        cuda_rt.synchronize()?;
    }

    let elapsed = start.elapsed();
    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / num_iterations as f64;

    println!();
    println!("=== Results ===");
    println!("  Total time: {:.4} s", elapsed.as_secs_f64());
    println!("  Avg time per token: {:.4} ms", avg_time_ms);
    println!("  Throughput: {:.2} tokens/sec", num_iterations as f64 / elapsed.as_secs_f64());
    println!();

    // Compare with llama.cpp baseline (~85 tokens/sec for Qwen2.5-0.5B on RTX 4070 Ti SUPER)
    let llama_baseline = 85.0;
    let speedup = (num_iterations as f64 / elapsed.as_secs_f64()) / llama_baseline;
    
    println!("Comparison:");
    println!("  llama.cpp baseline: {:.2} tokens/sec", llama_baseline);
    println!("  Our kernel:         {:.2} tokens/sec", num_iterations as f64 / elapsed.as_secs_f64());
    println!("  Speedup:            {:.2}x", speedup);

    Ok(())
}
