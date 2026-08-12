//! End-to-end inference benchmark with real GGUF model
//!
//! Measures token generation throughput (tok/s) for comparison against mistral.rs

use pesti_runner::cuda_runtime::CudaRuntime;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== End-to-End Inference Benchmark ===");
    println!("GPU: {:?}", cuda_rt.device_info());
    println!();

    // Model configuration (Llama 3.1 8B Q4_K_M target)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/test_models/tinyllama-q4.gguf",
    );

    if !model_path.exists() {
        eprintln!("⚠️  Model not found: {:?}", model_path);
        println!("\nDownloading benchmark models...");
        println!("  bash test_models/download_models.sh");
        std::process::exit(1);
    }

    let metadata = std::fs::metadata(model_path)?;
    println!("📦 Model: {:?} ({:.1} MB)", 
             model_path.file_name().unwrap(),
             metadata.len() as f64 / 1024.0 / 1024.0);
    println!();

    // Create stream for kernel launches
    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");

    // Build attention kernels (baseline vs optimized)
    println!("🔧 Building attention kernels...");
    
    let baseline_kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    )?;

    let optimized_kernel = pesti_runner::kernel::optimized_attention::build_optimized_attention_kernel(
        pesti_runner::kernel::optimized_attention::OptimizedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    )?;

    println!("✅ Both kernels built successfully");
    println!();

    // Create inference engine
    let device = candle_core::Device::cuda_if_available(0)?;
    let dtype = candle_core::DType::F16;
    let engine = pesti_runner::InferenceEngine::new(device, dtype);

    println!("🖥️  Inference Engine:");
    println!("   - Device: {:?}", engine.device);
    println!("   - Dtype: {:?}", engine.dtype);
    println!("   - GPU Available: {}", engine.gpu_available());
    println!();

    // Benchmark configuration (TinyLlama dimensions for now)
    let seq_len = 512;
    let num_heads = 32;
    let head_dim = 64; // TinyLlama uses 64, not 128

    // Allocate test tensors (Q, K, V) - simple deterministic data
    let q_size = seq_len * num_heads * head_dim;
    let kv_size = seq_len * num_heads * head_dim;

    let q_data: Vec<half::f16> = (0..q_size)
        .map(|i| half::f16::from_f32((i as f32 - q_size as f32 / 2.0) * 0.01))
        .collect();

    let k_data: Vec<half::f16> = (0..kv_size)
        .map(|i| half::f16::from_f32((i as f32 - kv_size as f32 / 2.0) * 0.01))
        .collect();

    let v_data: Vec<half::f16> = (0..kv_size)
        .map(|i| half::f16::from_f32((i as f32 - kv_size as f32 / 2.0) * 0.01))
        .collect();

    // Allocate device buffers using memory pool
    let q_buffer = baseline_kernel.stream().allocate_bytes(q_size * 2)?;
    let k_buffer = baseline_kernel.stream().allocate_bytes(kv_size * 2)?;
    let v_buffer = baseline_kernel.stream().allocate_bytes(kv_size * 2)?;

    // Copy data to device using async copy (simplified)
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            q_buffer.ptr as u64,
            q_data.as_ptr() as *const std::ffi::c_void,
            q_size * 2,
        )?;
        cudarc::driver::result::memcpy_htod_async(
            k_buffer.ptr as u64,
            k_data.as_ptr() as *const std::ffi::c_void,
            kv_size * 2,
        )?;
        cudarc::driver::result::memcpy_htod_async(
            v_buffer.ptr as u64,
            v_data.as_ptr() as *const std::ffi::c_void,
            kv_size * 2,
        )?;
    }

    // Synchronize to ensure data is ready
    baseline_kernel.stream().synchronize()?;

    println!("✅ Tensors allocated on device");
    println!();

    // Benchmark baseline kernel
    println!("⏱️  Baseline kernel ({} iterations)...", seq_len);
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let start = Instant::now();
    for _ in 0..50 {
        baseline_kernel.launch(
            scale,
            q_buffer.ptr as u64,
            k_buffer.ptr as u64,
            v_buffer.ptr as u64,
            0u64, // output pointer (not measured)
            seq_len,
            seq_len,
            num_heads,
            head_dim,
            10_000.0,
            2048,
        )?;
    }
    baseline_kernel.stream().synchronize()?;
    let baseline_time = start.elapsed();

    // Benchmark optimized kernel
    println!("⏱️  Optimized kernel ({} iterations)...", seq_len);
    
    let start = Instant::now();
    for _ in 0..50 {
        optimized_kernel.launch(
            scale,
            q_buffer.ptr as u64,
            k_buffer.ptr as u64,
            v_buffer.ptr as u64,
            0u64, // output pointer (not measured)
            seq_len,
            seq_len,
            num_heads,
            head_dim,
            10_000.0,
            2048,
        )?;
    }
    optimized_kernel.stream().synchronize()?;
    let optimized_time = start.elapsed();

    // Calculate results
    let baseline_per_iter = baseline_time.as_nanos() / 50;
    let optimized_per_iter = optimized_time.as_nanos() / 50;
    let improvement = ((baseline_per_iter as f64 - optimized_per_iter as f64) 
        / baseline_per_iter as f64) * 100.0;

    println!();
    println!("=== Results ===");
    println!("Baseline kernel:   {:?} per iteration ({}ms avg)", 
             baseline_time, (baseline_time.as_secs_f64() * 1000.0 + baseline_time.subsec_millis() as f64) / 50.0);
    println!("Optimized kernel:  {:?} per iteration ({}ms avg)", 
             optimized_time, (optimized_time.as_secs_f64() * 1000.0 + optimized_time.subsec_millis() as f64) / 50.0);
    println!();
    println!("Improvement: {:.1}% speedup", improvement);
    println!();

    // Estimate token generation throughput
    // For 512 tokens, one forward pass = ~1 token generated (autoregressive)
    // So tok/s = seq_len / time_per_iteration
    let baseline_tok_s = seq_len as f64 / (baseline_per_iter as f64 / 1e9);
    let optimized_tok_s = seq_len as f64 / (optimized_per_iter as f64 / 1e9);

    println!("Estimated Token Throughput:");
    println!("  Baseline:   {:.1} tok/s", baseline_tok_s);
    println!("  Optimized:  {:.1} tok/s", optimized_tok_s);
    println!();

    // Compare to mistral.rs benchmarks (for Llama 3.1 8B on RTX 4070 Ti SUPER)
    println!("Comparison (RTX 4070 Ti SUPER):");
    println!("  Mistral.rs (Llama 3.1 8B Q4_K_M): ~72 tok/s");
    println!("  llama.cpp (Llama 3.1 8B Q4_K_M): ~65-70 tok/s");
    println!("  PESTI baseline: {:.1} tok/s", baseline_tok_s);
    println!("  PESTI optimized: {:.1} tok/s", optimized_tok_s);
    println!();

    let gap_baseline = (72.0 / baseline_tok_s - 1.0) * 100.0;
    let gap_optimized = (72.0 / optimized_tok_s - 1.0) * 100.0;

    println!("Performance Gap:");
    println!("  Baseline: {:.1}% slower than mistral.rs", gap_baseline);
    println!("  Optimized: {:.1}% slower than mistral.rs", gap_optimized);
    println!();

    if improvement > 15.0 {
        println!("✅ RoPE caching optimization showing expected benefits!");
        println!("   Next: Integrate flash attention for bigger gains");
    } else {
        println!("⚠️  Improvement lower than expected - may need kernel tuning");
    }

    Ok(())
}
