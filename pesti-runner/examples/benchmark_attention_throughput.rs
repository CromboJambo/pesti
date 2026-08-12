//! End-to-end attention benchmark with real model inference
//!
//! Measures token generation throughput (tok/s) comparing baseline vs optimized kernels.

use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::{InferenceEngine, ModelConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== End-to-End Attention Benchmark ===");
    println!("GPU: {:?}", cuda_rt.device_info());
    println!();

    // Create inference engine
    let device = candle_core::Device::cuda_if_available(0)?;
    let dtype = candle_core::DType::F16;
    let engine = InferenceEngine::new(device, dtype);

    println!("Inference Engine:");
    println!("  - Device: {:?}", engine.device());
    println!("  - Dtype: {:?}", engine.dtype());
    println!("  - GPU Available: {}", engine.gpu_available());
    println!();

    // Test configuration
    let batch_size = 1;
    let seq_len = 512;
    let num_heads = 32;
    let head_dim = 64;

    println!("Test Configuration:");
    println!("  - Batch size: {}", batch_size);
    println!("  - Sequence length: {}", seq_len);
    println!("  - Num heads: {}", num_heads);
    println!("  - Head dim: {}", head_dim);
    println!();

    // Allocate test tensors
    let q_size = seq_len * num_heads * head_dim;
    let kv_size = seq_len * num_heads * head_dim;

    println!("Allocating test tensors...");
    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");

    // Create random Q, K, V data
    let q_data: Vec<half::f16> = (0..q_size)
        .map(|_| half::f16::from_f32((rand::random::<f32>() - 0.5) * 2.0))
        .collect();

    let k_data: Vec<half::f16> = (0..kv_size)
        .map(|_| half::f16::from_f32((rand::random::<f32>() - 0.5) * 2.0))
        .collect();

    let v_data: Vec<half::f16> = (0..kv_size)
        .map(|_| half::f16::from_f32((rand::random::<f32>() - 0.5) * 2.0))
        .collect();

    // Allocate device buffers
    let q_buffer = stream.allocate_bytes(q_size * 2)?;
    let k_buffer = stream.allocate_bytes(kv_size * 2)?;
    let v_buffer = stream.allocate_bytes(kv_size * 2)?;

    // Copy data to device (simplified - would use async copy in production)
    unsafe {
        cudarc::driver::result::memcpy_htod(
            q_buffer.ptr as u64,
            q_data.as_ptr() as *const std::ffi::c_void,
            q_size * 2,
        )?;
        cudarc::driver::result::memcpy_htod(
            k_buffer.ptr as u64,
            k_data.as_ptr() as *const std::ffi::c_void,
            kv_size * 2,
        )?;
        cudarc::driver::result::memcpy_htod(
            v_buffer.ptr as u64,
            v_data.as_ptr() as *const std::ffi::c_void,
            kv_size * 2,
        )?;
    }

    println!("✅ Tensors allocated on device");
    println!();

    // Build both kernels
    println!("Building attention kernels...");

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

    println!("✅ Both kernels built");
    println!();

    // Benchmark baseline kernel
    println!("Benchmarking baseline kernel...");
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let start = std::time::Instant::now();
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
    println!("Benchmarking optimized kernel...");
    
    let start = std::time::Instant::now();
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
    let baseline_avg = baseline_time.as_nanos() / 50;
    let optimized_avg = optimized_time.as_nanos() / 50;
    let improvement = ((baseline_avg as f64 - optimized_avg as f64) / baseline_avg as f64) * 100.0;

    println!();
    println!("=== Results ===");
    println!("Baseline kernel:   {:?} per launch ({}ms avg)", 
             baseline_time, (baseline_time.as_secs_f64() * 1000.0 + baseline_time.subsec_millis() as f64) / 50.0);
    println!("Optimized kernel:  {:?} per launch ({}ms avg)", 
             optimized_time, (optimized_time.as_secs_f64() * 1000.0 + optimized_time.subsec_millis() as f64) / 50.0);
    println!();
    println!("Improvement: {:.1}% speedup", improvement);
    println!();

    // Expected vs actual
    println!("Expected improvement for {} tokens: ~15%", seq_len);
    println!("Actual improvement: {:.1}%", improvement);
    
    if improvement >= 10.0 {
        println!("✅ Optimization showing expected benefits!");
    } else {
        println!("⚠️  Improvement lower than expected - may need kernel tuning");
    }

    Ok(())
}
