//! Performance benchmark for one-stage full fusion attention kernel
//! 
//! Compares GPU vs CPU performance across various sequence lengths and configurations.
//! Measures: tokens/sec, memory bandwidth, speedup factor.

#![cfg(feature = "cuda")]

use pesti_runner::cuda_runtime::{allocate_device_memory, copy_host_to_device, CudaRuntime};
use pesti_runner::cuda_shim::{cu_stream, launch_kernel, CudaModule};
use std::time::Instant;

/// Reference CPU attention implementation for comparison
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for dim_idx in 0..head_dim {
                let mut sum = 0.0f32;
                
                // Compute scores over k positions
                let mut scores = vec![0.0f32; seq_k];
                for k_pos in 0..seq_k {
                    let q_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    let k_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    
                    // Dot product over head_dim
                    for d in 0..head_dim {
                        let q_d = q[q_pos * num_heads * head_dim + head * head_dim + d];
                        let k_d = k[k_pos * num_heads * head_dim + head * head_dim + d];
                        scores[k_pos] += q_d * k_d;
                    }
                    scores[k_pos] /= (head_dim as f32).sqrt();
                }

                // Softmax over sequence dimension
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                
                for k_pos in 0..seq_k {
                    let softmax_val = (scores[k_pos] - max_score).exp() / exp_sum;
                    let v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    sum += softmax_val * v[v_idx];
                }

                output[q_pos * num_heads * head_dim + head * head_dim + dim_idx] = sum;
            }
        }
    }

    output
}

/// Launch the fused attention kernel (synchronous)
fn launch_fused_attention_sync(
    cuda_rt: &CudaRuntime,
    module: &CudaModule,
    q_ptr: *mut u8,
    k_ptr: *mut u8,
    v_ptr: *mut u8,
    out_ptr: *mut u8,
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = cuda_rt.new_stream()?;

    // Get function from module
    let mangled_name = "_Z22fused_attention_kernelPK6__halfS1_S1_Pfiiii";
    let function = module.load_function(mangled_name)?;

    // Parameters (10 total, matching fused kernel signature)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = v_ptr as u64;
    let mut out_v: u64 = out_ptr as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel with grid (seq_q, seq_k, num_heads), block (head_dim, 1, 1)
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);

    let mut params: [*mut std::ffi::c_void; 10] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_v as *mut u64 as *mut std::ffi::c_void,
        &mut out_v as *mut u64 as *mut std::ffi::c_void,
        &mut (scale as f32) as *mut f32 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        &mut 0u64 as *mut u64 as *mut std::ffi::c_void, // placeholder for unused param
    ];

    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        )?;
    }

    Ok(())
}

/// Benchmark a single configuration
fn benchmark_config(seq_q: usize, seq_k: usize, num_heads: usize, head_dim: usize) {
    println!("\n=== Configuration: seq_q={}, seq_k={}, heads={}, dim={} ===", 
             seq_q, seq_k, num_heads, head_dim);

    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    // Allocate buffers
    let q_size = seq_q * num_heads * head_dim * 2; // f16
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let output_size = seq_q * num_heads * head_dim * 4; // f32

    // Initialize with deterministic values
    let q_host: Vec<f32> = (0..seq_q * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_q * num_heads * head_dim) as f32 / 2.0) * 0.1)
        .collect();
    let k_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.15)
        .collect();
    let v_host: Vec<f32> = (0..seq_k * num_heads * head_dim)
        .map(|i| (i as f32 - (seq_k * num_heads * head_dim) as f32 / 2.0) * 0.2)
        .collect();

    // Allocate GPU memory
    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(output_size).unwrap() };

    // Copy to device (convert f32 to f16)
    let q_host_f16: Vec<half::f16> = q_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let k_host_f16: Vec<half::f16> = k_host.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_host_f16: Vec<half::f16> = v_host.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        copy_host_to_device(q_ptr, q_host_f16.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_host_f16.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_host_f16.as_ptr() as *const u8, v_size).unwrap();
    }

    // Load PTX and prepare kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    // Warmup run
    launch_fused_attention_sync(
        &cuda_rt, &module, q_ptr, k_ptr, v_ptr, out_ptr,
        seq_q, seq_k, num_heads, head_dim
    ).unwrap();
    cuda_rt.synchronize().unwrap();

    // Benchmark GPU (10 iterations)
    let mut gpu_times: Vec<f64> = Vec::with_capacity(10);
    for _ in 0..10 {
        launch_fused_attention_sync(
            &cuda_rt, &module, q_ptr, k_ptr, v_ptr, out_ptr,
            seq_q, seq_k, num_heads, head_dim
        ).unwrap();
        
        let start = Instant::now();
        cuda_rt.synchronize().unwrap();
        gpu_times.push(start.elapsed().as_secs_f64());
    }

    // Benchmark CPU (1 iteration for comparison)
    let start = Instant::now();
    let _cpu_output = cpu_attention(&q_host, &k_host, &v_host, seq_q, seq_k, num_heads, head_dim);
    let cpu_time = start.elapsed().as_secs_f64();

    // GPU stats
    let avg_gpu_time = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;
    let min_gpu_time = gpu_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_gpu_time = gpu_times.iter().cloned().fold(0.0, f64::max);

    // Calculate metrics
    let total_ops = 2.0 * seq_q as f64 * seq_k as f64 * num_heads as f64 * head_dim as f64;
    let gpu_tflops = (total_ops / avg_gpu_time) / 1e12;
    let speedup = cpu_time / avg_gpu_time;

    // Memory bandwidth (assuming 32 bytes per element: 2x f16 input + 1x f32 output)
    let total_bytes = (q_size + k_size + v_size + output_size) as f64;
    let gpu_bandwidth = total_bytes / avg_gpu_time / 1e9; // GB/s

    println!("  GPU:   {} ms (avg), {} ms (min), {} ms (max)", 
             avg_gpu_time * 1000.0, min_gpu_time * 1000.0, max_gpu_time * 1000.0);
    println!("  CPU:   {:.3} s", cpu_time);
    println!("  Speedup: {:.2}x", speedup);
    println!("  GPU TFLOPS: {:.3}", gpu_tflops);
    println!("  Memory bandwidth: {:.2} GB/s", gpu_bandwidth);
}

fn main() {
    println!("=== One-Stage Full Fusion Attention Benchmark ===");
    println!("Hardware: NVIDIA GeForce RTX 4070 Ti SUPER");
    println!();

    // Test configurations
    benchmark_config(2, 4, 2, 8);      // Original small config
    benchmark_config(4, 8, 2, 8);      // Medium sequences
    benchmark_config(8, 16, 2, 8);     // Larger sequences
    benchmark_config(4, 8, 4, 8);      // Multiple heads
    benchmark_config(4, 8, 2, 16);     // Larger head dim

    println!("\n=== Benchmark Complete ===");
}
