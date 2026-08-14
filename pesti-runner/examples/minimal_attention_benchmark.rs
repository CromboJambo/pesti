//! Minimal attention benchmark with target parameters (seq_k=512, heads=32)
//! 
//! This example benchmarks the one-stage full fusion attention kernel at the
//! target configuration: Qwen2 variant with 32 query heads, 8 KV heads,
//! embed_dim=128, head_dim=28.

#![cfg(feature = "cuda")]

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{launch_kernel, CudaModule};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Minimal Attention Benchmark ===");
    println!("Target: seq_q=1, seq_k=512, num_heads=32, head_dim=28\n");

    // Initialize CUDA
    let cuda = CudaRuntime::new(0)?;
    let stream = unsafe { cuda.new_stream()? };

    // Target parameters (Qwen2 variant)
    const SEQ_Q: usize = 1;
    const SEQ_K: usize = 512;
    const NUM_HEADS: usize = 32;
    const HEAD_DIM: usize = 28;
    const TOTAL_Q_TOKENS: usize = SEQ_Q * NUM_HEADS;
    const TOTAL_KV_TOKENS: usize = SEQ_K * NUM_HEADS;

    // Allocate device memory
    let q_size = TOTAL_Q_TOKENS * HEAD_DIM * std::mem::size_of::<f16>();
    let k_size = TOTAL_KV_TOKENS * HEAD_DIM * std::mem::size_of::<f16>();
    let v_size = TOTAL_KV_TOKENS * HEAD_DIM * std::mem::size_of::<f16>();
    let o_size = TOTAL_Q_TOKENS * HEAD_DIM * std::mem::size_of::<f32>();

    println!("Allocated {} bytes for Q", q_size);
    println!("Allocated {} bytes for K", k_size);
    println!("Allocated {} bytes for V", v_size);
    println!("Allocated {} bytes for O\n", o_size);

    // Allocate on GPU
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size)? };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size)? };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size)? };
    let output_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(o_size)? };

    // Initialize with simple values (no need for actual inference data)
    let q_host: Vec<f16> = (0..TOTAL_Q_TOKENS * HEAD_DIM)
        .map(|i| f16::from_f32((i as f32) * 0.01))
        .collect();
    let k_host: Vec<f16> = (0..TOTAL_KV_TOKENS * HEAD_DIM)
        .map(|i| f16::from_f32((i as f32 - TOTAL_KV_TOKENS as f32 / 2.0) * 0.1))
        .collect();
    let v_host: Vec<f16> = (0..TOTAL_KV_TOKENS * HEAD_DIM)
        .map(|i| f16::from_f32((i as f32 - TOTAL_KV_TOKENS as f32 / 2.0) * 0.1))
        .collect();

    // Copy Q, K, V to GPU
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(
            q_ptr,
            q_host.as_ptr() as *const u8,
            q_size,
        )?;
        pesti_runner::cuda_runtime::copy_host_to_device(
            k_ptr,
            k_host.as_ptr() as *const u8,
            k_size,
        )?;
        pesti_runner::cuda_runtime::copy_host_to_device(
            v_ptr,
            v_host.as_ptr() as *const u8,
            v_size,
        )?;
    }

    println!("✅ Inputs copied to GPU");

    // Load PTX and get function
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda.context(), &ptx_src)?;

    let mangled_name = "_Z22fused_attention_kernelPK6__halfS1_S1_Pfiiii";
    let function = module.load_function(mangled_name)?;

    // Prepare parameters
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = v_ptr as u64;
    let mut out_v: u64 = output_ptr as u64;
    let mut seq_q_v: u32 = SEQ_Q as u32;
    let mut seq_k_v: u32 = SEQ_K as u32;
    let mut num_heads_v: u32 = NUM_HEADS as u32;
    let mut head_dim_v: u32 = HEAD_DIM as u32;

    let grid = (SEQ_Q as u32, NUM_HEADS as u32, 1u32);
    let block = (HEAD_DIM as u32, 1u32, 1u32);

    println!("Launching kernel: {}", mangled_name);
    println!("Grid: {:?}, Block: {:?}", grid, block);

    // Warm-up run
    let mut params: [*mut std::ffi::c_void; 8] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_v as *mut u64 as *mut std::ffi::c_void,
        &mut out_v as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
    ];

    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        )?;
    }

    // Benchmark runs
    let num_runs = 100;
    let mut times = Vec::with_capacity(num_runs);

    println!("\nRunning benchmark ({} iterations)...", num_runs);
    
    for _ in 0..num_runs {
        let start = Instant::now();
        
        unsafe {
            launch_kernel(
                function.cu_function(),
                grid,
                block,
                0,
                pesti_runner::cuda_shim::cu_stream(&stream),
                &mut params,
            )?;
        }
        
        cuda.synchronize()?;
        times.push(start.elapsed());
    }

    // Calculate statistics
    let avg_time = times.iter().sum::<std::time::Duration>() / num_runs as u32;
    let min_time = times.iter().min().unwrap();
    let max_time = times.iter().max().unwrap();

    println!("\n=== Results ===");
    println!("Average latency: {:.4} ms", avg_time.as_secs_f64() * 1000.0);
    println!("Min latency: {:.4} ms", min_time.as_secs_f64() * 1000.0);
    println!("Max latency: {:.4} ms", max_time.as_secs_f64() * 1000.0);

    // Calculate throughput (tokens per second)
    let tokens_per_forward = SEQ_Q as f64 * NUM_HEADS as f64;
    let tokens_per_second = tokens_per_forward / avg_time.as_secs_f64();
    
    println!("\nThroughput: {:.2} tokens/sec", tokens_per_second);
    println!("Latency per forward pass: {:.4} ms", avg_time.as_secs_f64() * 1000.0);

    Ok(())
}
