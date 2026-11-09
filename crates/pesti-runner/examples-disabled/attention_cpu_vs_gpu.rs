//! CPU vs GPU Attention Kernel Benchmark
//!
//! Compares CpuAttentionKernel (naive triple-loop) against GemmBasedAttentionKernel
//! (GPU tensor cores via mma.sync) on matching workloads.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda --example attention_cpu_vs_gpu

use half::f16;
use pesti_runner::CudaRuntime;
use pesti_runner::kernel::Kvcache;
use pesti_runner::kernel::SoftmaxKernel;
use pesti_runner::kernel::attention::{
    AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel, GemmBasedAttentionKernel,
};
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::gemm::{CudaGemmKernelBuilder, GemmArch};
use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
use pesti_runner::kernel::softmax::CpuSoftmaxKernel;
use std::sync::Arc;
use std::time::Instant;

fn run_benchmark(
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    query_len: usize,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", label);
    println!(
        "  Config: {} heads, {} dim, seq_len={}, query_len={}",
        num_heads, head_dim, seq_len, query_len
    );

    let q_size = query_len * num_heads * head_dim;
    let kv_size = num_heads * head_dim * seq_len;

    // Generate deterministic Q, K, V data (same for both CPU and GPU)
    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32(((i as f32 * 0.1).sin() + 1.0) / 2.0))
        .collect();
    let k_host: Vec<f16> = (0..kv_size)
        .map(|i| f16::from_f32(((i as f32 * 0.07).sin() + 1.0) / 2.0))
        .collect();
    let v_host: Vec<f16> = (0..kv_size)
        .map(|i| f16::from_f32(((i as f32 * 0.03).sin() + 1.0) / 2.0))
        .collect();

    // --- CPU benchmark ---
    let cpu_kernel = CpuAttentionKernel::new(AttentionArch::Cpu);
    let cpu_config = AttentionConfig::new(num_heads, head_dim).with_max_seq(seq_len);

    // For CPU, create a host-backed Kvcache and populate it
    let mut kvc_cpu = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);

    // Populate Kvcache using write_kv_at for each position
    let head_stride = num_heads * head_dim;
    for pos in 0..seq_len {
        let k_offset = pos * head_stride;
        let v_offset = pos * head_stride;
        kvc_cpu.write_kv_at(
            pos,
            &k_host[k_offset..k_offset + head_stride],
            &v_host[v_offset..v_offset + head_stride],
        )?;
    }
    kvc_cpu.set_seq_len(seq_len);

    let q_cpu_buf = DeviceBuffer::from_host(q_host.clone());

    // Warmup
    let _ = cpu_kernel.forward(&q_cpu_buf, &kvc_cpu, &kvc_cpu, None, &cpu_config)?;

    let cpu_loops = if seq_len <= 128 { 20 } else { 5 };
    let cpu_start = Instant::now();
    let cpu_output = cpu_kernel.forward(&q_cpu_buf, &kvc_cpu, &kvc_cpu, None, &cpu_config)?;
    for _ in 0..cpu_loops {
        let _ = cpu_kernel.forward(&q_cpu_buf, &kvc_cpu, &kvc_cpu, None, &cpu_config)?;
    }
    let cpu_time = cpu_start.elapsed() / cpu_loops;
    let cpu_output_f32: Vec<f32> = cpu_output.to_host();
    // tok/s = query positions processed per second
    let cpu_tok_s = query_len as f32 / cpu_time.as_secs_f32();

    println!(
        "  CPU:  {:.2}ms/iter ({} iters), {:.1} tok/s",
        cpu_time.as_secs_f64() * 1000.0,
        cpu_loops,
        cpu_tok_s
    );

    // --- GPU benchmark ---
    let devices = pesti_runner::enumerate_devices().unwrap_or_default();
    if devices.is_empty() {
        println!("  GPU:  No CUDA devices found, skipping GPU benchmark");
        return Ok(());
    }

    println!("  Found {} CUDA device(s)", devices.len());

    // Use device 1 if available (RTX 5060 Ti), else 0
    let device_idx = if devices.len() > 1 { 1 } else { 0 };
    println!("  Using device {} for GPU benchmark", device_idx);

    let rt = CudaRuntime::new(device_idx)?;
    let stream = rt.new_stream()?;
    let device_info = rt.device_info().clone();
    let backend = Arc::new(CudaMemoryBackend::with_device_info(
        stream.clone(),
        device_info.clone(),
    ));

    println!(
        "  GPU Device: {} (sm_{}.{})",
        device_info.name, device_info.compute_capability.0, device_info.compute_capability.1
    );

    // Create GEMM kernel
    let gemm_kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        device_info.clone(),
    )
    .build()?;

    let attn_kernel = GemmBasedAttentionKernel::new(
        gemm_kernel,
        backend.clone(),
        Box::new(CpuSoftmaxKernel::new()) as Box<dyn SoftmaxKernel>,
    );

    // Create GPU Kvcache with same data
    let mut kvc_gpu = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
    let head_stride = num_heads * head_dim;
    for pos in 0..seq_len {
        let k_offset = pos * head_stride;
        let v_offset = pos * head_stride;
        kvc_gpu.write_kv_at(
            pos,
            &k_host[k_offset..k_offset + head_stride],
            &v_host[v_offset..v_offset + head_stride],
        )?;
    }

    // Copy Kvcache to device
    let host_slice = kvc_gpu.buffer().as_slice().expect("kvcache must be host");
    let kv_device_buf = DeviceBuffer::from_host_device(&*backend, host_slice)?;
    let kv_ptr = kv_device_buf.device_ptr();
    let mut kvc_device =
        unsafe { Kvcache::from_device(kv_ptr, num_heads, num_heads, head_dim, seq_len) };
    kvc_device.set_seq_len(seq_len);

    // Copy query to GPU
    let q_gpu_buf = DeviceBuffer::from_host_device(&*backend, &q_host)?;

    let gpu_config = AttentionConfig::new(num_heads, head_dim).with_max_seq(seq_len);

    // Warmup
    let _ = attn_kernel.forward(&q_gpu_buf, &kvc_device, &kvc_device, None, &gpu_config)?;

    // GPU benchmark
    let gpu_loops = 10;
    let gpu_start = Instant::now();
    let gpu_output =
        attn_kernel.forward(&q_gpu_buf, &kvc_device, &kvc_device, None, &gpu_config)?;
    for _ in 0..gpu_loops {
        let _ = attn_kernel.forward(&q_gpu_buf, &kvc_device, &kvc_device, None, &gpu_config)?;
    }
    let gpu_time = gpu_start.elapsed() / gpu_loops;
    // tok/s = query positions processed per second
    let gpu_tok_s = query_len as f32 / gpu_time.as_secs_f32();

    println!(
        "  GPU:  {:.2}ms/iter ({} iters), {:.1} tok/s",
        gpu_time.as_secs_f64() * 1000.0,
        gpu_loops,
        gpu_tok_s
    );

    // Speedup
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("  Speedup: {:.1}x", speedup);

    // Accuracy check
    let gpu_output_f32: Vec<f32> = gpu_output
        .to_host_vec(&*backend)
        .map_err(|e| format!("D2H failed: {e}"))?;

    let max_err = cpu_output_f32
        .iter()
        .zip(gpu_output_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_err = cpu_output_f32
        .iter()
        .zip(gpu_output_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / cpu_output_f32.len() as f32;

    println!(
        "  Accuracy: max_err={:.6e}, mean_err={:.6e}",
        max_err, mean_err
    );
    if max_err < 0.01 {
        println!("  ✅ CPU and GPU outputs match within tolerance");
    } else {
        println!("  ⚠️  Outputs differ — verify numerical precision");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PESTI CPU vs GPU Attention Benchmark ===");
    println!("  CpuAttentionKernel (naive triple-loop) vs GemmBasedAttentionKernel (mma.sync)\n");

    // Test 1: Small config (Qwen2.5-0.5B style)
    run_benchmark(8, 64, 256, 1, "Qwen2.5-0.5B: 8h x 64d x 256tok")?;

    // Test 2: Medium config
    run_benchmark(16, 128, 512, 1, "Medium: 16h x 128d x 512tok")?;

    // Test 3: Large config
    run_benchmark(32, 128, 1024, 1, "Large: 32h x 128d x 1024tok")?;

    // Test 4: Multi-query
    run_benchmark(8, 64, 512, 32, "Multi-query: 8h x 64d x 512tok, 32 query")?;

    println!("\n=== Benchmark Complete ===");
    Ok(())
}
