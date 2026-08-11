// Performance benchmark comparing baseline vs tiled attention kernels
// Measures throughput (M tokens/sec) and speedup factors

use pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder;
use pesti_runner::CudaRuntime;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tiled Kernel Performance Benchmark ===\n");
    
    // Configuration
    let seq_q = 128;
    let seq_k = 512;
    let num_heads = 4;
    let head_dim = 64;
    let rope_base: f32 = 10_000.0;
    
    // Initialize CUDA
    let cuda_rt = CudaRuntime::new(0)?;
    let stream = Arc::new(cuda_rt.new_stream()?);
    
    // Allocate and initialize host memory
    let q_size = seq_q * num_heads * head_dim * 2; // half precision
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4; // float output
    
    let mut h_q = vec![0.0f32; q_size / 2];
    let mut h_k = vec![0.0f32; k_size / 2];
    
    // Initialize with deterministic values
    for i in 0..q_size / 2 {
        h_q[i] = (i as f32 * 0.1).sin();
    }
    for i in 0..k_size / 2 {
        h_k[i] = (i as f32 * 0.05).cos();
    }
    
    println!("Configuration:");
    println!("  seq_q: {}, seq_k: {}", seq_q, seq_k);
    println!("  num_heads: {}, head_dim: {}", num_heads, head_dim);
    println!("  Total output elements: {} ({} MB)", 
             seq_q * num_heads * seq_k,
             seq_q * num_heads * seq_k * 4 / (1024 * 1024));
    println!();
    
    // Benchmark baseline kernel
    println!("🏃 Running baseline kernel...");
    let baseline_time = run_kernel(
        &cuda_rt,
        &stream,
        &h_q,
        &h_k,
        seq_q, seq_k, num_heads, head_dim, rope_base,
        "/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax.ptx",
        "_Z22fused_attention_kernelfPK6__halfS1_S1_Pfiiiifi"
    )?;
    
    // Benchmark tiled kernel
    println!("🏃 Running tiled kernel...");
    let tiled_time = run_kernel(
        &cuda_rt,
        &stream,
        &h_q,
        &h_k,
        seq_q, seq_k, num_heads, head_dim, rope_base,
        "/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax_tiled.ptx",
        "_Z28fused_attention_kernel_tiledfPK6__halfS1_S1_Pfiiiifi"
    )?;
    
    // Calculate metrics
    let total_tokens = seq_q as f64 * num_heads as f64;
    let baseline_mtps = total_tokens / (baseline_time.as_nanos() as f64) * 1e9;
    let tiled_mtps = total_tokens / (tiled_time.as_nanos() as f64) * 1e9;
    let speedup = baseline_time.as_secs_f64() / tiled_time.as_secs_f64();
    
    println!("\n📊 Results:");
    println!("  Baseline kernel: {:.3} ms ({:.2} M tokens/sec)", 
             baseline_time.as_secs_f64() * 1000.0, baseline_mtps);
    println!("  Tiled kernel:    {:.3} ms ({:.2} M tokens/sec)", 
             tiled_time.as_secs_f64() * 1000.0, tiled_mtps);
    println!("  Speedup:         {:.2}x", speedup);
    
    if speedup > 1.5 {
        println!("\n✅ Tiled kernel shows significant performance improvement!");
    } else {
        println!("\n⚠️  Speedup is modest - consider larger sequences for better utilization");
    }
    
    Ok(())
}

fn run_kernel(
    cuda_rt: &CudaRuntime,
    stream: &Arc<cudarc::driver::CudaStream>,
    h_q: &[f32],
    h_k: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
    ptx_path: &str,
    kernel_name: &str
) -> Result<Duration, Box<dyn std::error::Error>> {
    // Load kernel from PTX
    let kernel = FusedAttentionKernelBuilder::new(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        Arc::clone(stream),
    )
    .build_from_ptx_file(ptx_path, kernel_name)?;
    
    // Allocate device memory using DeviceBuffer
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4;
    
    let mut q_d = vec![0u8; q_size];
    let mut k_d = vec![0u8; k_size];
    let mut s_d = vec![0u8; s_size];
    
    // Copy to device (convert f32 to half)
    unsafe {
        for i in 0..q_size / 2 {
            let q_half = h_q[i] as u16;
            q_d[i * 2] = q_half as u8;
            q_d[i * 2 + 1] = (q_half >> 8) as u8;
        }
        for i in 0..k_size / 2 {
            let k_half = h_k[i] as u16;
            k_d[i * 2] = k_half as u8;
            k_d[i * 2 + 1] = (k_half >> 8) as u8;
        }
    }
    
    // Get device pointers (u64 addresses)
    let q_ptr = q_d.as_ptr() as u64;
    let k_ptr = k_d.as_ptr() as u64;
    let s_ptr = s_d.as_mut_ptr() as u64;
    
    // Warm-up run
    kernel.launch(
        1.0 / (head_dim as f32).sqrt(), // scale
        q_ptr, k_ptr, 0u64, s_ptr, // v_ptr is unused (null)
        seq_q, seq_k, num_heads, head_dim, rope_base, seq_q.max(seq_k),
    )?;
    stream.synchronize()?;
    
    // Benchmark run
    let start = Instant::now();
    kernel.launch(
        1.0 / (head_dim as f32).sqrt(),
        q_ptr, k_ptr, 0u64, s_ptr,
        seq_q, seq_k, num_heads, head_dim, rope_base, seq_q.max(seq_k),
    )?;
    stream.synchronize()?;
    let duration = start.elapsed();
    
    Ok(duration)
}
