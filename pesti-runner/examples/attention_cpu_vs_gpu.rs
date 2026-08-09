//! Compare CPU vs GPU attention implementations
//! Usage: cargo run --package pesti-runner --features cuda --example attention_cpu_vs_gpu

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use pesti_runner::CudaRuntime;
    use pesti_runner::kernel::Kvcache;
    use pesti_runner::kernel::attention::{
        AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel,
        GemmBasedAttentionKernel,
    };
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use pesti_runner::kernel::gemm::{CudaGemmKernelBuilder, GemmArch};
    use pesti_runner::kernel::memory::CudaMemoryBackend;
    use std::sync::Arc;
    use std::time::Instant;

    println!("=== CPU vs GPU Attention Comparison ===\n");

    // Initialize CUDA
    let rt = CudaRuntime::new(0)?;
    let stream = rt.new_stream()?;
    let info = rt.device_info();

    println!(
        "Device: {} (sm_{}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    );

    // Build GEMM kernel for GPU attention
    let gemm_kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    let backend = Arc::new(CudaMemoryBackend::with_device_info(
        stream.clone(),
        info.clone(),
    ));
    let gpu_kernel = GemmBasedAttentionKernel::new(gemm_kernel, backend);

    // Config
    let num_heads = 8;
    let head_dim = 64;
    let seq_len = 256;
    let query_len = 1;

    println!(
        "\nConfig: {} heads, {} dim, seq={}",
        num_heads, head_dim, seq_len
    );

    // Generate test data
    let q_size = query_len * num_heads * head_dim;
    let kv_size = num_heads * head_dim * seq_len;

    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
        .collect();

    let kv_host: Vec<f16> = (0..kv_size * 2) // K and V
        .map(|i| f16::from_f32((i as f32 * 0.05).cos()))
        .collect();

    // Create CPU Kvcache
    let mut kvc = Kvcache::new(num_heads, num_heads, head_dim, seq_len, false);
    let head_stride = num_heads * head_dim;
    for pos in 0..seq_len {
        let k_start = pos * head_stride;
        let v_start = pos * head_stride + kv_size / 2;
        kvc.write_kv_at(
            pos,
            &kv_host[k_start..k_start + head_stride],
            &kv_host[v_start..v_start + head_stride],
        )?;
    }

    let q_buf = DeviceBuffer::from_host(q_host);
    let config = AttentionConfig::default()
        .with_num_heads(num_heads)
        .with_head_dim(head_dim)
        .with_max_seq(seq_len);

    // Warmup
    let _ =
        CpuAttentionKernel::new(AttentionArch::Cpu).forward(&q_buf, &kvc, &kvc, None, &config)?;
    let _ = gpu_kernel.forward(&q_buf, &kvc, &kvc, None, &config)?;

    // Benchmark CPU
    let start = Instant::now();
    for _ in 0..10 {
        let _ = CpuAttentionKernel::new(AttentionArch::Cpu)
            .forward(&q_buf, &kvc, &kvc, None, &config)?;
    }
    let cpu_time = start.elapsed() / 10;

    // Benchmark GPU
    let start = Instant::now();
    for _ in 0..10 {
        let _ = gpu_kernel.forward(&q_buf, &kvc, &kvc, None, &config)?;
    }
    let gpu_time = start.elapsed() / 10;

    println!("\n=== Results ===");
    println!("CPU: {:.2}ms/iter", cpu_time.as_secs_f64() * 1000.0);
    println!("GPU: {:.2}ms/iter", gpu_time.as_secs_f64() * 1000.0);
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("Speedup: {:.1}x", speedup);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("⚠️  Requires --features cuda");
}
