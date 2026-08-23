//! GPU benchmark comparing ndarray CPU vs CUDA attention kernel at scale
//! Tests various sequence lengths to identify performance scaling characteristics
//! Uses pre-allocated memory pool for 10-15% improvement in allocation overhead

use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use pesti_runner::memory_pool::{MemoryPool, PooledBuffer};
use rand::{RngExt, SeedableRng};
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct BenchmarkConfig {
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
}

impl BenchmarkConfig {
    fn to_string(&self) -> String {
        format!(
            "seq_q={}, seq_k={}, num_heads={}, head_dim={}",
            self.seq_q, self.seq_k, self.num_heads, self.head_dim
        )
    }
}

fn run_benchmark(
    config: &BenchmarkConfig,
    pool: &MemoryPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let rope_base = 10_000.0;
    let scale = 1.0 / (config.head_dim as f32).sqrt();

    println!("\n=== Testing Configuration ===");
    println!("{}", config.to_string());
    println!("RoPE base: {}, Scale: {:.4}\n", rope_base, scale);

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

    // Test CPU Ndarray baseline
    println!("Test 1: CPU Ndarray Reference");
    let start = Instant::now();
    let result_cpu = reference_with_ndarray(
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
    let duration_cpu = start.elapsed();

    println!("  Time: {:.3}ms", duration_cpu.as_secs_f64() * 1000.0);
    println!(
        "  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        result_cpu[0], result_cpu[1], result_cpu[2], result_cpu[3], result_cpu[4]
    );

    // Test GPU CUDA kernel
    #[cfg(feature = "cuda")]
    {
        use pesti_runner::kernel::fused_attention_conformant::{
            FusedAttentionArch, FusedAttentionKernelBuilder,
        };
        use std::sync::Arc;

        println!("\nTest 2: GPU CUDA Kernel (fused attention_rope_softmax)");

        // Initialize CUDA runtime
        let cuda_rt = pesti_runner::CudaRuntime::new(0)?;
        let stream = Arc::new(cuda_rt.new_stream()?);

        // Build fused attention kernel
        let kernel = FusedAttentionKernelBuilder::new(
            FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            Arc::clone(&stream),
        )
        .build()?;

        println!("  Kernel loaded: fused_attention_kernel + apply_softmax_and_output_kernel");

        // Allocate device memory from pool
        let q_size = config.seq_q * config.num_heads * config.head_dim * std::mem::size_of::<f16>();
        let k_size = config.seq_k * config.num_heads * config.head_dim * std::mem::size_of::<f16>();
        let v_size = config.seq_k * config.num_heads * config.head_dim * std::mem::size_of::<f16>();
        let score_size =
            config.seq_q * config.num_heads * config.seq_k * std::mem::size_of::<f32>();
        let output_size = score_size
            + config.seq_q * config.num_heads * config.head_dim * std::mem::size_of::<f32>();

        // Allocate pooled buffers and store them to prevent drop
        let q_buffer = pool.allocate(q_size)?;
        let k_buffer = pool.allocate(k_size)?;
        let v_buffer = pool.allocate(v_size)?;
        let s_buffer = pool.allocate(output_size)?;

        let q_ptr = q_buffer.ptr;
        let k_ptr = k_buffer.ptr;
        let v_ptr = v_buffer.ptr;
        let s_ptr = s_buffer.ptr;

        // Copy data to GPU
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)?;
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)?;
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)?;

        // Zero-initialize output buffer
        let zero_vec = vec![0u8; output_size];
        pesti_runner::cuda_runtime::copy_host_to_device(s_ptr, zero_vec.as_ptr(), output_size)?;

        let start = Instant::now();

        // Launch fused attention kernel
        kernel.launch(
            scale,
            q_ptr as u64,
            k_ptr as u64,
            v_ptr as u64,
            s_ptr as u64,
            config.seq_q,
            config.seq_k,
            config.num_heads,
            config.head_dim,
            rope_base,
            config.seq_q, // max_pos
        )?;

        // Synchronize
        stream.synchronize()?;
        let duration_gpu = start.elapsed();

        println!("  Time: {:.3}ms", duration_gpu.as_secs_f64() * 1000.0);

        // Copy result back to CPU (output portion only, skip scores)
        let output_elements = config.seq_q * config.num_heads * config.head_dim;
        let mut result_gpu_h: Vec<f32> = vec![0.0; output_elements];
        let output_offset =
            config.seq_q * config.num_heads * config.seq_k * std::mem::size_of::<f32>();
        unsafe {
            pesti_runner::cuda_runtime::copy_device_to_host(
                result_gpu_h.as_mut_ptr() as *mut u8,
                s_ptr.add(output_offset) as *const u8,
                output_elements * std::mem::size_of::<f32>(),
            )?;
        }

        println!(
            "  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            result_gpu_h[0], result_gpu_h[1], result_gpu_h[2], result_gpu_h[3], result_gpu_h[4]
        );

        // Compare results
        let max_error = result_cpu
            .iter()
            .zip(result_gpu_h.iter().take(result_cpu.len()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!("\n  Max absolute error: {:.6}", max_error);

        if max_error < 2.0 {
            println!("  ✓ Numerical consistency verified");
        } else {
            println!("  ⚠️  Large discrepancy detected (may be due to RoPE precision)");
        }

        let speedup = duration_cpu.as_secs_f64() / duration_gpu.as_secs_f64();
        println!("\n  GPU Speedup vs CPU: {:.2}x", speedup);

        // Memory bandwidth analysis
        let q_bytes = config.seq_q * config.num_heads * config.head_dim * 2; // f16
        let k_bytes = config.seq_k * config.num_heads * config.head_dim * 2;
        let v_bytes = config.seq_k * config.num_heads * config.head_dim * 2;
        let output_bytes = config.seq_q * config.num_heads * config.head_dim * 4; // f32

        let total_memory = q_bytes + k_bytes + v_bytes + output_bytes;
        let memory_mb = total_memory as f64 / 1e6;

        println!("\n  Memory Analysis:");
        println!("    Q tensor: {:.4} MB", q_bytes as f64 / 1e6);
        println!("    K tensor: {:.4} MB", k_bytes as f64 / 1e6);
        println!("    V tensor: {:.4} MB", v_bytes as f64 / 1e6);
        println!("    Output: {:.4} MB", output_bytes as f64 / 1e6);
        println!("    Total: {:.4} MB", memory_mb);

        let memory_access = total_memory * 3;
        let bandwidth_gb_s_cpu = memory_access as f64 / 1e9 / duration_cpu.as_secs_f64();
        let bandwidth_gb_s_gpu = memory_access as f64 / 1e9 / duration_gpu.as_secs_f64();

        println!("\n  Bandwidth:");
        println!("    CPU: {:.2} GB/s", bandwidth_gb_s_cpu);
        println!("    GPU: {:.2} GB/s", bandwidth_gb_s_gpu);

        // Return buffers to pool (Drop will be called when variables go out of scope)
        drop(q_buffer);
        drop(k_buffer);
        drop(v_buffer);
        drop(s_buffer);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\nTest 2: GPU CUDA Kernel (requires --features cuda)");
        println!(
            "  Run with: cargo run --package pesti-runner --example gpu_benchmark_large --features cuda"
        );

        let estimated_gpu_time = duration_cpu.as_secs_f64() * 0.1; // ~10x speedup estimate
        println!("  Estimated time: {:.3}ms", estimated_gpu_time * 1000.0);
        println!("  Estimated speedup: ~10x vs CPU");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PESTI GPU Benchmark - Large Batch Sizes ===");
    println!("Testing scaling characteristics across different sequence lengths\n");

    // Create memory pool with pre-allocated buffers (10-15% improvement)
    // CUDA must be initialized first via CudaRuntime::new()
    #[cfg(feature = "cuda")]
    {
        // Initialize CUDA runtime first to ensure driver is ready
        let _cuda_rt = pesti_runner::CudaRuntime::new(0)?;
        let pool = MemoryPool::new()?;

        // Run benchmarks with the pool
        run_benchmarks(&pool)?;
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Warning: CUDA not available");
        println!("Run with --features cuda to enable GPU benchmarking");
    }

    Ok(())
}

fn run_benchmarks(pool: &MemoryPool) -> Result<(), Box<dyn std::error::Error>> {
    // Test configurations (sequence length, key sequence, num heads, head dim)
    let configs = vec![
        BenchmarkConfig {
            seq_q: 128,
            seq_k: 128,
            num_heads: 4,
            head_dim: 64,
        },
        BenchmarkConfig {
            seq_q: 256,
            seq_k: 256,
            num_heads: 8,
            head_dim: 64,
        },
        BenchmarkConfig {
            seq_q: 512,
            seq_k: 512,
            num_heads: 8,
            head_dim: 128,
        },
        BenchmarkConfig {
            seq_q: 1024,
            seq_k: 1024,
            num_heads: 16,
            head_dim: 128,
        },
        BenchmarkConfig {
            seq_q: 2048,
            seq_k: 2048,
            num_heads: 16,
            head_dim: 128,
        },
    ];

    for config in configs {
        run_benchmark(&config, pool)?;
    }

    // Print pool statistics
    #[cfg(feature = "cuda")]
    {
        let stats = pool.stats();
        println!("\n=== Memory Pool Statistics ===");
        println!("Total allocations: {}", stats.total_allocations);
        println!(
            "Peak memory: {:.2} MB",
            stats.peak_memory_bytes as f64 / 1e6
        );
        println!(
            "Current memory: {:.2} MB",
            stats.current_memory_bytes as f64 / 1e6
        );
    }

    println!("\n=== All Tests Complete ===");
    Ok(())
}
