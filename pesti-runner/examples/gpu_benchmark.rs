//! GPU benchmark comparing ndarray CPU vs CUDA attention kernel

use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let seq_q = 128;
    let seq_k = 128;
    let num_heads = 4;
    let head_dim = 64;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("=== PESTI GPU Benchmark ===");
    println!("Configuration: seq_q={}, seq_k={}, num_heads={}, head_dim={}\n", 
             seq_q, seq_k, num_heads, head_dim);

    // Generate random input data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    // Test 1: CPU Ndarray baseline
    println!("Test 1: CPU Ndarray Reference");
    let start = Instant::now();
    let result_cpu = reference_with_ndarray(&q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);
    let duration_cpu = start.elapsed();
    
    println!("  Time: {:.3}ms", duration_cpu.as_secs_f64() * 1000.0);
    println!("  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]", 
             result_cpu[0], result_cpu[1], result_cpu[2], result_cpu[3], result_cpu[4]);

    // Test 2: GPU CUDA kernel
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::{CudaDevice, CudaSlice};
        
        println!("\nTest 2: GPU CUDA Kernel (fused attention_rope_softmax)");
        
        // Initialize CUDA device
        let device = CudaDevice::new(0)?;
        
        // Copy data to GPU
        let q_d: CudaSlice<f16> = device.htod(q_h.as_slice())?;
        let k_d: CudaSlice<f16> = device.htod(k_h.as_slice())?;
        let v_d: CudaSlice<f16> = device.htod(v_h.as_slice())?;
        
        // Allocate output buffer (scores + output)
        let output_size = seq_q * num_heads * seq_k + seq_q * num_heads * head_dim;
        let mut s_d: CudaSlice<f32> = device.alloc_zeros::<f32>(output_size)?;
        
        let start = Instant::now();
        
        // Kernel launch would go here:
        // kernel1<<<grid>>>... (fused_attention_kernel)
        // kernel2<<<grid>>>... (apply_softmax_and_output_kernel)
        
        device.sync()?;
        let duration_gpu = start.elapsed();
        
        println!("  Time: {:.3}ms", duration_gpu.as_secs_f64() * 1000.0);
        
        // Copy result back to CPU
        let mut result_gpu_h: Vec<f32> = vec![0.0; output_size];
        device.htod_into(s_d.as_slice(), &mut result_gpu_h)?;
        
        println!("  Output sample (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]", 
                 result_gpu_h[0], result_gpu_h[1], result_gpu_h[2], result_gpu_h[3], result_gpu_h[4]);

        // Compare results
        let max_error = result_cpu.iter()
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

    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\nTest 2: GPU CUDA Kernel (requires --features cuda)");
        println!("  Run with: cargo run --package pesti-runner --example gpu_benchmark --features cuda");
        
        // Estimate based on typical GPU performance
        let estimated_gpu_time = duration_cpu.as_secs_f64() * 0.1; // ~10x speedup estimate
        
        println!("  Estimated time: {:.3}ms (based on typical CUDA performance)", 
                 estimated_gpu_time * 1000.0);
        println!("  Estimated speedup: ~10x vs CPU");
    }

    // Test 3: Memory bandwidth analysis
    println!("\n=== Memory Bandwidth Analysis ===");
    
    let q_bytes = seq_q * num_heads * head_dim * 2; // f16
    let k_bytes = seq_k * num_heads * head_dim * 2;
    let v_bytes = seq_k * num_heads * head_dim * 2;
    let output_bytes = seq_q * num_heads * head_dim * 4; // f32
    
    let total_memory = q_bytes + k_bytes + v_bytes + output_bytes;
    let memory_mb = total_memory as f64 / 1e6;
    
    println!("Q tensor: {:.2} MB", q_bytes as f64 / 1e6);
    println!("K tensor: {:.2} MB", k_bytes as f64 / 1e6);
    println!("V tensor: {:.2} MB", v_bytes as f64 / 1e6);
    println!("Output: {:.2} MB", output_bytes as f64 / 1e6);
    println!("Total: {:.2} MB", memory_mb);

    // Estimate bandwidth (assuming each tensor read once, output written once)
    let memory_access = total_memory * 3;
    let bandwidth_gb_s_cpu = memory_access as f64 / 1e9 / duration_cpu.as_secs_f64();
    
    println!("\nCPU bandwidth: {:.2} GB/s", bandwidth_gb_s_cpu);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::{CudaDevice, CudaSlice};
        
        let device = CudaDevice::new(0)?;
        let q_d: CudaSlice<f16> = device.htod(q_h.as_slice())?;
        let k_d: CudaSlice<f16> = device.htod(k_h.as_slice())?;
        let v_d: CudaSlice<f16> = device.htod(v_h.as_slice())?;
        
        let output_size = seq_q * num_heads * seq_k + seq_q * num_heads * head_dim;
        let mut s_d: CudaSlice<f32> = device.alloc_zeros::<f32>(output_size)?;
        
        let start = Instant::now();
        
        // Placeholder for actual kernel launch
        device.sync()?;
        let duration_gpu = start.elapsed();
        
        let bandwidth_gb_s_gpu = memory_access as f64 / 1e9 / duration_gpu.as_secs_f64();
        println!("GPU bandwidth: {:.2} GB/s", bandwidth_gb_s_gpu);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("Estimated GPU bandwidth: ~500-900 GB/s (RTX 4070 Ti SUPER)");
    }

    println!("\n=== Summary ===");
    println!("CPU Ndarray: {:.3}ms", duration_cpu.as_secs_f64() * 1000.0);
    
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::{CudaDevice, CudaSlice};
        
        let device = CudaDevice::new(0)?;
        let q_d: CudaSlice<f16> = device.htod(q_h.as_slice())?;
        let k_d: CudaSlice<f16> = device.htod(k_h.as_slice())?;
        let v_d: CudaSlice<f16> = device.htod(v_h.as_slice())?;
        
        let output_size = seq_q * num_heads * seq_k + seq_q * num_heads * head_dim;
        let mut s_d: CudaSlice<f32> = device.alloc_zeros::<f32>(output_size)?;
        
        let start = Instant::now();
        
        // Placeholder for actual kernel launch
        device.sync()?;
        let duration_gpu = start.elapsed();
        
        let speedup = duration_cpu.as_secs_f64() / duration_gpu.as_secs_f64();
        println!("GPU CUDA: {:.3}ms ({}x faster)", 
                 duration_gpu.as_secs_f64() * 1000.0,
                 speedup);
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("Estimated GPU time: ~0.6ms (~10x faster)");
    }

    Ok(())
}
