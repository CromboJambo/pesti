//! Numerical Conformance Test vs llama.cpp Baseline
//!
//! This test validates that our CUDA kernels produce numerically equivalent results
//! to a trusted CPU reference implementation (llama.cpp-style GEMM).

use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::{memory::{CudaMemoryBackend, MemoryBackend}, GemmKernel};
use pesti_runner::kernel::{CudaGemmKernelBuilder, GemmArch};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Numerical Conformance Test ===\n");

    // Initialize CUDA runtime
    let runtime = CudaRuntime::for_default_device()?;
    let device_info = runtime.device_info().clone();
    let stream: Arc<cudarc::driver::safe::CudaStream> = runtime.new_stream()?;

    println!("Device: {}", device_info.name);
    println!(
        "Compute Capability: {}.{}",
        device_info.compute_capability.0, device_info.compute_capability.1
    );
    println!();

    // Create CUDA memory backend with proper device info initialization
    let mut backend = CudaMemoryBackend::new(stream.clone());
    backend.try_init_device_info();

    // Build CUDA GEMM kernel for Ada Lovelace (mma.sync)
    println!("Building CUDA GEMM kernel (mma.sync for sm_8.9)...");
    let gemm_kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        stream.context().clone(), // Use stream's context
        stream.clone(),
        device_info.clone(),
    )
    .build()?;

    println!("✅ CUDA GEMM kernel built successfully");
    println!("  Architecture: {}", gemm_kernel.arch().name());
    println!();

    // Test parameters (matching llama.cpp typical dimensions)
    let m = 64usize; // Batch size × sequence length
    let n = 2048usize; // Hidden dimension (Qwen2.5-0.5B intermediate)
    let k = 512usize; // Intermediate dimension

    println!("Test Configuration:");
    println!("  GEMM dimensions: {} × {} × {}", m, k, n);
    println!(
        "  Input size (f16): {:.2} MB",
        (m * k + k * n) as f64 * 2.0 / 1e6
    );
    println!("  Output size (f32): {:.2} MB", (m * n) as f64 * 4.0 / 1e6);
    println!();

    // Generate deterministic test data using a simple LCG PRNG
    let mut rng = seed_rng(42); // Fixed seed for reproducibility
    let a_host: Vec<half::f16> = (0..m * k)
        .map(|_| half::f16::from_f32(lcg_f32(&mut rng)))
        .collect();

    let mut rng = seed_rng(123); // Different seed for B matrix
    let b_host: Vec<half::f16> = (0..k * n)
        .map(|_| half::f16::from_f32(lcg_f32(&mut rng)))
        .collect();

    println!("Generating test data with fixed seeds for reproducibility...");

    // Compute reference result on CPU (naive GEMM - same as llama.cpp)
    println!("Computing CPU reference result...");
    let c_cpu = gemm_naive_f32(&a_host, &b_host, m, k, n);

    // Upload data to GPU
    println!("Uploading tensors to device...");
    let a_buf = {
        let data: Vec<half::f16> = a_host.clone();
        pesti_runner::kernel::DeviceBuffer::from_host_device(&backend, &data)?
    };

    let b_buf = {
        let data: Vec<half::f16> = b_host.clone();
        pesti_runner::kernel::DeviceBuffer::from_host_device(&backend, &data)?
    };

    let mut c_buf = {
        let data: Vec<f32> = vec![0f32; m * n];
        pesti_runner::kernel::DeviceBuffer::from_host_device(&backend, &data)?
    };

    println!("Running CUDA GEMM kernel (C = 1.0 * A @ B + 0.0 * C)...");

    // **ACTUALLY CALL THE CUDA KERNEL** - this is the missing piece!
    gemm_kernel.matmul(
        1.0, // alpha
        &a_buf, &b_buf, 0.0, // beta (don't add old C values)
        &mut c_buf, m, n, k,
    )?;

    // Sync to ensure kernel completion
    backend.sync()?;

    // Read back results from GPU
    let c_gpu = {
        let result = c_buf.to_host_vec(&backend)?;
        result
    };

    println!("Comparing CPU vs GPU results...");

    // Compute numerical error metrics
    let mut max_error = 0.0f32;
    let mut mean_error = 0.0f32;
    let mut relative_errors: Vec<f32> = Vec::new();

    for i in 0..m * n {
        let abs_error = (c_cpu[i] - c_gpu[i]).abs();
        max_error = max_error.max(abs_error);
        mean_error += abs_error;

        // Relative error (avoid division by zero)
        let denom = c_cpu[i].abs().max(1e-6f32);
        relative_errors.push(abs_error / denom);
    }

    mean_error /= (m * n) as f32;
    let max_relative_error: f32 = relative_errors
        .iter()
        .cloned()
        .fold(0.0f32, |a, b| a.max(b));

    println!("\n=== Numerical Conformance Results ===");
    println!(
        "  Max absolute error:        {:.8} (target: < 1e-4)",
        max_error
    );
    println!("  Mean absolute error:       {:.8}", mean_error);
    println!(
        "  Max relative error:        {:.8} (target: < 1e-3)",
        max_relative_error
    );
    println!();

    // Check if results are within acceptable tolerance
    let tolerance_abs = 1e-4f32; // llama.cpp typically achieves < 1e-4
    let tolerance_rel = 1e-3f32; // Relative error < 0.1%

    if max_error <= tolerance_abs && max_relative_error <= tolerance_rel {
        println!("✅ PASS: Results within numerical tolerance!");
        println!(
            "  Absolute error {:.4} ≤ {} tolerance",
            max_error, tolerance_abs
        );
        println!(
            "  Relative error {:.4} ≤ {} tolerance",
            max_relative_error, tolerance_rel
        );
    } else {
        println!("❌ FAIL: Results exceed numerical tolerance!");
        if max_error > tolerance_abs {
            println!(
                "  Absolute error {:.4} > {} tolerance",
                max_error, tolerance_abs
            );
        }
        if max_relative_error > tolerance_rel {
            println!(
                "  Relative error {:.4} > {} tolerance",
                max_relative_error, tolerance_rel
            );
        }
        
        // Print sample comparisons even on failure for debugging
        println!("\nSample Output Comparisons (first 5 elements):");
        for i in 0..5 {
            println!(
                "  [{}] CPU: {:8.4} | GPU: {:8.4} | Diff: {:8.4e}",
                i,
                c_cpu[i],
                c_gpu[i],
                (c_cpu[i] - c_gpu[i]).abs()
            );
        }
        
        std::process::exit(1);
    }

    // Print some sample comparisons
    println!("\nSample Output Comparisons (first 5 elements):");
    for i in 0..5 {
        println!(
            "  [{}] CPU: {:8.4} | GPU: {:8.4} | Diff: {:8.4e}",
            i,
            c_cpu[i],
            c_gpu[i],
            (c_cpu[i] - c_gpu[i]).abs()
        );
    }

    Ok(())
}

// Simple LCG PRNG for deterministic test data
fn seed_rng(seed: u64) -> u64 {
    seed
}

fn lcg_f32(rng: &mut u64) -> f32 {
    // Linear congruential generator
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let value = (*rng >> 32) as f32 / (u32::MAX as f32);
    // Scale to [-1, 1] range
    value * 2.0 - 1.0
}

// Naive GEMM implementation (CPU reference - matches llama.cpp behavior)
fn gemm_naive_f32(a: &[half::f16], b: &[half::f16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                // A is row-major: [i, l] -> i*k + l
                // B is column-major (llama.cpp convention): [l, j] -> l*n + j
                let a_val = a[i * k + l].to_f32();
                let b_val = b[l * n + j].to_f32();
                sum += a_val * b_val;
            }
            c[i * n + j] = sum;
        }
    }

    c
}
