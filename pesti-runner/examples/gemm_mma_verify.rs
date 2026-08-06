//! Verify the real mma.sync GEMM kernel on GPU: numerical correctness vs CPU
//! reference, then benchmark GPU vs CPU throughput.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda --example gemm_mma_verify
//!
//! Builds CudaGemmKernel directly through the same path InferenceEngine uses
//! (CudaRuntime + CudaGemmKernelBuilder + CudaMemoryBackend), then runs the
//! kernel on device 1 (RTX 5060 Ti, sm_120) and device 0 (RTX 4070 Ti SUPER,
//! sm_8.9). Both are consumer GPUs with neither wgmma nor tcgen05, so the
//! kernel must select GemmArch::Mma (mma.sync).

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::memory::CudaMemoryBackend;
use pesti_runner::kernel::{
    CudaGemmKernelBuilder, GemmArch, GemmError, GemmKernel, MemoryBackend,
};
use std::sync::Arc;
use std::time::Instant;

fn cpu_gemm(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn run_device(
    ordinal: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Device {ordinal} ---");

    let rt = match CudaRuntime::new(ordinal) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            println!("  ❌ CudaRuntime::new({ordinal}) failed: {e}");
            return Ok(());
        }
    };
    let stream = rt.new_stream()?;

    // Build the GEMM kernel with the arch the engine would select for this
    // device: wgmma (Hopper) > tcgen05 (datacenter Blackwell) > mma.sync.
    let info = rt.device_info();
    let arch = if info.supports_wgmma() {
        GemmArch::Wgmma
    } else if info.supports_tcgen05() {
        GemmArch::Tcgen05
    } else {
        GemmArch::Mma
    };
    println!(
        "  Device: {} (sm_{}.{}) → arch {:?}",
        info.name, info.compute_capability.0, info.compute_capability.1, arch
    );

    let kernel = match CudaGemmKernelBuilder::new(arch, rt.context().clone(), stream.clone(), info.clone()).build() {
        Ok(k) => k,
        Err(GemmError::UnsupportedArch(msg)) => {
            println!("  ⏭️  {msg}");
            return Ok(());
        }
        Err(e) => {
            println!("  ❌ kernel build failed: {e}");
            return Ok(());
        }
    };
    println!("  ✅ Kernel loaded: {} (arch {:?})", "gemm_mma_kernel", arch);

    // Memory backend on the same stream (with real device info so
    // allocation capacity is nonzero)
    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());

    // Test data: A [m x k], B [k x n] f16
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 7) as f32) * 0.5 - 1.5))
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 11) as f32) * 0.25 - 1.0))
        .collect();

    let a_buf = DeviceBuffer::from_host_device(&backend, &a_host)?;
    let b_buf = DeviceBuffer::from_host_device(&backend, &b_host)?;
    let mut c_buf = DeviceBuffer::zeros_device(&backend, m * n)?;

    // Warmup
    kernel.matmul(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
    stream.synchronize()?;

    // Timed GPU run
    let t0 = Instant::now();
    kernel.matmul(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;
    stream.synchronize()?;
    let gpu_time = t0.elapsed();

    let c_gpu = c_buf.to_host_vec(&backend)?;

    // CPU reference
    let t1 = Instant::now();
    let c_cpu = cpu_gemm(&a_host, &b_host, m, n, k);
    let cpu_time = t1.elapsed();

    // Compare
    let mut max_err = 0.0f32;
    let mut n_bad = 0usize;
    for i in 0..m * n {
        let err = (c_gpu[i] - c_cpu[i]).abs();
        if err > max_err {
            max_err = err;
        }
        // tolerance: f16 inputs accumulate fp32; allow 1e-1 relative-ish
        if err > 0.5 && c_cpu[i].abs() > 1e-3 {
            n_bad += 1;
        }
    }

    println!("  GPU GEMM: {:.3} ms", gpu_time.as_secs_f64() * 1000.0);
    println!("  CPU GEMM: {:.3} ms", cpu_time.as_secs_f64() * 1000.0);
    println!("  Speedup: {:.1}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    println!("  Max error vs CPU: {:.3e}", max_err);
    println!("  Elements > 0.5 abs error: {n_bad}/{}", m * n);
    println!("  C[0][0] = {:.4} (cpu {:.4})", c_gpu[0], c_cpu[0]);

    if max_err < 1e-1 {
        println!("  ✅ CORRECT");
    } else {
        println!("  ❌ INCORRECT");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mma.sync GEMM verification ===\n");

    // Small correctness case with odd dims to exercise bounds checks
    run_device(0, 40, 24, 32)?;
    run_device(1, 40, 24, 32)?;

    // Benchmark case
    println!("\n--- Benchmark: 1024x1024x1024 ---");
    run_device(1, 1024, 1024, 1024)?;

    Ok(())
}
