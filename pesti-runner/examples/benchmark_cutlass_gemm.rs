//! Benchmark CUTLASS GEMM vs CPU fallback

use half::f16;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::gemm::{GemmArch, GemmKernel};
use pesti_runner::kernel::gemm_cutlass::CutlassGemmKernel;
use std::time::Instant;

fn main() {
    println!("=== CUTLASS GEMM Benchmark ===\n");

    // Test matrix sizes (typical LLM shapes)
    let test_sizes = vec![
        (64, 128, 512),   // Small batch
        (128, 256, 1024), // Medium
        (256, 512, 2048), // Large
    ];

    for (m, n, k) in test_sizes {
        println!("Testing {} x {} x {}", m, n, k);

        // Generate deterministic matrices for reproducibility
        let a_host: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
            .collect();
        let b_host: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32((i as f32 * 0.05).cos()))
            .collect();

        // Allocate buffers
        let a_buf = DeviceBuffer::from_cpu_device(a_host.clone());
        let b_buf = DeviceBuffer::from_cpu_device(b_host.clone());
        let mut c_buf = DeviceBuffer::zeros_cpu_device(m * n);

        // Create kernel
        let kernel = CutlassGemmKernel::new(GemmArch::Wgmma);
        assert!(kernel.is_available(), "Kernel should be available");

        // Warmup
        kernel
            .matmul(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)
            .expect("GEMM should succeed");

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            kernel
                .matmul(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)
                .expect("GEMM should succeed");
        }
        let elapsed = start.elapsed();

        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
        let tflops = (2.0 * m as f64 * n as f64 * k as f64)
            / (elapsed.as_secs_f64() * 1e12);

        println!(
            "  Avg: {:.2}ms, Throughput: {:.2} TFLOPS",
            avg_ms, tflops
        );
    }

    println!("\n=== Complete ===");
}
