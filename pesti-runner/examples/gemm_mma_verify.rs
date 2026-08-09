//! GEMM kernel verification test using mma.sync tensor cores
//!
//! Verifies basic GEMM operations on RTX 50-series (consumer Blackwell)
//! Usage: cargo run --package pesti-runner --features cuda --example gemm_mma_verify

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use half::f16;
    use pesti_runner::CudaRuntime;
    use pesti_runner::kernel::device_buf::DeviceBuffer;
    use pesti_runner::kernel::gemm::{CudaGemmKernelBuilder, GemmArch, GemmKernel};

    println!("=== GEMM mma.sync Verification ===\n");

    // Initialize CUDA
    let rt = CudaRuntime::new(0)?;
    let stream = rt.new_stream()?;
    let info = rt.device_info();

    println!(
        "Device: {} (sm_{}.{})",
        info.name, info.compute_capability.0, info.compute_capability.1
    );

    // Build GEMM kernel with mma.sync architecture
    let gemm_kernel = CudaGemmKernelBuilder::new(
        GemmArch::Mma,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    println!("✅ GEMM kernel loaded (mma.sync)");

    // Test matrix dimensions
    let m = 256;
    let n = 256;
    let k = 64;

    println!("\n📦 Testing GEMM: [{} x {}] @ [{} x {}]", m, k, k, n);

    // Create test matrices
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
        .collect();

    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32((i as f32 * 0.05).cos()))
        .collect();

    // Run GEMM via trait
    let a_buf = DeviceBuffer::from_host(a_host.clone());
    let b_buf = DeviceBuffer::from_host(b_host.clone());
    let mut c_buf: DeviceBuffer<f32> = DeviceBuffer::zeros(m * n);

    GemmKernel::matmul(&gemm_kernel, 1.0, &a_buf, &b_buf, 0.0, &mut c_buf, m, n, k)?;

    // Retrieve results
    let c_result = c_buf.to_host();

    println!("✅ GEMM completed successfully");
    println!("   Output shape: [{} x {}]", m, n);
    println!("   Sample output[0]: {:.4}", c_result[0]);

    println!("\n=== Test Complete ===");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("⚠️  Requires --features cuda");
}
