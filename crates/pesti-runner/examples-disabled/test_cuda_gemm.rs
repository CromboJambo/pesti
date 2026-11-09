//! Test CUDA GEMM kernel on consumer NVIDIA GPUs.
//!
//! Tests both RTX 4070 Ti SUPER (sm_8.9) and RTX 5060 Ti (sm_12.0).

use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use pesti_runner::cuda_runtime::CudaDeviceInfo;
use pesti_runner::kernel::MemoryBackend;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::gemm::{CudaGemmKernelBuilder, GemmArch, GemmKernel};
use std::sync::Arc;

/// Estimate compute capability from GPU name.
fn estimate_cc_from_name(name: &str) -> (i32, i32) {
    if name.contains("4070") || name.contains("4080") || name.contains("4090") {
        // Ada Lovelace architecture
        (8, 9)
    } else if name.contains("5060")
        || name.contains("5070")
        || name.contains("5080")
        || name.contains("5090")
    {
        // Blackwell architecture (consumer RTX 50 series)
        (12, 0)
    } else if name.contains("3070") || name.contains("3080") || name.contains("3090") {
        // Ampere architecture
        (8, 6)
    } else if name.contains("2080") || name.contains("2080 Ti") {
        // Turing architecture
        (7, 5)
    } else {
        // Default fallback - try to detect from cudarc later
        (0, 0)
    }
}

/// Initialize CUDA runtime and return device info + context + stream.
fn init_cuda(
    device_idx: usize,
) -> Result<(Arc<CudaContext>, Arc<CudaStream>, CudaDeviceInfo), Box<dyn std::error::Error>> {
    // First try NVML for reliable device detection
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(device_count) = nvml.device_count() {
            if device_count > device_idx as u32 {
                println!("Found {} CUDA devices via NVML", device_count);

                // Use specified device
                let device = device_idx;

                if let Ok(nvml_device) = nvml.device_by_index(device as u32) {
                    if let Ok(name) = nvml_device.name() {
                        println!("Using device {}: {}", device, name);

                        if let Ok(mem_info) = nvml_device.memory_info() {
                            // Estimate compute capability from GPU name
                            let cc = estimate_cc_from_name(&name);
                            println!("Compute capability: sm_{}.{}", cc.0, cc.1);

                            // Create context via cudarc - note: default_stream() returns Arc<CudaStream>
                            let context = CudaContext::new(device)?;
                            let stream = context.default_stream(); // Already Arc<CudaStream>

                            let device_info = CudaDeviceInfo {
                                ordinal: device,
                                name: name.to_string(),
                                compute_capability: cc,
                                total_memory: mem_info.total,
                                free_memory: mem_info.free,
                            };

                            return Ok((context, stream, device_info));
                        }
                    }
                }
            }
        }
    }

    // Fallback to cudarc context API
    let ordinal = device_idx;
    loop {
        match CudaContext::new(ordinal) {
            Ok(ctx) => {
                let name = ctx.name().unwrap_or_default();
                let cc = ctx.compute_capability().unwrap_or((0, 0));
                let (free, total) = ctx.mem_get_info().unwrap_or((0, 0));

                println!(
                    "Found device {}: {} (sm_{}.{}), {} MiB free",
                    ordinal,
                    name,
                    cc.0,
                    cc.1,
                    free / (1024 * 1024)
                );

                let context = ctx;
                let stream = context.default_stream(); // Already Arc<CudaStream>

                let device_info = CudaDeviceInfo {
                    ordinal,
                    name: name.clone(),
                    compute_capability: cc,
                    total_memory: total as u64,
                    free_memory: free as u64,
                };

                return Ok((context, stream, device_info));
            }
            Err(_) => break,
        }
    }

    Err("No CUDA devices found".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing CUDA GEMM on Consumer NVIDIA GPUs ===\n");

    // Use device index from environment variable or default to 0
    let device_idx = std::env::var("CUDA_DEVICE")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<usize>()
        .expect("Invalid CUDA_DEVICE value");

    // Initialize CUDA
    let (context, stream, device_info) = init_cuda(device_idx)?;

    // Test with mma.sync architecture (works on sm_8.9 and sm_12.0)
    let arch = GemmArch::Mma;
    println!("\nTesting GEMM kernel: {}", arch.name());

    // Create builder and build kernel
    let builder =
        CudaGemmKernelBuilder::new(arch, context.clone(), stream.clone(), device_info.clone());
    let kernel = builder.build()?;

    println!("✓ Kernel built successfully");
    println!("  Architecture: {:?}", kernel.arch());
    println!("  Available: {}", kernel.is_available());

    // Test dimensions (Q @ K^T for attention)
    let m = 32; // query_seq_len * num_heads
    let k = 64; // head_dim
    let n = 128; // cache_seq_len

    println!("\nTest dimensions: {} x {} → {}", m, k, n);

    // Create test matrices (A: [m, k], B: [k, n])
    let a_data: Vec<f16> = (0..(m * k))
        .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
        .collect();

    let b_data: Vec<f16> = (0..(k * n))
        .map(|i| f16::from_f32((i % 5) as f32 * 0.1))
        .collect();

    // Create memory backend with proper device info (use the one from NVML)
    let backend = pesti_runner::kernel::memory::CudaMemoryBackend::with_device_info(
        stream.clone(),
        device_info,
    );

    // Allocate device buffers manually using the backend
    let a_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 2) };
    let b_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 2) };

    let a_handle = backend.alloc(a_data.len() * 2)?;
    let b_handle = backend.alloc(b_data.len() * 2)?;
    let c_handle = backend.alloc(m * n * 4)?; // f32 = 4 bytes

    // Copy data to device
    backend.h2d(a_bytes, a_handle)?;
    backend.h2d(b_bytes, b_handle)?;

    // Create DeviceBuffer wrappers
    let a = DeviceBuffer::from_backend(a_handle, a_data.len());
    let b = DeviceBuffer::from_backend(b_handle, b_data.len());
    let mut c = DeviceBuffer::from_backend(c_handle, m * n);

    println!(
        "✓ Allocated device buffers: A[{}], B[{}], C[{}]",
        a.len(),
        b.len(),
        c.len()
    );

    // Launch GEMM: C = A @ B (alpha=1.0, beta=0.0)
    println!("\nLaunching kernel...");
    kernel.matmul(1.0, &a, &b, 0.0, &mut c, m, n, k)?;

    // Synchronize to ensure completion
    stream.synchronize()?;
    println!("✓ Kernel completed successfully");

    // Read back result
    let mut c_host = vec![0f32; m * n];
    let c_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, c_host.len() * 4) };
    backend.d2h(c_handle, c_bytes)?;

    // Verify first few values (manual calculation for small test)
    let expected_00 = (0..k)
        .map(|i| a_data[i] * b_data[i * n])
        .sum::<f16>()
        .to_f32();
    let actual_00 = c_host[0];

    println!("\nVerification:");
    println!("  C[0,0] expected: {:.4}", expected_00);
    println!("  C[0,0] actual:   {:.4}", actual_00);
    println!("  Difference:      {:.6}", (expected_00 - actual_00).abs());

    if (expected_00 - actual_00).abs() < 0.1 {
        println!("\n✓ GEMM test PASSED!");
    } else {
        println!("\n✗ GEMM test FAILED - values don't match");
        std::process::exit(1);
    }

    // Performance benchmark (optional)
    let iterations = 100;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        kernel.matmul(1.0, &a, &b, 0.0, &mut c, m, n, k)?;
    }

    stream.synchronize()?;
    let duration = start.elapsed();

    println!("\nPerformance:");
    println!("  {} iterations in {:?}", iterations, duration);
    println!(
        "  Average: {:.2} μs per GEMM",
        duration.as_secs_f64() * 1_000_000.0 / iterations as f64
    );

    Ok(())
}
