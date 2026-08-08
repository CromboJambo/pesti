//! Minimal end-to-end GPU test

use cuda_core::CudaContext;
use half::f16;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::memory::{CudaMemoryBackend, MemoryBackend};
use std::sync::Arc;

fn main() {
    println!("=== Minimal E2E GPU Test ===\n");

    // Initialize CUDA driver
    unsafe {
        cuda_core::init(0).expect("CUDA driver init failed");
    }
    println!("✅ CUDA driver initialized");

    // Create context
    let ctx = CudaContext::new(0).expect("CUDA context failed");
    println!("✅ CUDA context created");

    let stream = Arc::new(ctx.default_stream());
    println!("✅ Default stream obtained");

    // Create backend
    let mut backend = CudaMemoryBackend::new((*stream).clone());
    backend.try_init_device_info();
    println!("✅ Backend created\n");

    // Allocate
    let size = 1024;
    let data: Vec<f16> = (0..size).map(|i| f16::from_f32(i as f32 * 0.1)).collect();

    println!("Allocating {} f16 elements...", size);
    match DeviceBuffer::from_host_device(&backend, &data) {
        Ok(buf) => {
            println!("✅ Allocation succeeded!");

            // Read back
            let mut host_buf = vec![f16::default(); size];
            backend
                .d2h(buf.handle(), unsafe {
                    std::slice::from_raw_parts_mut(host_buf.as_mut_ptr() as *mut u8, size * 2)
                })
                .expect("D2H failed");

            let first_5: Vec<f32> = host_buf.iter().take(5).map(|&x| x.to_f32()).collect();
            println!("✅ First 5 values: {:?}", first_5);

            // Verify
            let match_count = data
                .iter()
                .zip(host_buf.iter())
                .filter(|(a, b)| (a.to_f32() - b.to_f32()).abs() < 1e-5)
                .count();

            if match_count == size {
                println!("\n✅ SUCCESS! All {} elements verified!", size);
            } else {
                println!("\n⚠️  Only {} of {} elements matched", match_count, size);
            }
        }
        Err(e) => {
            println!("❌ Allocation failed: {}", e);

            // Try simpler allocation
            println!("\nTrying simpler allocation...");
            let _buf: DeviceBuffer<f16> = DeviceBuffer::zeros(size);
            println!("✅ Host buffer created (fallback path)");
        }
    }

    println!("\n=== Test Complete ===");
}
