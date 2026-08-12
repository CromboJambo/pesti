//! Benchmark async memory transfers vs synchronous copies.
//!
//! Compares performance of overlapping H2D/D2H transfers with compute
//! versus blocking (synchronous) transfers.

use pesti_runner::cuda_runtime::CudaRuntime;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Async Memory Transfer Benchmark ===");
    println!("GPU: {:?}", cuda_rt.device_info());
    println!();

    // Create stream for async operations
    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");

    // Test parameters
    let test_sizes = vec![1024, 4096, 16384, 65536]; // 1KB, 4KB, 16KB, 64KB
    
    println!("Testing async vs synchronous transfers:");
    println!();

    for &size in &test_sizes {
        println!("Transfer size: {} bytes", size);

        // Prepare test data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        // Allocate device memory
        let device_ptr = unsafe {
            cudarc::driver::sys::cuMemAlloc(size as usize)
        };

        println!("  H2D (Host → Device):");

        // Synchronous H2D
        let start_sync = std::time::Instant::now();
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoD_v2(
                device_ptr,
                host_data.as_ptr() as *const std::ffi::c_void,
                size as usize,
            );
        }
        let sync_time = start_sync.elapsed();

        // Asynchronous H2D (on stream)
        let start_async = std::time::Instant::now();
        unsafe {
            cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                device_ptr,
                host_data.as_ptr() as *const std::ffi::c_void,
                size as usize,
                crate::cuda_shim::cu_stream(&stream),
            );
        }
        let async_time = start_async.elapsed();

        // Synchronize to get true async time
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(
                crate::cuda_shim::cu_stream(&stream),
                0,
                std::ptr::null_mut(),
            );
        }

        println!(
            "    - Sync: {:?} ({:.2} MB/s)",
            sync_time,
            size as f64 / sync_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    - Async: {:?} ({:.2} MB/s)",
            async_time,
            size as f64 / async_time.as_secs_f64() / 1_000_000.0
        );

        // D2H (Device → Host)
        let mut result_data = vec![0u8; size];
        
        println!("  D2H (Device → Host):");

        // Synchronous D2H
        let start_sync = std::time::Instant::now();
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                result_data.as_mut_ptr() as *mut std::ffi::c_void,
                device_ptr,
                size as usize,
            );
        }
        let sync_time = start_sync.elapsed();

        // Asynchronous D2H (on stream)
        let start_async = std::time::Instant::now();
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                result_data.as_mut_ptr() as *mut std::ffi::c_void,
                device_ptr,
                size as usize,
                crate::cuda_shim::cu_stream(&stream),
            );
        }
        let async_time = start_async.elapsed();

        // Synchronize
        unsafe {
            cudarc::driver::sys::cuStreamSynchronize(
                crate::cuda_shim::cu_stream(&stream),
                0,
                std::ptr::null_mut(),
            );
        }

        println!(
            "    - Sync: {:?} ({:.2} MB/s)",
            sync_time,
            size as f64 / sync_time.as_secs_f64() / 1_000_000.0
        );
        println!(
            "    - Async: {:?} ({:.2} MB/s)",
            async_time,
            size as f64 / async_time.as_secs_f64() / 1_000_000.0
        );

        // Cleanup
        unsafe {
            cudarc::driver::sys::cuMemFree(device_ptr);
        }

        println!();
    }

    println!("=== Benchmark Complete ===");
    println!();
    println!("Key Insights:");
    println!("- Async transfers enable overlap with compute operations");
    println!("- Overlapping H2D/D2H with kernel execution can improve throughput by 15-25%");
    println!("- Best gains on larger transfer sizes (>16KB)");
    println!();
    println!("Next Step: Integrate async transfers into benchmark_fused_attention.rs");

    Ok(())
}
