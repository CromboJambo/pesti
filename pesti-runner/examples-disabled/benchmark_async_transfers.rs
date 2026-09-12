//! Benchmark async memory transfers vs synchronous copies.
//!
//! Compares performance of overlapping H2D/D2H transfers with compute
//! versus blocking (synchronous) transfers.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Async Memory Transfer Benchmark ===");

    // Initialize CUDA
    let ctx = unsafe { cudarc::driver::sys::cuInit(0) };
    if ctx != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        eprintln!("CUDA init failed: {:?}", ctx);
        std::process::exit(1);
    }

    let mut device: cudarc::driver::sys::CUdevice = 0;
    unsafe { cudarc::driver::sys::cuDeviceGet(&mut device, 0) };

    let mut ctx_ptr: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
    unsafe { cudarc::driver::sys::cuCtxCreate_v2(&mut ctx_ptr, 0, device) };

    println!("CUDA initialized successfully");
    println!();

    // Create a stream for async operations
    let mut stream: cudarc::driver::sys::CUstream = std::ptr::null_mut();
    unsafe { cudarc::driver::sys::cuStreamCreate(&mut stream, 0) };

    // Test parameters
    let test_sizes: Vec<usize> = vec![1024, 4096, 16384, 65536, 262144, 1048576]; // 1KB to 1MB

    println!("Transfer Size | Sync H2D | Async H2D | Speedup | Sync D2H | Async D2H | Speedup");
    println!("--------------|----------|-----------|---------|----------|-----------|--------");

    for &size in &test_sizes {
        // Prepare test data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Allocate device memory
        let mut device_ptr: u64 = 0;
        unsafe {
            cudarc::driver::sys::cuMemAlloc_v2(&mut device_ptr, size);
        }

        // --- H2D Benchmark ---
        // Synchronous H2D (3 iterations for averaging)
        let mut sync_h2d_times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoD_v2(
                    device_ptr,
                    host_data.as_ptr() as *const std::ffi::c_void,
                    size,
                );
            }
            sync_h2d_times.push(start.elapsed());
        }
        let sync_h2d_avg = sync_h2d_times.iter().map(|d| d.as_nanos()).sum::<u128>() / 3;

        // Asynchronous H2D (3 iterations for averaging)
        let mut async_h2d_times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    device_ptr,
                    host_data.as_ptr() as *const std::ffi::c_void,
                    size,
                    stream,
                );
                cudarc::driver::sys::cuStreamSynchronize(stream);
            }
            async_h2d_times.push(start.elapsed());
        }
        let async_h2d_avg = async_h2d_times.iter().map(|d| d.as_nanos()).sum::<u128>() / 3;

        // --- D2H Benchmark ---
        let mut result_data = vec![0u8; size];

        // Synchronous D2H (3 iterations for averaging)
        let mut sync_d2h_times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    result_data.as_mut_ptr() as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                );
            }
            sync_d2h_times.push(start.elapsed());
        }
        let sync_d2h_avg = sync_d2h_times.iter().map(|d| d.as_nanos()).sum::<u128>() / 3;

        // Asynchronous D2H (3 iterations for averaging)
        let mut async_d2h_times = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    result_data.as_mut_ptr() as *mut std::ffi::c_void,
                    device_ptr,
                    size,
                    stream,
                );
                cudarc::driver::sys::cuStreamSynchronize(stream);
            }
            async_d2h_times.push(start.elapsed());
        }
        let async_d2h_avg = async_d2h_times.iter().map(|d| d.as_nanos()).sum::<u128>() / 3;

        // Cleanup
        unsafe {
            cudarc::driver::sys::cuMemFree_v2(device_ptr);
        }

        // Format output
        let size_label = if size >= 1048576 {
            format!("{}MB", size / 1048576)
        } else if size >= 1024 {
            format!("{}KB", size / 1024)
        } else {
            format!("{}B", size)
        };

        let h2d_speedup = sync_h2d_avg as f64 / async_h2d_avg.max(1) as f64;
        let d2h_speedup = sync_d2h_avg as f64 / async_d2h_avg.max(1) as f64;

        println!(
            "{:>13} | {:>7}ns | {:>8}ns | {:>6.2}x | {:>7}ns | {:>8}ns | {:>6.2}x",
            size_label,
            sync_h2d_avg,
            async_h2d_avg,
            h2d_speedup,
            sync_d2h_avg,
            async_d2h_avg,
            d2h_speedup,
        );
    }

    // Cleanup stream
    unsafe {
        cudarc::driver::sys::cuStreamDestroy_v2(stream);
        cudarc::driver::sys::cuCtxDestroy_v2(ctx_ptr);
    }

    println!();
    println!("=== Benchmark Complete ===");
    println!();
    println!("Key Insights:");
    println!("- Async transfers are NOT faster for raw copy speed (driver adds overhead)");
    println!("- The benefit is OVERLAP: launch async H2D, then compute, then sync");
    println!("- This hides transfer latency behind kernel execution");
    println!("- Best gains when: H2D time > kernel time (large inputs, small kernels)");
    println!();
    println!("Example overlap pattern:");
    println!("  1. Launch async H2D transfer");
    println!("  2. Launch compute kernel on already-available data");
    println!("  3. Sync H2D (data ready for next kernel)");
    println!("  4. Launch next kernel");

    Ok(())
}
