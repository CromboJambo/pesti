//! Direct cuBLAS test to isolate the issue.

use cudarc::cublas::result::{create_handle, destroy_handle, hgemm};
use cudarc::driver::{CudaContext, CudaStream};
use half::f16;

fn main() {
    println!("=== Direct cuBLAS Hgemm Test ===");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to init CUDA");
    let stream = ctx.default_stream();
    println!("CUDA context initialized");

    // Create cuBLAS handle
    let handle = create_handle().expect("Failed to create cuBLAS handle");
    println!("cuBLAS handle created");

    // Simple test: 2x3 * 3x4 = 2x4 matrix multiply
    // Use identity-like matrices for easy verification
    let m = 2;
    let k = 3;
    let n = 4;

    // A: [m,k] = [2,3] - simple values
    let a_host: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
        f16::from_f32(5.0),
        f16::from_f32(6.0),
    ];

    // B: [k,n] = [3,4] - simple values
    let b_host: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(0.0),
    ];

    // Allocate device memory
    let a_dev = cudarc::driver::CudaMallocAsync::cuda_malloc_async::<f16>(&stream, m * k)
        .expect("Failed to alloc A");
    let b_dev = cudarc::driver::CudaMallocAsync::cuda_malloc_async::<f16>(&stream, k * n)
        .expect("Failed to alloc B");
    let c_dev = cudarc::driver::CudaMallocAsync::cuda_malloc_async::<f16>(&stream, m * n)
        .expect("Failed to alloc C");

    // Copy data to device
    stream.copy_h2d(a_dev, &a_host).expect("Failed to copy A");
    stream.copy_h2d(b_dev, &b_host).expect("Failed to copy B");

    // Compute C = A * B (row-major)
    // In cuBLAS terms: C^T = B^T * A^T
    let alpha = f16::from_f32(1.0);
    let beta = f16::from_f32(0.0);

    unsafe {
        let result = hgemm(
            handle,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n as i32, // result rows (n)
            m as i32, // result cols (m)
            k as i32, // inner dim (k)
            &alpha,
            b_dev.as_ptr(), // B^T treated as first operand
            n as i32,       // leading dim of B
            a_dev.as_ptr(), // A^T treated as second operand
            k as i32,       // leading dim of A
            &beta,
            c_dev.as_ptr(), // C^T output
            n as i32,       // leading dim of C
        );

        match result {
            Ok(_) => println!("cuBLAS Hgemm succeeded!"),
            Err(e) => {
                println!("cuBLAS Hgemm failed: {}", e);
                destroy_handle(handle).ok();
                std::process::exit(1);
            }
        }
    }

    // Sync and read back
    stream.synchronize().expect("Failed to sync");

    let mut c_host = vec![f16::from_f32(0.0); m * n];
    stream
        .copy_d2h(&c_host, c_dev)
        .expect("Failed to copy C back");

    println!("Result (should be A*B):");
    for row in 0..m {
        let mut line = String::new();
        for col in 0..n {
            line.push_str(&format!("{:.1} ", c_host[row * n + col].to_f32()));
        }
        println!("{}", line);
    }

    // Cleanup
    stream.cuda_free_async(a_dev).ok();
    stream.cuda_free_async(b_dev).ok();
    stream.cuda_free_async(c_dev).ok();
    destroy_handle(handle).ok();

    println!("\nDirect cuBLAS test: COMPLETE");
}
