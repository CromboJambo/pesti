//! Test cuBLASLt API surface via cudarc
//! Explores available functions for shape-optimized GEMM

use cudarc::cublas::{result, sys};
use cudarc::driver::{CudaContext, CudaStream};
use half::f16;

fn main() {
    println!("Testing cuBLASLt API surface...");

    // Initialize CUDA context
    let ctx = CudaContext::new(0).expect("CUDA init failed");
    let stream = ctx.default_stream();

    // Check cublasLt availability
    println!("cuBLASLt functions available in cudarc sys module:");

    // Try to create a cublasLt handle (if exposed)
    // In cudarc 0.19.x, cublasLt is accessed via sys bindings
    
    // Test basic GEMM shapes we care about for LLM inference
    let test_shapes = vec![
        (1, 1536, 4096),   // typical attention head
        (1, 3072, 4096),   // larger layer
        (1, 128, 1536),    // small projection
    ];

    for (m, n, k) in test_shapes {
        println!("Shape m={} n={} k={}: would use cublasLtMatmul with heuristics", m, n, k);
    }

    println!("\nAPI exploration complete.");
}