//! Direct CUDA cuBLAS bridge for true F16 inference (Phase 1).
//!
//! Uses cudarc's result API directly to call cublasHgemm — the correct
//! cuBLAS function for F16 inputs with F32 accumulation. The safe API's
//! gemm() internally calls cublasGemmEx which has mathType issues for
//! our F16-in/F32-out use case.

use cudarc::cublas::{result, CudaBlas};
use cudarc::driver::{CudaStream, LaunchAsync};
use std::sync::Once;

/// Global cuBLAS handle (created once, used for all GEMM operations).
static mut CUBLAS: Option<CudaBlas> = None;
static INIT: Once = Once::new();

/// Initialize the CUDA cuBLAS bridge. Call once at startup.
pub fn init_cuda_bridge(stream: &CudaStream) -> anyhow::Result<()> {
    unsafe {
        let blas = CudaBlas::new(stream.stream())?;
        CUBLAS = Some(blas);
    }
    Ok(())
}

/// Get the global cuBLAS handle.
pub fn get_cublas_handle() -> &'static CudaBlas {
    unsafe { CUBLAS.as_ref().expect("CUDA bridge not initialized") }
}

/// Perform F16 GEMM with F32 accumulation: C = alpha * A * B + beta * C
///
/// Uses cublasHgemm directly via cudarc's result API.
/// A is (k x m), B is (n x k), C is (k x n) — column-major layout as expected by cuBLAS.
/// Weights are pre-transposed and stored in column-major format on GPU.
pub fn gemm_f16f32(
    stream: &CudaStream,
    a: &[u16],
    b: &[u16],
    c: &mut [u16],
    m: usize,
    n: usize,
    k: usize,
) -> anyhow::Result<()> {
    let blas = get_cublas_handle();

    // cublasHgemm computes C = alpha * op(A) * op(B) + beta * C
    // with F16 inputs and F32 accumulation.
    // A is (k x m), B is (n x k), C is (k x n) in column-major layout.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    result::hgemm(
        blas,
        false, // transA = N (A is already transposed on GPU)
        false, // transB = N (B is row-major on device, treated as column-major via layout)
        n as i32,   // m: rows of result C
        m as i32,   // n: cols of result C
        k as i32,   // k: inner dimension
        &alpha,
        b.as_ptr(),     // A in cuBLAS = B from caller (row input)
        n as i32,       // lda = row stride of B = n
        a.as_ptr(),     // B in cuBLAS = A from caller (weight)
        k as i32,       // ldb = row stride of A = k
        &beta,
        c.as_mut_ptr(), // C output
        m as i32,       // ldc = row stride of C = m
    )
    .map_err(|e| anyhow::anyhow!("cuda_bridge::gemm_f16f32: cublasHgemm failed: {:?}", e))?;

    stream.synchronize()?;

    Ok(())
}
