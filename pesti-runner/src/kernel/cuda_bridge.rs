//! CUDA cuBLAS GEMM bridge using cudarc's result API.
//!
//! Uses cudarc's result::hgemm which calls cublasHgemm directly via libloading,
//! avoiding the gemm_ex parameter issues entirely.

use cudarc::cublas::{result, CudaBlas, CUBLAS_OP_N};

/// Compute C = alpha * A * B + beta * C using cuBLAS F16 GEMM.
///
/// # Layout
/// All tensors in row-major order (PyTorch convention). cuBLAS is column-major,
/// so we transpose the operation: compute C^T = B^T * A^T in column-major space,
/// which equals (A * B)^T — the transpose of what we want. We swap operands and
/// use CUBLAS_OP_N to get the correct result directly into row-major layout.
///
/// # Arguments
/// - `a`: Left operand [m, k], F16 values
/// - `b`: Right operand [k, n], F16 values  
/// - `c`: Output buffer [m, n], F16 values (overwritten)
/// - `stream`: CUDA stream for execution
pub fn gemm_f16f32(
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
    m: usize,
    n: usize,
    k: usize,
    stream: &cudarc::driver::CudaStream,
) -> Result<(), String> {
    let handle = result::create_handle();
    result::set_stream(handle, stream.as_ptr());

    // cublasHgemm signature (column-major): C = alpha*op(A)*op(B) + beta*C
    // A is MxK, B is KxN, C is MxN in column-major layout.
    //
    // For row-major inputs [m,k] and [k,n], we need to compute:
    //   C_row[m,n] = sum_j(A_row[m,j] * B_row[j,n])
    // 
    // In column-major terms, A_row^T is KxM and B_row^T is NxK.
    // Compute: (A_row * B_row)^T = B_row^T * A_row^T
    // So in column-major: C_col[N,M] = B_col[K,N]^T * A_col[M,K]^T
    // But cublasHgemm expects op(A) to be MxK and op(B) to be KxN.
    // 
    // Strategy: swap operands, compute C^T = B^T * A^T with transposed dims.
    // cublasHgemm(transpose_b=k, transpose_a=n, k, m, n, ...)
    
    let alpha = 1.0f16;
    let beta = 0.0f16;

    unsafe {
        result::hgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha,
            b.as_ptr() as *const _,
            n as i32,
            a.as_ptr() as *const _,
            k as i32,
            &beta,
            c.as_mut_ptr() as *mut _,
            m as i32,
        ).map_err(|e| format!("cuda_bridge::gemm_f16f32: cublasHgemm failed: {:?}", e))?;
    }

    result::destroy(handle);
    Ok(())
}