//! Direct CUDA cuBLAS bridge for true F16 inference (Phase 1).
//!
//! Uses raw cublasGemmEx via libloading for precise control over compute type
//! and mathType parameters required for F16 input / F32 accumulation.

use cudarc::driver::{CudaStream, LaunchAsync};
use std::ffi::c_void;

// cuBLAS handle (opaque pointer)
type cublasHandle_t = *mut c_void;

// cuBLAS status codes
const CUBLAS_STATUS_SUCCESS: i32 = 0;

// CUDA data types for cublasGemmEx
const CUDA_R_16F: i32 = 14;
const CUDA_R_32F: i32 = 0;

// cuBLAS operation types
const CUBLAS_OP_N: u8 = 0;
const CUBLAS_OP_T: u8 = 1;

// Math type for F16 compute with F32 accumulation
const CUBLAS_DEFAULT_MATH: i32 = 0;

extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> i32;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> i32;
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudarc::driver::sys::CUstream_st) -> i32;
    fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: u8,
        transb: u8,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        A: *const c_void,
        Atype: i32,
        lda: i32,
        B: *const c_void,
        Btype: i32,
        ldb: i32,
        beta: *const f32,
        C: *mut c_void,
        Ctype: i32,
        ldc: i32,
        computeType: i32,
        mathType: i32,
    ) -> i32;
}

/// CUDA cuBLAS handle wrapper for the lifetime of inference.
pub struct CublasHandle {
    handle: cublasHandle_t,
}

impl CublasHandle {
    /// Create a new cuBLAS handle and attach it to the given stream.
    pub fn new(stream: &CudaStream) -> anyhow::Result<Self> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        unsafe {
            let status = cublasCreate_v2(&mut handle);
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow::anyhow!(
                    "cublasCreate_v2 failed with status {}",
                    status
                ));
            }
            // Attach to our CUDA stream for async execution
            let stream_ptr = stream.stream();
            cublasSetStream_v2(handle, stream_ptr);
        }
        Ok(Self { handle })
    }

    /// Perform F16 GEMM with F32 accumulation: C = alpha * A * B + beta * C
    ///
    /// A is (k x m), B is (n x k), C is (k x n) — column-major layout as expected by cuBLAS.
    /// Weights are pre-transposed and stored in column-major format on GPU.
    pub fn gemm_f16f32(
        &self,
        stream: &CudaStream,
        a: &[u16],
        b: &[u16],
        c: &mut [u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> anyhow::Result<()> {
        // Convert slices to device pointers
        let a_ptr = a.as_ptr() as *const c_void;
        let b_ptr = b.as_ptr() as *const c_void;
        let c_ptr = c.as_mut_ptr() as *mut c_void;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // cublasGemmEx: C = alpha * op(A) * op(B) + beta * C
        // A is (k x m), B is (n x k), C is (k x n) in column-major
        let status = unsafe {
            cublasGemmEx(
                self.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32,
                a_ptr,
                CUDA_R_16F,
                k as i32,
                b_ptr,
                CUDA_R_16F,
                n as i32,
                &beta as *const f32,
                c_ptr,
                CUDA_R_16F,
                m as i32,
                CUDA_R_32F, // compute in F32 for precision
                CUBLAS_DEFAULT_MATH,
            )
        };

        if status != CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuda_bridge::gemm_f16f32: cublasGemmEx failed with status {}",
                status
            ));
        }

        // Synchronize stream to ensure completion
        stream.synchronize()?;

        Ok(())
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.handle);
        }
    }
}

/// Global cuBLAS handle (created once, used for all GEMM operations).
static mut CUBLAS: Option<CublasHandle> = None;

/// Initialize the CUDA cuBLAS bridge. Call once at startup.
pub fn init_cuda_bridge(stream: &CudaStream) -> anyhow::Result<()> {
    unsafe {
        let handle = CublasHandle::new(stream)?;
        CUBLAS = Some(handle);
    }
    Ok(())
}

/// Get the global cuBLAS handle.
pub fn get_cublas_handle() -> &'static CublasHandle {
    unsafe { CUBLAS.as_ref().expect("CUDA bridge not initialized") }
}
