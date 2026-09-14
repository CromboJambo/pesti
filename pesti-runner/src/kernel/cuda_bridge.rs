//! Direct CUDA cuBLAS bridge for true F16 inference (Phase 1).
//!
//! Uses cudarc's hgemm() wrapper which calls cublasHgemm directly.
//! Bypasses candle_core's F16→F32 conversion overhead.

use std::sync::Arc;

use cudarc::cublas::{result, sys};
use cudarc::driver::{CudaContext, CudaStream};
use half::f16;

/// Direct cuBLAS F16 bridge.
pub struct CudaBridge {
    handle: sys::cublasHandle_t,
    stream: Arc<CudaStream>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle on device 0.
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("Failed to init CUDA device: {}", e))?;
        let stream = Arc::new(ctx.default_stream());

        // Create cuBLAS handle via cudarc's result API
        let handle = result::create_handle().map_err(|e| format!("Failed to create cuBLAS handle: {}", e))?;

        // Set stream for async execution
        let cu_stream = cudarc::driver::sys::CUstream_st::from(stream.as_raw());
        result::set_stream(handle, cu_stream).map_err(|e| format!("Failed to set stream: {}", e))?;

        Ok(Self { handle, stream })
    }

    /// Perform F16 GEMM: C = A @ B where A is [m,k] F16, B is [k,n] F16, result is [m,n] F32.
    /// Uses cublasHgemm for true half-precision compute on GPU.
    pub fn gemm_f16f32(
        &self,
        a: &[f16],
        b: &[f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        // Upload inputs to device
        let a_dev = self.stream.clone_htod(a).map_err(|e| format!("Failed to upload A: {}", e))?;
        let b_dev = self.stream.clone_htod(b).map_err(|e| format!("Failed to upload B: {}", e))?;

        // Allocate output on device (F16 for cublas Hgemm, convert to F32 after)
        let c_size = m * n;
        let mut c_dev = self.stream.alloc_zeros::<f16>(c_size).map_err(|e| format!("Failed to allocate C: {}", e))?;

        // cuBLAS is column-major. For row-major A×B, compute (B^T × A^T)^T.
        // Use CUBLAS_OP_N with swapped operands and dimensions instead of transpose flags.
        let alpha = f16::from_f32(1.0);
        let beta = f16::from_f32(0.0);

        unsafe {
            result::hgemm(
                self.handle,
                sys::cublasOperation_t::CUBLAS_OP_N,
                sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,  // m (swapped for column-major)
                m as i32,  // n (swapped for column-major)
                k as i32,
                &alpha,
                b_dev.as_ptr() as *const f16,
                n as i32,  // lda (swapped)
                a_dev.as_ptr() as *const f16,
                k as i32,  // ldb
                &beta,
                c_dev.as_mut_ptr() as *mut f16,
                n as i32,  // ldc (swapped)
            ).map_err(|e| format!("cublasHgemm failed: {}", e))?;
        }

        // Sync and download result (F16)
        self.stream.synchronize().map_err(|e| format!("sync failed: {}", e))?;
        let c_host = self.stream.clone_dtoh(&c_dev).map_err(|e| format!("Failed to download C: {}", e))?;

        // Convert F16 results back to F32 for the caller
        Ok(c_host.iter().map(|v| v.to_f32()).collect())
    }

    /// Synchronize the CUDA stream.
    pub fn sync(&self) -> Result<(), String> {
        self.stream.synchronize().map_err(|e| format!("sync failed: {}", e))
    }
}

impl Drop for CudaBridge {
    fn drop(&mut self) {
        // Destroy cuBLAS handle
        unsafe {
            result::destroy_handle(self.handle).ok();
        }
    }
}
