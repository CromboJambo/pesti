//! CUDA cuBLAS GEMM bridge using cudarc's result API.
//!
//! Uses cudarc's result::hgemm which calls cublasHgemm directly via libloading,
//! avoiding the gemm_ex parameter issues entirely.

use std::sync::Arc;
use cudarc::cublas::result::{create_handle, destroy_handle, hgemm};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
use half::f16;

/// Direct cuBLAS F16 bridge using cudarc result API.
pub struct CudaBridge {
    handle: cudarc::cublas::sys::cublasHandle_t,
    stream: Option<Arc<CudaStream>>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle on device 0.
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("Failed to init CUDA device: {}", e))?;
        let stream = ctx.default_stream();

        let handle = create_handle().map_err(|e| format!("Failed to create cuBLAS handle: {}", e))?;

        Ok(Self {
            handle,
            stream: Some(stream),
        })
    }

    /// Perform F16 GEMM: C = A @ B where A is [m,k] F16, B is [k,n] F16, result is [m,n] F32.
    /// Uses cublasHgemm directly for true F16 compute.
    pub fn gemm_f16f32(
        &self,
        a: &[f16],
        b: &[f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        let stream = self.stream.as_ref().ok_or("CUDA stream not initialized")?;

        // Upload inputs to device
        let a_dev = stream
            .clone_htod(a)
            .map_err(|e| format!("Failed to upload A: {}", e))?;
        let b_dev = stream
            .clone_htod(b)
            .map_err(|e| format!("Failed to upload B: {}", e))?;

        // Allocate output on device (F16 for cublas Hgemm)
        let c_size = m * n;
        let mut c_dev = stream
            .alloc_zeros::<f16>(c_size)
            .map_err(|e| format!("Failed to allocate C: {}", e))?;

        // cuBLAS is column-major. For row-major A×B, compute (B^T × A^T)^T.
        let alpha = f16::from_f32(1.0);
        let beta = f16::from_f32(0.0);

        // Get device pointers via DevicePtr trait
        let (a_ptr, _a_guard) = a_dev.device_ptr(stream);
        let (b_ptr, _b_guard) = b_dev.device_ptr(stream);
        let (c_ptr, _c_guard) = c_dev.device_ptr(stream);

        unsafe {
            hgemm(
                self.handle,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b_ptr as *const f16,
                n as i32,
                a_ptr as *const f16,
                k as i32,
                &beta,
                c_ptr as *mut f16,
                m as i32,
            )
            .map_err(|e| format!("cublasHgemm failed: {}", e))?;
        }

        // Sync and download result, convert F16→F32 on host
        stream
            .synchronize()
            .map_err(|e| format!("sync failed: {}", e))?;
        let c_host = stream
            .clone_dtoh(&c_dev)
            .map_err(|e| format!("Failed to download C: {}", e))?;

        // Convert F16 results back to F32 for the caller
        Ok(c_host.iter().map(|v| v.to_f32()).collect())
    }

    /// Synchronize the CUDA stream.
    pub fn sync(&self) -> Result<(), String> {
        if let Some(stream) = &self.stream {
            stream
                .synchronize()
                .map_err(|e| format!("sync failed: {}", e))?;
        }
        Ok(())
    }
}

impl Drop for CudaBridge {
    fn drop(&mut self) {
        unsafe {
            let _ = destroy_handle(self.handle);
        }
    }
}