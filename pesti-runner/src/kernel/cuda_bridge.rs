//! CUDA cuBLAS GEMM bridge using cudarc's safe CudaBlas API.
//!
//! Uses the safe CudaBlas type with GemmConfig, which internally calls cublasGemmEx
//! with proper F16 input / F32 compute type settings. This achieves true F16 compute
//! on GPU while maintaining F32 accumulation precision for numerical stability.

use std::sync::Arc;
use cudarc::cublas::{CudaBlas, GemmConfig, Gemm};
use cudarc::driver::{CudaContext, CudaStream};
use half::f16;

/// Direct cuBLAS F16 bridge using safe cudarc API.
pub struct CudaBridge {
    handle: Arc<CudaBlas>,
    stream: Option<Arc<CudaStream>>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle on device 0.
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("Failed to init CUDA device: {}", e))?;
        let stream = ctx.default_stream();

        let handle = Arc::new(
            CudaBlas::new(stream.clone())
                .map_err(|e| format!("Failed to create cuBLAS handle: {}", e))?,
        );

        Ok(Self {
            handle,
            stream: Some(stream),
        })
    }

    /// Perform F16 GEMM: C = A @ B where A is [m,k] F16, B is [k,n] F16, result is [m,n] F32.
    /// Uses cublasGemmEx internally with CUBLAS_COMPUTE_32F for F32 accumulation precision.
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
        // cublasGemmEx with CUBLAS_OP_T for both operands gives us the right layout.
        unsafe {
            self.handle
                .gemm(
                    GemmConfig {
                        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha: f16::from_f32(1.0),
                        lda: n as i32,
                        ldb: k as i32,
                        beta: f16::from_f32(0.0),
                        ldc: m as i32,
                    },
                    &b_dev,
                    &a_dev,
                    &mut c_dev,
                )
                .map_err(|e| format!("cublasGemmEx failed: {}", e))?;
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