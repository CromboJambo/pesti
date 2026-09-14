//! Direct CUDA cuBLAS bridge for true F16 inference (Phase 1).
//!
//! Uses cudarc's CudaBlas safe API with the proven test pattern:
//! - CUBLAS_OP_N with swapped operands for row-major layout
//! - F16 compute via cublasGemmEx internally
//! - F32 accumulation for precision

use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig};
use cudarc::driver::{CudaContext, CudaStream};
use half::f16;

/// Direct cuBLAS F16 bridge.
pub struct CudaBridge {
    blas: Arc<CudaBlas>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle on device 0.
    pub fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("Failed to init CUDA device: {}", e))?;
        let stream = ctx.default_stream();

        // Create cuBLAS handle via cudarc's safe API
        let blas = CudaBlas::new(stream.clone()).map_err(|e| {
            format!("Failed to create cuBLAS handle: {:?}", e)
        })?;

        Ok(Self {
            blas: Arc::new(blas),
        })
    }

    /// Perform F16 GEMM: C = A @ B where A is [m,k] F16, B is [k,n] F16, result is [m,n] F32.
    /// Uses cudarc's proven safe gemm() API with row-major layout conversion.
    pub fn gemm_f16f32(
        &self,
        a: &[f16],
        b: &[f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        // Upload inputs to device
        let a_dev = self.blas.stream().clone_htod(a).map_err(|e| format!("Failed to upload A: {:?}", e))?;
        let b_dev = self.blas.stream().clone_htod(b).map_err(|e| format!("Failed to upload B: {:?}", e))?;

        // Allocate output on device (F16)
        let c_size = m * n;
        let mut c_dev = self.blas.stream().alloc_zeros::<f16>(c_size).map_err(|e| {
            format!("Failed to allocate C: {:?}", e)
        })?;

        // cuBLAS is column-major. For row-major A×B, compute (B^T × A^T)^T.
        // Use CUBLAS_OP_N with swapped operands and dimensions instead of transpose flags.
        let alpha = f16::from_f32(1.0);
        let beta = f16::from_f32(0.0);

        unsafe {
            self.blas.gemm(
                GemmConfig {
                    transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,  // swapped for column-major layout
                    n: m as i32,  // swapped for column-major layout
                    k: k as i32,
                    alpha,
                    lda: n as i32,  // leading dim of B (swapped)
                    ldb: k as i32,  // leading dim of A
                    beta,
                    ldc: n as i32,  // leading dim of C (swapped)
                },
                &b_dev,
                &a_dev,
                &mut c_dev,
            ).map_err(|e| format!("cublas gemm failed: {:?}", e))?;
        }

        // Sync and download result (F16)
        self.blas.stream().synchronize().map_err(|e| {
            format!("sync failed: {:?}", e)
        })?;
        let c_host = self.blas.stream().clone_dtoh(&c_dev).map_err(|e| {
            format!("Failed to download C: {:?}", e)
        })?;

        // Convert F16 results back to F32 for the caller
        Ok(c_host.iter().map(|v| v.to_f32()).collect())
    }

    /// Synchronize the CUDA stream.
    pub fn sync(&self) -> Result<(), String> {
        self.blas.stream().synchronize().map_err(|e| {
            format!("sync failed: {:?}", e)
        })
    }
}
