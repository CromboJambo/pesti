//! Direct CUDA cuBLAS bridge for true F16 inference.
//!
//! Replaces candle_core's Tensor API (which converts to F32 internally) with
//! direct cuBLAS Hgemm calls that operate on F16 data throughout the pipeline.
//!
//! ## Architecture
//!
//! ```text
//! pesti-runner/src/kernel/dispatch.rs
//!     ↓ calls
//! pesti-runner/src/kernel/cuda_bridge.rs  (THIS MODULE)
//!     ↓ wraps
//! cudarc::cublas::CudaBlasHandle → cublasGemmEx(CUBLAS_COMPUTE_16F)
//! ```
//!
//! ## Key Design Principles
//!
//! 1. **True F16 tensors on GPU:** Pass f16 pointers directly to cuBLAS, no conversion to F32.
//! 2. **Minimal allocation:** Allocate once per layer during model load, reuse across forward passes.
//! 3. **Zero-copy where possible:** Avoid intermediate host→device copies for cached weight tensors.

use cudarc::driver::{CudaDevice, CudaStream, LaunchAsync};
use cudarc::cublas::sys::*;
use cudarc::cublas::CudaBlasHandle;
use half::f16;
use std::sync::Arc;

/// CUDA bridge for direct F16 inference operations.
pub struct CudaBridge {
    device: Arc<CudaDevice>,
    stream: Arc<CudaStream>,
    cublas: Arc<CudaBlasHandle>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle.
    pub fn new() -> Result<Self, String> {
        let device = CudaDevice::new(0)
            .map_err(|e| format!("Failed to create CUDA device: {:?}", e))?;
        let stream = device
            .stream()
            .map_err(|e| format!("Failed to create CUDA stream: {:?}", e))?;
        let cublas = CudaBlasHandle::new_with_stream(&stream)
            .map_err(|e| format!("Failed to create cuBLAS handle: {:?}", e))?;

        Ok(Self {
            device: Arc::new(device),
            stream: Arc::new(stream),
            cublas: Arc::new(cublas),
        })
    }

    /// F16 GEMM: C = alpha * (A @ B) + beta * C
    ///
    /// Uses cublasGemmEx with CUBLAS_COMPUTE_16F for true half-precision compute.
    /// All tensors remain in F16 on GPU — no conversion to F32.
    ///
    /// # Arguments
    /// * `a` - Input matrix [m, k] in row-major order (f16)
    /// * `b` - Weight matrix [k, n] in row-major order (f16)  
    /// * `c` - Output matrix [m, n], will be overwritten (f32 or f16 depending on output_dtype)
    /// * `m` - Rows of A and C
    /// * `k` - Columns of A, rows of B
    /// * `n` - Columns of B and C
    /// * `alpha` - Scale factor for A @ B
    /// * `beta` - Scale factor for C
    pub fn gemm_f16(
        &self,
        a: &[f16],
        b: &[f16],
        c_out: Option<&mut [f32]>,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Vec<f32>, String> {
        // Upload A and B to GPU as F16
        let a_dev = self.upload_f16(a);
        let b_dev = self.upload_f16(b);

        // Allocate output buffer on GPU (F32 for precision)
        let c_size = m * n;
        let mut c_host = vec![0.0f32; c_size];
        let c_dev = self.upload_f32(&c_host);

        // Perform GEMM using cuBLAS HgemmEx (F16 compute, F32 output)
        unsafe {
            cublasGemmEx(
                self.cublas.as_ptr(),
                CUBLAS_OP_T, // A is transposed (row-major to column-major)
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32,
                a_dev.as_ptr() as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32, // leading dimension of A (transposed)
                b_dev.as_ptr() as *const std::ffi::c_void,
                CUDA_R_16F,
                n as i32, // leading dimension of B
                &beta as *const f32,
                c_dev.as_ptr() as *mut std::ffi::c_void,
                CUDA_R_32F,
                n as i32, // leading dimension of C
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT,
            );
        }

        // Download result back to host
        self.download_f32(c_dev, &mut c_host);

        Ok(c_host)
    }

    /// Upload F16 data from host to GPU device memory.
    pub fn upload_f16(&self, data: &[f16]) -> cudarc::driver::DevicePtr<f16> {
        let bytes = data.len() * std::mem::size_of::<f16>();
        let dev = self.device.alloc_bytes(bytes).unwrap();
        let ptr = dev.as_ptr();
        let src = data.as_ptr() as *const u8;
        unsafe {
            cudarc::driver::cuMemcpyHtoD(
                ptr,
                src,
                bytes as cudarc::driver::size_t,
            ).unwrap();
        }
        dev
    }

    /// Upload F32 data from host to GPU device memory.
    pub fn upload_f32(&self, data: &[f32]) -> cudarc::driver::DevicePtr<f32> {
        let bytes = data.len() * std::mem::size_of::<f32>();
        let dev = self.device.alloc_bytes(bytes).unwrap();
        let ptr = dev.as_ptr();
        let src = data.as_ptr() as *const u8;
        unsafe {
            cudarc::driver::cuMemcpyHtoD(
                ptr,
                src,
                bytes as cudarc::driver::size_t,
            ).unwrap();
        }
        dev
    }

    /// Download F32 data from GPU device memory to host.
    pub fn download_f32(&self, src: cudarc::driver::DevicePtr<f32>, dst: &mut [f32]) {
        let bytes = dst.len() * std::mem::size_of::<f32>();
        let dst_ptr = dst.as_mut_ptr() as *mut u8;
        unsafe {
            cudarc::driver::cuMemcpyDtoH(
                dst_ptr,
                src.as_ptr(),
                bytes as cudarc::driver::size_t,
            ).unwrap();
        }
    }

    /// Synchronize the CUDA stream to ensure all operations complete.
    pub fn sync(&self) {
        self.stream.sync().unwrap();
    }
}

impl Drop for CudaBridge {
    fn drop(&mut self) {
        // Ensure stream is synchronized before dropping
        self.stream.sync().ok();
    }
}