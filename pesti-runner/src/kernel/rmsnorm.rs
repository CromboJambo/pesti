//! RMSNorm kernel for GPU (Phase 3).
//!
//! Computes: output[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)
//! Row-wise normalization, each row processed by a block.

use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// GPU RMSNorm kernel.
pub struct CudaRmsnormKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Builder for CudaRmsnormKernel that handles PTX loading.
pub struct CudaRmsnormKernelBuilder {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaRmsnormKernelBuilder {
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { context, stream }
    }

    /// Build the RMSNorm kernel by loading PTX module.
    pub fn build(self) -> Result<CudaRmsnormKernel, String> {
        let ptx_src = include_str!("ptx/rmsnorm.ptx");

        let module = CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| format!("RMSNorm module load failed: {:?}", e))?;

        let function = module
            .load_function("rmsnorm_kernel")
            .map_err(|e| format!("RMSNorm function load failed: {:?}", e))?;

        Ok(CudaRmsnormKernel {
            context: self.context,
            stream: self.stream,
            function,
        })
    }
}

impl CudaRmsnormKernel {
    /// Apply RMSNorm on GPU.
    ///
    /// x: [batch * seq_len, hidden] f16
    /// weight: [hidden] f16 (learned scale)
    /// out: [batch * seq_len, hidden] f16
    pub fn forward(
        &self,
        x: &DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
        out: &mut DeviceBuffer<f16>,
        num_rows: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Result<(), String> {
        let grid_x = num_rows as u32;

        // Kernel params: x_ptr, weight_ptr, out_ptr, num_rows, hidden_dim, eps
        let mut params: [*mut std::ffi::c_void; 6] = [
            &{ x.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &{ weight.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &{ out.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &(num_rows as u32) as *const u32 as *mut std::ffi::c_void,
            &(hidden_dim as u32) as *const u32 as *mut std::ffi::c_void,
            &eps as *const f32 as *mut std::ffi::c_void,
        ];

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                (grid_x, 1, 1),
                (256, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params,
            )
            .map_err(|e| format!("RMSNorm kernel launch failed: {:?}", e))?;

            crate::cuda_shim::stream_synchronize(&self.stream)
                .map_err(|e| format!("RMSNorm stream sync failed: {:?}", e))?;
        }

        Ok(())
    }
}

/// CPU reference implementation for conformance testing.
pub fn rmsnorm_cpu(x: &[f32], weight: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; hidden_dim];
    
    // Compute RMS
    let mut sum_sq = 0.0;
    for i in 0..hidden_dim {
        sum_sq += x[i] * x[i];
    }
    let rms = (sum_sq / hidden_dim as f32).sqrt();
    
    // Normalize and scale
    for i in 0..hidden_dim {
        out[i] = weight[i] * x[i] / (rms + eps);
    }
    
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_cpu_known_values() {
        // Simple case: [1, 2, 3] with weight all 1s
        let x = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0, 1.0];
        let out = rmsnorm_cpu(&x, &weight, 3, 1e-6);
        
        // RMS of [1,2,3] = sqrt(14/3) ≈ 2.160
        // Output should be x / rms
        assert!((out[0] - 0.463).abs() < 0.01);
        assert!((out[1] - 0.925).abs() < 0.01);
        assert!((out[2] - 1.387).abs() < 0.01);
    }
}