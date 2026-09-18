//! SwiGLU activation kernel for GPU (Phase 3).
//!
//! Computes: silu(gate) * up where silu(x) = x * sigmoid(x)
//! Element-wise operation, trivially parallelizable.

use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// GPU SwiGLU activation kernel.
pub struct CudaSwigluKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Builder for CudaSwigluKernel that handles PTX loading.
pub struct CudaSwigluKernelBuilder {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaSwigluKernelBuilder {
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { context, stream }
    }

    /// Build the SwiGLU kernel by loading PTX module.
    pub fn build(self) -> Result<CudaSwigluKernel, String> {
        let ptx_src = include_str!("ptx/swiglu.ptx");

        let module = CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| format!("SwiGLU module load failed: {:?}", e))?;

        let function = module
            .load_function("swiglu_kernel")
            .map_err(|e| format!("SwiGLU function load failed: {:?}", e))?;

        Ok(CudaSwigluKernel {
            context: self.context,
            stream: self.stream,
            function,
        })
    }
}

impl CudaSwigluKernel {
    /// Compute SwiGLU activation on GPU.
    ///
    /// gate and up are f16 tensors of shape [batch * seq_len, hidden].
    /// Output is written to out (f16).
    pub fn forward(
        &self,
        gate: &DeviceBuffer<f16>,
        up: &DeviceBuffer<f16>,
        out: &mut DeviceBuffer<f16>,
        num_elements: usize,
    ) -> Result<(), String> {
        let grid_x = (num_elements as u32).div_ceil(256);

        // Kernel params: gate_ptr, up_ptr, out_ptr, num_elements
        let mut params: [*mut std::ffi::c_void; 4] = [
            &{ gate.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &{ up.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &{ out.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &(num_elements as u32) as *const u32 as *mut std::ffi::c_void,
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
            .map_err(|e| format!("SwiGLU kernel launch failed: {:?}", e))?;

            crate::cuda_shim::stream_synchronize(&self.stream)
                .map_err(|e| format!("SwiGLU stream sync failed: {:?}", e))?;
        }

        Ok(())
    }
}

/// CPU reference implementation for conformance testing.
pub fn swiglu_cpu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let n = gate.len();
    let mut out = vec![0.0; n];
    for i in 0..n {
        // silu(x) = x / (1 + exp(-x))
        let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
        out[i] = gate[i] * sigmoid * up[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_cpu_known_values() {
        // silu(0) = 0, so SwiGLU(0, x) = 0 for any x
        let gate = vec![0.0];
        let up = vec![5.0];
        let out = swiglu_cpu(&gate, &up);
        assert!((out[0] - 0.0).abs() < 1e-6);

        // silu(1) ≈ 0.731, so SwiGLU(1, 2) ≈ 1.462
        let gate = vec![1.0];
        let up = vec![2.0];
        let out = swiglu_cpu(&gate, &up);
        assert!((out[0] - 1.462).abs() < 0.01);

        // Negative: silu(-1) ≈ -0.269
        let gate = vec![-1.0];
        let up = vec![3.0];
        let out = swiglu_cpu(&gate, &up);
        assert!((out[0] + 0.807).abs() < 0.01);
    }
}