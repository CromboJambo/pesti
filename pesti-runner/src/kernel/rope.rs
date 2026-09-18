//! RoPE (Rotary Positional Embeddings) kernel for GPU (Phase 3).
//!
//! Applies rotary embeddings to query and key vectors in-place.
//! Each thread handles one rotation pair across all sequence positions and heads.

use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// Trait for RoPE kernels (GPU or CPU implementations).
pub trait RopeKernel: Send + Sync {
    /// Apply rotary embeddings to query and key tensors.
    /// Both q and k are modified in-place.
    fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<(), String>;
}

/// GPU RoPE kernel.
pub struct CudaRopeKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Builder for CudaRopeKernel that handles PTX loading.
pub struct CudaRopeKernelBuilder {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl CudaRopeKernelBuilder {
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { context, stream }
    }

    /// Build the RoPE kernel by loading PTX module.
    pub fn build(self) -> Result<CudaRopeKernel, String> {
        let ptx_src = include_str!("ptx/rope.ptx");

        let module = CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| format!("RoPE module load failed: {:?}", e))?;

        let function = module
            .load_function("rope_kernel")
            .map_err(|e| format!("RoPE function load failed: {:?}", e))?;

        Ok(CudaRopeKernel {
            context: self.context,
            stream: self.stream,
            function,
        })
    }
}

impl CudaRopeKernel {
    /// Apply RoPE to query and key tensors in-place on GPU.
    ///
    /// q: [batch, seq_len, num_heads, head_dim] flattened row-major f32
    /// k: [batch, seq_len, num_heads, head_dim] flattened row-major f32
    pub fn apply_gpu(
        &self,
        q: &mut DeviceBuffer<f32>,
        k: &mut DeviceBuffer<f32>,
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
        head_dim: usize,
        base: f32,
    ) -> Result<(), String> {
        let grid_x = (seq_len * num_heads).div_ceil(256);

        // Kernel params: data_ptr, base, seq_len, num_heads, head_dim, start_pos
        let mut params_q: [*mut std::ffi::c_void; 6] = [
            &{ q.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &base as *const f32 as *mut std::ffi::c_void,
            &(seq_len as u32) as *const u32 as *mut std::ffi::c_void,
            &(num_heads as u32) as *const u32 as *mut std::ffi::c_void,
            &(head_dim as u32) as *const u32 as *mut std::ffi::c_void,
            &(start_pos as u32) as *const u32 as *mut std::ffi::c_void,
        ];

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                ((grid_x as u32), 1, 1),
                (256, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params_q,
            )
            .map_err(|e| format!("RoPE kernel launch failed for Q: {:?}", e))?;

            crate::cuda_shim::stream_synchronize(&self.stream)
                .map_err(|e| format!("RoPE stream sync failed for Q: {:?}", e))?;
        }

        // Apply to K (same parameters, different tensor)
        let mut params_k: [*mut std::ffi::c_void; 6] = [
            &{ k.device_ptr() } as *const u64 as *mut std::ffi::c_void,
            &base as *const f32 as *mut std::ffi::c_void,
            &(seq_len as u32) as *const u32 as *mut std::ffi::c_void,
            &(num_heads as u32) as *const u32 as *mut std::ffi::c_void,
            &(head_dim as u32) as *const u32 as *mut std::ffi::c_void,
            &(start_pos as u32) as *const u32 as *mut std::ffi::c_void,
        ];

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                ((grid_x as u32), 1, 1),
                (256, 1, 1),
                0,
                self.stream.cu_stream(),
                &mut params_k,
            )
            .map_err(|e| format!("RoPE kernel launch failed for K: {:?}", e))?;

            crate::cuda_shim::stream_synchronize(&self.stream)
                .map_err(|e| format!("RoPE stream sync failed for K: {:?}", e))?;
        }

        Ok(())
    }
}

/// CPU reference implementation of RopeKernel for conformance testing and fallback.
pub struct CpuRopeKernel {
    base: f32,
}

impl CpuRopeKernel {
    pub fn new(base: f32) -> Self {
        Self { base }
    }
}

impl RopeKernel for CpuRopeKernel {
    fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<(), String> {
        let head_dim = q.len() / (num_heads * seq_len);
        rope_cpu(q, num_heads, seq_len, start_pos, head_dim, self.base);
        rope_cpu(k, num_heads, seq_len, start_pos, head_dim, self.base);
        Ok(())
    }
}

/// CPU reference implementation for conformance testing.
pub fn rope_cpu(
    data: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    start_pos: usize,
    head_dim: usize,
    base: f32,
) {
    let dim_half = head_dim / 2;
    let theta: Vec<f32> = (0..dim_half)
        .map(|i| base.powf(-(i as f32) / dim_half as f32))
        .collect();

    for pos in 0..seq_len {
        let actual_pos = start_pos + pos;
        for head in 0..num_heads {
            for (i, &freq) in theta.iter().enumerate() {
                let angle = actual_pos as f32 * freq;
                let cos = angle.cos();
                let sin = angle.sin();

                let idx = pos * num_heads * head_dim + head * head_dim + i;
                let next = idx + dim_half;

                let orig = data[idx];
                data[idx] = orig * cos - data[next] * sin;
                data[next] = orig * sin + data[next] * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_cpu_known_values() {
        // RoPE at position 0 should be identity (cos(0)=1, sin(0)=0)
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        rope_cpu(&mut data, 1, 1, 0, 4, 10000.0);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);

        // At position 1, rotation should change values
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        rope_cpu(&mut data, 1, 1, 1, 4, 10000.0);
        // cos(1/100) ~ 0.99995, sin(1/100) ~ 0.01
        assert!((data[0] - 0.99995).abs() < 0.001);
    }

    #[test]
    fn test_cpu_rope_kernel_trait() {
        let kernel = CpuRopeKernel::new(10000.0);
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        
        // Should not panic
        kernel.apply(&mut q, &mut k, 1, 1, 0).unwrap();
    }
}
