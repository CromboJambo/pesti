//! Fused attention kernel for autoregressive decode (batch=1, seq=1).
//!
//! Computes softmax(q @ K^T / sqrt(d)) @ V in a single kernel launch,
//! replacing the CPU-only attention loop in layer.rs.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    safe::{CudaContext, CudaStream},
    sys,
};

/// Configuration for fused decode attention.
#[derive(Debug)]
pub struct FusedDecodeAttentionConfig {
    pub head_dim: usize,
    pub scale: f32, // Pre-computed 1/sqrt(head_dim)
}

impl Default for FusedDecodeAttentionConfig {
    fn default() -> Self {
        let head_dim = 64;
        Self {
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA fused decode attention kernel.
pub struct FusedDecodeAttentionKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: crate::cuda_shim::CudaModule,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl FusedDecodeAttentionKernel {
    /// Launch fused attention kernel for single query position.
    ///
    /// # Arguments
    /// * `q` - Query vector on device [head_dim] f32
    /// * `k_cache` - Key cache on device [seq_len, head_dim] f32 (row-major)
    /// * `v_cache` - Value cache on device [seq_len, head_dim] f32 (row-major)
    /// * `scale` - 1/sqrt(head_dim)
    /// * `seq_len` - Number of cached positions
    /// * `head_dim` - Head dimension
    /// * `output` - Output buffer on device [head_dim] f32
    pub fn launch(
        &self,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        scale: f32,
        seq_len: usize,
        head_dim: usize,
        output: u64,
    ) -> Result<(), String> {
        let mut q_v: u64 = q;
        let mut k_v: u64 = k_cache;
        let mut v_v: u64 = v_cache;
        let mut scale_v: f32 = scale;
        let mut seq_len_v: i32 = seq_len as i32;
        let mut head_dim_v: i32 = head_dim as i32;
        let mut out_v: u64 = output;

        let params: [*mut std::ffi::c_void; 7] = [
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut seq_len_v as *mut i32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut i32 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
        ];

        let grid = (1u32, 1u32, 1u32);
        let block = (head_dim as u32, 1u32, 1u32);

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0u32,
                crate::cuda_shim::cu_stream(&self.stream),
                &params,
            )
            .map_err(|e| format!("kernel launch failed: {:?}", e))?;
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub fn build_fused_decode_attention_kernel(
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> Result<FusedDecodeAttentionKernel, String> {
    let ptx_src = include_str!("ptx/fused_decode_attention.ptx");

    let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_src)
        .map_err(|e| format!("module load failed: {:?}", e))?;

    // Entry point name from PTX compilation (no mangling for __global__ extern "C")
    let function = module
        .load_function("fused_attention_kernel")
        .map_err(|e| format!("function lookup failed: {:?}", e))?;

    Ok(FusedDecodeAttentionKernel {
        context,
        stream,
        module,
        function,
    })
}
