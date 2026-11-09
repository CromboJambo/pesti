//! Fused decode attention kernel with Q4_K quantized KV cache (Phase 4).
//!
//! Computes softmax(q @ K^T / sqrt(d)) @ V where K and V are stored in Q4_K format.
//! Dequantization happens on-the-fly within the kernel to minimize memory bandwidth.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    safe::{CudaContext, CudaStream},
    sys,
};

/// Configuration for Q4_K fused decode attention.
#[derive(Debug)]
pub struct FusedDecodeAttentionQ4KConfig {
    pub head_dim: usize,
    pub scale: f32, // Pre-computed 1/sqrt(head_dim)
}

impl Default for FusedDecodeAttentionQ4KConfig {
    fn default() -> Self {
        let head_dim = 64;
        Self {
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA fused decode attention kernel with Q4_K quantized KV cache.
#[derive(Clone)]
pub struct FusedDecodeAttentionQ4KKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl FusedDecodeAttentionQ4KKernel {
    /// Load and compile the Q4_K fused attention kernel.
    pub fn load(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, String> {
        // Compile PTX source at runtime
        let ptx_source = include_str!("ptx/fused_decode_attention_q4k.ptx");

        let module = crate::cuda_shim::CudaModule::from_ptx(&context, ptx_source)?;
        let function = module.get_function("fused_attention_q4k")?;

        Ok(Self {
            context,
            stream,
            module: std::sync::Arc::new(module),
            function,
        })
    }

    /// Launch fused attention kernel with Q4_K quantized KV cache.
    ///
    /// # Arguments
    /// * `q` - Query vector on device [head_dim] f32
    /// * `k_quant` - Quantized K cache on device [num_blocks * 144] u8 (Q4_K format)
    /// * `v_quant` - Quantized V cache on device [num_blocks * 144] u8 (Q4_K format)
    /// * `scale` - 1/sqrt(head_dim)
    /// * `seq_len` - Number of cached positions
    /// * `head_dim` - Head dimension
    /// * `output` - Output buffer on device [head_dim] f32
    pub fn launch(
        &self,
        q: u64,
        k_quant: u64,
        v_quant: u64,
        scale: f32,
        seq_len: usize,
        head_dim: usize,
        output: u64,
    ) -> Result<(), String> {
        let mut q_v: u64 = q;
        let mut k_v: u64 = k_quant;
        let mut v_v: u64 = v_quant;
        let mut scale_v: f32 = scale;
        let mut seq_len_v: i32 = seq_len as i32;
        let mut head_dim_v: i32 = head_dim as i32;
        let mut out_v: u64 = output;

        let mut params = [
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
                params.as_mut_ptr(),
                None,
            )?;
        }

        Ok(())
    }
}
