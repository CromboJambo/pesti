//! Fused attention kernel with delta-encoded KV cache (Route C).
//!
//! Reads i8 delta codes directly in PTX, exploiting prefix-sum structure
//! for O(n) score computation without reconstructing full K/V vectors.
//!
//! Mathematical basis:
//!   K[pos] = base + sum_{j=1}^{pos} delta_k[j]
//!   score[pos] = q . K[pos] * scale
//!   output[i] = v_base[i]*total_w + sum_j delta_v[j][i]*v_step*suffix_w[j]

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaContext, CudaStream};

/// Configuration for delta-encoded fused decode attention.
#[derive(Debug)]
pub struct DeltaFusedDecodeAttentionConfig {
    pub head_dim: usize,
    pub scale: f32,
}

impl Default for DeltaFusedDecodeAttentionConfig {
    fn default() -> Self {
        let head_dim = 64;
        Self {
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA fused attention kernel with delta-encoded KV cache.
#[derive(Clone)]
pub struct DeltaFusedDecodeAttentionKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl DeltaFusedDecodeAttentionKernel {
    /// Launch fused attention kernel with delta-encoded KV cache.
    ///
    /// # Arguments
    /// * `q` - Query vector on device [head_dim] f32
    /// * `k_base` - K base vector (position 0) on device [head_dim] f32
    /// * `v_base` - V base vector (position 0) on device [head_dim] f32
    /// * `k_deltas` - K delta codes on device [(seq_len-1) x head_dim] i8, row-major
    /// * `v_deltas` - V delta codes on device [(seq_len-1) x head_dim] i8, row-major
    /// * `k_step` - Quantization step size for K deltas
    /// * `v_step` - Quantization step size for V deltas
    /// * `scale` - 1/sqrt(head_dim)
    /// * `seq_len` - Total sequence length (including position 0)
    /// * `head_dim` - Head dimension (must equal block size)
    /// * `output` - Output buffer on device [head_dim] f32
    pub fn launch(
        &self,
        q: u64,
        k_base: u64,
        v_base: u64,
        k_deltas: u64,
        v_deltas: u64,
        k_step: f32,
        v_step: f32,
        scale: f32,
        seq_len: usize,
        head_dim: usize,
        output: u64,
    ) -> Result<(), String> {
        let mut q_v: u64 = q;
        let mut k_base_v: u64 = k_base;
        let mut v_base_v: u64 = v_base;
        let mut k_deltas_v: u64 = k_deltas;
        let mut v_deltas_v: u64 = v_deltas;
        let mut k_step_v: f32 = k_step;
        let mut v_step_v: f32 = v_step;
        let mut scale_v: f32 = scale;
        let mut seq_len_v: u32 = seq_len as u32;
        let mut head_dim_v: u32 = head_dim as u32;
        let mut out_v: u64 = output;

        let mut params = [
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_base_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_base_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_deltas_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_deltas_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_step_v as *mut f32 as *mut std::ffi::c_void,
            &mut v_step_v as *mut f32 as *mut std::ffi::c_void,
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut seq_len_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
        ];

        let grid = (1u32, 1u32, 1u32);
        let block = (head_dim as u32, 1u32, 1u32);

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0u32, // shared memory declared in PTX
                crate::cuda_shim::cu_stream(&self.stream),
                &mut params,
            )
            .map_err(|e| format!("delta attention kernel launch failed: {:?}", e))?;
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub fn build_delta_fused_decode_attention_kernel(
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> Result<DeltaFusedDecodeAttentionKernel, String> {
    let ptx_src = include_str!("ptx/fused_decode_attention_delta.ptx");

    let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_src)
        .map_err(|e| format!("delta attention module load failed: {:?}", e))?;

    let function = module
        .load_function("fused_attention_delta")
        .map_err(|e| format!("delta attention function lookup failed: {:?}", e))?;

    Ok(DeltaFusedDecodeAttentionKernel {
        context,
        stream,
        module,
        function,
    })
}
