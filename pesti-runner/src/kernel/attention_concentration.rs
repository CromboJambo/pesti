//! Attention concentration validation kernel for early-exit inference.
//!
//! Computes max attention weight across all positions for each head as a
//! proxy for "attention confidence." High concentration (> threshold) suggests
//! the model is confident about where to attend, potentially enabling early exit.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::safe::{CudaContext, CudaStream};

/// Configuration for attention concentration validation.
#[derive(Debug, Clone)]
pub struct AttentionConcentrationConfig {
    pub seq_len: usize,
    pub num_heads: usize,
    pub batch_size: usize,
}

impl Default for AttentionConcentrationConfig {
    fn default() -> Self {
        Self {
            seq_len: 1024,
            num_heads: 32,
            batch_size: 1,
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA attention concentration kernel.
#[derive(Clone)]
pub struct AttentionConcentrationKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl AttentionConcentrationKernel {
    /// Launch attention concentration kernel.
    ///
    /// # Arguments
    /// * `attention_weights` - Attention weights on device [batch*heads*seq_len] f32
    /// * `concentrations` - Output buffer for concentration scores [batch*heads] f32
    /// * `seq_len` - Sequence length
    /// * `num_heads` - Number of attention heads
    /// * `batch_size` - Batch size
    pub fn launch(
        &self,
        attention_weights: u64,
        concentrations: u64,
        seq_len: usize,
        num_heads: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        let mut attn_v: u64 = attention_weights;
        let mut conc_v: u64 = concentrations;
        let mut seq_len_v: i32 = seq_len as i32;
        let mut num_heads_v: i32 = num_heads as i32;
        let mut batch_size_v: i32 = batch_size as i32;

        let mut params = [
            &mut attn_v as *mut u64 as *mut std::ffi::c_void,
            &mut conc_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_len_v as *mut i32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
            &mut batch_size_v as *mut i32 as *mut std::ffi::c_void,
        ];

        let grid = (batch_size as u32, num_heads as u32, 1u32);
        let block = (256u32, 1u32, 1u32);

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0u32,
                crate::cuda_shim::cu_stream(&self.stream),
                &mut params,
            )
            .map_err(|e| format!("kernel launch failed: {:?}", e))?;
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub fn build_attention_concentration_kernel(
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> Result<AttentionConcentrationKernel, String> {
    let ptx_src = include_str!("ptx/attention_concentration.ptx");

    let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_src)
        .map_err(|e| format!("module load failed: {:?}", e))?;

    let function = module
        .load_function("attention_concentration_kernel")
        .map_err(|e| format!("function lookup failed: {:?}", e))?;

    Ok(AttentionConcentrationKernel {
        context,
        stream,
        module,
        function,
    })
}
