//! Fused attention wrapper implementing AttentionKernel trait.

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::Kvcache;
#[cfg(feature = "cuda")]
use crate::kernel::{AttentionArch, AttentionConfig};
#[cfg(not(feature = "cuda"))]
use crate::kernel::attention_stub::{AttentionArch, AttentionConfig};

// Import fused kernel types
#[cfg(feature = "cuda")]
use super::fused_attention::{build_fused_attention_kernel, FusedAttentionKernel};

/// Wrapper that implements `AttentionKernel` trait for fused attention.
#[derive(Clone)]
pub struct FusedAttentionWrapper {
    #[cfg(feature = "cuda")]
    kernel: FusedAttentionKernel,
    #[cfg(not(feature = "cuda"))]
    _phantom: std::marker::PhantomData<*const ()>,
}

impl Default for FusedAttentionWrapper {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Placeholder - real kernel created via builder
            let dummy_kernel = unsafe {
                std::mem::transmute::<u64, FusedAttentionKernel>(0)
            };
            Self { kernel: dummy_kernel }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self { _phantom: std::marker::PhantomData }
        }
    }
}

#[cfg(feature = "cuda")]
impl FusedAttentionWrapper {
    pub fn new(kernel: FusedAttentionKernel, _backend: std::sync::Arc<crate::kernel::memory::CudaMemoryBackend>) -> Self {
        Self { kernel }
    }
}

/// Attention errors for fused wrapper.
#[derive(Debug, thiserror::Error)]
pub enum FusedAttentionError {
    #[error("not available")]
    NotAvailable,

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),
}

#[cfg(feature = "cuda")]
impl crate::kernel::AttentionKernel for FusedAttentionWrapper {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, crate::kernel::AttentionError> {
        // Extract device pointers from buffers
        let q_ptr = query.device_ptr();
        let k_ptr = key_cache.buffer().device_ptr();
        let v_ptr = value_cache.buffer().device_ptr();

        // Allocate output buffer for softmax scores
        let num_heads = config.num_heads;
        let seq_q = query.len() / (num_heads * config.head_dim);
        let seq_k = key_cache.seq_len();
        
        let s_len = seq_q * seq_k;
        let mut output = DeviceBuffer::<f32>::zeros_device(&crate::kernel::memory::CudaMemoryBackend::default(), s_len)?;
        let s_ptr = output.device_ptr();

        // Launch fused kernel
        self.kernel.launch(
            config.scale,  // 1/sqrt(head_dim)
            q_ptr,
            k_ptr,
            v_ptr,
            s_ptr,
            seq_q,
            seq_k,
            num_heads,
            config.head_dim,
            config.rope_base,
            config.max_pos,
        )?;

        Ok(output)
    }

    fn is_available(&self) -> bool {
        if cfg!(feature = "cuda") { true } else { false }
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::default() // Placeholder
    }
}

#[cfg(not(feature = "cuda"))]
impl crate::kernel::AttentionKernel for FusedAttentionWrapper {
    fn forward(
        &self,
        _query: &DeviceBuffer<f16>,
        _key_cache: &Kvcache,
        _value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        _config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, crate::kernel::AttentionError> {
        Err(crate::kernel::AttentionError::NotAvailable)
    }

    fn is_available(&self) -> bool {
        false
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::default()
    }
}
