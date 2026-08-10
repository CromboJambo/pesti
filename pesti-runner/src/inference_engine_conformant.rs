//! Inference engine for LLM inference with CUDA GPU acceleration.
//!
//! This is the main orchestrator that ties together kernels, memory, and KV cache.
//! Uses dispatch patterns to route computation to optimal backend (CPU vs GPU).

use crate::kernel::{
    AttentionArch, CudaGemmKernelBuilder, GemmArch, MemoryBackend, SoftmaxKernel,
};
use candle_core::Device;

// --- CUDA-specific imports ---
#[cfg(feature = "cuda")]
use crate::cuda_runtime::{CudaRuntime, is_available as cuda_is_available};
#[cfg(feature = "cuda")]
use crate::kernel::{fused_attention_conformant::build_fused_attention_kernel_conformant, AttentionKernel, CpuAttentionKernel, FusedAttentionArchConformant, FusedAttentionConfigConformant, FusedAttentionKernelConformant, FusedAttentionWrapperConformant};

/// Inference engine for tensor computation with computational inertia support.
pub struct InferenceEngine {
    pub device: Device,
    pub dtype: DType,
    attention: Box<dyn AttentionKernel + Send + Sync>,
    #[cfg(feature = "cuda")]
    cuda_runtime: Option<Arc<CudaRuntime>>,
}

impl InferenceEngine {
    /// Create a new inference engine with optional CUDA device.
    pub fn new(on_device: bool, config: InferenceConfig) -> Result<Self, Error> {
        let device = if on_device && cfg!(feature = "cuda") && cuda_is_available() {
            Device::Cuda(cudarc::driver::CudaDevice::new(0)?)
        } else {
            Device::Cpu
        };

        let dtype = DType::F16; // LLM inference uses f16 for weights, f32 for activations

        #[cfg(feature = "cuda")]
        if on_device && cuda_is_available() {
            // Try fused attention kernel first (rope + softmax in one launch) with llama.cpp layout conformance
            let arch = FusedAttentionArchConformant::MmaSync;
            match build_fused_attention_kernel_conformant(arch, config.cuda_context.clone(), config.cuda_stream.clone()) {
                Ok(fused_kernel) => {
                    tracing::info!("Using fused attention kernel (RoPE + softmax in one launch)");
                    // Wrap fused kernel in AttentionKernel trait interface
                    let wrapped = FusedAttentionWrapperConformant::new(fused_kernel);
                    return Ok(Self {
                        device,
                        dtype,
                        attention: Box::new(wrapped),
                        #[cfg(feature = "cuda")]
                        cuda_runtime: Some(config.cuda_runtime.clone()),
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Fused attention kernel failed, falling back to GEMM-based");
                }
            }

            // Fallback: GEMM-based attention (Option A from roadmap)
            let gemm_kernel = CudaGemmKernelBuilder::new(GemmArch::MmaSync, config.cuda_context.clone(), config.cuda_stream.clone())?
                .build()?;

            let softmax_kernel: Box<dyn SoftmaxKernel> = Box::new(CpuSoftmaxKernel::new());
            let attention_kernel = crate::kernel::GemmBasedAttentionKernel::new(gemm_kernel, config.backend.clone(), softmax_kernel);

            tracing::info!("Using GEMM-based attention kernel (Option A)");
            return Ok(Self {
                device,
                dtype,
                attention: Box::new(attention_kernel),
                #[cfg(feature = "cuda")]
                cuda_runtime: Some(config.cuda_runtime.clone()),
            });
        }

        // CPU-only fallback for no GPU or non-cuda builds
        let cpu_attention = CpuAttentionKernel::new(AttentionArch::Cpu);

        tracing::info!("Using CPU attention kernel (fallback)");
        Ok(Self {
            device,
            dtype,
            attention: Box::new(cpu_attention),
            #[cfg(feature = "cuda")]
            cuda_runtime: None,
        })
    }

    /// Get the current device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the data type used for weights.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Configuration for InferenceEngine creation.
pub struct InferenceConfig {
    #[cfg(feature = "cuda")]
    pub cuda_context: Arc<cudarc::driver::safe::CudaContext>,
    #[cfg(feature = "cuda")]
    pub cuda_stream: Arc<cudarc::driver::safe::CudaStream>,
    #[cfg(feature = "cuda")]
    pub backend: Arc<crate::kernel::memory::CudaMemoryBackend>,
    #[cfg(feature = "cuda")]
    pub cuda_runtime: Arc<CudaRuntime>,
}

#[cfg(feature = "cuda")]
impl InferenceConfig {
    pub fn new(cuda_rt: &Arc<CudaRuntime>) -> Self {
        let context = cuda_rt.context().clone();
        let stream = cuda_rt.stream().clone();
        let backend = Arc::new(crate::kernel::memory::CudaMemoryBackend::default());

        Self {
            cuda_context: context,
            cuda_stream: stream,
            backend,
            cuda_runtime: cuda_rt.clone(),
        }
    }
}
