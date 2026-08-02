use crate::cuda_runtime::{enumerate_devices, is_available, CudaRuntime};
use crate::error::RunnerError;
use crate::kernel::{
    AttentionArch, AttentionConfig, AttentionError, AttentionKernel, CpuAttentionKernel,
    CudaAttentionKernelBuilder, CudaGemmKernelBuilder, GemmArch, GemmError, GemmKernel,
    MemoryManager,
};
#[cfg(feature = "mistralrs")]
use crate::kernel::mistralrs_backend::MistralRsBackend;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use half::f16;
use std::sync::Arc;
use tracing::warn;
/// Inference engine for tensor computation.
///
/// actual tensor computation layer. separate from PESTI host.
pub struct InferenceEngine {
    pub device: candle_core::Device,
    pub dtype: DType,
    gemm: Box<dyn GemmKernel + Send + Sync>,
    attention: Box<dyn AttentionKernel + Send + Sync>,
    /// CUDA runtime for device memory management (None = CPU-only mode).
    cuda_runtime: Option<Arc<CudaRuntime>>,
    /// CUDA stream for async operations.
    stream: Option<Arc<cuda_core::CudaStream>>,
    /// Memory manager for allocating device/host buffers.
    memory_manager: MemoryManager,
    /// Backup CPU GEMM kernel for runtime fallback.
    cpu_gemm: crate::kernel::CpuGemmKernel,
    /// Backup CPU attention kernel for runtime fallback.
    cpu_attention: CpuAttentionKernel,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        // Try to initialize CUDA if device preference is GPU
        let (cuda_runtime, stream) = if matches!(device, Device::Cuda(_)) || is_available() {
            match CudaRuntime::for_default_device() {
                Ok(rt) => {
                    let rt = Arc::new(rt);
                    match rt.new_stream() {
                        Ok(stream) => (Some(rt), Some(stream)),
                        Err(_) => (Some(rt), None),
                    }
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        // Initialize GEMM kernel: prefer mistral.rs if feature enabled and available,
        // then fall back to CUDA PTX, then CPU.
        let gemm: Box<dyn GemmKernel + Send + Sync> = if let (Some(cuda_rt), Some(s)) = (&cuda_runtime, &stream) {
            let arch = if cuda_rt.device_info().supports_wgmma() {
                GemmArch::Wgmma
            } else if cuda_rt.device_info().supports_tcgen05() {
                GemmArch::Tcgen05
            } else {
                GemmArch::Wgmma
            };

            // Try mistral.rs backend first if enabled
            #[cfg(feature = "mistralrs")]
            {
                let mr = MistralRsBackend::default();
                if let Some(kernel) = mr.create_gemm_kernel(arch) {
                    tracing::info!("Using mistral.rs GEMM kernel (arch={})", arch.name());
                    return Self {
                        device, dtype, gemm: kernel, attention: Box::new(CpuAttentionKernel::new()),
                        cuda_runtime, stream, memory_manager: MemoryManager::new(),
                        cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                        cpu_attention: CpuAttentionKernel::new(),
                    };
                }
            }

            match CudaGemmKernelBuilder::new(arch, cuda_rt.context().clone(), s.clone(), cuda_rt.device_info().clone()).build() {
                Ok(kernel) => Box::new(kernel),
                Err(e) => {
                    eprintln!("Failed to initialize CUDA GEMM kernel: {}. Falling back to CPU.", e);
                    Box::new(crate::kernel::CpuGemmKernel::new())
                }
            }
        } else {
            Box::new(crate::kernel::CpuGemmKernel::new())
        };

        // Initialize attention kernel: same priority order
        let attention: Box<dyn AttentionKernel + Send + Sync> = if is_available() {
            #[cfg(feature = "mistralrs")]
            {
                let mr = MistralRsBackend::default();
                if let Some(kernel) = mr.create_attention_kernel(AttentionArch::Wgmma) {
                    tracing::info!("Using mistral.rs attention kernel");
                    return Self {
                        device, dtype, gemm, attention: kernel,
                        cuda_runtime, stream, memory_manager: MemoryManager::new(),
                        cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                        cpu_attention: CpuAttentionKernel::new(),
                    };
                }
            }
            
            // Try CUDA attention kernel builder (similar to GEMM)
            if let (Some(cuda_rt), Some(s)) = (&cuda_runtime, &stream) {
                let arch = if cuda_rt.device_info().supports_wgmma() {
                    AttentionArch::Wgmma
                } else if cuda_rt.device_info().supports_tcgen05() {
                    AttentionArch::Tcgen05
                } else {
                    AttentionArch::Wgmma
                };
                
                match CudaAttentionKernelBuilder::new(
                    arch, 
                    cuda_rt.context().clone(), 
                    s.clone(), 
                    cuda_rt.device_info().clone()
                ).build() {
                    Ok(kernel) => Box::new(kernel),
                    Err(e) => {
                        eprintln!("Failed to initialize CUDA attention kernel: {}. Falling back to CPU.", e);
                        Box::new(CpuAttentionKernel::new())
                    }
                }
            } else {
                Box::new(CpuAttentionKernel::new())
            }
        } else {
            Box::new(crate::kernel::CpuAttentionKernel::new())
        };

        Self {
            device,
            dtype,
            gemm,
            attention,
            cuda_runtime,
            stream,
            memory_manager: MemoryManager::new(),
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: CpuAttentionKernel::new(),
        }
    }

    /// Create engine with a specific GEMM kernel.
    pub fn with_gemm(device: Device, dtype: DType, gemm: Box<dyn GemmKernel + Send + Sync>) -> Self {
        let attention = Box::new(CpuAttentionKernel::new());

        let (cuda_runtime, stream) = if is_available() {
            match CudaRuntime::for_default_device() {
                Ok(rt) => {
                    let rt = Arc::new(rt);
                    match rt.new_stream() {
                        Ok(stream) => (Some(rt), Some(stream)),
                        Err(_) => (Some(rt), None),
                    }
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        };

        Self {
            device,
            dtype,
            gemm,
            attention,
            cuda_runtime,
            stream,
            memory_manager: MemoryManager::new(),
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: CpuAttentionKernel::new(),
        }
    }

    /// Get the CUDA stream for device operations.
    fn get_stream(&self) -> Option<&Arc<cuda_core::CudaStream>> {
        self.stream.as_ref()
    }

    /// Check if GPU path is available.
    pub fn gpu_available(&self) -> bool {
        self.cuda_runtime.is_some() && self.gemm.is_available()
    }

    /// Get device info including CUDA details if available.
    pub fn full_device_info(&self) -> Result<String, RunnerError> {
        let base = self.device_info()?;
        if let Some(cuda) = &self.cuda_runtime {
            let info = cuda.device_info();
            Ok(format!(
                "{} | GPU: {} (sm_{}.{}) free={:.1}GiB/total={:.1}GiB",
                base,
                info.name,
                info.compute_capability.0,
                info.compute_capability.1,
                info.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            ))
        } else {
            Ok(base)
        }
    }

    /// Run a GEMM operation: C = alpha * A @ B + beta * C.
    ///
    /// A: [m x k] f16, B: [k x n] f16, C: [m x n] f32
    ///
    /// Falls back to CPU if the GPU kernel fails at runtime (OOM, invalid context, etc.).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        alpha: f32,
        a: &crate::kernel::DeviceBuffer<f16>,
        b: &crate::kernel::DeviceBuffer<f16>,
        beta: f32,
        c: &mut crate::kernel::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), RunnerError> {
        // Try GPU first
        match self.gemm.matmul(alpha, a, b, beta, c, m, n, k) {
            Ok(()) => Ok(()),
            Err(GemmError::NotAvailable) => {
                // GPU not available — fall through to CPU
                warn!(m, n, k, "GEMM: GPU not available, falling back to CPU");
                self.cpu_gemm
                    .matmul(alpha, a, b, beta, c, m, n, k)
                    .map_err(|e| RunnerError::Tensor(format!("GEMM CPU fallback failed: {e}")))
            }
            Err(e) => {
                // GPU failed — try CPU fallback
                warn!(
                    error = %e,
                    m, n, k,
                    "GEMM: GPU kernel failed, falling back to CPU"
                );
                self.cpu_gemm
                    .matmul(alpha, a, b, beta, c, m, n, k)
                    .map_err(|e| RunnerError::Gemm {
                        arch: self.gemm.arch().name().to_string(),
                        m,
                        n,
                        k,
                        detail: e,
                    })
            }
        }
    }

    /// Get the GEMM kernel's target architecture.
    pub fn gemm_arch(&self) -> GemmArch {
        self.gemm.arch()
    }

    /// Check if the GEMM kernel is available on this system.
    pub fn gemm_available(&self) -> bool {
        self.gemm.is_available()
    }

    /// Run inference on a loaded model.
    pub fn infer(&self, model: &impl Module, input: Tensor) -> Result<Tensor, RunnerError> {
        model
            .forward(&input)
            .map_err(|e: candle_core::Error| RunnerError::Tensor(e.to_string()))
    }

    /// Materialize lazy-loaded tensor from manifest.
    pub fn materialize_tensor(
        &self,
        file_path: &str,
        _tensor_name: &str,
    ) -> Result<Tensor, RunnerError> {
        let data = std::fs::read(file_path)
            .map_err(|e: std::io::Error| RunnerError::Asset(e.to_string()))?;
        Tensor::from_raw_buffer(&data, self.dtype, &[1], &self.device)
            .map_err(|e: candle_core::Error| RunnerError::Tensor(e.to_string()))
    }

    /// Get device info.
    pub fn device_info(&self) -> Result<String, RunnerError> {
        Ok(match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(ordinal) => format!("cuda:{ordinal:?}"),
            Device::Metal(_) => "metal".to_string(),
        })
    }

    /// Get dtype info.
    pub fn dtype_info(&self) -> Result<String, RunnerError> {
        Ok(match self.dtype {
            DType::F32 => "F32".to_string(),
            DType::F16 => "F16".to_string(),
            DType::I64 => "I64".to_string(),
            DType::I32 => "I32".to_string(),
            DType::U8 => "U8".to_string(),
            _ => "unknown".to_string(),
        })
    }

    /// Run scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V.
    ///
    /// `query` — [query_seq_len x (num_heads * head_dim)] f16
    /// `key_cache` — KV cache containing K tensor
    /// `value_cache` — KV cache containing V tensor
    /// `mask` — optional [query_seq_len x cache_seq_len] f32 mask
    /// `config` — attention configuration (num_heads, head_dim, max_seq, arch)
    ///
    /// Returns output tensor [query_seq_len x (num_heads * head_dim)] f32
    ///
    /// Falls back to CPU if the GPU kernel fails at runtime.
    pub fn attention(
        &self,
        query: &crate::kernel::DeviceBuffer<f16>,
        key_cache: &crate::kernel::Kvcache,
        value_cache: &crate::kernel::Kvcache,
        mask: Option<&crate::kernel::DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<crate::kernel::DeviceBuffer<f32>, RunnerError> {
        // Try GPU first
        match self.attention.forward(query, key_cache, value_cache, mask, config) {
            Ok(output) => Ok(output),
            Err(AttentionError::NotAvailable) => {
                // GPU not available — fall through to CPU
                warn!("Attention: GPU not available, falling back to CPU");
                self.cpu_attention
                    .forward(query, key_cache, value_cache, mask, config)
                    .map_err(|e| RunnerError::Tensor(format!("Attention CPU fallback failed: {e}")))
            }
            Err(e) => {
                // GPU failed — try CPU fallback
                warn!(
                    error = %e,
                    "Attention: GPU kernel failed, falling back to CPU"
                );
                let num_heads = config.num_heads;
                let head_dim = config.head_dim;
                let seq = key_cache.seq_len();
                self.cpu_attention
                    .forward(query, key_cache, value_cache, mask, config)
                    .map_err(|_| RunnerError::Attention {
                        num_heads,
                        head_dim,
                        seq,
                        detail: e,
                    })
            }
        }
    }

    /// Get the attention kernel's target architecture.
    pub fn attention_arch(&self) -> AttentionArch {
        self.attention.arch()
    }

    /// Check if the attention kernel is available on this system.
    pub fn attention_available(&self) -> bool {
        self.attention.is_available()
    }

    /// List available CUDA devices.
    pub fn list_devices() -> Result<Vec<crate::cuda_runtime::CudaDeviceInfo>, RunnerError> {
        enumerate_devices().map_err(|e| RunnerError::Device(e.to_string()))
    }

    /// Get a description of the active inference backend.
    pub fn backend_description(&self) -> String {
        #[cfg(feature = "mistralrs")]
        if self.cuda_runtime.is_some() && self.gemm.is_available() {
            // Check if we're using mistral.rs by checking the kernel arch
            match self.gemm_arch() {
                GemmArch::Wgmma | GemmArch::Tcgen05 => {
                    // Could be either backend — use runtime info if available
                    return format!("GPU ({})", self.full_device_info().unwrap_or_else(|_| "unknown".to_string()));
                }
            }
        }

        if self.gemm.is_available() {
            "GPU (CUDA PTX)".to_string()
        } else {
            "CPU (reference)".to_string()
        }
    }
}
