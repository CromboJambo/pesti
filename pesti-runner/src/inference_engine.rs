//! Inference engine for tensor computation.
//!
//! actual tensor computation layer. separate from PESTI host.

use std::sync::Arc;

use crate::error::RunnerError;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use half::f16;
use tracing::warn;

// Import GEMM trait to ensure CpuGemmKernel methods are in scope
#[cfg(not(feature = "cuda"))]
use crate::kernel::gemm_stub::GemmKernel;
#[cfg(feature = "cuda")]
use crate::kernel::gemm::GemmKernel;

// Import AttentionKernel trait and its Error type to ensure CpuAttentionKernel methods are in scope
#[cfg(not(feature = "cuda"))]
use crate::kernel::AttentionKernel;
#[cfg(feature = "cuda")]
use crate::kernel::attention::AttentionKernel;

/// Inference engine for tensor computation.
pub struct InferenceEngine {
    pub device: candle_core::Device,
    pub dtype: DType,
    gemm: Box<dyn crate::kernel::GemmKernel + Send + Sync>,
    attention: Box<dyn crate::kernel::AttentionKernel + Send + Sync>,
    /// CUDA runtime for device memory management (None = CPU-only mode).
    #[cfg(feature = "cuda")]
    cuda_runtime: Option<Arc<crate::cuda_runtime::CudaRuntime>>,
    /// CUDA stream for async operations.
    #[cfg(feature = "cuda")]
    stream: Option<Arc<cuda_core::CudaStream>>,
    /// Memory manager for allocating device/host buffers.
    memory_manager: crate::kernel::MemoryManager,
    /// Backup CPU GEMM kernel for runtime fallback.
    cpu_gemm: crate::kernel::CpuGemmKernel,
    /// Backup CPU attention kernel for runtime fallback.
    cpu_attention: crate::kernel::CpuAttentionKernel,
    /// Whether a real CUDA GEMM kernel was successfully built (cuda feature).
    #[cfg(feature = "cuda")]
    gpu_gemm: bool,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        // CPU-only mode - just use CPU kernels
        #[cfg(not(feature = "cuda"))]
        {
            use crate::kernel::attention_stub::AttentionArch;
            return Self {
                device,
                dtype,
                gemm: Box::new(crate::kernel::CpuGemmKernel::new()),
                attention: Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu)),
                memory_manager: crate::kernel::MemoryManager::Cpu(
                    crate::kernel::CpuMemoryBackend::new(1024 * 1024),
                ),
                cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
            };
        }

        #[cfg(feature = "cuda")]
        {
            use crate::cuda_runtime::{enumerate_devices, is_available, CudaRuntime};
            use crate::kernel::{
                AttentionArch, AttentionConfig, AttentionKernel,
                CudaAttentionKernelBuilder, CudaGemmKernelBuilder, GemmArch, GemmError,
            };

            // Try to initialize CUDA only if the caller requested a CUDA device.
            // (NVML-based `is_available()` must not trigger CUDA context creation
            // for a CPU-only engine.) The requested ordinal is honored so a
            // multi-GPU host can pin inference to a specific device.
            let (cuda_runtime, stream) = match &device {
                Device::Cuda(cuda_dev) => {
                    // Extract the requested ordinal from candle's device
                    // location. (candle's DeviceId is a unique counter, not
                    // the GPU index — location() exposes the real ordinal.)
                    let ordinal = match cuda_dev.location() {
                        candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                        _ => 0,
                    };
                    match CudaRuntime::new(ordinal) {
                        Ok(rt) => {
                            let rt = Arc::new(rt);
                            match rt.new_stream() {
                                Ok(stream) => {
                                    (Some(rt), Some(stream))
                                },
                                Err(e) => {
                                    (Some(rt), None)
                                }
                            }
                        }
                        Err(e) => {
                            (None, None)
                        }
                    }
                }
                _ => {
                    (None, None)
                }
            };

            // Initialize GEMM kernel: prefer mistral.rs if feature enabled and available,
            // then fall back to CUDA PTX, then CPU.
            // `gpu_gemm` records whether a real CUDA kernel was built, so
            // `gpu_available()`/`backend_description()` report the truth even
            // when the builder falls back to CPU.
            let mut gpu_gemm = false;
            let gemm: Box<dyn crate::kernel::GemmKernel + Send + Sync> =
                if let (Some(cuda_rt), Some(s)) = (&cuda_runtime, &stream) {
                    // Select the best arch this device actually supports.
                    //  - wgmma:    Hopper (sm_90a) only
                    //  - tcgen05:  datacenter Blackwell (sm_100a) only
                    //  - mma.sync: every tensor-core GPU (sm_80..sm_120),
                    //    including consumer Blackwell RTX 50-series which has
                    //    neither wgmma nor tcgen05.
                    let arch = if cuda_rt.device_info().supports_wgmma() {
                        Some(GemmArch::Wgmma)
                    } else if cuda_rt.device_info().supports_tcgen05() {
                        Some(GemmArch::Tcgen05)
                    } else {
                        Some(GemmArch::Mma)
                    };

                    // Try mistral.rs backend first if enabled
                    #[cfg(feature = "mistralrs")]
                    {
                        use crate::kernel::mistralrs_backend::MistralRsBackend;
                        let mr = MistralRsBackend::default();
                        if let Some(arch) = arch {
                            if let Some(kernel) = mr.create_gemm_kernel(arch) {
                                tracing::info!("Using mistral.rs GEMM kernel (arch={})", arch.name());
                                gpu_gemm = true;
                                return Self {
                                    device,
                                    dtype,
                                    gemm: kernel,
                                    attention: Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu)),
                                    #[cfg(feature = "cuda")]
                                    cuda_runtime,
                                    #[cfg(feature = "cuda")]
                                    stream,
                                    memory_manager: crate::kernel::MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024)),
                                    cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                                    cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
                                    #[cfg(feature = "cuda")]
                                    gpu_gemm: true,
                                };
                            }
                        }
                    }

                    match arch {
                        Some(arch) => match CudaGemmKernelBuilder::new(
                            arch,
                            cuda_rt.context().clone(),
                            s.clone(),
                            cuda_rt.device_info().clone(),
                        )
                        .build()
                        {
                            Ok(kernel) => {
                                gpu_gemm = true;
                                Box::new(kernel)
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to initialize CUDA GEMM kernel, falling back to CPU");
                                Box::new(crate::kernel::CpuGemmKernel::new())
                            }
                        },
                        None => {
                            tracing::info!("No CUDA GEMM kernel for this device (needs sm_100+); using CPU");
                            Box::new(crate::kernel::CpuGemmKernel::new())
                        }
                    }
                } else {
                    Box::new(crate::kernel::CpuGemmKernel::new())
                };

            // Initialize attention kernel: same priority order
            let attention: Box<dyn crate::kernel::AttentionKernel + Send + Sync> =
                if is_available() {
                    #[cfg(feature = "mistralrs")]
                    {
                        use crate::kernel::mistralrs_backend::MistralRsBackend;
                        let mr = MistralRsBackend::default();
                        if let Some(kernel) =
                            mr.create_attention_kernel(AttentionArch::Wgmma)
                        {
                            tracing::info!("Using mistral.rs attention kernel");
                            return Self {
                                device,
                                dtype,
                                gemm,
                                attention: kernel,
                                #[cfg(feature = "cuda")]
                                cuda_runtime,
                                #[cfg(feature = "cuda")]
                                stream,
                                memory_manager: crate::kernel::MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024)),
                                cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                                cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
                                #[cfg(feature = "cuda")]
                                gpu_gemm,
                            };
                        }
                    }

                    // Try CUDA attention kernel builder (similar to GEMM)
                    if let (Some(cuda_rt), Some(s)) = (&cuda_runtime, &stream) {
                        // Same capability gate as GEMM: both tensor-core
                        // attention paths require Blackwell (sm_100+).
                        let arch = if cuda_rt.device_info().supports_wgmma() {
                            Some(AttentionArch::Wgmma)
                        } else if cuda_rt.device_info().supports_tcgen05() {
                            Some(AttentionArch::Tcgen05)
                        } else {
                            None
                        };

                        match arch {
                            Some(arch) => {
                                match CudaAttentionKernelBuilder::new(
                                    arch,
                                    cuda_rt.context().clone(),
                                    s.clone(),
                                    cuda_rt.device_info().clone(),
                                )
                                .build()
                                {
                                    Ok(kernel) => Box::new(kernel),
                                    Err(e) => {
                                        tracing::warn!(error = %e, "Failed to initialize CUDA attention kernel, falling back to CPU");
                                        Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu))
                                    }
                                }
                            }
                            None => {
                                tracing::info!("No CUDA attention kernel for this device (needs sm_100+); using CPU");
                                Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu))
                            }
                        }
                    } else {
                        Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu))
                    }
                } else {
                    Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu))
                };

            Self {
                device,
                dtype,
                gemm,
                attention,
                #[cfg(feature = "cuda")]
                cuda_runtime,
                #[cfg(feature = "cuda")]
                stream,
                memory_manager: crate::kernel::MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024)),
                cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
                #[cfg(feature = "cuda")]
                gpu_gemm,
            }
        }
    }

    /// Create engine with a specific GEMM kernel.
    pub fn with_gemm(
        device: Device,
        dtype: DType,
        gemm: Box<dyn crate::kernel::GemmKernel + Send + Sync>,
    ) -> Self {
        #[cfg(feature = "cuda")]
        use crate::kernel::attention::AttentionArch;
        #[cfg(not(feature = "cuda"))]
        use crate::kernel::attention_stub::AttentionArch;
        
        let attention = Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu));

        Self {
            device,
            dtype,
            gemm,
            attention,
            #[cfg(feature = "cuda")]
            cuda_runtime: None,
            #[cfg(feature = "cuda")]
            stream: None,
            memory_manager: crate::kernel::MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024)),
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
            #[cfg(feature = "cuda")]
            gpu_gemm: false,
        }
    }

    /// Get the CUDA stream for device operations.
    #[cfg(feature = "cuda")]
    fn get_stream(&self) -> Option<&Arc<cuda_core::CudaStream>> {
        self.stream.as_ref()
    }

    /// Check if GPU path is available.
    pub fn gpu_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_runtime.is_some() && self.gpu_gemm
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get device info including CUDA details if available.
    pub fn full_device_info(&self) -> Result<String, RunnerError> {
        let base = self.device_info()?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda_runtime {
            let info = cuda.device_info();
            return Ok(format!(
                "{} | GPU: {} (sm_{}.{}) free={:.1}GiB/total={:.1}GiB",
                base,
                info.name,
                info.compute_capability.0,
                info.compute_capability.1,
                info.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            ));
        }
        Ok(base)
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
            Err(crate::kernel::GemmError::NotAvailable) => {
                // GPU not available — fall through to CPU
                warn!(m, n, k, "GEMM: GPU not available, falling back to CPU");
                self.cpu_gemm
                    .matmul(alpha, a, b, beta, c, m, n, k)
                    .map_err(|e| {
                        RunnerError::Tensor(format!("GEMM CPU fallback failed: {e}"))
                    })
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
    pub fn gemm_arch(&self) -> crate::kernel::GemmArch {
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
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => format!("cuda:{ordinal:?}"),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_ordinal) => "cuda (stub)".to_string(),
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
        config: &crate::kernel::AttentionConfig,
    ) -> Result<crate::kernel::DeviceBuffer<f32>, RunnerError> {
        // Try GPU first
        match self.attention.forward(query, key_cache, value_cache, mask, config) {
            Ok(output) => Ok(output),
            Err(crate::kernel::AttentionError::NotAvailable) => {
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
                    .map_err(|e: crate::kernel::AttentionError| {
                        // Map kernel error to runner error (using the generic one from error.rs)
                        let detail = match e {
                            crate::kernel::AttentionError::LaunchFailed(msg) => {
                                crate::error::AttentionError::LaunchFailed(msg)
                            }
                            crate::kernel::AttentionError::NotAvailable => {
                                crate::error::AttentionError::NotAvailable
                            }
                            _ => crate::error::AttentionError::LaunchFailed(e.to_string()),
                        };
                        RunnerError::Attention {
                            num_heads,
                            head_dim,
                            seq,
                            detail,
                        }
                    })
            }
        }
    }

    /// Get the attention kernel's target architecture.
    pub fn attention_arch(&self) -> crate::kernel::AttentionArch {
        self.attention.arch()
    }

    /// Check if the attention kernel is available on this system.
    pub fn attention_available(&self) -> bool {
        self.attention.is_available()
    }

    /// List available CUDA devices.
    #[cfg(feature = "cuda")]
    pub fn list_devices() -> Result<Vec<crate::cuda_runtime::CudaDeviceInfo>, RunnerError> {
        crate::cuda_runtime::enumerate_devices().map_err(|e| {
            RunnerError::Device(e.to_string())
        })
    }

    /// Get a description of the active inference backend.
    pub fn backend_description(&self) -> String {
        #[cfg(feature = "cuda")]
        if self.gpu_gemm {
            return format!(
                "GPU ({})",
                self.full_device_info()
                    .unwrap_or_else(|_| "unknown".to_string())
            );
        }

        "CPU (reference)".to_string()
    }
}