//! Inference engine for tensor computation.
//!
//! Actual tensor computation layer. Separate from PESTI host.
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

use crate::error::RunnerError;
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use half::f16;
use std::sync::Arc;
use tracing::warn;

// Import InertiaManager for computational inertia support
use crate::inertia::InertiaManager;

// Import GEMM trait to ensure CpuGemmKernel methods are in scope
#[cfg(feature = "cuda")]
use crate::kernel::gemm::{CudaGemmKernel, GemmKernel};
#[cfg(not(feature = "cuda"))]
use crate::kernel::gemm_stub::GemmKernel;

// Import AttentionKernel trait and its Error type to ensure CpuAttentionKernel methods are in scope
#[cfg(not(feature = "cuda"))]
use crate::kernel::AttentionKernel;
#[cfg(feature = "cuda")]
use crate::kernel::attention::{AttentionKernel, GemmBasedAttentionKernel};

/// Inference engine for tensor computation with computational inertia support.
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
    stream: Option<Arc<cudarc::driver::safe::CudaStream>>,
    /// Memory manager for allocating device/host buffers.
    memory_manager: crate::kernel::MemoryManager,
    /// Backup CPU GEMM kernel for runtime fallback.
    cpu_gemm: crate::kernel::CpuGemmKernel,
    /// Backup CPU attention kernel for runtime fallback.
    cpu_attention: crate::kernel::CpuAttentionKernel,
    /// Whether a real CUDA GEMM kernel was successfully built (cuda feature).
    #[cfg(feature = "cuda")]
    gpu_gemm: bool,
    /// Computational inertia manager: logs demand when GPU unavailable.
    inertia_manager: InertiaManager,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        // CPU-only mode - just use CPU kernels
        #[cfg(not(feature = "cuda"))]
        {
            use crate::kernel::attention_stub::AttentionArch;
            Self {
                device,
                dtype,
                gemm: Box::new(crate::kernel::CpuGemmKernel::new()),
                attention: Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu)),
                memory_manager: crate::kernel::MemoryManager::Cpu(
                    crate::kernel::CpuMemoryBackend::new(1024 * 1024),
                ),
                cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
                inertia_manager: InertiaManager::new(1024), // default queue size
            }
        }

        #[cfg(feature = "cuda")]
        {
            use crate::cuda_runtime::{CudaRuntime, is_available};
            use crate::kernel::{AttentionArch, CudaGemmKernelBuilder, GemmArch};

            // Try to initialize CUDA only if the caller requested a CUDA device.
            let (cuda_runtime, stream) = match &device {
                Device::Cuda(cuda_dev) => {
                    let ordinal = match cuda_dev.location() {
                        candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                        _ => 0,
                    };
                    match CudaRuntime::new(ordinal) {
                        Ok(rt) => {
                            let rt = Arc::new(rt);
                            match rt.new_stream() {
                                Ok(s) => (Some(rt), Some(s)),
                                Err(e) => (Some(rt), None),
                            }
                        }
                        Err(e) => (None, None),
                    }
                }
                _ => (None, None),
            };

            // Initialize GEMM kernel first
            let mut gpu_gemm = false;
            let arch = if let Some(cuda_rt) = &cuda_runtime {
                if cuda_rt.device_info().supports_wgmma() {
                    Some(GemmArch::Wgmma)
                } else if cuda_rt.device_info().supports_tcgen05() {
                    Some(GemmArch::Tcgen05)
                } else {
                    Some(GemmArch::Mma)
                }
            } else {
                None
            };

            let (gemm, cuda_gemm_for_attention): (
                Box<dyn GemmKernel + Send + Sync>,
                Option<CudaGemmKernel>,
            ) = if let (Some(cuda_rt), Some(s), Some(arch)) = (&cuda_runtime, &stream, &arch) {
                match CudaGemmKernelBuilder::new(
                    arch.clone(),
                    cuda_rt.context().clone(),
                    s.clone(),
                    cuda_rt.device_info().clone(),
                )
                .build()
                {
                    Ok(kernel) => {
                        gpu_gemm = true;
                        // Keep a clone for attention kernel
                        let kernel_clone = kernel.clone();
                        (Box::new(kernel), Some(kernel_clone))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to initialize CUDA GEMM kernel, falling back to CPU");
                        (Box::new(crate::kernel::CpuGemmKernel::new()), None)
                    }
                }
            } else {
                (Box::new(crate::kernel::CpuGemmKernel::new()), None)
            };

            // Initialize attention kernel using the GEMM kernel if available
            let (attention, backend): (
                Box<dyn AttentionKernel + Send + Sync>,
                Option<Arc<crate::kernel::memory::CudaMemoryBackend>>,
            ) = if is_available() {
                if let (Some(cuda_rt), Some(gemm_kernel)) = (&cuda_runtime, cuda_gemm_for_attention)
                {
                    let s = stream.as_ref().unwrap();
                    let info = cuda_rt.device_info().clone();
                    let backend = Arc::new(
                        crate::kernel::memory::CudaMemoryBackend::with_device_info(s.clone(), info),
                    );

                    // GemmBasedAttentionKernel::new() returns the struct directly (no Result)
                    let softmax_kernel: Box<dyn crate::kernel::SoftmaxKernel> =
                        Box::new(crate::kernel::CpuSoftmaxKernel::new());
                    let attention_kernel =
                        GemmBasedAttentionKernel::new(gemm_kernel, backend.clone(), softmax_kernel);
                    tracing::info!("Using GEMM-based attention kernel (Option A)");
                    (Box::new(attention_kernel), Some(backend))
                } else {
                    (
                        Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu)),
                        None,
                    )
                }
            } else {
                (
                    Box::new(crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu)),
                    None,
                )
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
                memory_manager: crate::kernel::MemoryManager::Cpu(
                    crate::kernel::CpuMemoryBackend::new(1024 * 1024),
                ),
                cpu_gemm: crate::kernel::CpuGemmKernel::new(),
                cpu_attention: crate::kernel::CpuAttentionKernel::new(AttentionArch::Cpu),
                #[cfg(feature = "cuda")]
                gpu_gemm,
                inertia_manager: InertiaManager::new(1024), // default queue size
            }
        }
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
        match self.gemm.matmul(alpha, a, b, beta, c, m, n, k) {
            Ok(()) => Ok(()),
            Err(crate::kernel::GemmError::NotAvailable) => {
                warn!(m, n, k, "GEMM: GPU not available, falling back to CPU");
                self.cpu_gemm
                    .matmul(alpha, a, b, beta, c, m, n, k)
                    .map_err(|e| RunnerError::Tensor(format!("GEMM CPU fallback failed: {e}")))
            }
            Err(e) => {
                warn!(error = %e, m, n, k, "GEMM: GPU kernel failed, falling back to CPU");
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

    /// Get pending work for execution when GPU becomes available.
    pub fn get_pending_for_execution(&mut self) -> Vec<crate::inertia::Demand> {
        self.inertia_manager.get_pending_for_execution()
    }

    /// Get inertia manager stats.
    pub fn inertia_stats(&self) -> crate::inertia::InertiaStats {
        self.inertia_manager.stats()
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

    /// Get backend description string.
    pub fn backend_description(&self) -> String {
        if self.gpu_available() {
            format!(
                "GPU ({} @ {})",
                self.gemm_arch().name(),
                self.device_info().unwrap_or_default()
            )
        } else {
            "CPU".to_string()
        }
    }

    /// Run scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V.
    pub fn attention(
        &self,
        query: &crate::kernel::DeviceBuffer<f16>,
        key_cache: &crate::kernel::Kvcache,
        value_cache: &crate::kernel::Kvcache,
        mask: Option<&crate::kernel::DeviceBuffer<f32>>,
        config: &crate::kernel::AttentionConfig,
    ) -> Result<crate::kernel::DeviceBuffer<f32>, RunnerError> {
        match self
            .attention
            .forward(query, key_cache, value_cache, mask, config)
        {
            Ok(output) => Ok(output),
            Err(crate::kernel::AttentionError::NotAvailable) => {
                warn!("Attention: GPU not available, falling back to CPU");
                self.cpu_attention
                    .forward(query, key_cache, value_cache, mask, config)
                    .map_err(|e| RunnerError::Tensor(format!("Attention CPU fallback failed: {e}")))
            }
            Err(e) => {
                warn!(error = %e, "Attention: GPU kernel failed, falling back to CPU");
                self.cpu_attention
                    .forward(query, key_cache, value_cache, mask, config)
                    .map_err(|e| RunnerError::Tensor(format!("Attention CPU fallback failed: {e}")))
            }
        }
    }
}
