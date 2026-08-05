//! GEMM kernel interface and configuration.
//!
//! Provides the core matmul abstraction used by the LLM inference engine.
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! The differentiator: proving tcgen05 matmul works for LLM workloads
//! with non-matrix layouts (KV cache updates: M:1×K, K:N×K shapes).

use crate::kernel::device_buf::DeviceBuffer;
use half::f16;

/// Tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GemmArch {
    /// WGMMA — warp group matrix multiply (sm_120, consumer Blackwell)
    Wgmma,
    /// tcgen05 — tensor core with tensor memory (sm_100, datacenter Blackwell)
    #[default]
    Tcgen05,
}

impl GemmArch {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgmma => "wgmma",
            Self::Tcgen05 => "tcgen05",
        }
    }

    pub fn supports_tma(&self) -> bool {
        // tcgen05 has native TMA support; WGMMA uses TMA for GMEM->SMEM copies
        true
    }

    pub fn tile_size(&self) -> usize {
        match self {
            Self::Wgmma => 64,    // 64x64x64 tiles
            Self::Tcgen05 => 128, // 128x128x16 tiles
        }
    }
}

/// Configuration for a GEMM kernel launch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GemmConfig {
    /// Target architecture
    pub arch: GemmArch,
    /// Whether to use TMA for async GMEM->SMEM copies
    pub use_tma: bool,
    /// Custom block size override (0 = use arch default)
    pub block_size: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            arch: GemmArch::default(),
            use_tma: true,
            block_size: 0,
        }
    }
}

impl GemmConfig {
    pub fn effective_block_size(&self) -> usize {
        if self.block_size > 0 {
            self.block_size
        } else {
            self.arch.tile_size()
        }
    }

    pub fn with_arch(mut self, arch: GemmArch) -> Self {
        self.arch = arch;
        self
    }

    pub fn with_tma(mut self, use_tma: bool) -> Self {
        self.use_tma = use_tma;
        self
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
}

// --- GemmKernel Trait ---

/// Trait for GEMM (matrix multiply) kernels.
///
/// Both GPU (CUDA) and CPU implementations implement this trait.
pub trait GemmKernel: Send + Sync {
    /// Perform GEMM: C = alpha * A @ B + beta * C
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError>;

    /// Check if the kernel is available on the current device.
    fn is_available(&self) -> bool;

    /// Get the target architecture.
    fn arch(&self) -> GemmArch;
}

// --- CPU Implementation (Fallback) ---

/// Simple CPU GEMM implementation for testing and fallback.
pub struct CpuGemmKernel;

impl CpuGemmKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuGemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmKernel for CpuGemmKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError> {
        let a_host = a.as_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: m * k,
            got: 0,
        })?;
        let b_host = b.as_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: k * n,
            got: 0,
        })?;
        let c_host = c.as_mut_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: m * n,
            got: 0,
        })?;

        // Simple O(m*n*k) GEMM
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a_host[i * k + l].to_f32() * b_host[l * n + j].to_f32();
                }
                c_host[i * n + j] = alpha * sum + beta * c_host[i * n + j];
            }
        }

        Ok(())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> GemmArch {
        GemmArch::default()
    }
}

// --- GPU Implementation (Real cuda-oxide backed) ---

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
/// CUDA implementation for GEMM kernel using cuda-oxide.
pub struct CudaGemmKernel {
    arch: GemmArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    module: Arc<cuda_core::CudaModule>,
    function: cuda_core::CudaFunction,
}

#[cfg(feature = "cuda")]
/// Builder for CudaGemmKernel that handles PTX loading and kernel resolution.
pub struct CudaGemmKernelBuilder {
    arch: GemmArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    device_info: crate::cuda_runtime::CudaDeviceInfo,
}

#[cfg(feature = "cuda")]
impl CudaGemmKernelBuilder {
    pub fn new(
        arch: GemmArch,
        context: Arc<cuda_core::CudaContext>,
        stream: Arc<cuda_core::CudaStream>,
        device_info: crate::cuda_runtime::CudaDeviceInfo,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
            device_info,
        }
    }

    /// Build the kernel by loading PTX module and resolving function.
    pub fn build(self) -> Result<CudaGemmKernel, GemmError> {
        // Pre-flight architecture check
        match self.arch {
            GemmArch::Wgmma if !self.device_info.supports_wgmma() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "WGMMA requires sm_120+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0,
                    self.device_info.compute_capability.1
                )));
            }
            GemmArch::Tcgen05 if !self.device_info.supports_tcgen05() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "tcgen05 requires sm_100+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0,
                    self.device_info.compute_capability.1
                )));
            }
            _ => {}
        }

        // Select PTX based on architecture
        let ptx_src = match self.arch {
            GemmArch::Wgmma => include_str!("ptx/gemm_wgmma.ptx"),
            GemmArch::Tcgen05 => include_str!("ptx/gemm_tcgen05.ptx"),
        };

        // Load module from PTX source
        let module = self
            .context
            .load_module_from_ptx_src(ptx_src)
            .map_err(|e| GemmError::Cuda(format!("module load failed: {}", e)))?;

        // Resolve kernel function
        let kernel_name = match self.arch {
            GemmArch::Wgmma => "gemm_wgmma_kernel",
            GemmArch::Tcgen05 => "gemm_tcgen05_kernel",
        };
        let function = module
            .load_function(kernel_name)
            .map_err(|e| GemmError::Cuda(format!("function load failed: {}", e)))?;

        Ok(CudaGemmKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

#[cfg(feature = "cuda")]
impl CudaGemmKernel {
    /// Get the cuda-oxide context for external operations
    pub fn context(&self) -> &Arc<cuda_core::CudaContext> {
        &self.context
    }

    /// Get the cuda-oxide stream
    pub fn stream(&self) -> &Arc<cuda_core::CudaStream> {
        &self.stream
    }

    /// Launch the GEMM kernel on the given streams.
    pub fn launch(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError> {
        use crate::cuda_runtime::CudaRuntime;

        // Launch kernel with WGMMA or tcgen05 instructions
        unsafe {
            cuda_core::launch_kernel_on_stream(
                self.function,
                &[m, n, k, alpha, beta],
                &[
                    a.device_ptr().ok_or(GemmError::LaunchFailed("A not on device"))?,
                    b.device_ptr().ok_or(GemmError::LaunchFailed("B not on device"))?,
                    c.device_ptr().ok_or(GemmError::LaunchFailed("C not on device"))?,
                ],
                self.stream.cu_stream(),
            )
            .map_err(|e| GemmError::LaunchFailed(format!("kernel launch failed: {}", e)))?;
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl GemmKernel for CudaGemmKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GemmError> {
        self.launch(alpha, a, b, beta, c, m, n, k)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> GemmArch {
        self.arch
    }
}

// --- Error Types ---

/// Errors that can occur during GEMM execution.
#[derive(Debug, thiserror::Error)]
pub enum GemmError {
    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("launch failed: {0}")]
    LaunchFailed(String),

    #[error("PTX compilation failed: {0}")]
    PtxCompile(String),
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_gemm_2x2() {
        // Simple 2x2 matrix multiplication: A @ B = C
        let a_host = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0), f16::from_f32(4.0)];
        let b_host = vec![f16::from_f32(5.0), f16::from_f32(6.0), f16::from_f32(7.0), f16::from_f32(8.0)];
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[1,2],[3,4]] @ [[5,6],[7,8]]

        let a_buf = DeviceBuffer::from_cpu(&a_host).expect("A alloc should succeed");
        let b_buf = DeviceBuffer::from_cpu(&b_host).expect("B alloc should succeed");
        let mut c_buf = DeviceBuffer::zeros_cpu_device(4);

        let kernel = CpuGemmKernel::new();
        kernel.matmul(1.0, &a_buf, &b_buf, 0.0, &mut c_buf, 2, 2, 2).expect("GEMM should succeed");

        let c_host = c_buf.as_slice().expect("C should be on host");
        assert_eq!(c_host, &expected);
    }

    #[test]
    fn test_gemm_arch_names() {
        assert_eq!(GemmArch::Wgmma.name(), "wgmma");
        assert_eq!(GemmArch::Tcgen05.name(), "tcgen05");
    }

    #[test]
    fn test_gemm_config_defaults() {
        let config = GemmConfig::default();
        assert_eq!(config.arch, GemmArch::Tcgen05);
        assert!(config.use_tma);
        assert_eq!(config.effective_block_size(), 128);
    }
}
