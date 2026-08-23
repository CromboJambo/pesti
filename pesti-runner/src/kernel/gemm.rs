//! GEMM kernel interface and configuration.
//!
//! Provides the core matmul abstraction used by the LLM inference engine.
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! The differentiator: proving tcgen05 matmul works for LLM workloads
//! with non-matrix layouts (KV cache updates: M:1×K, K:N×K shapes).
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

use crate::cuda_runtime::{CudaRuntime, IntoResult};
use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// Tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GemmArch {
    /// WGMMA — warpgroup matrix multiply (sm_90a, consumer Blackwell: RTX 5060 Ti / 5090)
    Wgmma,
    /// tcgen05 — tensor memory MMA (sm_100a/sm_103a, datacenter Blackwell)
    Tcgen05,
    /// mma.sync — classic warp-level tensor core MMA (sm_80..sm_120, incl.
    /// consumer Blackwell RTX 50-series, which has neither wgmma nor tcgen05)
    #[default]
    Mma,
}

impl GemmArch {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgmma => "wgmma",
            Self::Tcgen05 => "tcgen05",
            Self::Mma => "mma.sync",
        }
    }

    pub fn supports_tma(&self) -> bool {
        // tcgen05 has native TMA support; WGMMA uses TMA for GMEM->SMEM copies
        match self {
            Self::Wgmma | Self::Tcgen05 => true,
            Self::Mma => false,
        }
    }

    pub fn tile_size(&self) -> usize {
        match self {
            Self::Wgmma => 128,   // 128×128 tiles (WGMMA)
            Self::Tcgen05 => 128, // 128×128×16 tiles
            Self::Mma => 16,      // 16×8×16 tiles
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
        _alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        _beta: f32,
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

impl CpuGemmKernel {
    /// Direct f32 GEMM for testing: C = alpha * A @ B + beta * C
    /// A: [m x k] f32, B: [k x n] f32, C: [m x n] f32
    pub fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), GemmError> {
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(GemmError::BufferSizeMismatch {
                expected: m * k,
                got: a.len().min(m * k),
            });
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }

        Ok(())
    }
}

// --- GPU Implementation (Real cudarc backed) ---

#[cfg(feature = "cuda")]
/// CUDA implementation for GEMM kernel using cudarc.
#[derive(Clone)]
pub struct CudaGemmKernel {
    arch: GemmArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    function: CudaFunction,
}

#[cfg(feature = "cuda")]
/// Builder for CudaGemmKernel that handles PTX loading and kernel resolution.
pub struct CudaGemmKernelBuilder {
    arch: GemmArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_info: crate::cuda_runtime::CudaDeviceInfo,
}

#[cfg(feature = "cuda")]
impl CudaGemmKernelBuilder {
    pub fn new(
        arch: GemmArch,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
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
        // Pre-flight architecture check. mma.sync works on every tensor-core
        // GPU (sm_80..sm_120), so it needs no capability gate.
        match self.arch {
            GemmArch::Wgmma if !self.device_info.supports_wgmma() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "WGMMA requires sm_90a (Hopper), but device is sm_{}.{}",
                    self.device_info.compute_capability.0, self.device_info.compute_capability.1
                )));
            }
            GemmArch::Tcgen05 if !self.device_info.supports_tcgen05() => {
                return Err(GemmError::UnsupportedArch(format!(
                    "tcgen05 requires sm_100a (datacenter Blackwell), but device is sm_{}.{}",
                    self.device_info.compute_capability.0, self.device_info.compute_capability.1
                )));
            }
            _ => {}
        }

        // Select PTX based on architecture
        let ptx_src = match self.arch {
            GemmArch::Wgmma => include_str!("ptx/gemm_wgmma_sm90.ptx"), // Hopper (sm_9.0)
            GemmArch::Tcgen05 => include_str!("ptx/gemm_tcgen05_sm89.ptx"), // Ada Lovelace (sm_8.9)
            GemmArch::Mma => include_str!("ptx/gemm_mma_sync.ptx"),
        };

        // Load module from PTX source
        let module = CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| GemmError::Cuda(format!("module load failed: {e:?}")))?;

        // Resolve kernel function
        let kernel_name = match self.arch {
            GemmArch::Wgmma => "gemm_wgmma_kernel",
            GemmArch::Tcgen05 => "gemm_tcgen05_kernel",
            GemmArch::Mma => "gemm_mma_kernel",
        };
        let function = module
            .load_function(kernel_name)
            .map_err(|e| GemmError::Cuda(format!("function load failed: {e:?}")))?;

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
    /// Get the cudarc context for external operations
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the cudarc stream
    pub fn stream(&self) -> &Arc<CudaStream> {
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
        use cudarc::driver::sys;

        // Kernel signature varies by architecture:
        // - WGMMA (sm_90a/sm_120): gemm_wgmma_kernel(f32 alpha, u64 A, u64 B, f32 beta, u64 C, u32 m, u32 n, u32 k)
        // - tcgen05: gemm_tcgen05_kernel(f32 alpha, u64 A, u64 B, f32 beta, u64 C, u32 m, u32 n, u32 k)
        // - mma.sync: gemm_mma_kernel(f32 alpha, u64 A, u64 B, f32 beta, u64 C, u32 m, u32 n, u32 k)

        let mut a_v: u64 = a.device_ptr();
        let mut b_v: u64 = b.device_ptr();
        let mut c_v: u64 = c.device_ptr();
        let mut alpha_v: f32 = alpha;
        let mut beta_v: f32 = beta;
        let mut m_v: u32 = m as u32;
        let mut n_v: u32 = n as u32;
        let mut k_v: u32 = k as u32;

        let mut params: [*mut std::ffi::c_void; 8] = [
            &mut alpha_v as *mut f32 as *mut std::ffi::c_void,
            &mut a_v as *mut u64 as *mut std::ffi::c_void,
            &mut b_v as *mut u64 as *mut std::ffi::c_void,
            &mut beta_v as *mut f32 as *mut std::ffi::c_void,
            &mut c_v as *mut u64 as *mut std::ffi::c_void,
            &mut m_v as *mut u32 as *mut std::ffi::c_void,
            &mut n_v as *mut u32 as *mut std::ffi::c_void,
            &mut k_v as *mut u32 as *mut std::ffi::c_void,
        ];

        // Grid/block configuration varies by architecture:
        // - WGMMA: 128×128 tiles per block, 128 threads (4 warp groups × 32 warps)
        // - tcgen05: 128×128 tiles per block, 128 threads
        // - mma.sync: 16×8 tiles per block, 32 threads (1 warp)
        let (grid_x, grid_y, block_size) = match self.arch {
            GemmArch::Wgmma => (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 128u32),
            GemmArch::Tcgen05 => (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 128u32),
            GemmArch::Mma => (((n + 7) / 8) as u32, ((m + 15) / 16) as u32, 32u32),
        };

        let grid = (grid_x, grid_y, 1u32);
        let block = (block_size, 1u32, 1u32);

        // Launch: grid, block, shared_mem_bytes, stream, params
        unsafe {
            use crate::cuda_shim::launch_kernel;
            launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0, // shared_mem_bytes (WGMMA uses implicit SMEM)
                self.stream.cu_stream(),
                &mut params,
            )
            .map_err(|e| GemmError::LaunchFailed(format!("kernel launch failed: {e:?}")))?;
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

    #[error("not available")]
    NotAvailable,

    #[error("invalid dimensions: {m}x{n} vs {k}")]
    InvalidDimensions { m: usize, n: usize, k: usize },
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_arch_name() {
        assert_eq!(GemmArch::Wgmma.name(), "wgmma");
        assert_eq!(GemmArch::Tcgen05.name(), "tcgen05");
        assert_eq!(GemmArch::Mma.name(), "mma.sync");
    }

    #[test]
    fn test_gemm_config_default() {
        let config = GemmConfig::default();
        assert_eq!(config.arch, GemmArch::Mma);
        assert!(config.use_tma);
        assert_eq!(config.block_size, 0);
    }

    #[test]
    fn test_cpu_gemm_kernel() {
        let kernel = CpuGemmKernel::new();
        assert!(kernel.is_available());
        assert_eq!(kernel.arch(), GemmArch::Mma);
    }

    #[test]
    fn test_cpu_gemm_matmul_identity() {
        let kernel = CpuGemmKernel::new();
        let m = 2usize;
        let n = 2usize;
        let k = 2usize;

        // Identity matrix test: A * I = A
        let a = DeviceBuffer::from_host(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ]);
        let b = DeviceBuffer::from_host(vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(1.0),
        ]);
        let mut c = DeviceBuffer::zeros(m * n);

        kernel.matmul(1.0, &a, &b, 0.0, &mut c, m, n, k).unwrap();

        let c_host = c.to_host();
        assert!((c_host[0] - 1.0).abs() < 0.01);
        assert!((c_host[1] - 2.0).abs() < 0.01);
        assert!((c_host[2] - 3.0).abs() < 0.01);
        assert!((c_host[3] - 4.0).abs() < 0.01);
    }
}
