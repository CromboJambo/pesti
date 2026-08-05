//! Stub GEMM module for CPU-only builds.
//!
//! Provides stub implementations of GEMM types to allow compilation
//! without CUDA dependencies.

use crate::kernel::device_buf::DeviceBuffer;
use half::f16;

/// Stub tensor core architecture selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
        true
    }

    pub fn tile_size(&self) -> usize {
        match self {
            Self::Wgmma => 64,
            Self::Tcgen05 => 128,
        }
    }
}

/// Stub configuration for a GEMM kernel launch
#[derive(Debug, Clone)]
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

// --- Stub GemmKernel Trait ---

/// Stub trait for GEMM (matrix multiply) kernels.
pub trait GemmKernel: Send + Sync {
    /// Perform GEMM: C = alpha * A @ B + beta * C
    fn matmul(
        &self,
        _alpha: f32,
        _a: &DeviceBuffer<f16>,
        _b: &DeviceBuffer<f16>,
        _beta: f32,
        _c: &mut DeviceBuffer<f32>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), GemmError>;

    /// Check if the kernel is available on the current device.
    fn is_available(&self) -> bool;

    /// Get the target architecture.
    fn arch(&self) -> GemmArch;
}

// --- Stub CPU Implementation (Fallback) ---

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
        _alpha: f32,
        _a: &DeviceBuffer<f16>,
        _b: &DeviceBuffer<f16>,
        _beta: f32,
        _c: &mut DeviceBuffer<f32>,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), GemmError> {
        Ok(())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> GemmArch {
        GemmArch::default()
    }
}

// --- Stub Error Types ---

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
