//! Stub builder module for CPU-only builds.
//!
//! Provides stub implementations matching the real GEMM API
//! to allow compilation without CUDA dependencies.

/// Dummy PTX source trait (stub)
pub trait PtxSource {
    fn as_bytes(&self) -> &[u8];
}

/// Dummy kernel from PTX (stub)
pub trait KernelFromPtx {
    fn from_ptx(_ptx: &[u8], _name: &str) -> Result<Self::Kernel, crate::kernel::GemmError>
    where
        Self: Sized;

    type Kernel;
}

/// Stub configuration for a GEMM kernel launch (mirrors real GemmConfig)
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// Target architecture (stub)
    pub arch: crate::kernel::GemmArch,
    /// Whether to use TMA for async GMEM->SMEM copies
    pub use_tma: bool,
    /// Custom block size override (0 = use arch default)
    pub block_size: usize,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            arch: crate::kernel::GemmArch::default(),
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

    pub fn with_arch(mut self, arch: crate::kernel::GemmArch) -> Self {
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

/// Stub GEMM builder (CPU-only stub)
pub struct GemmBuilder {
    pub arch: crate::kernel::GemmArch,
    pub config: GemmConfig,
}

impl GemmBuilder {
    pub fn new(arch: crate::kernel::GemmArch) -> Self {
        Self {
            arch,
            config: GemmConfig::default(),
        }
    }

    pub fn with_config(mut self, config: GemmConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<(), crate::kernel::GemmError> {
        // CPU-only stub - just return Ok
        Ok(())
    }
}

impl Default for GemmBuilder {
    fn default() -> Self {
        Self::new(crate::kernel::GemmArch::default())
    }
}
