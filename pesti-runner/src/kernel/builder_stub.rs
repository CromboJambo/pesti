//! Stub builder module for CPU-only builds.

use crate::kernel::gemm::{GemmArch, GemmConfig};

/// Dummy PTX source trait
pub trait PtxSource {
    fn as_bytes(&self) -> &[u8];
}

/// Dummy kernel from PTX
pub trait KernelFromPtx {
    fn from_ptx(_ptx: &[u8], _name: &str) -> Result<Self::Kernel, crate::kernel::GemmError>
    where
        Self: Sized;

    type Kernel;
}

/// Dummy GEMM builder (CPU-only stub)
pub struct GemmBuilder {
    pub arch: GemmArch,
    pub config: GemmConfig,
}

impl GemmBuilder {
    pub fn new(arch: GemmArch) -> Self {
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
        Self::new(GemmArch::default())
    }
}
