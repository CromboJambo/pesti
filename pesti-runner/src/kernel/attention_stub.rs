//! Stub attention module for CPU-only builds.

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache_stub::Kvcache;
use half::f16;

/// Dummy attention architecture
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttentionArch {
    Wgmma,
    Tcgen05,
}

impl Default for AttentionArch {
    fn default() -> Self {
        Self::Tcgen05
    }
}

/// Dummy attention config
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub arch: AttentionArch,
    pub use_tma: bool,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub block_size: usize,
    pub rope_base: f32,
    pub max_pos: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            arch: AttentionArch::default(),
            use_tma: true,
            num_heads: 32,
            head_dim: 64,
            max_seq: 4096,
            block_size: 128,
            rope_base: 10000.0,
            max_pos: 4096,
        }
    }
}

impl AttentionConfig {
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    pub fn with_max_seq(mut self, max_seq: usize) -> Self {
        self.max_seq = max_seq;
        self
    }

    pub fn with_arch(mut self, arch: AttentionArch) -> Self {
        self.arch = arch;
        self
    }

    pub fn with_tma(mut self, use_tma: bool) -> Self {
        self.use_tma = use_tma;
        self
    }
}

/// Dummy attention error
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),
    #[error("attention not available")]
    NotAvailable,
}

/// Dummy attention kernel trait (CPU-only stub)
pub trait AttentionKernel: Send + Sync {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&[bool]>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError>;

    fn is_available(&self) -> bool;

    fn arch(&self) -> AttentionArch;
}

/// CPU attention kernel (stub - uses candle SDPA)
pub struct CpuAttentionKernel;

impl CpuAttentionKernel {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuAttentionKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionKernel for CpuAttentionKernel {
    fn forward(
        &self,
        _query: &DeviceBuffer<f16>,
        _key_cache: &Kvcache,
        _value_cache: &Kvcache,
        _mask: Option<&[bool]>,
        _config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        Err(AttentionError::NotAvailable)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::default()
    }
}

/// Stub CUDA attention kernel
#[cfg(feature = "cuda")]
pub struct CudaAttentionKernel;

#[cfg(feature = "cuda")]
impl AttentionKernel for CudaAttentionKernel {
    fn forward(
        &self,
        _query: &DeviceBuffer<f16>,
        _key_cache: &Kvcache,
        _value_cache: &Kvcache,
        _mask: Option<&[bool]>,
        _config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        Err(AttentionError::NotAvailable)
    }

    fn is_available(&self) -> bool {
        false
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::default()
    }
}

/// Stub attention slice
pub struct AttentionSlice {
    pub key_cache: DeviceBuffer<f16>,
    pub value_cache: DeviceBuffer<f16>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AttentionSlice {
    pub fn new(
        _key_cache: DeviceBuffer<f16>,
        _value_cache: DeviceBuffer<f16>,
        _seq_len: usize,
        _num_heads: usize,
        _head_dim: usize,
    ) -> Self {
        Self {
            key_cache: _key_cache,
            value_cache: _value_cache,
            seq_len: _seq_len,
            num_heads: _num_heads,
            head_dim: _head_dim,
        }
    }
}

/// Stub CUDA attention kernel builder
#[cfg(feature = "cuda")]
pub struct CudaAttentionKernelBuilder;

#[cfg(feature = "cuda")]
impl CudaAttentionKernelBuilder {
    pub fn new(
        _arch: AttentionArch,
        _config: AttentionConfig,
        _context: std::sync::Arc<cuda_core::CudaContext>,
        _stream: std::sync::Arc<cuda_core::CudaStream>,
    ) -> Self {
        Self
    }

    pub fn build(self) -> Result<CudaAttentionKernel, AttentionError> {
        Ok(CudaAttentionKernel)
    }
}
