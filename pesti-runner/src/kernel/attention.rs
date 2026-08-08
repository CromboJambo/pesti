//! Attention kernel interface and configuration for Blackwell tensor cores.
//!
//! **Current implementation**: Uses GEMM-based approach (Option A)
//! - Q @ K^T via existing mma.sync GEMM kernel  
//! - Softmax on CPU (transfer to host, softmax, back to device or use directly)
//! - V @ S^T via another GEMM
//!
//! **Future optimization** (Option B): Dedicated WGMMA/tcgen05 attention PTX
//! - Single-kernel softmax + fused multiply-add
//! - Better for very long sequences

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::gemm::{CudaGemmKernel, GemmArch, GemmKernel};
use half::f16;
use std::simd::prelude::*;

/// Attention architecture selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionArch {
    /// CPU-only attention (reference implementation).
    Cpu,
    /// WGMMA-based attention for Blackwell tensor cores.
    Wgmma,
    /// Tcgen05-based attention for Blackwell tensor cores.
    Tcgen05,
}

impl Default for AttentionArch {
    fn default() -> Self {
        Self::Cpu
    }
}

// Serialize/Deserialize support
impl serde::Serialize for AttentionArch {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Cpu => "cpu".serialize(serializer),
            Self::Wgmma => "wgmma".serialize(serializer),
            Self::Tcgen05 => "tcgen05".serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for AttentionArch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "cpu" => Ok(Self::Cpu),
            "wgmma" => Ok(Self::Wgmma),
            "tcgen05" => Ok(Self::Tcgen05),
            _ => Err(Error::custom(format!("Unknown AttentionArch: {}", s))),
        }
    }
}

// Conversion from GemmArch to AttentionArch
impl From<GemmArch> for AttentionArch {
    fn from(arch: GemmArch) -> Self {
        match arch {
            GemmArch::Wgmma => Self::Wgmma,
            GemmArch::Tcgen05 => Self::Tcgen05,
            GemmArch::Mma => Self::Cpu, // Fallback to CPU for mma.sync
        }
    }
}

/// Configuration for attention computation.
#[derive(Debug)]
pub struct AttentionConfig {
    pub arch: AttentionArch,
    pub use_tma: bool,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub block_size: usize,
    pub rope_base: f32,
    pub max_pos: usize,
    pub scale: f32, // Pre-computed scaling factor (1/sqrt(head_dim))
}

impl AttentionConfig {
    /// Create a new attention config with standard scaled dot-product scaling.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            arch: AttentionArch::Cpu,
            use_tma: true,
            num_heads,
            head_dim,
            max_seq: 4096,
            block_size: 128,
            rope_base: 10000.0,
            max_pos: 4096,
            scale,
        }
    }

    /// Default config with standard dimensions.
    pub fn default() -> Self {
        Self::new(32, 64)
    }

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

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
}

/// Attention kernel trait - implemented by CPU and GPU backends.
pub trait AttentionKernel: Send + Sync {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError>;

    fn is_available(&self) -> bool;

    fn arch(&self) -> AttentionArch;
}

/// CPU-based attention kernel (reference implementation).
pub struct CpuAttentionKernel {
    pub arch: AttentionArch,
}

impl CpuAttentionKernel {
    pub fn new(arch: AttentionArch) -> Self {
        Self { arch }
    }
}

impl AttentionKernel for CpuAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        // Extract host data from device buffers
        let q_host: Vec<f32> = query.to_host().iter().map(|&x| f16::to_f32(x)).collect();
        let k_host: Vec<f32> = key_cache
            .buffer()
            .to_host()
            .iter()
            .map(|&x| f16::to_f32(x))
            .collect();
        let v_host: Vec<f32> = value_cache
            .buffer()
            .to_host()
            .iter()
            .map(|&x| f16::to_f32(x))
            .collect();

        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let n = key_cache.seq_len();
        let query_seq_len = q_host.len() / (num_heads * head_dim);

        // Step 1: Q @ K^T -> scores [query_seq_len, num_heads, cache_seq_len]
        let mut scores = vec![0.0f32; query_seq_len * num_heads * n];

        // SIMD inner product helper: process 8 elements at a time
        #[inline]
        fn simd_dot_product(q_slice: &[f32], k_slice: &[f32], head_dim: usize) -> f32 {
            const LANES: usize = 8;

            let mut sum = 0.0f32;
            let simd_len = (head_dim / LANES) * LANES;

            for i in (0..simd_len).step_by(LANES) {
                let q_vec = f32x8::from_slice(&q_slice[i..]);
                let k_vec = f32x8::from_slice(&k_slice[i..]);
                sum += (q_vec * k_vec).reduce_sum();
            }

            // Handle remainder
            for i in simd_len..head_dim {
                sum += q_slice[i] * k_slice[i];
            }

            sum
        }

        for qs in 0..query_seq_len {
            for h in 0..num_heads {
                let q_base = (qs * num_heads + h) * head_dim;
                for s in 0..n {
                    let k_base = (h * n + s) * head_dim;
                    let sum = simd_dot_product(&q_host[q_base..], &k_host[k_base..], head_dim);
                    scores[qs * num_heads * n + h * n + s] = sum * config.scale;
                }
            }
        }

        // Step 2: Softmax on cache_seq_len dimension
        let mut softmax_scores = vec![0.0f32; scores.len()];
        for qs in 0..query_seq_len {
            for h in 0..num_heads {
                let start = (qs * num_heads + h) * n;
                let mut max_val = f32::NEG_INFINITY;
                for s in 0..n {
                    if scores[start + s] > max_val {
                        max_val = scores[start + s];
                    }
                }
                let mut sum = 0.0f32;
                for s in 0..n {
                    let exp_val = (scores[start + s] - max_val).exp();
                    softmax_scores[start + s] = exp_val;
                    sum += exp_val;
                }
                if sum > 0.0 {
                    for s in 0..n {
                        softmax_scores[start + s] /= sum;
                    }
                }
            }
        }

        // Step 3: Softmax @ V -> output [query_seq_len, num_heads, head_dim]
        let mut output = vec![0.0f32; query_seq_len * num_heads * head_dim];

        // SIMD vectorized dot product for softmax @ V (8 lanes)
        #[inline]
        fn simd_softmax_v_dot(
            softmax_row: &[f32],
            v_slice: &[f32],
            n: usize,
            head_dim: usize,
            d: usize,
        ) -> f32 {
            const LANES: usize = 8;

            let mut sum = 0.0f32;
            let simd_len = (n / LANES) * LANES;

            for i in (0..simd_len).step_by(LANES) {
                let s_vec = f32x8::from_slice(&softmax_row[i..]);
                let v_start = i * head_dim + d;
                // Load V values - note: may need to handle unaligned access for non-multiple-of-8 dims
                let mut v_vals = [0.0f32; LANES];
                for j in 0..LANES {
                    if i + j < n {
                        v_vals[j] = v_slice[v_start + j * head_dim];
                    }
                }
                let v_vec = f32x8::from_array(v_vals);
                sum += (s_vec * v_vec).reduce_sum();
            }

            // Handle remainder
            for i in simd_len..n {
                sum += softmax_row[i] * v_slice[i * head_dim + d];
            }

            sum
        }

        for qs in 0..query_seq_len {
            for h in 0..num_heads {
                let softmax_start = (qs * num_heads + h) * n;
                let softmax_row = &softmax_scores[softmax_start..softmax_start + n];

                for d in 0..head_dim {
                    let sum = simd_softmax_v_dot(softmax_row, &v_host, n, head_dim, d);
                    output[qs * num_heads * head_dim + h * head_dim + d] = sum;
                }
            }
        }

        // Convert back to device buffer
        Ok(DeviceBuffer::from_host(output))
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }
}

/// Slice of KV cache for attention computation (stub for now).
#[derive(Debug, Clone)]
pub struct AttentionSlice {
    /// Base device pointer for the entire K+V buffer.
    pub gmem_addr: u64,
    /// Number of heads in the cache.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Maximum sequence length (needed for V base calculation).
    pub max_seq: usize,
    /// Head index (0..num_heads).
    pub head_idx: usize,
    /// Sequence start position.
    pub seq_start: usize,
    /// Number of sequence positions (box Y).
    pub seq_len: usize,
    /// Whether this is the K tensor (true) or V tensor (false).
    pub is_key: bool,
}

impl AttentionSlice {
    pub fn new(
        gmem_addr: u64,
        num_heads: usize,
        head_dim: usize,
        max_seq: usize,
        head_idx: usize,
        seq_start: usize,
        seq_len: usize,
        is_key: bool,
    ) -> Self {
        Self {
            gmem_addr,
            num_heads,
            head_dim,
            max_seq,
            head_idx,
            seq_start,
            seq_len,
            is_key,
        }
    }

    pub fn base_addr(&self) -> u64 {
        let head_stride = self.num_heads * self.head_dim;
        let head_offset = self.head_idx * self.head_dim;
        if self.is_key {
            self.gmem_addr + (head_stride * head_offset) as u64 * 2
        } else {
            self.gmem_addr + (head_stride * self.max_seq + head_stride * head_offset) as u64 * 2
        }
    }
}

/// **Option B**: Dedicated WGMMA/tcgen05 attention kernel (future optimization).
pub struct CudaAttentionKernel {
    // TODO: Add PTX module and tensor core parameters
}

impl CudaAttentionKernel {
    pub fn new(_arch: AttentionArch) -> Self {
        Self {}
    }
}

impl AttentionKernel for CudaAttentionKernel {
    fn forward(
        &self,
        _query: &DeviceBuffer<f16>,
        _key_cache: &Kvcache,
        _value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        _config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        // For now, return zeros - this is a placeholder until we integrate with real GPU backend
        let num_heads = 1;
        let head_dim = 64;
        Ok(DeviceBuffer::zeros(num_heads * head_dim))
    }

    fn is_available(&self) -> bool {
        false // Not yet implemented
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::Wgmma
    }
}

/// Builder for GPU attention kernels.
pub struct CudaAttentionKernelBuilder {
    arch: AttentionArch,
}

impl CudaAttentionKernelBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arch: AttentionArch,
        _context: std::sync::Arc<cuda_core::CudaContext>,
        _stream: std::sync::Arc<cuda_core::CudaStream>,
        _device_info: crate::cuda_runtime::CudaDeviceInfo,
    ) -> Self {
        Self { arch }
    }

    pub fn build(self) -> Result<CudaAttentionKernel, AttentionError> {
        Ok(CudaAttentionKernel::new(self.arch))
    }
}

/// **Option A**: GEMM-based attention using existing mma.sync kernel.
pub struct GemmBasedAttentionKernel {
    gemm_kernel: CudaGemmKernel,
    backend: std::sync::Arc<crate::kernel::memory::CudaMemoryBackend>,
}

impl GemmBasedAttentionKernel {
    pub fn new(
        gemm_kernel: CudaGemmKernel,
        backend: std::sync::Arc<crate::kernel::memory::CudaMemoryBackend>,
    ) -> Self {
        Self {
            gemm_kernel,
            backend,
        }
    }
}

impl AttentionKernel for GemmBasedAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let n = key_cache.seq_len();
        let query_seq_len = (query.len() / (num_heads * head_dim)) as usize;

        // Step 1: Q @ K^T -> scores [query_seq_len, num_heads, cache_seq_len]
        // Q: [query_seq_len, num_heads, head_dim] -> reshape to [query_seq_len * num_heads, head_dim]
        // K: [num_heads, cache_seq_len, head_dim] -> transpose to [head_dim, num_heads * cache_seq_len]

        let q_m = query_seq_len * num_heads;
        let q_k = head_dim;
        let k_n = n; // We'll do Q @ K^T, so K is transposed

        // Allocate scores buffer on device: [q_m, k_n]
        let backend = &*self.backend;
        let mut scores_buffer = DeviceBuffer::<f32>::zeros_device(backend, q_m * k_n)
            .map_err(|e| AttentionError::Cuda(format!("scores alloc: {e}")))?;

        // Launch Q @ K^T via GEMM
        self.gemm_kernel
            .matmul(
                1.0, // alpha
                query,
                key_cache.buffer(),
                0.0, // beta (don't accumulate)
                &mut scores_buffer,
                q_m, // m = query_seq_len * num_heads
                k_n, // n = cache_seq_len (K is transposed)
                q_k, // k = head_dim
            )
            .map_err(|e| AttentionError::Gemm(e))?;

        // Synchronize stream to ensure GEMM completes before reading back
        self.gemm_kernel
            .stream()
            .synchronize()
            .map_err(|e| AttentionError::Cuda(format!("sync after QK: {e}")))?;

        // Step 2: Apply scaling factor and softmax on CPU
        let mut scores_host = scores_buffer
            .to_host_vec(backend)
            .map_err(|e| AttentionError::Transfer(e))?;
        let mut softmax_scores = vec![0.0f32; scores_host.len()];

        // Apply scale and softmax per head
        for qs in 0..query_seq_len {
            for h in 0..num_heads {
                let start = (qs * num_heads + h) * n;

                // Apply scaling
                let mut max_val = f32::NEG_INFINITY;
                for s in 0..n {
                    let idx = start + s;
                    scores_host[idx] *= config.scale;
                    if scores_host[idx] > max_val {
                        max_val = scores_host[idx];
                    }
                }

                // Compute softmax
                let mut sum = 0.0f32;
                for s in 0..n {
                    let idx = start + s;
                    let exp_val = (scores_host[idx] - max_val).exp();
                    softmax_scores[idx] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                if sum > 0.0 {
                    for s in 0..n {
                        let idx = start + s;
                        softmax_scores[idx] /= sum;
                    }
                }
            }
        }

        // Step 3: S @ V -> output [query_seq_len, num_heads, head_dim]
        // S: [query_seq_len * num_heads, cache_seq_len] (softmax scores)
        // V: [num_heads, cache_seq_len, head_dim] -> reshape to [cache_seq_len, head_dim]

        let s_m = query_seq_len * num_heads;
        let s_k = n; // cache_seq_len
        let v_n = head_dim;

        // Convert softmax scores from f32 to f16 for GEMM input, allocate on device
        let softmax_scores_f16: Vec<f16> =
            softmax_scores.iter().map(|&x| f16::from_f32(x)).collect();
        let softmax_buf = DeviceBuffer::from_host_device(backend, &softmax_scores_f16)
            .map_err(|e| AttentionError::Cuda(format!("softmax buf alloc: {e}")))?;

        let mut output_buffer = DeviceBuffer::<f32>::zeros_device(backend, s_m * v_n)
            .map_err(|e| AttentionError::Cuda(format!("output alloc: {e}")))?;

        // Launch S @ V via GEMM (need to transpose V)
        self.gemm_kernel
            .matmul(
                1.0,                  // alpha
                &softmax_buf,         // S (f16)
                value_cache.buffer(), // V (f16)
                0.0,                  // beta
                &mut output_buffer,
                s_m, // m = query_seq_len * num_heads
                v_n, // n = head_dim
                s_k, // k = cache_seq_len (V is transposed)
            )
            .map_err(|e| AttentionError::Gemm(e))?;

        // Synchronize before returning
        self.gemm_kernel
            .stream()
            .synchronize()
            .map_err(|e| AttentionError::Cuda(format!("sync after SV: {e}")))?;

        Ok(output_buffer)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> AttentionArch {
        AttentionArch::Wgmma
    }
}

/// Attention errors.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("attention config invalid: num_heads={num_heads}, head_dim={head_dim}")]
    InvalidConfig { num_heads: usize, head_dim: usize },

    #[error("buffer size mismatch: expected={expected}, got={got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("invalid dimensions: num_heads={num_heads}, head_dim={head_dim}")]
    InvalidDimensions {
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    },

    #[error("GEMM error: {0}")]
    Gemm(#[from] crate::kernel::gemm::GemmError),

    #[error("kernel launch failed: {detail}")]
    KernelLaunchFailed { detail: String },

    #[error("attention not available")]
    NotAvailable,

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("transfer error: {0}")]
    Transfer(#[from] crate::kernel::device_buf::DeviceBufferError),

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("CUDA error: {0}")]
    Cuda(String),
}

// Import Kvcache here to avoid circular dependency issues
use crate::kernel::kvcache::Kvcache;
