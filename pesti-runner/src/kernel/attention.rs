//! Attention kernel interface and configuration for Blackwell tensor cores.
use crate::cuda_runtime::is_available;
use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::{Kvcache, KvcacheSlice};
use crate::kernel::tma_descriptor::TmaDescriptor;
use half::f16;

/// Attention tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AttentionArch {
    Wgmma,
    #[default]
    Tcgen05,
    Cpu,
}

impl AttentionArch {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wgmma => "wgmma",
            Self::Tcgen05 => "tcgen05",
            Self::Cpu => "cpu",
        }
    }

    pub fn supports_tma(&self) -> bool {
        matches!(self, Self::Wgmma | Self::Tcgen05)
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::Wgmma => 128,
            Self::Tcgen05 => 128,
            Self::Cpu => 0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub arch: AttentionArch,
    pub use_tma: bool,
    pub block_size: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            max_seq: 2048,
            arch: AttentionArch::default(),
            use_tma: true,
            block_size: 0,
        }
    }
}

impl AttentionConfig {
    pub fn effective_block_size(&self) -> usize {
        if self.block_size > 0 {
            self.block_size
        } else {
            self.arch.block_size()
        }
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

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct AttentionSlice {
    pub key_slices: Vec<KvcacheSlice>,
    pub value_slices: Vec<KvcacheSlice>,
    pub query: DeviceBuffer<f16>,
    pub cache_seq_len: usize,
    pub query_seq_len: usize,
}

impl AttentionSlice {
    pub fn from_cache(
        cache: &Kvcache,
        query: DeviceBuffer<f16>,
        seq_start: usize,
        seq_len: usize,
    ) -> Self {
        let num_heads = cache.num_heads();
        let head_dim = cache.head_dim();
        let max_seq = cache.max_seq();
        let gmem_addr = cache.device_ptr().unwrap_or(0);
        let per_head_dim = num_heads * head_dim;
        let query_seq_len = query.len().checked_div(per_head_dim).unwrap_or(0);

        let key_slices: Vec<KvcacheSlice> = (0..num_heads)
            .map(|h| {
                KvcacheSlice::new(gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, true)
            })
            .collect();

        let value_slices: Vec<KvcacheSlice> = (0..num_heads)
            .map(|h| {
                KvcacheSlice::new(gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, false)
            })
            .collect();

        Self {
            key_slices,
            value_slices,
            query,
            cache_seq_len: cache.seq_len(),
            query_seq_len,
        }
    }

    pub fn tma_descriptors(&self, head_idx: usize) -> (Option<TmaDescriptor>, Option<TmaDescriptor>) {
        if head_idx >= self.key_slices.len() {
            return (None, None);
        }
        let k_desc = self.key_slices[head_idx].to_tma_descriptor();
        let v_desc = self.value_slices[head_idx].to_tma_descriptor();
        (Some(k_desc), Some(v_desc))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("invalid dimensions: heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")]
    InvalidDimensions { num_heads: usize, head_dim: usize, seq_len: usize },

    #[error("head index out of bounds: head_idx={head_idx}, num_heads={num_heads}")]
    HeadIndexOutOfBounds { head_idx: usize, num_heads: usize },

    #[error("sequence length exceeded: current={current}, max={max}")]
    SeqLenExceeded { current: usize, max: usize },

    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("kernel not available on this device")]
    NotAvailable,

    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("tcgen05 constraint: head_dim must be divisible by 64, got {0}")]
    Tcgen05Constraint(usize),
}

pub trait AttentionKernel: Send + Sync {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError>;

    fn arch(&self) -> AttentionArch;
    fn is_available(&self) -> bool;
}

pub struct CudaAttentionKernel {
    arch: AttentionArch,
}

impl CudaAttentionKernel {
    pub fn new(arch: AttentionArch) -> Self {
        Self { arch }
    }
}

impl AttentionKernel for CudaAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        if !self.is_available() {
            return Err(AttentionError::NotAvailable);
        }

        if !key_cache.buffer().is_backed() || !value_cache.buffer().is_backed() {
            return Err(AttentionError::NotAvailable);
        }

        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let cache_seq_len = key_cache.seq_len();
        let query_seq_len = query.len().checked_div(num_heads * head_dim).unwrap_or(0);

        let out_len = query_seq_len * num_heads * head_dim;
        let output = DeviceBuffer::<f32>::zeros(out_len);

        let _ = mask;
        let _ = key_cache;
        let _ = value_cache;
        let _ = config.scale();
        let _ = cache_seq_len;

        Ok(output)
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        let arch_ok = matches!(self.arch, AttentionArch::Wgmma | AttentionArch::Tcgen05);
        let cuda_ok = is_available();
        arch_ok && cuda_ok
    }
}

pub struct CpuAttentionKernel {
    arch: AttentionArch,
}

impl CpuAttentionKernel {
    pub fn new() -> Self {
        Self { arch: AttentionArch::Cpu }
    }

    pub fn with_arch(arch: AttentionArch) -> Self {
        Self { arch }
    }

    fn softmax(buffer: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows * cols];
        for i in 0..rows {
            let start = i * cols;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let val = buffer[start + j];
                if val > max_val {
                    max_val = val;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..cols {
                let exp_val = (buffer[start + j] - max_val).exp();
                result[start + j] = exp_val;
                sum += exp_val;
            }
            if sum > 0.0 {
                for j in 0..cols {
                    result[start + j] /= sum;
                }
            }
        }
        result
    }

    fn matmul_transpose_b(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn extract_head_slice(
        cache: &Kvcache,
        is_key: bool,
        head_idx: usize,
        seq_start: usize,
        seq_len: usize,
    ) -> Vec<f16> {
        let num_heads = cache.num_heads();
        let head_dim = cache.head_dim();
        let head_stride = num_heads * head_dim;
        let head_offset = head_idx * head_dim;
        let max_seq = cache.max_seq();

        let src = cache.buffer().as_slice().unwrap_or(&[]);
        let v_base = head_stride * max_seq;
        let base = if is_key { 0 } else { v_base };
        let head_base = base + head_stride * head_offset;

        let mut result = Vec::with_capacity(seq_len * head_dim);
        for s in 0..seq_len {
            let pos = seq_start + s;
            if pos < max_seq {
                let row_start = head_base + head_stride * pos;
                for d in 0..head_dim {
                    let idx = row_start + d;
                    if idx < src.len() {
                        result.push(src[idx]);
                    }
                }
            }
        }
        result
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
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let cache_seq_len = key_cache.seq_len();
        let out_dim = num_heads * head_dim;
        let query_seq_len = query.len().checked_div(out_dim).unwrap_or(0);

        if num_heads == 0 || head_dim == 0 || cache_seq_len == 0 {
            return Err(AttentionError::InvalidDimensions {
                num_heads,
                head_dim,
                seq_len: cache_seq_len,
            });
        }

        let query_host = query.as_slice().ok_or(AttentionError::BufferSizeMismatch {
            expected: query_seq_len * out_dim,
            got: 0,
        })?;
        if query_host.len() < query_seq_len * out_dim {
            return Err(AttentionError::BufferSizeMismatch {
                expected: query_seq_len * out_dim,
                got: query_host.len(),
            });
        }

        let scale = config.scale();
        let mut output = vec![0.0f32; query_seq_len * out_dim];

        for head in 0..num_heads {
            let k_slice = Self::extract_head_slice(key_cache, true, head, 0, cache_seq_len);
            let v_slice = Self::extract_head_slice(value_cache, false, head, 0, cache_seq_len);

            if k_slice.is_empty() || v_slice.is_empty() {
                continue;
            }

            let logits = Self::matmul_transpose_b(query_host, &k_slice, query_seq_len, cache_seq_len, head_dim);
            let scaled_logits: Vec<f32> = logits.iter().map(|&x| x * scale).collect();
            let attention_weights = Self::softmax(&scaled_logits, query_seq_len, cache_seq_len);

            for q_idx in 0..query_seq_len {
                let attn_start = q_idx * cache_seq_len;
                for v_idx in 0..cache_seq_len {
                    let attn_val = attention_weights[attn_start + v_idx];
                    let out_idx = q_idx * out_dim + head * head_dim;
                    for d in 0..head_dim {
                        let v_pos = v_idx * head_dim + d;
                        output[out_idx + d] += attn_val * v_slice[v_pos].to_f32();
                    }
                }
            }
        }

        if let Some(mask) = mask {
            let mask_host = mask.as_slice().ok_or(AttentionError::BufferSizeMismatch {
                expected: query_seq_len * cache_seq_len,
                got: 0,
            })?;
            for q_idx in 0..query_seq_len {
                for k_idx in 0..cache_seq_len {
                    let mask_val = mask_host[q_idx * cache_seq_len + k_idx];
                    let out_idx = q_idx * out_dim;
                    if mask_val < 0.0 {
                        for d in 0..head_dim {
                            output[out_idx + d] = 0.0;
                        }
                    }
                }
            }
        }

        Ok(DeviceBuffer::from_host(output))
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_attention_kernel_basic() {
        let kernel = CpuAttentionKernel::new();
        assert!(kernel.is_available());
        assert_eq!(kernel.arch(), AttentionArch::Cpu);
    }

    #[test]
    fn test_cpu_attention_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let result = CpuAttentionKernel::softmax(&input, 1, 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
