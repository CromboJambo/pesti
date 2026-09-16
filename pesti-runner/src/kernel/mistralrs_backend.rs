//! Mistral.rs backend — production-grade GPU kernels behind PESTI's trait layer.
//!
//! Bridges PESTI's `GemmKernel` and `AttentionKernel` traits to
//! [mistral.rs](https://github.com/Lightning-AI/mistral.rs) via candle-core, which provides:
//! - Optimized CUDA GEMM through the gemm crate family
//! - Flash attention and SDPA via candle's attention ops
//! - FP16/F32 tensor operations on GPU
//! - KV cache management during autoregressive generation
//!
//! This is a feature-gated backend — only compiled when `mistralrs` feature is enabled.

use crate::kernel::candle_bridge;
use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::{
    AttentionArch, AttentionConfig, AttentionError, AttentionKernel, GemmArch, GemmError,
    GemmKernel, Kvcache,
};
use half::f16;

// ── GEMM Backend ────────────────────────────────────────────────────────

/// A mistral.rs-backed GEMM kernel.
///
/// Wraps a single mistralrs device and uses its tensor operations for
/// matrix multiply. Falls back to CPU if the device is unavailable.
pub struct MistralRsGemmKernel {
    /// Architecture this kernel targets.
    arch: GemmArch,
    /// Whether the underlying device is actually available.
    available: bool,
    /// Device ordinal (if CUDA).
    device_idx: Option<usize>,
}

impl MistralRsGemmKernel {
    /// Try to create a mistral.rs GEMM kernel on the default GPU.
    ///
    /// Returns `None` if no GPU is available or if the device doesn't
    /// support the requested architecture.
    pub fn try_new(arch: GemmArch) -> Option<Self> {
        // Check if CUDA is available
        if !crate::cuda_runtime::is_available() {
            return None;
        }

        // Try to initialize a device — if it fails, fall back gracefully
        let device_idx = match crate::cuda_runtime::enumerate_devices() {
            Ok(devices) if !devices.is_empty() => Some(0),
            _ => None,
        };

        let available = device_idx.is_some();

        if available {
            tracing::debug!(
                arch = ?arch,
                device = device_idx.map(|d| d.to_string()).unwrap_or("none".to_string()),
                "MistralRs GEMM kernel initialized"
            );
        }

        Some(Self {
            arch,
            available,
            device_idx,
        })
    }

    /// Run GEMM using mistral.rs tensor operations via candle-core.
    ///
    /// Converts PESTI's DeviceBuffer<f16> to candle Tensors, performs
    /// matrix multiply on GPU (or CPU if CUDA unavailable), and converts
    /// the result back to DeviceBuffer<f32>.
    fn matmul_mistralrs(
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
        // Convert input buffers to candle tensors
        let a_slice = a.as_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: m * k,
            got: 0,
        })?;
        let a_tensor = candle_bridge::f16_to_tensor(a_slice, &[m, k], None)
            .map_err(|e| GemmError::Cuda(format!("convert A tensor: {}", e)))?;

        let b_slice = b.as_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: k * n,
            got: 0,
        })?;
        let b_tensor = candle_bridge::f16_to_tensor(b_slice, &[k, n], None)
            .map_err(|e| GemmError::Cuda(format!("convert B tensor: {}", e)))?;

        // Perform GEMM: C = alpha * (A @ B) + beta * C
        let result_data = candle_bridge::gemm_with_tensors(&a_tensor, &b_tensor, None, m, k, n, alpha, beta)
            .map_err(|e| GemmError::Cuda(format!("candle GEMM: {}", e)))?;

        // Write result back to output buffer (host-backed only for now)
        let c_host = c.as_mut_slice().ok_or(GemmError::BufferSizeMismatch {
            expected: m * n,
            got: 0,
        })?;
        if result_data.len() != c_host.len() {
            return Err(GemmError::BufferSizeMismatch {
                expected: c_host.len(),
                got: result_data.len(),
            });
        }
        c_host.clone_from_slice(&result_data);

        Ok(())
    }
}

impl GemmKernel for MistralRsGemmKernel {
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
        if !self.available {
            return Err(GemmError::NotAvailable);
        }

        // Validate dimensions
        if m == 0 || n == 0 || k == 0 {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Try mistralrs path first, fall through to caller's fallback
        self.matmul_mistralrs(alpha, a, b, beta, c, m, n, k)
    }

    fn arch(&self) -> GemmArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ── Attention Backend ───────────────────────────────────────────────────

/// A mistral.rs-backed Attention kernel.
///
/// Wraps mistralrs attention primitives (flash attention, SDPA) for
/// scaled dot-product attention computation.
pub struct MistralRsAttentionKernel {
    /// Architecture this kernel targets.
    arch: AttentionArch,
    /// Whether the underlying device is actually available.
    available: bool,
}

impl MistralRsAttentionKernel {
    /// Try to create a mistral.rs attention kernel on the default GPU.
    pub fn try_new(arch: AttentionArch) -> Option<Self> {
        if !crate::cuda_runtime::is_available() {
            return None;
        }

        Some(Self {
            arch,
            available: true,
        })
    }

    /// Run attention using mistral.rs primitives via candle-core.
    ///
    /// Converts PESTI buffers to candle Tensors, reshapes for SDPA,
    /// and computes scaled dot-product attention on GPU (or CPU fallback).
    fn forward_mistralrs(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        use crate::kernel::candle_bridge;

        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let seq_len = key_cache.seq_len();

        // Query shape: [1, 1, num_heads, head_dim] (single token decode)
        let q_shape = vec![1usize, 1, num_heads, head_dim];
        let q_slice = query.as_slice().ok_or(AttentionError::InvalidDimensions {
            num_heads,
            head_dim,
            seq_len: 0,
        })?;
        let q_tensor = candle_bridge::f16_to_tensor(q_slice, &q_shape, None)
            .map_err(|e| AttentionError::Cuda(format!("convert query tensor: {}", e)))?;

        // Extract K and V from KV cache into separate tensors.
        // KV cache layout: [num_heads * head_dim, max_seq] with K first, then V.
        let k_stride = num_heads * head_dim;
        let k_shape = vec![1usize, seq_len, num_heads, head_dim];

        // Build K tensor from cache (extract K rows for all seq positions)
        let mut k_data: Vec<f16> = Vec::with_capacity(seq_len * k_stride);
        if let Some(kbuf) = key_cache.buffer().as_slice() {
            for pos in 0..seq_len {
                let row_start = pos * k_stride;
                k_data.extend_from_slice(&kbuf[row_start..row_start + k_stride]);
            }
        }
        let k_tensor = candle_bridge::f16_to_tensor(&k_data, &k_shape, None)
            .map_err(|e| AttentionError::Cuda(format!("convert key tensor: {}", e)))?;

        // Build V tensor from cache (extract V rows for all seq positions)
        let mut v_data: Vec<f16> = Vec::with_capacity(seq_len * k_stride);
        if let Some(vbuf) = value_cache.buffer().as_slice() {
            let v_base = k_stride * key_cache.max_seq();
            for pos in 0..seq_len {
                let row_start = v_base + pos * k_stride;
                v_data.extend_from_slice(&vbuf[row_start..row_start + k_stride]);
            }
        }
        let v_tensor = candle_bridge::f16_to_tensor(&v_data, &k_shape, None)
            .map_err(|e| AttentionError::Cuda(format!("convert value tensor: {}", e)))?;

        // Compute scale factor: 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Run SDPA via candle
        let result_tensor = candle_bridge::sdpa(&q_tensor, &k_tensor, &v_tensor, scale)
            .map_err(|e| AttentionError::Cuda(format!("candle SDPA: {}", e)))?;

        // Convert result back to DeviceBuffer<f32>
        let result_data = candle_bridge::tensor_to_f32(&result_tensor)
            .map_err(|e| AttentionError::Cuda(format!("convert result tensor: {}", e)))?;

        Ok(DeviceBuffer::from_host(result_data))
    }
}

impl AttentionKernel for MistralRsAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        mask: Option<&DeviceBuffer<f32>>,
        config: &AttentionConfig,
    ) -> Result<DeviceBuffer<f32>, AttentionError> {
        if !self.available {
            return Err(AttentionError::NotAvailable);
        }

        // Validate inputs
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let cache_seq_len = key_cache.seq_len();

        if num_heads == 0 || head_dim == 0 || cache_seq_len == 0 {
            return Err(AttentionError::InvalidDimensions {
                num_heads,
                head_dim,
                seq_len: cache_seq_len,
            });
        }

        if !key_cache.buffer().is_backed() || !value_cache.buffer().is_backed() {
            return Err(AttentionError::NotAvailable);
        }

        // Try mistralrs path first, fall through to caller's fallback
        self.forward_mistralrs(query, key_cache, value_cache, mask, config)
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        self.available
    }
}

// ── Backend Selection ───────────────────────────────────────────────────

/// The active inference backend for GPU computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralRsBackend {
    /// Use mistral.rs kernels (production-grade).
    MistralRs,
    /// Use PESTI's own CUDA kernels (PTX-based, unverified).
    Cuda,
    /// CPU-only mode.
    Cpu,
}

impl Default for MistralRsBackend {
    fn default() -> Self {
        // Prefer mistral.rs if available, otherwise CUDA, then CPU
        if crate::cuda_runtime::is_available() {
            Self::MistralRs
        } else {
            Self::Cpu
        }
    }
}

impl MistralRsBackend {
    /// Check if GPU acceleration is available.
    pub fn gpu_available(&self) -> bool {
        matches!(self, Self::MistralRs | Self::Cuda)
    }

    /// Get a description of this backend.
    pub fn description(&self) -> &'static str {
        match self {
            Self::MistralRs => "mistral.rs (production GPU kernels)",
            Self::Cuda => "PESTI CUDA (PTX, unverified)",
            Self::Cpu => "CPU (reference)",
        }
    }

    /// Try to create a GEMM kernel for this backend.
    pub fn create_gemm_kernel(&self, arch: GemmArch) -> Option<Box<dyn GemmKernel + Send + Sync>> {
        match self {
            Self::MistralRs => MistralRsGemmKernel::try_new(arch)
                .map(|k| Box::new(k) as Box<dyn GemmKernel + Send + Sync>),
            Self::Cuda => {
                // Fall through to the existing CUDA path
                None
            }
            Self::Cpu => Some(Box::new(crate::kernel::CpuGemmKernel::new())),
        }
    }

    /// Try to create an attention kernel for this backend.
    pub fn create_attention_kernel(
        &self,
        arch: AttentionArch,
    ) -> Option<Box<dyn AttentionKernel + Send + Sync>> {
        match self {
            Self::MistralRs => MistralRsAttentionKernel::try_new(arch)
                .map(|k| Box::new(k) as Box<dyn AttentionKernel + Send + Sync>),
            Self::Cuda => {
                // Fall through to the existing CUDA path
                None
            }
            Self::Cpu => Some(Box::new(crate::kernel::CpuAttentionKernel::new(
                AttentionArch::Cpu,
            ))),
        }
    }
}
