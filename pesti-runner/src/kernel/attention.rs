//! Attention kernel interface and configuration for Blackwell tensor cores.
//!
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! The kernel computes scaled dot-product attention:
//!   S = Q @ K^T / sqrt(D)
//!   O = softmax(S) @ V

use crate::cuda_runtime::CudaDeviceInfo;
use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::{Kvcache, KvcacheSlice};
use half::f16;
use std::sync::Arc;

/// Attention tensor core architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum AttentionArch {
    /// WGMMA — warp group matrix multiply (sm_120, consumer Blackwell)
    Wgmma,
    /// tcgen05 — tensor core with tensor memory (sm_100, datacenter Blackwell)
    #[default]
    Tcgen05,
    /// CPU fallback
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

    pub fn tile_size(&self) -> usize {
        match self {
            Self::Wgmma => 64,
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
    /// RoPE base frequency (typically 10000.0)
    pub rope_base: f32,
    /// Maximum position for RoPE scaling
    pub max_pos: usize,
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
            rope_base: 10000.0,
            max_pos: 32768,
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

    /// Compute softmax normalization constant for numerical stability
    pub fn softmax_scale(&self) -> f32 {
        self.scale()
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
                KvcacheSlice::new(
                    gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, true,
                )
            })
            .collect();

        let value_slices: Vec<KvcacheSlice> = (0..num_heads)
            .map(|h| {
                KvcacheSlice::new(
                    gmem_addr, num_heads, head_dim, max_seq, h, seq_start, seq_len, false,
                )
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

    pub fn tma_descriptors(
        &self,
        head_idx: usize,
    ) -> (Option<TmaDescriptor>, Option<TmaDescriptor>) {
        if head_idx >= self.key_slices.len() {
            return (None, None);
        }
        let k_desc = self.key_slices[head_idx].to_tma_descriptor();
        let v_desc = self.value_slices[head_idx].to_tma_descriptor();
        (Some(k_desc), Some(v_desc))
    }
}

use crate::kernel::tma_descriptor::TmaDescriptor;

#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("invalid dimensions: heads={num_heads}, head_dim={head_dim}, seq_len={seq_len})")]
    InvalidDimensions {
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    },

    #[error("head index out of bounds: head_idx={head_idx}, num_heads={num_heads})")]
    HeadIndexOutOfBounds { head_idx: usize, num_heads: usize },

    #[error("sequence length exceeded: current={current}, max={max})")]
    SeqLenExceeded { current: usize, max: usize },

    #[error("buffer size mismatch: expected {expected}, got {got})")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("kernel not available on this device")]
    NotAvailable,

    #[error("kernel launch failed: {0})")]
    LaunchFailed(String),

    #[error("CUDA error: {0})")]
    Cuda(String),

    #[error("unsupported architecture: {0})")]
    UnsupportedArch(String),

    #[error("tcgen05 constraint: head_dim must be divisible by 64, got {0})")]
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

// --- GPU Implementation (Real cuda-oxide backed) ---

use cuda_core::CudaFunction;

/// CUDA implementation for attention kernel using WGMMA tensor cores.
pub struct CudaAttentionKernel {
    arch: AttentionArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    module: Arc<cuda_core::CudaModule>,
    function: CudaFunction,
}

/// Builder for CudaAttentionKernel that handles PTX loading and kernel resolution.
pub struct CudaAttentionKernelBuilder {
    arch: AttentionArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    device_info: CudaDeviceInfo,
}

impl CudaAttentionKernelBuilder {
    pub fn new(
        arch: AttentionArch,
        context: Arc<cuda_core::CudaContext>,
        stream: Arc<cuda_core::CudaStream>,
        device_info: CudaDeviceInfo,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
            device_info,
        }
    }

    /// Build the kernel by loading PTX module and resolving function.
    pub fn build(self) -> Result<CudaAttentionKernel, AttentionError> {
        // Pre-flight architecture check
        match self.arch {
            AttentionArch::Wgmma if !self.device_info.supports_wgmma() => {
                return Err(AttentionError::UnsupportedArch(format!(
                    "WGMMA requires sm_120+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0, self.device_info.compute_capability.1
                )));
            }
            AttentionArch::Tcgen05 if !self.device_info.supports_tcgen05() => {
                return Err(AttentionError::UnsupportedArch(format!(
                    "tcgen05 requires sm_100+, but device is sm_{}.{}",
                    self.device_info.compute_capability.0, self.device_info.compute_capability.1
                )));
            }
            _ => {}
        }

        // Select PTX based on architecture
        let ptx_src = match self.arch {
            AttentionArch::Wgmma => include_str!("ptx/attention_wgmma.ptx"),
            AttentionArch::Tcgen05 => include_str!("ptx/attention_tcgen05.ptx"),
            AttentionArch::Cpu => {
                return Err(AttentionError::UnsupportedArch(
                    "Cpu architecture requires CPU kernel, not GPU".to_string(),
                ));
            }
        };

        // Load module from PTX source
        let module = self
            .context
            .load_module_from_ptx_src(ptx_src)
            .map_err(|e| AttentionError::Cuda(format!("module load failed: {}", e)))?;

        // Resolve kernel function
        let kernel_name = match self.arch {
            AttentionArch::Wgmma => "attention_wgmma_kernel",
            AttentionArch::Tcgen05 => "attention_tcgen05_kernel",
            AttentionArch::Cpu => {
                return Err(AttentionError::UnsupportedArch(
                    "Cpu architecture requires CPU kernel, not GPU".to_string(),
                ));
            }
        };
        let function = module
            .load_function(kernel_name)
            .map_err(|e| AttentionError::Cuda(format!("function load failed: {}", e)))?;

        Ok(CudaAttentionKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

impl CudaAttentionKernel {
    /// Create a new CUDA attention kernel with the given architecture.
    pub fn new(_arch: AttentionArch) -> Result<Self, AttentionError> {
        // For now, return an error to force builder usage
        Err(AttentionError::UnsupportedArch(
            "Use CudaAttentionKernelBuilder instead of direct construction".to_string(),
        ))
    }

    /// Get the cuda-oxide context for external operations
    pub fn context(&self) -> &Arc<cuda_core::CudaContext> {
        &self.context
    }

    /// Get the cuda-oxide stream
    pub fn stream(&self) -> &Arc<cuda_core::CudaStream> {
        &self.stream
    }
}

impl AttentionKernel for CudaAttentionKernel {
    fn forward(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        _mask: Option<&DeviceBuffer<f32>>,
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

        // Validate dimensions
        if num_heads == 0 || head_dim == 0 || cache_seq_len == 0 {
            return Err(AttentionError::InvalidDimensions {
                num_heads,
                head_dim,
                seq_len: cache_seq_len,
            });
        }

        // Output: attention scores [query_seq_len, num_heads, cache_seq_len]
        let out_len = query_seq_len * num_heads * cache_seq_len;
        let output = DeviceBuffer::<f32>::zeros(out_len);

        // Extract kernel parameters from config and inputs
        let seq_q = query_seq_len;
        let seq_k = cache_seq_len;
        let head_dim = config.head_dim;
        let scale = config.scale();

        // Launch WGMMA attention kernel (compute Q @ K^T / sqrt(D))
        // PTX expects: (alpha=scale, q_ptr, k_ptr, beta=0, s_ptr, seq_q, seq_k, head_dim)
        match self.arch {
            AttentionArch::Wgmma => {
                unsafe {
                    // Get device pointers (unwrap because we validated earlier)
                    let q_ptr = query.device_ptr();
                    let k_ptr = key_cache.device_ptr()
                        .ok_or(AttentionError::LaunchFailed("Key cache not on device".into()))?;
                    let s_ptr = output.device_ptr();

                    // Calculate grid dimensions (64x64 tiles)
                    let grid_x = (seq_k as u32).div_ceil(64);
                    let grid_y = (seq_q as u32).div_ceil(64);
                    let block_size = 128; // 128 threads per block

                    // Build kernel parameters - store values so addresses don't go out of scope
                    let alpha_val = scale.to_bits();
                    let beta_val = 0.0f32; // No bias term in attention logits
                    let seq_q_val = seq_q as u32;
                    let seq_k_val = seq_k as u32;
                    let head_dim_val = head_dim as u32;

                    // Create aligned arrays to hold parameter addresses
                    // These stay in scope for the duration of launch_kernel
                    let _kernel_params_buf = [0usize; 8];
                    let param_ptrs: [*const std::ffi::c_void; 8] = [
                        &alpha_val as *const u32 as *const std::ffi::c_void,
                        &(q_ptr as u64) as *const u64 as *const std::ffi::c_void,
                        &(k_ptr as u64) as *const u64 as *const std::ffi::c_void,
                        &beta_val as *const f32 as *const std::ffi::c_void,
                        &(s_ptr as u64) as *const u64 as *const std::ffi::c_void,
                        &seq_q_val as *const u32 as *const std::ffi::c_void,
                        &seq_k_val as *const u32 as *const std::ffi::c_void,
                        &head_dim_val as *const u32 as *const std::ffi::c_void,
                    ];

                    // Convert to mutable pointers for launch_kernel
                    let mut param_ptrs_mut: [*mut std::ffi::c_void; 8] = 
                        param_ptrs.map(|p| p as *mut std::ffi::c_void);

                    // Launch WGMMA tensor core kernel
                    cuda_core::launch_kernel(
                        self.function.cu_function(),
                        (grid_x, grid_y, 1),
                        (block_size, 1, 1),
                        0, // dynamic shared memory (we use static .shared in PTX)
                        self.stream.cu_stream(),
                        &mut param_ptrs_mut,
                    )
                    .map_err(|e| AttentionError::LaunchFailed(format!("WGMMA launch failed: {e}")))?;

                    // Synchronize stream to ensure completion
                    self.stream
                        .synchronize()
                        .map_err(|e| AttentionError::LaunchFailed(format!("Stream sync failed: {e}")))?;
                    
                    // Log successful kernel launch
                    eprintln!("[WGMMA] Launched attention kernel: Q[{seq_q}] x K[{seq_k}] -> S[{seq_q}x{seq_k}]");
                    eprintln!("  Grid: ({grid_x}, {grid_y}, 1), Block: {block_size} threads");
                    eprintln!("  Scale: {scale:.6}, Head dim: {head_dim}");
                }
            },
            AttentionArch::Tcgen05 => {
                // TODO: Implement tcgen05 path with TMA descriptors for async prefetching
                // Datacenter Blackwell (sm_100) uses cuTensorMapEncodeTiled()
                // Similar launch but with different thread config and TMA bindings
                return Err(AttentionError::NotAvailable);
            },
            _ => {
                // CPU fallback handled by caller
                return Err(AttentionError::NotAvailable);
            }
        }

        Ok(output)
    }

    fn arch(&self) -> AttentionArch {
        self.arch
    }

    fn is_available(&self) -> bool {
        // Check that kernel function is valid (not zeroed)
        unsafe { !self.function.cu_function().is_null() }
    }
}

// --- CPU Fallback Implementation ---

pub struct CpuAttentionKernel {
    arch: AttentionArch,
}

impl CpuAttentionKernel {
    pub fn new() -> Self {
        Self {
            arch: AttentionArch::Cpu,
        }
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

            let logits = Self::matmul_transpose_b(
                query_host,
                &k_slice,
                query_seq_len,
                cache_seq_len,
                head_dim,
            );
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

// --- Tests ---

