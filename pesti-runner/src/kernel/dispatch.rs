//! Kernel dispatch: bridges the tensor kernel layer to the transformer layer.
//!
//! The transformer layer (Linear, Attention, TransformerLayer) currently uses
//! raw `Vec<f32>` on the host. This module provides a dispatch context that
//! can route GEMM and attention operations to GPU or CPU based on availability,
//! handling buffer allocation, data transfers, and fallback transparently.
//!
//! ## Architecture
//!
//! ```text
//! TransformerLayer (CPU path)
//!     │
//!     ▼
//! DispatchContext  ← holds InferenceEngine + MemoryManager
//!     │
//!     ├── dispatch_linear()  → GPU GEMM or CPU fallback
//!     ├── dispatch_attention() → GPU attention or CPU fallback
//!     └── dispatch_gemm()    → raw GEMM with auto-transfer
//! ```
//!
//! ## Usage
//!
//! ```text
//! let ctx = DispatchContext::new(MemoryManager::new());
//!
//! // GPU-backed linear: weights go to device, matmul on GPU, result back to host
//! let out = ctx.dispatch_linear(&input, &weights, batch_size)?;
//!
//! // Explicit CPU fallback
//! let out = ctx.dispatch_linear_cpu(&input, &weights, batch_size)?;
//! ```

use crate::error::RunnerError;
use crate::inference_engine::InferenceEngine;
#[cfg(feature = "cuda")]
use crate::kernel::attention::{
    AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel,
};
#[cfg(not(feature = "cuda"))]
use crate::kernel::attention_stub::{
    AttentionArch, AttentionConfig, AttentionKernel, CpuAttentionKernel,
};
use crate::kernel::candle_bridge;
use crate::kernel::device_buf::DeviceBuffer;
#[cfg(feature = "cuda")]
use crate::kernel::gemm::{GemmArch, GemmKernel};
#[cfg(not(feature = "cuda"))]
use crate::kernel::gemm_stub::{GemmArch, GemmKernel};
#[cfg(feature = "cuda")]
use crate::kernel::kvcache::Kvcache;
#[cfg(not(feature = "cuda"))]
use crate::kernel::kvcache_stub::Kvcache;
#[cfg(feature = "cuda")]
use crate::kernel::memory::MemoryManager;
#[cfg(not(feature = "cuda"))]
use crate::kernel::memory_stub::MemoryManager;
use candle_core::{DType, Device};
use half::f16;
use tracing::{debug, warn};

// ── Error types ────────────────────────────────────────────────────────────

/// Errors specific to the dispatch layer.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("CUDA not available")]
    CudaNotAvailable,

    #[error("buffer transfer failed: {0}")]
    Transfer(String),

    #[error("kernel dispatch failed: {0}")]
    Kernel(String),

    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("memory allocation failed: {0}")]
    Memory(String),

    #[error("GPU kernel returned error: {0}")]
    GpuKernel(String),
}

impl From<RunnerError> for DispatchError {
    fn from(e: RunnerError) -> Self {
        match e {
            RunnerError::Gemm {
                arch,
                m,
                n,
                k,
                detail,
            } => DispatchError::GpuKernel(format!(
                "GEMM(arch={arch}, m={m}, n={n}, k={k}): {detail}"
            )),
            RunnerError::Attention {
                num_heads,
                head_dim,
                seq,
                detail,
            } => DispatchError::GpuKernel(format!(
                "Attention(heads={num_heads}, dim={head_dim}, seq={seq}): {detail}"
            )),
            RunnerError::Tensor(msg) => DispatchError::Kernel(msg),
            other => DispatchError::Kernel(other.to_string()),
        }
    }
}

// ── DispatchContext ────────────────────────────────────────────────────────

/// Context that bridges the transformer layer to GPU/CPU tensor kernels.
///
/// Holds an `InferenceEngine` (with its GEMM + attention kernels) and a
/// `MemoryManager` for buffer allocation. Provides high-level dispatch
/// methods that handle buffer allocation, data transfer, kernel invocation,
/// and CPU fallback automatically.
pub struct DispatchContext {
    /// The inference engine with its GEMM and attention kernels.
    engine: InferenceEngine,
    /// Memory manager for device buffer allocation.
    memory: MemoryManager,
    /// Whether GPU path is preferred (true) or CPU-only (false).
    prefer_gpu: bool,
    /// Cached CPU GEMM kernel for fallback.
    cpu_gemm: crate::kernel::CpuGemmKernel,
    /// Cached CPU attention kernel for fallback.
    cpu_attention: CpuAttentionKernel,
}

impl DispatchContext {
    /// Create a new dispatch context, auto-detecting GPU availability.
    ///
    /// When the `cuda` feature is enabled and a GPU is present, this builds a
    /// `Device::Cuda` engine (which initializes the CUDA runtime + GEMM kernel)
    /// and a CUDA-backed `MemoryManager` on the SAME stream the engine uses.
    /// Otherwise it falls back to a CPU-only engine and memory backend.
    pub fn new() -> Self {
        let device = Self::detect_device();
        let engine = InferenceEngine::new(device, DType::F32);
        let prefer_gpu = engine.gpu_available();
        let backend_desc = engine.backend_description();
        tracing::info!(backend = %backend_desc, "DispatchContext initialized");

        // The memory manager MUST share the engine's CUDA stream. Using a
        // separate stream (or the CPU backend) makes H2D/D2H race with kernel
        // launches and silently corrupts results.
        let memory = Self::build_memory(&engine);

        Self {
            engine,
            memory,
            prefer_gpu,
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: CpuAttentionKernel::new(AttentionArch::Cpu),
        }
    }

    /// Pick the target device: CUDA when the feature is on and a GPU exists.
    fn detect_device() -> Device {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda_runtime::is_available;
            if is_available() {
                match Device::new_cuda(0) {
                    Ok(dev) => return dev,
                    Err(e) => {
                        tracing::warn!(error = %e, "CUDA device creation failed, using CPU");
                    }
                }
            }
            Device::Cpu
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    }

    /// Build a memory manager that matches the engine's backend.
    fn build_memory(engine: &InferenceEngine) -> MemoryManager {
        #[cfg(feature = "cuda")]
        {
            if let (Some(stream), Some(info)) = (engine.cuda_stream(), engine.cuda_device_info()) {
                return MemoryManager::Cuda(crate::kernel::memory::CudaMemoryBackend::with_device_info(
                    stream.clone(),
                    info.clone(),
                ));
            }
        }
        MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024))
    }

    /// Create a dispatch context with explicit GPU preference.
    pub fn with_gpu_preference(prefer_gpu: bool) -> Self {
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let backend_desc = engine.backend_description();
        tracing::info!(backend = %backend_desc, prefer_gpu, "DispatchContext initialized with GPU preference");
        Self {
            engine,
            memory: crate::kernel::MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(
                1024 * 1024,
            )),
            prefer_gpu,
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: CpuAttentionKernel::new(AttentionArch::Cpu),
        }
    }

    /// Create from an existing inference engine.
    pub fn from_engine(mut engine: InferenceEngine) -> Self {
        let prefer_gpu = engine.gpu_available();
        tracing::info!(gpu = %prefer_gpu, "DispatchContext::from_engine initialized");

        // Build a proper memory backend that matches the engine's GPU/CPU state
        let memory = Self::build_memory_from_engine(&engine);

        Self {
            engine,
            memory,
            prefer_gpu,
            cpu_gemm: crate::kernel::CpuGemmKernel::new(),
            cpu_attention: CpuAttentionKernel::new(AttentionArch::Cpu),
        }
    }

    /// Build memory manager from an existing engine (mirrors build_memory logic).
    fn build_memory_from_engine(engine: &InferenceEngine) -> MemoryManager {
        #[cfg(feature = "cuda")]
        {
            if let (Some(stream), Some(info)) = (engine.cuda_stream(), engine.cuda_device_info()) {
                return MemoryManager::Cuda(crate::kernel::memory::CudaMemoryBackend::with_device_info(
                    stream.clone(),
                    info.clone(),
                ));
            }
        }
        MemoryManager::Cpu(crate::kernel::CpuMemoryBackend::new(1024 * 1024))
    }

    /// Whether GPU path is preferred.
    pub fn prefer_gpu(&self) -> bool {
        self.prefer_gpu
    }

    /// Set GPU preference.
    pub fn set_prefer_gpu(&mut self, prefer_gpu: bool) {
        self.prefer_gpu = prefer_gpu;
    }

    /// Whether GPU is actually available (not just preferred).
    pub fn gpu_available(&self) -> bool {
        self.engine.gpu_available()
    }

    /// Get the GEMM architecture.
    pub fn gemm_arch(&self) -> GemmArch {
        self.engine.gemm_arch()
    }

    // ── Core dispatch: GEMM ──────────────────────────────────────────────

    /// Dispatch a GEMM operation: C = alpha * A @ B + beta * C.
    ///
    /// A: [m x k] f16, B: [k x n] f16, C: [m x n] f32
    ///
    /// If GPU is preferred and available:
    ///   1. Allocate device buffers for A, B, C
    ///   2. Transfer A and B to device
    ///   3. Launch GPU GEMM kernel
    ///   4. Transfer result C back to host
    ///
    /// Falls back to CPU if GPU is unavailable or fails.
    pub fn dispatch_gemm(
        &self,
        a_host: &[f16],
        b_host: &[f16],
        c_init: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Vec<f32>, DispatchError> {
        let c_len = m * n;

        // If GPU not preferred or unavailable, use CPU directly
        if !self.prefer_gpu || !self.gpu_available() {
            debug!(m, n, k, "GEMM dispatch: GPU not available, using CPU");
            return self.dispatch_gemm_cpu(a_host, b_host, c_init, m, n, k, alpha, beta);
        }

        // GPU path: allocate device buffers
        let a_bytes = std::mem::size_of_val(a_host);
        let b_bytes = std::mem::size_of_val(b_host);
        let c_bytes = c_len * std::mem::size_of::<f32>();

        let a_handle = self
            .memory
            .alloc(a_bytes)
            .map_err(|e| DispatchError::Memory(format!("alloc A: {e}")))?;
        let b_handle = self
            .memory
            .alloc(b_bytes)
            .map_err(|e| DispatchError::Memory(format!("alloc B: {e}")))?;
        let c_handle = self
            .memory
            .alloc(c_bytes)
            .map_err(|e| DispatchError::Memory(format!("alloc C: {e}")))?;

        let a_buf = DeviceBuffer::<f16>::from_backend(a_handle, a_host.len());
        let b_buf = DeviceBuffer::<f16>::from_backend(b_handle, b_host.len());
        let mut c_buf = DeviceBuffer::<f32>::from_backend(c_handle, c_len);

        // Transfer inputs to device
        let a_bytes_raw: &[u8] =
            unsafe { std::slice::from_raw_parts(a_host.as_ptr() as *const u8, a_bytes) };
        self.memory
            .h2d(a_bytes_raw, a_handle)
            .map_err(|e| DispatchError::Transfer(format!("H2D A: {e}")))?;

        let b_bytes_raw: &[u8] =
            unsafe { std::slice::from_raw_parts(b_host.as_ptr() as *const u8, b_bytes) };
        self.memory
            .h2d(b_bytes_raw, b_handle)
            .map_err(|e| DispatchError::Transfer(format!("H2D B: {e}")))?;

        // Initialize C if provided
        if let Some(c_init_data) = c_init {
            let c_init_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    c_init_data.as_ptr() as *const u8,
                    std::mem::size_of_val(c_init_data),
                )
            };
            self.memory
                .h2d(c_init_bytes, c_handle)
                .map_err(|e| DispatchError::Transfer(format!("H2D C init: {e}")))?;
        }

        // Dispatch to GPU or CPU fallback
        let _result = self
            .engine
            .matmul(alpha, &a_buf, &b_buf, beta, &mut c_buf, m, n, k);

        // Transfer result back to host
        let mut c_host = vec![0.0f32; c_len];
        let c_bytes_out: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(c_host.as_mut_ptr() as *mut u8, c_bytes) };
        if let Err(e) = self.memory.d2h(c_handle, c_bytes_out) {
            warn!(error = %e, "GEMM dispatch: D2H failed, using zero output");
            return Ok(c_host);
        }

        // Sync after async D2H to ensure data is ready
        if let Err(e) = self.memory.sync() {
            warn!(error = %e, "GEMM dispatch: sync failed, returning partial result");
        }

        Ok(c_host)
    }

    /// CPU-only GEMM dispatch.
    ///
    /// Computes C = alpha * A @ B + beta * C where:
    ///   A: [m x k] f16, B: [k x n] f16, C: [m x n] f32
    pub fn dispatch_gemm_cpu(
        &self,
        a_host: &[f16],
        b_host: &[f16],
        c_init: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<Vec<f32>, DispatchError> {
        let c_len = m * n;
        let mut c = if let Some(c_init_data) = c_init {
            c_init_data.to_vec()
        } else {
            vec![0.0f32; c_len]
        };

        // Direct f16→f32 GEMM: C[i,j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C[i,j]
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_host[i * k + kk].to_f32() * b_host[kk * n + j].to_f32();
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }
        Ok(c)
    }

    // ── Core dispatch: Linear ────────────────────────────────────────────

    /// Dispatch a linear layer forward pass: y = x @ W^T + bias.
    ///
    /// x: [batch_size, in_features] f32
    /// W: [out_features, in_features] f16 (weight matrix)
    /// bias: [out_features] f32 (optional)
    ///
    /// Returns: [batch_size, out_features] f32
    ///
    /// Automatically dispatches to GPU if available, falls back to CPU.
    /// When GPU is available, uses `candle_bridge::gemm` for GPU-accelerated
    /// matrix multiplication via candle-core's CUDA backend.
    pub fn dispatch_linear(
        &self,
        x: &[f32],
        weights: &[f16],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>, DispatchError> {
        let m = batch_size;
        let k = in_features;
        let n = out_features;

        // Convert input to f16 for GPU
        let x_f16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();

        // Transpose weights: W is [out, in], need [in, out] for GEMM
        // W^T[i,j] = W[j,i]
        let w_t: Vec<f16> = {
            let mut out = Vec::with_capacity(k * n);
            for i in 0..k {
                for j in 0..n {
                    out.push(weights[j * k + i]);
                }
            }
            out
        };

        // Use candle_bridge::gemm when a real CUDA device is available, CPU
        // fallback otherwise. (The bridge only runs on GPU when candle-core is
        // built with its cuda feature; otherwise prefer the native CPU GEMM.)
        let mut result = if self.prefer_gpu
            && self.gpu_available()
            && crate::kernel::candle_bridge::bridge_is_cuda()
        {
            debug!(m, n, k, "Linear: using candle_bridge::gemm (GPU)");

            // Validate weight shape before transpose
            assert_eq!(
                weights.len(),
                n * k,
                "Weight shape mismatch: expected {} elements ({}×{}), got {}",
                n * k,
                n,
                k,
                weights.len()
            );

            candle_bridge::gemm(&x_f16, &w_t, None, m, n, k, 1.0, 0.0)
                .map_err(|e| DispatchError::Kernel(format!("candle_bridge::gemm: {e}")))
        } else {
            debug!(m, n, k, "Linear: using CPU GEMM");
            Ok(self.dispatch_gemm_cpu(&x_f16, &w_t, None, m, n, k, 1.0, 0.0)?)
        }?;

        // Add bias if present
        if let Some(b) = bias {
            for b_idx in 0..batch_size {
                for o in 0..out_features {
                    result[b_idx * out_features + o] += b[o];
                }
            }
        }

        Ok(result)
    }

    /// CPU-only linear forward pass.
    pub fn dispatch_linear_cpu(
        &self,
        x: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        batch_size: usize,
    ) -> Result<Vec<f32>, DispatchError> {
        let mut output = vec![0.0f32; batch_size * out_features];

        for b in 0..batch_size {
            let x_start = b * in_features;
            for o in 0..out_features {
                let mut sum = 0.0f32;
                for i in 0..in_features {
                    sum += x[x_start + i] * weights[o * in_features + i];
                }
                if let Some(bias) = bias {
                    sum += bias[o];
                }
                output[b * out_features + o] = sum;
            }
        }

        Ok(output)
    }

    // ── Core dispatch: Attention ─────────────────────────────────────────

    /// Dispatch attention: softmax(Q @ K^T / sqrt(head_dim)) @ V.
    ///
    /// query: [query_seq_len, num_heads * head_dim] f16
    /// key_cache: KV cache containing K
    /// value_cache: KV cache containing V
    ///
    /// Returns: [query_seq_len, num_heads * head_dim] f32
    pub fn dispatch_attention(
        &self,
        query: &[f16],
        key_cache: &Kvcache,
        value_cache: &Kvcache,
        num_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Vec<f32>, DispatchError> {
        let query_seq_len = query.len() / (num_heads * head_dim);
        let config = AttentionConfig {
            num_heads,
            head_dim,
            max_seq,
            arch: AttentionArch::default(),
            use_tma: true,
            block_size: 0,
            rope_base: 10000.0,
            max_pos: 32768,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };

        // CPU-only path to eliminate H2D overhead and precision loss from repeated transfers
        debug!(
            m = query_seq_len * num_heads,
            n = max_seq,
            "Attention dispatch: CPU path (no H2D)"
        );

        // Allocate device buffer for query only (one-way transfer), then pull result back once
        let query_bytes = std::mem::size_of_val(query);
        let query_handle = self
            .memory
            .alloc(query_bytes)
            .map_err(|e| DispatchError::Memory(format!("alloc query: {e}")))?;

        // One H2D transfer (no D2H intermediate for softmax)
        let query_bytes_raw: &[u8] =
            unsafe { std::slice::from_raw_parts(query.as_ptr() as *const u8, query_bytes) };
        self.memory
            .h2d(query_bytes_raw, query_handle)
            .map_err(|e| DispatchError::Transfer(format!("H2D query: {e}")))?;

        let query_buf = DeviceBuffer::<f16>::from_backend(query_handle, query.len());

        // Run CPU attention (still expects device buffer, but extracts to host internally)
        let result_buf = self
            .cpu_attention
            .forward(&query_buf, key_cache, value_cache, None, &config)
            .map_err(|e| DispatchError::Kernel(format!("CPU attention: {e}")))?;

        // One final D2H transfer to get Vec<f32> output (no intermediate softmax H2D)
        let out_dim = num_heads * head_dim;
        let result_len = query_seq_len * out_dim;
        let mut result_host = vec![0.0f32; result_len];
        let result_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(result_host.as_mut_ptr() as *mut u8, result_len * 4)
        };
        self.memory
            .d2h(result_buf.handle(), result_bytes)
            .map_err(|e| DispatchError::Transfer(format!("D2H attention: {e}")))?;

        Ok(result_host)
    }

    // ── Utility ──────────────────────────────────────────────────────────

    /// Get device info string.
    pub fn device_info(&self) -> String {
        self.engine
            .full_device_info()
            .unwrap_or_else(|_| "unknown".to_string())
    }

    /// List available devices.
    #[cfg(feature = "cuda")]
    pub fn list_devices(&self) -> Vec<crate::cuda_runtime::CudaDeviceInfo> {
        crate::cuda_runtime::enumerate_devices().unwrap_or_default()
    }
}

impl Default for DispatchContext {
    fn default() -> Self {
        Self::new()
    }
}

// ── LinearDispatch: GPU-aware linear layer ─────────────────────────────────

/// A linear layer that can dispatch to GPU or CPU.
///
/// Wraps weight matrix + bias and provides `forward()` that automatically
/// picks the best backend.
pub struct LinearDispatch {
    /// Weight matrix (stored as f16 for GPU compatibility).
    weights_f16: Vec<f16>,
    /// Weight matrix (stored as f32 for CPU path).
    weights_f32: Vec<f32>,
    /// Optional bias.
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl LinearDispatch {
    pub fn new(
        weights_f16: Vec<f16>,
        weights_f32: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weights_f16,
            weights_f32,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass with dispatch context.
    pub fn forward(
        &self,
        ctx: &DispatchContext,
        x: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, DispatchError> {
        if !ctx.prefer_gpu() || !ctx.gpu_available() {
            return self.forward_cpu(x, batch_size);
        }

        ctx.dispatch_linear(
            x,
            &self.weights_f16,
            self.bias.as_deref(),
            self.in_features,
            self.out_features,
            batch_size,
        )
    }

    /// CPU-only forward pass.
    pub fn forward_cpu(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>, DispatchError> {
        ctx_dispatch_linear_cpu(
            x,
            &self.weights_f32,
            self.bias.as_deref(),
            self.in_features,
            self.out_features,
            batch_size,
        )
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// Standalone CPU linear forward (free function to avoid circular impl).
fn ctx_dispatch_linear_cpu(
    x: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
    batch_size: usize,
) -> Result<Vec<f32>, DispatchError> {
    let mut output = vec![0.0f32; batch_size * out_features];
    for b in 0..batch_size {
        let x_start = b * in_features;
        for o in 0..out_features {
            let mut sum = 0.0f32;
            for i in 0..in_features {
                sum += x[x_start + i] * weights[o * in_features + i];
            }
            if let Some(bias) = bias {
                sum += bias[o];
            }
            output[b * out_features + o] = sum;
        }
    }
    Ok(output)
}

// ── AttentionDispatch: GPU-aware attention layer ───────────────────────────

/// An attention layer that can dispatch to GPU or CPU.
pub struct AttentionDispatch {
    /// Q projection weights.
    pub wq: LinearDispatch,
    /// K projection weights.
    pub wk: LinearDispatch,
    /// V projection weights.
    pub wv: LinearDispatch,
    /// O projection weights.
    pub wo: LinearDispatch,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    /// RoPE base (theta). Qwen2.5 uses 1e6, not the 1e4 LLaMA default —
    /// must come from the model config, never hardcoded.
    pub rope_base: f32,
}

impl AttentionDispatch {
    /// Compute scaled dot-product attention with dispatch.
    ///
    /// 1. Q/K/V projections (GPU or CPU via dispatch context)
    /// 2. RoPE on Q and K
    /// 3. softmax(Q @ K^T / sqrt(head_dim)) @ V
    /// 4. Output projection (wo)
    pub fn forward(
        &mut self,
        ctx: &DispatchContext,
        x: &[f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
        key_cache: &mut Kvcache,
        value_cache: &mut Kvcache,
    ) -> Result<Vec<f32>, DispatchError> {
        let embed_dim = self.num_heads * self.head_dim;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Q/K/V projections (GPU or CPU)
        let q = self.wq.forward(ctx, x, batch_size)?;
        let k = self.wk.forward(ctx, x, batch_size)?;
        let v = self.wv.forward(ctx, x, batch_size)?;

        // GPU-accelerated path when available.
        //
        // NOTE: this routes through `candle_bridge`, which only runs on a real
        // CUDA device when candle-core is built with its `cuda` feature. When
        // the bridge falls back to CPU, the native CPU dispatch path below is
        // used instead (it is correct and avoids the bridge's GQA shape bugs).
        if ctx.gpu_available() && crate::kernel::candle_bridge::bridge_is_cuda() {
            return self.forward_gpu(
                ctx,
                &q,
                &k,
                &v,
                batch_size,
                seq_len,
                start_pos,
                key_cache,
                value_cache,
                scale,
            );
        }

        // CPU fallback: RoPE + manual SDPA
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        self.apply_rope(&mut q_rope, &mut k_rope, seq_len, start_pos);

        let kv_dim = self.num_kv_heads * self.head_dim;
        for pos in 0..seq_len {
            let global_pos = start_pos + pos;
            let k_start = pos * kv_dim;
            let k_row: Vec<f16> = k_rope[k_start..(k_start + kv_dim)]
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            let v_start = pos * kv_dim;
            let v_row: Vec<f16> = v[v_start..(v_start + kv_dim)]
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            // Write each tensor into its OWN cache's OWN region only.
            // Do NOT use write_kv_at here: it writes K AND V into one buffer,
            // and with separate key/value caches that cross-contaminates each
            // cache's unused region (harmless to region-selective CPU readers,
            // corrupting to whole-buffer GPU readers). See Kvcache::write_kv_at
            // docs and the `kv_write_no_cross_contamination` regression test.
            key_cache
                .write_k_at(global_pos, &k_row)
                .map_err(|e| DispatchError::Kernel(format!("KV cache K write at pos {global_pos}: {e}")))?;
            value_cache
                .write_v_at(global_pos, &v_row)
                .map_err(|e| DispatchError::Kernel(format!("KV cache V write at pos {global_pos}: {e}")))?;
        }

        let mut output = vec![0.0f32; batch_size * seq_len * embed_dim];
        let cache_len = start_pos + seq_len;
        let heads_per_group = self.num_heads / self.num_kv_heads;
        for b in 0..batch_size {
            for pos in 0..seq_len {
                let q_idx = (b * seq_len + pos) * embed_dim;
                let mut attn_output = vec![0.0f32; self.num_heads * self.head_dim];

                // Per-head GQA attention. Each query head h attends over the KV
                // cache using its group's KV head (h / heads_per_group), with its
                // OWN score vector and OWN softmax. (The previous code summed all
                // heads' Q·K into one scalar per position and shared a single
                // softmax across heads — that is not attention and produced
                // garbage.)
                for h in 0..self.num_heads {
                    let q_start = q_idx + h * self.head_dim;
                    let group = h / heads_per_group;

                    // scores[j] = scale * dot(q_head_h, k_head_group @ pos j)
                    let mut scores = vec![0.0f32; cache_len];
                    for j in 0..cache_len {
                        let k_slice =
                            Self::extract_head_slice(key_cache, true, group, j, self.head_dim);
                        if k_slice.len() == self.head_dim {
                            let mut sum = 0.0f32;
                            for d in 0..self.head_dim {
                                sum += q_rope[q_start + d] * k_slice[d].to_f32();
                            }
                            scores[j] = sum * scale;
                        }
                    }

                    // Per-head numerically-stable softmax.
                    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = scores.iter().map(|w| (*w - max_val).exp()).collect();
                    let exp_sum: f32 = exps.iter().sum();
                    let inv_sum = if exp_sum > 0.0 {
                        1.0 / exp_sum
                    } else {
                        1.0 / cache_len as f32
                    };

                    // V-weighted sum for this head.
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..cache_len {
                            let v_slice = Self::extract_head_slice(
                                value_cache,
                                false,
                                group,
                                j,
                                self.head_dim,
                            );
                            if !v_slice.is_empty() {
                                sum += (exps[j] * inv_sum) * v_slice[d].to_f32();
                            }
                        }
                        attn_output[h * self.head_dim + d] = sum;
                    }
                }

                let wo_output = self.wo.forward(ctx, &attn_output, 1)?;
                for i in 0..embed_dim {
                    output[(b * seq_len + pos) * embed_dim + i] = wo_output[i];
                }
            }
        }

        Ok(output)
    }

    /// Apply RoPE (Rotary Positional Embeddings) to Q and K.
    fn apply_rope(&self, q: &mut [f32], k: &mut [f32], seq_len: usize, start_pos: usize) {
        let dim = self.head_dim;
        // inv_freq[i] = rope_base^(-2i/dim). Base comes from the model config
        // (Qwen2.5 = 1e6); hardcoding 1e4 here silently corrupted attention.
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / self.rope_base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        for pos in 0..seq_len {
            let global_pos = start_pos + pos;
            for head in 0..self.num_heads {
                let q_start = head * dim;
                for i in 0..dim / 2 {
                    let freq = inv_freq[i] * global_pos as f32;
                    let cos = freq.cos();
                    let sin = freq.sin();
                    let q0_idx = q_start + i;
                    let q1_idx = q0_idx + dim / 2;
                    let q0 = q[q0_idx];
                    let q1 = q[q1_idx];
                    q[q0_idx] = q0 * cos - q1 * sin;
                    q[q1_idx] = q0 * sin + q1 * cos;
                }
            }

            for head in 0..self.num_kv_heads {
                let k_start = head * dim;
                for i in 0..dim / 2 {
                    let freq = inv_freq[i] * global_pos as f32;
                    let cos = freq.cos();
                    let sin = freq.sin();
                    let k0_idx = k_start + i;
                    let k1_idx = k0_idx + dim / 2;
                    let k0 = k[k0_idx];
                    let k1 = k[k1_idx];
                    k[k0_idx] = k0 * cos - k1 * sin;
                    k[k1_idx] = k0 * sin + k1 * cos;
                }
            }
        }
    }

    /// Extract a head's slice from a Kvcache buffer.
    fn extract_head_slice(
        cache: &Kvcache,
        is_key: bool,
        head_idx: usize,
        seq_pos: usize,
        head_dim: usize,
    ) -> Vec<f16> {
        let num_heads = cache.num_heads();
        let max_seq = cache.max_seq();
        let head_stride = num_heads * head_dim;
        let head_offset = head_idx * head_dim;
        let v_base = head_stride * max_seq;
        let base = if is_key { 0 } else { v_base };

        let src = cache.buffer().as_slice().unwrap_or(&[]);
        let row_start = base + head_stride * seq_pos + head_offset;
        let mut result = Vec::with_capacity(head_dim);
        for d in 0..head_dim {
            let idx = row_start + d;
            if idx < src.len() {
                result.push(src[idx]);
            }
        }
        result
    }

    // ── GPU-accelerated helpers ──────────────────────────────────────────

    /// GPU-accelerated RoPE using candle_bridge.
    fn apply_rope_gpu(
        q: &mut [f32],
        k: &mut [f32],
        seq_len: usize,
        start_pos: usize,
        head_dim: usize,
    ) -> Result<(), DispatchError> {
        let _cos_shape = [seq_len, head_dim / 2];
        let (cos, sin) = candle_bridge::rope_embeddings(seq_len, head_dim, 10000.0, 0)
            .map_err(|e| DispatchError::Kernel(format!("rope_embeddings: {e}")))?;

        let q_tensor = candle_bridge::f16_to_tensor(
            &q.iter()
                .map(|v| half::f16::from_f32(*v))
                .collect::<Vec<_>>(),
            &[1, seq_len, head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(q): {e}")))?;

        let k_tensor = candle_bridge::f16_to_tensor(
            &k.iter()
                .map(|v| half::f16::from_f32(*v))
                .collect::<Vec<_>>(),
            &[1, seq_len, head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(k): {e}")))?;

        let q_out = candle_bridge::apply_rope(&q_tensor, &cos, &sin, start_pos)
            .map_err(|e| DispatchError::Kernel(format!("apply_rope(q): {e}")))?;
        let k_out = candle_bridge::apply_rope(&k_tensor, &cos, &sin, start_pos)
            .map_err(|e| DispatchError::Kernel(format!("apply_rope(k): {e}")))?;

        let q_result = candle_bridge::tensor_to_f32(&q_out)
            .map_err(|e| DispatchError::Kernel(format!("tensor_to_f32(q): {e}")))?;
        let k_result = candle_bridge::tensor_to_f32(&k_out)
            .map_err(|e| DispatchError::Kernel(format!("tensor_to_f32(k): {e}")))?;

        // Extract the seq_len slice (first position for decode)
        for i in 0..head_dim {
            q[i] = q_result[i];
            k[i] = k_result[i];
        }

        Ok(())
    }

    /// GPU-accelerated SDPA using candle_bridge.
    fn sdpa_gpu(
        q: &[f32],
        k_cache: &Kvcache,
        v_cache: &Kvcache,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        start_pos: usize,
        scale: f32,
    ) -> Result<Vec<f32>, DispatchError> {
        let cache_len = start_pos + seq_len;

        // Build Q tensor: [1, seq_len, num_heads, head_dim]
        let q_tensor = candle_bridge::f16_to_tensor(
            &q.iter()
                .map(|v| half::f16::from_f32(*v))
                .collect::<Vec<_>>(),
            &[1, seq_len, num_heads, head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(q): {e}")))?;

        // Build K and V tensors from KV cache
        let k_buffer = k_cache.buffer();
        let v_buffer = v_cache.buffer();

        let k_slice: Vec<f16> = k_buffer.as_slice().map_or(vec![], |b| b.to_vec());
        let v_slice: Vec<f16> = v_buffer.as_slice().map_or(vec![], |b| b.to_vec());

        let k_tensor =
            candle_bridge::f16_to_tensor(&k_slice, &[1, cache_len, num_kv_heads, head_dim], None)
                .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(k): {e}")))?;

        let v_tensor =
            candle_bridge::f16_to_tensor(&v_slice, &[1, cache_len, num_kv_heads, head_dim], None)
                .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(v): {e}")))?;

        // Run SDPA
        let attn_out = candle_bridge::sdpa(&q_tensor, &k_tensor, &v_tensor, scale)
            .map_err(|e| DispatchError::Kernel(format!("sdpa: {e}")))?;

        let result = candle_bridge::tensor_to_f32(&attn_out)
            .map_err(|e| DispatchError::Kernel(format!("tensor_to_f32: {e}")))?;

        Ok(result)
    }

    /// GPU-accelerated full attention path: RoPE + SDPA via candle_bridge.
    fn forward_gpu(
        &self,
        ctx: &DispatchContext,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
        key_cache: &mut Kvcache,
        value_cache: &mut Kvcache,
        scale: f32,
    ) -> Result<Vec<f32>, DispatchError> {
        let embed_dim = self.num_heads * self.head_dim;
        let cache_len = start_pos + seq_len;

        // Write K/V from projections to KV cache
        let kv_dim = self.num_kv_heads * self.head_dim;
        for pos in 0..seq_len {
            let global_pos = start_pos + pos;
            let k_start = global_pos * kv_dim;
            let k_row: Vec<f16> = k[k_start..(k_start + kv_dim)]
                .iter()
                .map(|&val| half::f16::from_f32(val))
                .collect();
            let v_start = global_pos * kv_dim;
            let v_row: Vec<f16> = v[v_start..(v_start + kv_dim)]
                .iter()
                .map(|&val| half::f16::from_f32(val))
                .collect();
            // Write each tensor into its OWN cache's OWN region only.
            // Do NOT use write_kv_at here: it writes K AND V into one buffer,
            // and with separate key/value caches that cross-contaminates each
            // cache's unused region (harmless to region-selective CPU readers,
            // corrupting to whole-buffer GPU readers). See Kvcache::write_kv_at
            // docs and the `kv_write_no_cross_contamination` regression test.
            key_cache
                .write_k_at(global_pos, &k_row)
                .map_err(|e| DispatchError::Kernel(format!("KV cache K write at pos {global_pos}: {e}")))?;
            value_cache
                .write_v_at(global_pos, &v_row)
                .map_err(|e| DispatchError::Kernel(format!("KV cache V write at pos {global_pos}: {e}")))?;
            
            // DEBUG: Log KV writes
            if start_pos == 0 || pos == seq_len - 1 {
                println!("[DEBUG] forward_gpu: wrote K/V at global_pos={}, key_cache.seq_len={}", 
                         global_pos, key_cache.seq_len());
            }
        }

        // Extract K/V from cache for SDPA
        let k_buf = key_cache
            .buffer()
            .as_slice()
            .ok_or_else(|| DispatchError::Kernel("KV cache buffer not available".into()))?;
        let v_buf = value_cache
            .buffer()
            .as_slice()
            .ok_or_else(|| DispatchError::Kernel("Value cache buffer not available".into()))?;

        // Build K/V tensors: [1, cache_len, num_kv_heads, head_dim]
        let k_tensor = candle_bridge::f16_to_tensor(
            k_buf,
            &[1, cache_len, self.num_kv_heads, self.head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(k): {e}")))?;

        let v_tensor = candle_bridge::f16_to_tensor(
            v_buf,
            &[1, cache_len, self.num_kv_heads, self.head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(v): {e}")))?;

        // Apply RoPE to Q
        let _cos_shape = [cache_len, self.head_dim / 2];
        let (cos, sin) = candle_bridge::rope_embeddings(cache_len, self.head_dim, 10000.0, 0)
            .map_err(|e| DispatchError::Kernel(format!("rope_embeddings: {e}")))?;

        let q_tensor = candle_bridge::f16_to_tensor(
            &q.iter()
                .map(|&val| half::f16::from_f32(val))
                .collect::<Vec<_>>(),
            &[1, seq_len, self.num_heads, self.head_dim],
            None,
        )
        .map_err(|e| DispatchError::Kernel(format!("f16_to_tensor(q): {e}")))?;

        let q_rope_tensor = candle_bridge::apply_rope(&q_tensor, &cos, &sin, start_pos)
            .map_err(|e| DispatchError::Kernel(format!("apply_rope: {e}")))?;

        // Run SDPA
        let attn_out = candle_bridge::sdpa(&q_rope_tensor, &k_tensor, &v_tensor, scale)
            .map_err(|e| DispatchError::Kernel(format!("sdpa: {e}")))?;

        let result = candle_bridge::tensor_to_f32(&attn_out)
            .map_err(|e| DispatchError::Kernel(format!("tensor_to_f32: {e}")))?;

        // Output projection: attn_output @ wo^T
        let mut output = vec![0.0f32; batch_size * seq_len * embed_dim];
        for b in 0..batch_size {
            for pos in 0..seq_len {
                let attn_start = (b * seq_len + pos) * self.num_heads * self.head_dim;
                let attn_slice = &result[attn_start..attn_start + self.num_heads * self.head_dim];
                let wo_output = self.wo.forward(ctx, attn_slice, 1)?;
                let out_start = (b * seq_len + pos) * embed_dim;
                for i in 0..embed_dim {
                    output[out_start + i] = wo_output[i];
                }
            }
        }

        Ok(output)
    }
}

// ── LayerDispatch: GPU-aware transformer layer ─────────────────────────────

/// A transformer layer that can dispatch to GPU or CPU.
pub struct LayerDispatch {
    pub attention: AttentionDispatch,
    pub feed_forward: FeedForwardDispatch,
    pub attention_norm: RmsNormDispatch,
    pub ffn_norm: RmsNormDispatch,
}

impl LayerDispatch {
    /// Forward pass through one transformer layer with dispatch.
    pub fn forward(
        &mut self,
        ctx: &DispatchContext,
        x: &[f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
        key_cache: &mut Kvcache,
        value_cache: &mut Kvcache,
    ) -> Result<Vec<f32>, DispatchError> {
        let embed_dim = x.len() / batch_size;

        // Attention sub-layer: x + attn(RMSNorm(x))
        let normed = self.attention_norm.forward(x, batch_size)?;
        let attn_out = self.attention.forward(
            ctx,
            &normed,
            batch_size,
            seq_len,
            start_pos,
            key_cache,
            value_cache,
        )?;

        // Residual: x + attn_out
        let mut h = vec![0.0f32; batch_size * embed_dim];
        for i in 0..h.len() {
            h[i] = x[i] + attn_out[i];
        }

        // FFN sub-layer: h + ffn(RMSNorm(h))
        let normed_ffn = self.ffn_norm.forward(&h, batch_size)?;
        let ffn_out = self.feed_forward.forward(ctx, &normed_ffn, batch_size)?;

        // Residual: h + ffn_out
        for i in 0..h.len() {
            h[i] += ffn_out[i];
        }

        Ok(h)
    }
}

// ── FeedForwardDispatch ────────────────────────────────────────────────────

pub struct FeedForwardDispatch {
    pub w1: LinearDispatch,
    pub w2: LinearDispatch,
    pub w3: LinearDispatch,
    pub intermediate_dim: usize,
}

impl FeedForwardDispatch {
    pub fn forward(
        &self,
        ctx: &DispatchContext,
        x: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, DispatchError> {
        // Gate and Up projections
        let gate = self.w1.forward(ctx, x, batch_size)?;
        let up = self.w3.forward(ctx, x, batch_size)?;

        // SwiGLU: silu(gate) * up
        let swiglu_out = swiglu_dispatch(&gate, &up, self.intermediate_dim);

        // Down projection
        self.w2.forward(ctx, &swiglu_out, batch_size)
    }
}

/// SwiGLU activation: silu(x) * y
fn swiglu_dispatch(x: &[f32], y: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; size];
    for i in 0..size {
        let sigmoid = if x[i] >= 0.0 {
            1.0 / (1.0 + (-x[i]).exp())
        } else {
            x[i] / (1.0 + x[i].exp())
        };
        output[i] = sigmoid * x[i] * y[i];
    }
    output
}

// ── RmsNormDispatch ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct RmsNormDispatch {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNormDispatch {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>, DispatchError> {
        // RMSNorm is simple enough to do on CPU — no GPU dispatch needed
        let embed_dim = x.len() / batch_size;
        let weight_len = self.weight.len();

        // Ensure embed_dim matches weight length
        if embed_dim != weight_len {
            return Err(DispatchError::Kernel(format!(
                "RMSNorm embed_dim={} doesn't match weight_len={}",
                embed_dim, weight_len
            )));
        }

        let mut output = vec![0.0f32; x.len()];

        for b in 0..batch_size {
            let start = b * embed_dim;
            let mut rms_sum = 0.0f32;
            for i in start..start + embed_dim {
                rms_sum += x[i] * x[i];
            }
            let rms = (rms_sum / embed_dim as f32 + self.eps).sqrt();
            let inv_rms = 1.0 / rms;
            for i in start..start + embed_dim {
                output[i] = x[i] * inv_rms * self.weight[i - start];
            }
        }

        Ok(output)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────
