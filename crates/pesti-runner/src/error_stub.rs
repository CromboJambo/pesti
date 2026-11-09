//! Stub error types for CPU-only builds.

use thiserror::Error;

/// Dummy CUDA error
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA not available")]
    NotAvailable,
    #[error("CUDA not initialized: {0}")]
    NotInitialized(String),
    #[error("CUDA error: {0}")]
    Other(String),
}

/// Dummy attention error (kept since it's used by CPU kernels too)
#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),
    #[error("attention not available")]
    NotAvailable,
}

/// Stub RunnerError without CUDA dependency
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("GEMM error ({arch}, {m}x{n}x{k}): {detail}")]
    Gemm {
        arch: String,
        m: usize,
        n: usize,
        k: usize,
        #[source]
        detail: crate::kernel::GemmError,
    },

    #[error("attention error (heads={num_heads}, dim={head_dim}, seq={seq}): {detail}")]
    Attention {
        num_heads: usize,
        head_dim: usize,
        seq: usize,
        #[source]
        detail: AttentionError,
    },

    #[error("tensor computation error: {0}")]
    Tensor(String),

    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    #[error("model error: {0}")]
    Model(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("remote discovery error: {0}")]
    Remote(String),
}

pub type Result<T> = std::result::Result<T, RunnerError>;
