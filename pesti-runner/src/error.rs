//! Error types for pesti-runner.
//!
//! Provides `RunnerError` and `Result` type alias used throughout the crate.
//! When CUDA is disabled, CUDA-related errors are represented as string messages.

use thiserror::Error;

/// CUDA error (stub when feature="cuda" is disabled)
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA not available")]
    NotAvailable,
    #[error("CUDA not initialized: {0}")]
    NotInitialized(String),
    #[error("CUDA error: {0}")]
    Other(String),
}

/// Attention error (used by both CPU and GPU kernels)
#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),
    #[error("attention not available")]
    NotAvailable,
}

/// Core runner error type
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

    #[error("model loading error: {0}")]
    ModelLoad(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("device backend error: {0}")]
    Device(String),

    #[error("plug-in protocol error: {0}")]
    Protocol(#[from] pesti_plug_in::PlugInError),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("GGUF parse error: {0}")]
    Gguf(#[from] pesti_gguf::GgufError),

    #[error("asset loading error: {0}")]
    Asset(String),

    #[error("unspecified internal error: {0}")]
    Internal(String),

    #[error("dequantization error for tensor '{0}': {1}")]
    Dequant(String, String),

    #[error("GGUF header missing required field: {0}")]
    MissingHeaderField(String),

    #[error("dispatch error: {0}")]
    Dispatch(#[from] crate::kernel::DispatchError),
}

/// Result type alias for pesti-runner operations.
pub type Result<T> = std::result::Result<T, RunnerError>;
