//! Parameter-Efficient Fine-Tuning (PEFT) adapters for PESTI.
//!
//! This module provides trait-based adapter interfaces that can be composed with
//! existing linear layers, similar to HuggingFace PEFT but designed for the
//! candle/Rust ecosystem.
//!
//! ## Design Philosophy
//!
//! - **Trait-like interfaces**: Adapters implement traits that define their behavior
//! - **Composable**: Multiple adapters can be stacked or merged
//! - **Low overhead**: Pure matrix operations, no dynamic dispatch in hot path
//! - **Merge/unmerge**: Support for merging adapter weights into base model
//!
//! ## Adapter Types
//!
//! - `LoRA`: Low-Rank Adaptation - the most common PEFT method
//! - `QLoRA`: Quantized LoRA - adapter on top of quantized weights
//! - `AdapterBlock`: Feed-forward adapter block (less common)
//!
//! ## Usage Pattern
//!
//! ```rust,ignore
//! // Create a base linear layer
//! let linear = Linear::new(weights, bias, in_features, out_features);
//!
//! // Create a LoRA adapter
//! let lora_a = LoRAMatrix::random(in_features, r);
//! let lora_b = LoRAMatrix::zeros(out_features, r);
//! let lora_adapter = LoRAAdapter::new(lora_a, lora_b, scaling);
//!
//! // Compose: forward goes through adapter first, then base linear
//! let output = linear_with_adapter.forward(x, &lora_adapter);
//!
//! // Merge adapter into base weights (optional)
//! let merged_linear = lora_adapter.merge(&linear);
//! ```
//!
//! ## Performance Considerations
//!
//! - Adapter forward adds O(r * batch_size) overhead where r is the rank
//! - Merged adapters have zero runtime overhead
//! - Use `merge_weights()` before deployment for best performance

pub mod adapter;
pub mod lora;
pub mod matrices;

// Re-export all public types from submodules
pub use adapter::{Adapter, AdapterBuilder, AdapterType};
pub use lora::{LoRAAdapter, LoRABuilder};
pub use matrices::{MatA, MatB};

/// Configuration for an adapter.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Rank of the adapter (number of latent dimensions).
    pub rank: usize,
    /// Scaling factor applied to adapter output.
    pub scaling: f32,
    /// Type of adapter to use.
    pub adapter_type: AdapterType,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            rank: 8, // Common default rank for LoRA
            scaling: 16.0, // Typical scaling = rank * alpha (alpha=2)
            adapter_type: AdapterType::LoRA,
        }
    }
}

impl AdapterConfig {
    /// Create a new LoRA configuration with specified rank and scaling.
    pub fn lora(rank: usize, scaling: f32) -> Self {
        Self {
            rank,
            scaling,
            adapter_type: AdapterType::LoRA,
        }
    }

    /// Calculate the effective scaling factor.
    ///
    /// In PEFT, scaling is typically `rank * alpha`. The scaling parameter
    /// allows tuning the adapter's impact without changing the rank.
    pub fn effective_scaling(&self) -> f32 {
        self.scaling / self.rank as f32
    }
}

/// Error type for adapter operations.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Adapter not initialized")]
    NotInitialized,

    #[error("Rank must be > 0")]
    InvalidRank,

    #[error("Scaling factor must be positive")]
    InvalidScaling,

    #[error("Base linear layer dimension mismatch with adapter")]
    LinearMismatch,
}

/// Type alias for adapter results.
pub type Result<T> = std::result::Result<T, AdapterError>;

/// Type alias for LoRA matrix (re-exported for convenience)
pub type LoRAMatrix = f32;
