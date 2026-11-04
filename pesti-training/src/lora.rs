//! QLoRA adapter setup for decision engine fine-tuning.
//!
//! Wraps pesti-runner's PEFT infrastructure with default configurations
//! optimized for the Kev-9B decision engine use case.

use crate::error::{Result, TrainingError};
use pesti_runner::peft::{AdapterConfig, AdapterType, LoRAAdapter, LoRABuilder};

/// Default QLoRA configuration for decision engine training.
pub const DEFAULT_LORA_RANK: usize = 8;
pub const DEFAULT_LORA_ALPHA: f32 = 16.0;
pub const DEFAULT_LORA_DROPOUT: f32 = 0.05;

/// Target modules for LoRA adaptation (attention + FFN projections).
pub const TARGET_MODULES: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// Configuration for QLoRA fine-tuning.
#[derive(Debug, Clone)]
pub struct QLoRAConfig {
    /// LoRA rank (low-rank decomposition dimension)
    pub rank: usize,
    /// Scaling factor (lora_alpha)
    pub alpha: f32,
    /// Dropout rate for adapter layers
    pub dropout: f32,
    /// Target modules to adapt
    pub target_modules: Vec<String>,
}

impl Default for QLoRAConfig {
    fn default() -> Self {
        Self {
            rank: DEFAULT_LORA_RANK,
            alpha: DEFAULT_LORA_ALPHA,
            dropout: DEFAULT_LORA_DROPOUT,
            target_modules: TARGET_MODULES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl QLoRAConfig {
    /// Create adapter config from this QLoRA config.
    pub fn to_adapter_config(&self) -> AdapterConfig {
        AdapterConfig {
            rank: self.rank,
            scaling: self.alpha / self.rank as f32,
            adapter_type: AdapterType::LoRA,
        }
    }

    /// Create a smaller-rank config for faster training/testing.
    pub fn test_config() -> Self {
        Self {
            rank: 4,
            alpha: 8.0,
            dropout: 0.05,
            target_modules: TARGET_MODULES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Build a LoRA adapter for the given layer dimensions.
pub fn build_adapter(
    in_features: usize,
    out_features: usize,
    config: &QLoRAConfig,
) -> Result<LoRAAdapter> {
    let builder = LoRABuilder::new()
        .with_rank(config.rank)
        .with_scaling(config.alpha);

    // For training, use zero initialization (B matrix) for stable starts
    let adapter = builder.build_zeros(in_features, out_features).map_err(|e| {
        TrainingError::Lora(format!("Failed to build LoRA adapter: {}", e))
    })?;

    Ok(adapter)
}

/// Save adapter weights to a checkpoint file.
pub fn save_adapter<P: AsRef<std::path::Path>>(
    adapter: &LoRAAdapter,
    path: P,
) -> Result<()> {
    adapter
        .save_to_checkpoint(path)
        .map_err(|e| TrainingError::Lora(format!("Failed to save adapter: {}", e)))
}

/// Load adapter weights from a checkpoint file.
pub fn load_adapter<P: AsRef<std::path::Path>>(
    path: P,
    in_features: usize,
    out_features: usize,
) -> Result<LoRAAdapter> {
    let builder = LoRABuilder::new();
    let mut adapter = builder.build_zeros(in_features, out_features).map_err(|e| {
        TrainingError::Lora(format!("Failed to build adapter for loading: {}", e))
    })?;

    adapter
        .load_from_checkpoint(path, in_features, out_features)
        .map_err(|e| TrainingError::Lora(format!("Failed to load adapter: {}", e)))?;

    Ok(adapter)
}
