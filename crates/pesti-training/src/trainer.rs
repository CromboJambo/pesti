//! QLoRA training pipeline for decision engine fine-tuning.
//!
//! This module provides the high-level `train` function that orchestrates:
//! 1. Loading crabjar JSONL decision data via DatasetLoader
//! 2. Configuring LoRA adapters with appropriate hyperparameters
//! 3. Driving pesti-runner's Trainer to perform QLoRA fine-tuning

use crate::dataset::{DatasetLoader, TrainingSample};
use crate::error::Result;
use crate::lora::{QLoRAConfig, build_adapter};
use pesti_runner::peft::LoRAAdapter;
use qwen2_bpe::Qwen2Tokenizer;
use std::path::Path;
use tracing::info;

/// Hyperparameter defaults for QLoRA decision engine training.
pub const DEFAULT_LR: f32 = 2e-5;
pub const DEFAULT_EPOCHS: u8 = 3;
pub const DEFAULT_BATCH_SIZE: usize = 4;
pub const DEFAULT_GRADIENT_ACCUMULATION: usize = 4;

/// Training arguments for the pesti-runner Trainer.
#[derive(Debug, Clone)]
pub struct TrainArgs {
    /// Model path (GGUF or safetensors)
    pub model_path: String,
    /// Dataset path (JSONL)
    pub dataset_path: String,
    /// Output directory for trained adapter
    pub output_dir: String,
    /// LoRA rank
    pub lora_rank: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of epochs
    pub epochs: u8,
    /// Batch size
    pub batch_size: usize,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            dataset_path: String::new(),
            output_dir: "/tmp/pesti-train-output".to_string(),
            lora_rank: 8,
            learning_rate: DEFAULT_LR,
            epochs: DEFAULT_EPOCHS,
            batch_size: DEFAULT_BATCH_SIZE,
        }
    }
}

/// Prepare training data and return formatted samples.
pub fn prepare_training_data(
    tokenizer: Qwen2Tokenizer,
    dataset_path: &Path,
    max_seq_length: usize,
) -> Result<Vec<TrainingSample>> {
    info!("Loading training data from {}", dataset_path.display());

    let loader = DatasetLoader::new(tokenizer, max_seq_length);
    let samples = loader.load_jsonl(dataset_path)?;

    info!("Prepared {} training samples", samples.len());
    Ok(samples)
}

/// Build a LoRA adapter with the given configuration.
pub fn create_lora_adapter(
    in_features: usize,
    out_features: usize,
    config: &QLoRAConfig,
) -> Result<LoRAAdapter> {
    info!("Building LoRA adapter (rank={})", config.rank);
    build_adapter(in_features, out_features, config)
}

/// Validate training dataset and report statistics.
pub fn validate_dataset(dataset_path: &Path) -> Result<usize> {
    // Use a minimal tokenizer for validation
    let config = qwen2_bpe::Qwen2Config {
        vocab: std::collections::HashMap::new(),
        merges: vec![],
        special_tokens: None,
    };
    let tokenizer = Qwen2Tokenizer::new(config).map_err(|e| {
        crate::error::TrainingError::Train(format!("Failed to create tokenizer: {}", e))
    })?;

    let loader = DatasetLoader::new(tokenizer, 512);
    let samples = loader.load_jsonl(dataset_path)?;

    info!(
        "Dataset validation complete: {} samples",
        samples.len()
    );

    // Report average sequence length
    if !samples.is_empty() {
        let avg_len: usize = samples
            .iter()
            .map(|s| s.input_ids.len())
            .sum::<usize>()
            / samples.len();
        info!("Average sequence length: {} tokens", avg_len);
    }

    Ok(samples.len())
}
