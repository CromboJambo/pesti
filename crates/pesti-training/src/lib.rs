//! pesti-training: QLoRA fine-tuning pipeline for crabjar decision engine training data.
//!
//! This crate provides a complete training infrastructure built on top of
//! pesti-runner's PEFT/TRL modules. It loads crabjar's JSONL decision training
//! data, formats it for instruction-following fine-tuning, and drives QLoRA
//! adapter training through pesti-runner's trainer.
//!
//! # Architecture
//!
//! ```text
//! crabjar JSONL dump → pesti-training (dataset loader) → pesti-runner TRL Trainer
//!                                                             ↓
//!                                                     LoRA adapters on model layers
//!                                                             ↓
//!                                                 Trained adapter weights (.safetensors)
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use pesti_training::{prepare_training_data, create_lora_adapter, QLoRAConfig};
//! use qwen2_bpe::Qwen2Tokenizer;
//! use std::path::Path;
//!
//! // Load tokenizer and dataset
//! let tokenizer = Qwen2Tokenizer::load_from_json("/path/to/tokenizer.json").unwrap();
//! let samples = prepare_training_data(tokenizer, Path::new("training.jsonl"), 512).unwrap();
//!
//! // Build LoRA adapter for a layer
//! let config = QLoRAConfig::default();
//! let adapter = create_lora_adapter(4096, 4096, &config).unwrap();
//! ```

pub mod dataset;
pub mod error;
pub mod lora;
pub mod trainer;

// Re-export key types for ergonomic use
pub use crate::dataset::{DatasetLoader, TrainingSample};
pub use crate::error::{Result, TrainingError};
pub use crate::lora::{QLoRAConfig, build_adapter, save_adapter, load_adapter};
pub use crate::trainer::{
    prepare_training_data, create_lora_adapter, validate_dataset, TrainArgs,
    DEFAULT_LR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_GRADIENT_ACCUMULATION,
};
