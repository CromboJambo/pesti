//! TRL-like training orchestrator for PESTI.
//!
//! This module provides high-level training orchestration similar to HuggingFace's TRL,
//! but designed for Rust's type system and the existing PESTI architecture.
//!
//! ## Core Concepts
//!
//! - **Trainer**: Composable training loop interface
//! - **Callback**: Hooks into training lifecycle events
//! - **State**: Training state management (epochs, steps, metrics)
//! - **Config**: High-level configuration for training runs
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use pesti_runner::trl::{Trainer, TrainingConfig, State};
//! use pesti_runner::peft::{Adapter, LoRAAdapter};
//!
//! // Create trainer with LoRA adapter
//! let config = TrainingConfig::default();
//! let state = State::new();
//! let mut trainer = Trainer::new(model, adapter, config);
//!
//! // Train for N epochs
//! for epoch in 0..config.num_epochs {
//!     trainer.train_epoch(&dataset, &mut state)?;
//!     trainer.evaluate(&eval_dataset, &state)?;
//! }
//! ```

pub mod callbacks;
pub mod config;
pub mod dataset;
pub mod loss;
pub mod state;
pub mod trainer;

// Re-exports
pub use callbacks::{Callback, Callbacks, CheckpointCallback, LoggingCallback, ProgressBarCallback};
pub use config::{OptimizerConfig, TrainingConfig};
pub use dataset::{Batch, Dataset, DatasetLoader, InMemoryDataset};
pub use loss::{CrossEntropyLoss, KLDivergenceLoss, LossFunction, LossType, PairwiseRankingLoss};
pub use state::{Metrics, State};
pub use trainer::{Trainer, TrainerBuilder};
