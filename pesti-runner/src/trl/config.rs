//! Training configuration for TRL-like orchestrator.

use serde::{Deserialize, Serialize};

/// High-level training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs to train
    pub num_epochs: usize,
    /// Learning rate for optimizer
    pub learning_rate: f32,
    /// Weight decay for regularization
    pub weight_decay: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Number of evaluation steps per epoch
    pub eval_steps: usize,
    /// Number of save steps per epoch
    pub save_steps: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether to use mixed precision (f16)
    pub use_fp16: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            learning_rate: 2e-4,
            weight_decay: 0.1,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            eval_steps: 100,
            save_steps: 500,
            max_seq_len: 2048,
            use_fp16: false,
        }
    }
}

impl TrainingConfig {
    /// Create a new training config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of epochs.
    pub fn with_num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set gradient accumulation steps.
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation_steps = steps;
        self
    }

    /// Enable mixed precision (FP16).
    pub fn with_fp16(mut self, enabled: bool) -> Self {
        self.use_fp16 = enabled;
        self
    }
}

/// Optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Optimizer type (adam, adamw, sgd)
    pub optimizer_type: OptimizerType,
    /// Learning rate
    pub learning_rate: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Beta1 for Adam optimizers
    pub beta1: f32,
    /// Beta2 for Adam optimizers
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::AdamW,
            learning_rate: 2e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
        }
    }
}

/// Supported optimizer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    AdamW,
    Adam,
    SGD,
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::AdamW
    }
}
