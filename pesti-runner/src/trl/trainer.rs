//! Main trainer implementation.

use crate::peft::Adapter;

#[cfg(feature = "cuda")]
use crate::transformer::LlamaModel;

#[cfg(not(feature = "cuda"))]
use crate::transformer_stub::LlamaModel;

use super::callbacks::{Callback, Callbacks};
use super::config::{OptimizerConfig, TrainingConfig};
use super::dataset::{Batch, Dataset};
use super::loss::{CrossEntropyLoss, LossFunction};
use super::state::{Metrics, State};

/// Main trainer for fine-tuning with adapters.
pub struct Trainer<A: Adapter> {
    /// Base model (frozen during training)
    model: LlamaModel,
    /// Adapter being trained
    adapter: A,
    /// Training configuration
    config: TrainingConfig,
    /// Optimizer configuration
    optimizer_config: OptimizerConfig,
    /// Loss function
    loss_fn: Box<dyn LossFunction>,
    /// Callbacks
    callbacks: Callbacks,
    /// Current state
    state: State,
}

impl<A: Adapter> Trainer<A> {
    /// Create a new trainer.
    pub fn new(
        model: LlamaModel,
        adapter: A,
        config: TrainingConfig,
        optimizer_config: OptimizerConfig,
    ) -> Self {
        Self {
            model,
            adapter,
            config,
            optimizer_config,
            loss_fn: Box::new(CrossEntropyLoss::default()),
            callbacks: Callbacks::new(),
            state: State::new(),
        }
    }

    /// Set the loss function.
    pub fn with_loss<F: LossFunction + 'static>(mut self, loss_fn: F) -> Self {
        self.loss_fn = Box::new(loss_fn);
        self
    }

    /// Add a callback.
    pub fn add_callback<C: Callback + 'static>(&mut self, callback: C) {
        self.callbacks.add(callback);
    }

    /// Get reference to state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Get mutable reference to state.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Train for one epoch.
    pub fn train_epoch<D: Dataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        self.callbacks.on_train_start(&self.state);

        let mut total_loss = 0.0f32;
        let num_batches = dataset.len().div_ceil(batch_size);

        for batch_idx in 0..num_batches {
            // Get batch
            let batch = self.get_batch(dataset, batch_idx * batch_size)?;

            // Forward pass through adapter
            let output = self.adapter.forward(&batch.input_ids, batch.batch_size())?;

            // Compute loss
            let loss = if let Some(labels) = &batch.labels {
                self.loss_fn.compute(&output, labels)
            } else {
                // Default: use input as labels (self-supervised) - convert token IDs to u32
                let token_ids: Vec<u32> = batch.input_ids.iter().map(|&x| x as u32).collect();
                self.loss_fn.compute(&output, &token_ids)
            };

            // Backward pass (gradient accumulation placeholder)
            self.backward(&output, &batch)?;

            // Update state
            total_loss += loss;
            self.state.record_loss(loss);
            self.state.increment_step(1);

            // Call step callback
            self.callbacks.on_step_end(&self.state, loss);

            // Checkpoint every N steps
            if self.state.epoch_step.is_multiple_of(self.config.save_steps) {
                println!("Checkpoint at step {}", self.state.global_step);
            }

            // Evaluation every N steps
            if self.state.epoch_step.is_multiple_of(self.config.eval_steps) {
                let eval_loss = self.evaluate(dataset)?;
                println!("Eval loss: {:.4}", eval_loss);
            }
        }

        // End of epoch
        let avg_loss = total_loss / num_batches as f32;
        self.state.increment_epoch();
        self.callbacks.on_epoch_end(&self.state);

        Ok(avg_loss)
    }

    /// Evaluate on a dataset.
    pub fn evaluate<D: Dataset>(
        &mut self,
        dataset: &D,
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        self.callbacks.on_eval_start(&self.state);

        let mut total_loss = 0.0f32;
        let num_batches = dataset.len().div_ceil(self.config.batch_size);

        for batch_idx in 0..num_batches {
            let batch = self.get_batch(dataset, batch_idx * self.config.batch_size)?;

            let output = self.adapter.forward(&batch.input_ids, batch.batch_size())?;

            let loss = if let Some(labels) = &batch.labels {
                self.loss_fn.compute(&output, labels)
            } else {
                // Self-supervised: convert token IDs to u32
                let token_ids: Vec<u32> = batch.input_ids.iter().map(|&x| x as u32).collect();
                self.loss_fn.compute(&output, &token_ids)
            };

            total_loss += loss;
        }

        let avg_loss = total_loss / num_batches as f32;
        let perplexity = (-avg_loss).exp();

        let metrics = Metrics::new(avg_loss).with_perplexity(perplexity);
        self.state.record_eval(metrics.clone());

        self.callbacks.on_eval_end(&self.state, &metrics);

        Ok(avg_loss)
    }

    /// Get a batch from dataset.
    fn get_batch<D: Dataset>(
        &self,
        dataset: &D,
        start_idx: usize,
    ) -> Result<Batch, Box<dyn std::error::Error + Send + Sync>> {
        if start_idx >= dataset.len() {
            return Err("Start index out of bounds".into());
        }

        // Get single sample for now (simplified batching)
        let batch = dataset.get(start_idx).ok_or("Failed to get sample")?;
        Ok(batch)
    }

    /// Backward pass (gradient computation placeholder).
    fn backward(
        &mut self,
        _output: &[f32],
        _batch: &Batch,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement gradient computation
        // For now, just zero gradients and mark as ready for update

        // In a full implementation, this would:
        // 1. Compute dL/d(adapter_weights) using chain rule
        // 2. Accumulate gradients (for gradient accumulation)
        // 3. Store gradients in adapter for optimizer step

        Ok(())
    }

    /// Save checkpoint.
    pub fn save_checkpoint(
        &self,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement actual checkpoint saving
        println!("Would save checkpoint to: {}", path);
        Ok(())
    }

    /// Load checkpoint.
    pub fn load_checkpoint<P: AsRef<std::path::Path>>(
        &mut self,
        _path: P,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // TODO: Implement actual checkpoint loading
        println!("Would load checkpoint from: {:?}", _path.as_ref());
        Ok(())
    }

    /// Run full training loop.
    pub fn train<D: Dataset>(
        &mut self,
        train_dataset: &D,
        eval_dataset: Option<&D>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for _epoch in 0..self.config.num_epochs {
            let avg_loss = self.train_epoch(train_dataset, self.config.batch_size)?;
            println!("Epoch {} - Avg loss: {:.4}", self.state.epoch, avg_loss);

            if let Some(eval) = eval_dataset {
                self.evaluate(eval)?;
            }
        }

        self.state.finish();
        Ok(())
    }
}

/// Builder for creating trainers with common configurations.
pub struct TrainerBuilder<A: Adapter> {
    model: Option<LlamaModel>,
    adapter: Option<A>,
    config: TrainingConfig,
    optimizer_config: OptimizerConfig,
}

impl<A: Adapter> TrainerBuilder<A> {
    /// Create new trainer builder.
    pub fn new() -> Self {
        Self {
            model: None,
            adapter: None,
            config: TrainingConfig::default(),
            optimizer_config: OptimizerConfig::default(),
        }
    }

    /// Set the base model.
    pub fn with_model(mut self, model: LlamaModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the adapter.
    pub fn with_adapter(mut self, adapter: A) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Set training config.
    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set optimizer config.
    pub fn with_optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.optimizer_config = optimizer;
        self
    }

    /// Build the trainer.
    pub fn build(self) -> Result<Trainer<A>, Box<dyn std::error::Error + Send + Sync>> {
        let model = self.model.ok_or("Model not set")?;
        let adapter = self.adapter.ok_or("Adapter not set")?;

        Ok(Trainer::new(
            model,
            adapter,
            self.config,
            self.optimizer_config,
        ))
    }
}

impl<A: Adapter> Default for TrainerBuilder<A> {
    fn default() -> Self {
        Self::new()
    }
}
