//! Callback system for training lifecycle hooks.

use super::state::{Metrics, State};

/// Callback trait for hooking into training events.
pub trait Callback: Send + Sync {
    /// Called at the start of training.
    fn on_train_start(&mut self, state: &State) {}

    /// Called at the end of each epoch.
    fn on_epoch_end(&mut self, _state: &State) {}

    /// Called after each training step.
    fn on_step_end(&mut self, _state: &State, _loss: f32) {}

    /// Called before evaluation.
    fn on_eval_start(&mut self, _state: &State) {}

    /// Called after evaluation.
    fn on_eval_end(&mut self, _state: &State, _metrics: &Metrics) {}

    /// Called when a checkpoint is saved.
    fn on_checkpoint_saved(&mut self, _path: &str) {}
}

/// Collection of callbacks.
pub struct Callbacks {
    callbacks: Vec<Box<dyn Callback>>,
}

impl Callbacks {
    /// Create new callbacks collection.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback.
    pub fn add<C: Callback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Call on_train_start for all callbacks.
    pub fn on_train_start(&mut self, state: &State) {
        for cb in &mut self.callbacks {
            cb.on_train_start(state);
        }
    }

    /// Call on_epoch_end for all callbacks.
    pub fn on_epoch_end(&mut self, state: &State) {
        for cb in &mut self.callbacks {
            cb.on_epoch_end(state);
        }
    }

    /// Call on_step_end for all callbacks.
    pub fn on_step_end(&mut self, state: &State, loss: f32) {
        for cb in &mut self.callbacks {
            cb.on_step_end(state, loss);
        }
    }

    /// Call on_eval_start for all callbacks.
    pub fn on_eval_start(&mut self, state: &State) {
        for cb in &mut self.callbacks {
            cb.on_eval_start(state);
        }
    }

    /// Call on_eval_end for all callbacks.
    pub fn on_eval_end(&mut self, state: &State, metrics: &Metrics) {
        for cb in &mut self.callbacks {
            cb.on_eval_end(state, metrics);
        }
    }

    /// Call on_checkpoint_saved for all callbacks.
    pub fn on_checkpoint_saved(&mut self, path: &str) {
        for cb in &mut self.callbacks {
            cb.on_checkpoint_saved(path);
        }
    }
}

impl Default for Callbacks {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress bar callback (terminal output).
pub struct ProgressBarCallback {
    current_step: usize,
    total_steps: usize,
    last_print: usize,
    print_interval: usize,
}

impl ProgressBarCallback {
    /// Create new progress bar callback.
    pub fn new(total_steps: usize) -> Self {
        Self {
            current_step: 0,
            total_steps,
            last_print: 0,
            print_interval: 50,
        }
    }

    /// Update step counter.
    pub fn update(&mut self, step: usize) {
        self.current_step = step;
    }
}

impl Callback for ProgressBarCallback {
    fn on_epoch_end(&mut self, state: &State) {
        if state.epoch_step - self.last_print >= self.print_interval {
            let progress = (state.global_step as f32 / self.total_steps as f32) * 100.0;
            println!("\rEpoch {}: {:.1}% complete", state.epoch, progress);
            self.last_print = state.epoch_step;
        }
    }

    fn on_eval_end(&mut self, _state: &State, metrics: &Metrics) {
        if let Some(ppl) = metrics.perplexity {
            println!("  Eval loss: {:.4}, perplexity: {:.2}", metrics.loss, ppl);
        } else {
            println!("  Eval loss: {:.4}", metrics.loss);
        }
    }
}

/// Checkpoint callback (saves model state).
pub struct CheckpointCallback {
    save_path: String,
    save_interval: usize,
}

impl CheckpointCallback {
    /// Create new checkpoint callback.
    pub fn new(save_path: &str, save_interval: usize) -> Self {
        Self {
            save_path: save_path.to_string(),
            save_interval,
        }
    }
}

impl Callback for CheckpointCallback {
    fn on_epoch_end(&mut self, state: &State) {
        if state.global_step % self.save_interval == 0 {
            let checkpoint_path = format!("{}/epoch_{}.ckpt", self.save_path, state.epoch);
            println!("Saving checkpoint: {}", checkpoint_path);
            // TODO: Implement actual checkpoint saving
            self.on_checkpoint_saved(&checkpoint_path);
        }
    }
}

/// Logging callback (saves to file).
pub struct LoggingCallback {
    log_path: String,
}

impl LoggingCallback {
    /// Create new logging callback.
    pub fn new(log_path: &str) -> Self {
        Self {
            log_path: log_path.to_string(),
        }
    }
}

impl Callback for LoggingCallback {
    fn on_epoch_end(&mut self, state: &State) {
        if let Some(avg_loss) = state.avg_loss() {
            println!("Epoch {} - Avg loss: {:.4}", state.epoch, avg_loss);
            // TODO: Write to log file
        }
    }
}
