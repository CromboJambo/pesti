//! Training state management.

use serde::{Deserialize, Serialize};

/// Current training state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Total steps completed
    pub global_step: usize,
    /// Steps in current epoch
    pub epoch_step: usize,
    /// Training loss history
    pub losses: Vec<f32>,
    /// Evaluation metrics history
    pub eval_metrics: Vec<Metrics>,
    /// Whether training is complete
    pub is_finished: bool,
}

impl State {
    /// Create a new training state.
    pub fn new() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            epoch_step: 0,
            losses: Vec::new(),
            eval_metrics: Vec::new(),
            is_finished: false,
        }
    }

    /// Reset state for a new training run.
    pub fn reset(&mut self) {
        self.epoch = 0;
        self.global_step = 0;
        self.epoch_step = 0;
        self.losses.clear();
        self.eval_metrics.clear();
        self.is_finished = false;
    }

    /// Increment epoch counter.
    pub fn increment_epoch(&mut self) {
        self.epoch += 1;
    }

    /// Increment step counter.
    pub fn increment_step(&mut self, steps: usize) {
        self.global_step += steps;
        self.epoch_step += steps;
    }

    /// Record a training loss.
    pub fn record_loss(&mut self, loss: f32) {
        self.losses.push(loss);
    }

    /// Record evaluation metrics.
    pub fn record_eval(&mut self, metrics: Metrics) {
        self.eval_metrics.push(metrics);
    }

    /// Mark training as complete.
    pub fn finish(&mut self) {
        self.is_finished = true;
    }

    /// Get average loss so far.
    pub fn avg_loss(&self) -> Option<f32> {
        if self.losses.is_empty() {
            None
        } else {
            Some(self.losses.iter().sum::<f32>() / self.losses.len() as f32)
        }
    }

    /// Get best evaluation metric so far.
    pub fn best_eval(&self) -> Option<&Metrics> {
        if self.eval_metrics.is_empty() {
            None
        } else {
            // Best = lowest loss (lower is better for loss metrics)
            self.eval_metrics
                .iter()
                .min_by(|a, b| a.loss.total_cmp(&b.loss))
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Loss value
    pub loss: f32,
    /// Accuracy (if applicable)
    pub accuracy: Option<f32>,
    /// Perplexity (for language modeling)
    pub perplexity: Option<f32>,
    /// Custom metrics map
    pub custom: std::collections::HashMap<String, f32>,
}

impl Metrics {
    /// Create new metrics with loss.
    pub fn new(loss: f32) -> Self {
        Self {
            loss,
            accuracy: None,
            perplexity: None,
            custom: std::collections::HashMap::new(),
        }
    }

    /// Set accuracy.
    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Set perplexity.
    pub fn with_perplexity(mut self, ppl: f32) -> Self {
        self.perplexity = Some(ppl);
        self
    }

    /// Add custom metric.
    pub fn with_custom(mut self, name: &str, value: f32) -> Self {
        self.custom.insert(name.to_string(), value);
        self
    }
}
