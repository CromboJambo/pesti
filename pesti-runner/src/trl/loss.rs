//! Loss functions for training.

use serde::{Deserialize, Serialize};

/// Supported loss function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    /// Cross-entropy loss (standard language modeling)
    CrossEntropy,
    /// KL divergence (for distillation or RLHF)
    KLDivergence,
    /// Pairwise ranking loss (for DPO)
    PairwiseRanking,
    /// Regression MSE loss
    MSELoss,
}

impl Default for LossType {
    fn default() -> Self {
        Self::CrossEntropy
    }
}

/// Loss function trait.
pub trait LossFunction: Send + Sync {
    /// Compute loss given predictions and targets.
    ///
    /// # Arguments
    /// * `logits` - Model output [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
    /// * `targets` - Target labels [batch_size, seq_len]
    ///
    /// # Returns
    /// Scalar loss value
    fn compute(&self, logits: &[f32], targets: &[u32]) -> f32;

    /// Get the type of loss.
    fn loss_type(&self) -> LossType;

    /// Compute gradient w.r.t. logits (for backprop).
    fn gradient(&self, logits: &mut [f32], targets: &[u32], scale: f32) {
        // Default implementation: cross-entropy gradient
        match self.loss_type() {
            LossType::CrossEntropy => {
                let batch_size = logits.len() / targets.len().max(1);
                let vocab_size = targets.len();

                for (i, &target) in targets.iter().enumerate() {
                    let logit_idx = i * vocab_size + target as usize;
                    if logit_idx < logits.len() {
                        logits[logit_idx] -= 1.0; // d(CE)/d(logit) = softmax - one_hot
                    }
                }

                for (i, logit) in logits.iter_mut().enumerate() {
                    *logit *= scale;
                }
            }
            LossType::KLDivergence => {
                // KL divergence gradient: d(KL)/d(logits) = softmax - q
                self.compute(logits, targets); // Placeholder
            }
            LossType::PairwiseRanking => {
                // Pairwise ranking gradient
                self.compute(logits, targets); // Placeholder
            }
            LossType::MSELoss => {
                // MSE gradient: d(MSE)/d(logits) = 2 * (pred - target) / N
                let scale = 2.0 / logits.len() as f32;
                for logit in logits.iter_mut() {
                    *logit *= scale;
                }
            }
        }
    }
}

/// Cross-entropy loss function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntropyLoss {
    /// Label smoothing factor
    pub label_smoothing: f32,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {
            label_smoothing: 0.0,
        }
    }
}

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, logits: &[f32], targets: &[u32]) -> f32 {
        let batch_size = targets.len();
        let vocab_size = logits.len() / batch_size;

        let mut total_loss = 0.0f32;

        for (b, &target) in targets.iter().enumerate() {
            let logit_idx = b * vocab_size + target as usize;
            if logit_idx < logits.len() {
                // Softmax + CE: -log(softmax(logits[target]))
                let max_logit = logits[b * vocab_size..(b + 1) * vocab_size]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                let exp_sum: f32 = logits[b * vocab_size..(b + 1) * vocab_size]
                    .iter()
                    .map(|&l| (l - max_logit).exp())
                    .sum();

                let log_softmax_target = (logits[logit_idx] - max_logit) - exp_sum.ln();
                total_loss -= log_softmax_target;
            }
        }

        // Apply label smoothing if enabled
        if self.label_smoothing > 0.0 {
            let smoothing = self.label_smoothing / vocab_size as f32;
            let no_smoothing = (1.0 - self.label_smoothing) / vocab_size as f32;

            for b in 0..batch_size {
                let logit_idx = b * vocab_size + targets[b] as usize;
                if logit_idx < logits.len() {
                    total_loss += smoothing * (vocab_size as f32 - 1.0);
                }
            }
        }

        total_loss / batch_size as f32
    }

    fn loss_type(&self) -> LossType {
        LossType::CrossEntropy
    }
}

/// KL Divergence loss for distillation or RLHF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KLDivergenceLoss;

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self
    }
}

impl LossFunction for KLDivergenceLoss {
    fn compute(&self, logits: &[f32], targets: &[u32]) -> f32 {
        // KL(P || Q) = sum(P * (log P - log Q))
        // For one-hot target distribution P, this reduces to -log(Q[target])
        let batch_size = targets.len();
        let vocab_size = logits.len() / batch_size;

        let mut total_loss = 0.0f32;

        for b in 0..batch_size {
            let start = b * vocab_size;
            let end = start + vocab_size;

            // Compute softmax of the predicted distribution Q
            let max_logit = logits[start..end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits[start..end]
                .iter()
                .map(|&l| (l - max_logit).exp())
                .sum();

            // Target is one-hot encoded at targets[b]
            let target_idx = targets[b] as usize;
            if target_idx < vocab_size {
                // Cross-entropy: -log(Q[target])
                let log_prob_target = logits[start + target_idx] - max_logit - exp_sum.ln();
                total_loss -= log_prob_target; // Negative because log(prob) is negative
            }
        }

        total_loss / batch_size as f32
    }

    fn loss_type(&self) -> LossType {
        LossType::KLDivergence
    }
}

/// Pairwise ranking loss for DPO (Direct Preference Optimization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseRankingLoss {
    /// Temperature scaling
    pub temperature: f32,
}

impl Default for PairwiseRankingLoss {
    fn default() -> Self {
        Self { temperature: 0.1 }
    }
}

impl LossFunction for PairwiseRankingLoss {
    fn compute(&self, logits: &[f32], targets: &[u32]) -> f32 {
        // DPO loss: -log(sigmoid(r_w - r_l))
        let batch_size = targets.len();
        let vocab_size = logits.len() / batch_size;

        let mut total_loss = 0.0f32;

        for b in 0..batch_size {
            let start = b * vocab_size;
            let end = start + vocab_size;

            // Compute log probabilities for chosen and rejected
            let max_logit = logits[start..end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits[start..end]
                .iter()
                .map(|&l| (l - max_logit).exp())
                .sum();

            // Target encodes preference (e.g., target[b] = 1 means chosen > rejected)
            if targets[b] == 1 {
                let log_prob_chosen = logits[start + 1] - max_logit - exp_sum.ln();
                let log_prob_rejected = logits[start] - max_logit - exp_sum.ln();

                let diff = (log_prob_chosen - log_prob_rejected) / self.temperature;
                total_loss += (-diff).exp().recip().ln(); // -log(sigmoid(diff))
            }
        }

        total_loss / batch_size as f32
    }

    fn loss_type(&self) -> LossType {
        LossType::PairwiseRanking
    }
}
