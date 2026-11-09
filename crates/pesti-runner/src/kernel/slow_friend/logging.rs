//! Session logging for routing-alignment training data (G5).
//!
//! Captures per-step data: hidden state norms, divergence scores, activation
//! pattern hashes. Output is JSONL format for downstream analysis and
//! training-data generation.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

/// Per-step log entry capturing routing-relevant signals.
#[derive(Debug)]
pub struct StepLog {
    pub step: u64,
    pub token_id: u32,
    pub hidden_norm: f32,
    pub divergence_score: Option<f32>,
    pub activation_hash: u64,
}

/// Session logger that accumulates per-step data.
pub struct SessionLogger {
    entries: Vec<StepLog>,
    step: u64,
}

impl Default for SessionLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionLogger {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            step: 0,
        }
    }

    /// Log a step's hidden state and divergence (if available).
    pub fn log_step(&mut self, token_id: u32, hidden: &[f32], divergence_score: Option<f32>) {
        let norm = Self::hidden_norm(hidden);
        let hash = Self::activation_hash(hidden);

        self.entries.push(StepLog {
            step: self.step,
            token_id,
            hidden_norm: norm,
            divergence_score,
            activation_hash: hash,
        });

        self.step += 1;
    }

    /// Compute L2 norm of hidden state.
    fn hidden_norm(h: &[f32]) -> f32 {
        let sum: f64 = h.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        (sum.sqrt()) as f32
    }

    /// Compute a deterministic hash of the activation pattern.
    /// Uses quantized binning to create a coarse signature that's stable
    /// across small numerical perturbations but captures routing-relevant structure.
    fn activation_hash(h: &[f32]) -> u64 {
        // Quantize to 8 bins per dimension for coarse-grained signature
        let mut hasher = DefaultHasher::new();
        for &x in h {
            // Map f32 to a small integer bin (saturating)
            let v = x.clamp(-10.0, 10.0);
            let bin = ((v + 10.0) / 20.0 * 7.0).floor() as i8;
            hasher.write_i8(bin);
        }
        hasher.finish()
    }

    /// Export as JSONL string (one object per line).
    pub fn to_jsonl(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            let div_str = match entry.divergence_score {
                Some(d) => format!(",\"divergence\":{:.6}", d),
                None => String::new(),
            };
            out.push_str(&format!(
                "{{\"step\":{},\"token_id\":{},\"hidden_norm\":{:.6}{},\"activation_hash\":\"{:x}\"}}\n",
                entry.step, entry.token_id, entry.hidden_norm, div_str, entry.activation_hash
            ));
        }
        out
    }

    /// Export summary statistics.
    pub fn summary(&self) -> String {
        let total_steps = self.entries.len();
        if total_steps == 0 {
            return "no entries".to_string();
        }

        let avg_norm: f64 = self
            .entries
            .iter()
            .map(|e| e.hidden_norm as f64)
            .sum::<f64>()
            / total_steps as f64;

        // Count unique activation hashes (coarse routing diversity)
        let mut hashes = std::collections::HashSet::new();
        for entry in &self.entries {
            hashes.insert(entry.activation_hash);
        }

        format!(
            "steps={}, avg_hidden_norm={:.4}, unique_activation_hashes={}",
            total_steps,
            avg_norm,
            hashes.len()
        )
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_steps_and_exports_jsonl() {
        let mut logger = SessionLogger::new();
        logger.log_step(1, &[1.0, 2.0, 3.0], Some(0.05));
        logger.log_step(2, &[4.0, 5.0, 6.0], None);

        assert_eq!(logger.len(), 2);
        let jsonl = logger.to_jsonl();
        assert!(jsonl.contains("\"step\":0"));
        assert!(jsonl.contains("\"step\":1"));
        assert!(jsonl.contains("\"divergence\":0.050000"));
    }

    #[test]
    fn activation_hash_is_deterministic() {
        let h = vec![1.0, 2.0, 3.0];
        let hash1 = SessionLogger::activation_hash(&h);
        let hash2 = SessionLogger::activation_hash(&h);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn activation_hash_differs_for_different_inputs() {
        let h1 = vec![1.0, 2.0, 3.0];
        let h2 = vec![4.0, 5.0, 6.0];
        assert_ne!(
            SessionLogger::activation_hash(&h1),
            SessionLogger::activation_hash(&h2)
        );
    }

    #[test]
    fn summary_computes_stats() {
        let mut logger = SessionLogger::new();
        logger.log_step(1, &[1.0, 2.0], None);
        logger.log_step(2, &[3.0, 4.0], None);

        let summary = logger.summary();
        assert!(summary.contains("steps=2"));
        assert!(summary.contains("avg_hidden_norm="));
    }
}
