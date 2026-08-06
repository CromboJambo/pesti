//! Token sampling strategies for LLM inference.
//!
//! Converts logits to token IDs using temperature scaling, top-p (nucleus), and top-k filtering.
//!

#![allow(clippy::needless_borrow)]
use rand::RngExt;
use rand::rngs::StdRng;

/// Sampling configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingConfig {
    /// Temperature for logit scaling (1.0 = no scaling, < 1.0 = sharper, > 1.0 = softer).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (0.0 = disabled).
    pub top_p: f32,
    /// Top-k sampling threshold (0 = disabled).
    pub top_k: usize,
    /// Seed for reproducible sampling (None = random seed from OS).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            seed: None,
        }
    }
}

/// Sample a token ID from logits using the configured strategy.
///
/// `logits` — unnormalized log probabilities over vocabulary, length = vocab_size
/// `rng` — random number generator for stochastic sampling
/// Returns: sampled token ID
pub fn sample(logits: &[f32], config: &SamplingConfig, rng: &mut StdRng) -> u32 {
    let _vocab_size = logits.len();

    // Apply temperature scaling
    let scaled_logits: Vec<f32> = if config.temperature > 0.0 {
        logits
            .iter()
            .map(|&logit| logit / config.temperature)
            .collect()
    } else {
        logits.to_vec()
    };

    // Convert to probabilities via softmax
    let probs = softmax(&scaled_logits);

    // Apply top-k filtering
    let mut indexed_probs: Vec<(f32, usize)> =
        probs.iter().enumerate().map(|(i, &p)| (p, i)).collect();

    if config.top_k > 0 {
        indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = config.top_k.min(indexed_probs.len());
        indexed_probs.truncate(k);
    }

    // Apply top-p filtering
    if config.top_p > 0.0 && config.top_k == 0 {
        indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed_probs.len();
        for (i, &(p, _)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff_idx);
    }

    // Normalize probabilities after filtering
    let total_prob: f32 = indexed_probs.iter().map(|(p, _)| p).sum();
    if total_prob <= 0.0 {
        // Fallback to argmax
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    let normalized_probs: Vec<(f32, usize)> = indexed_probs
        .iter()
        .map(|(p, i)| (*p / total_prob, *i))
        .collect();

    // Weighted random sampling
    let r: f32 = rng.random();
    let mut cumsum = 0.0f32;
    for &(p, idx) in &normalized_probs {
        cumsum += p;
        if r <= cumsum {
            return idx as u32;
        }
    }

    // Fallback to last token
    normalized_probs
        .last()
        .map(|(_, idx)| *idx as u32)
        .unwrap_or(0)
}

/// Argmax sampling (greedy decoding).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Compute softmax of logit vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

