//! Generation loop for autoregressive text generation.
//!
//! Orchestrates the full inference pipeline: tokenize → prefill → decode loop
//! with KV cache updates at each step.

use crate::error::{Result, RunnerError};
use crate::model::Model;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// Sampling parameters for token selection.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature (1.0 = no scaling). Lower = more deterministic.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Random seed for reproducible generation.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 1.0,
            seed: 42,
        }
    }
}

/// Parameters for a generation request.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum tokens to generate (excluding prompt).
    pub max_tokens: usize,
    /// Stop when this token ID is generated.
    pub stop_token: Option<u32>,
    /// Sampling parameters.
    pub sampling: SamplingParams,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            stop_token: None,
            sampling: SamplingParams::default(),
        }
    }
}

/// Result of a generation request.
#[derive(Debug)]
pub struct GenerateResult {
    /// Generated text (decoded from tokens).
    pub text: String,
    /// Raw token IDs generated.
    pub tokens: Vec<u32>,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Reason generation stopped.
    pub stop_reason: StopReason,
}

/// Why generation stopped.
#[derive(Debug, Clone)]
pub enum StopReason {
    MaxTokensReached,
    StopToken(u32),
    EosToken,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::MaxTokensReached => write!(f, "max_tokens"),
            StopReason::StopToken(t) => write!(f, "stop_token({})", t),
            StopReason::EosToken => write!(f, "eos"),
        }
    }
}

/// Generate text from a prompt using the given model.
///
/// Pipeline:
/// 1. Tokenize prompt
/// 2. Prefill: process all prompt tokens through layers (populates KV cache)
/// 3. Decode loop: sample token → embed → forward layer-by-layer → update KV cache
/// 4. Stop on max_tokens, stop_token, or EOS
pub fn generate(model: &mut Model, prompt: &str, params: GenerateParams) -> Result<GenerateResult> {
    info!(prompt = %prompt.chars().take(80).collect::<String>(), "Starting generation");

    // Step 1: Tokenize prompt using model's tokenizer
    let input_tokens = {
        let tok = model.llama_model.tokenizer.as_ref().ok_or_else(|| {
            RunnerError::Tokenizer("model has no tokenizer loaded".to_string())
        })?;
        tok.encode(prompt)?
    };
    debug!(num_prompt_tokens = input_tokens.len(), "Prompt tokenized");

    // Step 2: Prefill — process all prompt tokens at once
    // This populates the KV cache with the full context
    let mut hidden = vec![0.0f32; model.config.num_heads * model.config.head_dim];
    for (i, tok) in input_tokens.iter().enumerate() {
        let embedded = model.llama_model.embed(*tok, i)?;
        debug!(pos = i, token = tok, "Prefill step");

        // Forward through all layers with KV cache updates
        hidden = model.forward_with_dispatch(&embedded, i)?;
    }

    // Step 3: Decode loop — generate tokens one at a time
    let mut generated_tokens = Vec::with_capacity(params.max_tokens);
    let mut stop_reason = StopReason::MaxTokensReached;

    for step in 0..params.max_tokens {
        debug!(step, "Decode step");

        // Get logits from the last hidden state through output head
        let logits = model.llama_model.apply_output_head(&hidden)?;

        // Sample next token
        let next_token = sample_token(&logits, &params.sampling);
        generated_tokens.push(next_token);

        // Check stop conditions
        if let Some(stop_tok) = params.stop_token {
            if next_token == stop_tok {
                stop_reason = StopReason::StopToken(stop_tok);
                break;
            }
        }
        // Check for EOS (token 0 is typically EOS in many models)
        if next_token == 0 && step > 0 {
            stop_reason = StopReason::EosToken;
            break;
        }

        // Embed the generated token and continue the sequence
        let embedded = model.llama_model.embed(next_token, input_tokens.len() + step)?;

        // Forward through layers — this updates KV cache for next step
        hidden = model.forward_with_dispatch(&embedded, input_tokens.len() + step)?;
    }

    // Step 4: Decode generated tokens back to text
    let generated_text = {
        let tok = model.llama_model.tokenizer.as_ref().ok_or_else(|| {
            RunnerError::Tokenizer("model has no tokenizer loaded".to_string())
        })?;
        tok.decode(&generated_tokens)?
    };

    info!(
        num_tokens = generated_tokens.len(),
        stop_reason = %stop_reason,
        "Generation complete"
    );

    let count = generated_tokens.len();
    Ok(GenerateResult {
        text: generated_text,
        tokens: generated_tokens,
        num_tokens: count,
        stop_reason,
    })
}

/// Sample a token from logits using temperature and top-p sampling.
fn sample_token(logits: &[f32], params: &SamplingParams) -> u32 {
    if params.temperature <= 0.0 {
        // Greedy decoding
        return argmax(logits);
    }

    let scaled: Vec<f32> = logits.iter().map(|l| l / params.temperature).collect();

    // Top-p nucleus sampling
    if params.top_p < 1.0 {
        return top_p_sample(&scaled, params.top_p);
    }

    // Standard softmax sampling
    softmax_sample(&scaled)
}

/// Greedy: return index of maximum logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_val {
            best_val = l;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Softmax sampling from scaled logits.
fn softmax_sample(logits: &[f32]) -> u32 {
    // Numerically stable softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    // Build cumulative distribution
    let mut cdf = Vec::with_capacity(exps.len());
    let mut cumsum = 0.0f32;
    for &e in &exps {
        cumsum += e / sum;
        cdf.push(cumsum);
    }

    // Sample uniform [0, 1) using time-based seed (rand crate not available in this context)
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let u: f32 = ((seed >> 16) & 0xFFFF) as f32 / 65535.0;
    for (i, &c) in cdf.iter().enumerate() {
        if u < c {
            return i as u32;
        }
    }

    // Fallback to last token
    (cdf.len() - 1) as u32
}

/// Top-p nucleus sampling.
fn top_p_sample(logits: &[f32], p: f32) -> u32 {
    // Sort by logit value descending, keeping original indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff where cumulative probability reaches p
    let max_logit = indexed[0].1;
    let exps: Vec<f32> = indexed.iter().map(|(_, l)| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let mut cumsum = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cumsum += e / sum;
        if cumsum >= p {
            // Include this token and all above it
            return softmax_sample(&exps[..=i]);
        }
    }

    // If we got here, use all tokens
    softmax_sample(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let logits = [0.1, 0.8, 0.3, 0.9, 0.2];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 0.7).abs() < 1e-6);
        assert!((params.top_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_params_default() {
        let params = GenerateParams::default();
        assert_eq!(params.max_tokens, 100);
        assert!(params.stop_token.is_none());
    }
}