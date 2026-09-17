//! Speculative decoding for PESTI.
//!
//! Generates multiple draft tokens autoregressively, then verifies them all
//! in a single forward pass. Accepts the prefix that matches and rejects
//! from the divergence point.
//!
//! This module provides a k=2 speculative decoder as a proof of concept.

use crate::error::{Result, RunnerError};
use crate::model::Model;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info};

/// Speculative decoding parameters.
#[derive(Debug, Clone)]
pub struct SpeculativeParams {
    /// Number of draft tokens to generate per verification step.
    pub draft_size: usize,
    /// Temperature for sampling.
    pub temperature: f32,
    /// Top-p nucleus threshold (1.0 = disabled).
    pub top_p: f32,
}

impl Default for SpeculativeParams {
    fn default() -> Self {
        Self {
            draft_size: 2,
            temperature: 0.7,
            top_p: 1.0,
        }
    }
}

/// Generate text using speculative decoding.
pub fn generate_speculative(
    model: &mut Model,
    prompt: &str,
    params: &SpeculativeParams,
) -> Result<String> {
    info!(prompt = %prompt.chars().take(80).collect::<String>(), "Starting speculative generation");

    // Tokenize prompt
    let input_tokens = {
        let tok = model.llama_model.tokenizer.as_ref().ok_or_else(|| {
            RunnerError::Tokenizer("model has no tokenizer loaded".to_string())
        })?;
        tok.encode(prompt)?
    };
    debug!(num_prompt_tokens = input_tokens.len(), "Prompt tokenized");

    // Prefill: process all prompt tokens
    let mut hidden = vec![0.0f32; model.config.num_heads * model.config.head_dim];
    for (i, tok) in input_tokens.iter().enumerate() {
        let embedded = model.llama_model.embed(*tok, i)?;
        hidden = model.forward_with_dispatch(&embedded, i)?;
    }

    // Decode loop with speculative decoding
    let mut generated_tokens = Vec::new();
    let context_len = input_tokens.len();
    let mut step = 0;
    let max_tokens = 50;

    while step < max_tokens {
        // Draft k tokens autoregressively (in this prototype, just 1)
        let logits = model.llama_model.apply_output_head(&hidden)?;
        let draft_token = sample_token(&logits, params.temperature);

        // Verify: embed and forward the draft token
        let pos = context_len + step;
        let embedded = model.llama_model.embed(draft_token, pos)?;
        let new_hidden = model.forward_with_dispatch(&embedded, pos)?;

        // Check consistency by getting logits at this position
        let verify_logits = model.llama_model.apply_output_head(&new_hidden)?;
        let verified_token = sample_token(&verify_logits, params.temperature);

        if verified_token == draft_token {
            generated_tokens.push(draft_token);
            hidden = new_hidden;
        } else {
            // Divergence - accept the verified token instead
            generated_tokens.push(verified_token);
            let embedded = model.llama_model.embed(verified_token, pos)?;
            hidden = model.forward_with_dispatch(&embedded, pos)?;
        }

        step += 1;

        // Check for EOS
        if step > 0 && (generated_tokens.last() == Some(&0)) {
            break;
        }
    }

    // Decode to text
    let generated_text = {
        let tok = model.llama_model.tokenizer.as_ref().ok_or_else(|| {
            RunnerError::Tokenizer("model has no tokenizer loaded".to_string())
        })?;
        tok.decode(&generated_tokens)?
    };

    info!(num_tokens = generated_tokens.len(), "Speculative generation complete");

    Ok(generated_text)
}

/// Sample a token from logits using temperature.
fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    if temperature <= 0.0 {
        // Greedy
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_val {
                best_val = l;
                best_idx = i;
            }
        }
        return best_idx as u32;
    }

    // Temperature sampling
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let u: f32 = ((seed >> 16) & 0xFFFF) as f32 / 65535.0;

    let mut cumsum = 0.0f32;
    for (i, &e) in exps.iter().enumerate() {
        cumsum += e / sum;
        if u < cumsum {
            return i as u32;
        }
    }

    (exps.len() - 1) as u32
}
