//! Stub implementations for CPU-only builds (no CUDA).
//!
//! This module provides minimal placeholder implementations that allow the
//! workspace to compile without CUDA dependencies, using only standard library
//! and existing workspace crates.

use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufHeader;
use std::path::Path;
use tokenizers::Tokenizer as TokenizerType;

// ── Core Types ───────────────────────────────────────────────────────────────

/// Stub LlamaModel (mirrors real LlamaModel from transformer/model.rs)
#[derive(Debug, Default)]
pub struct LlamaModel {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub max_seq_len: usize,
    // Additional fields needed for conformance testing
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub final_norm: Option<RmsNorm>,
    pub output: Option<Linear>,
}

/// Stub RmsNorm (minimal implementation for conformance testing)
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub eps: f32,
    pub weight: Vec<f32>,
}

impl RmsNorm {
    pub fn new(eps: f32, dim: usize) -> Self {
        Self {
            eps,
            weight: vec![1.0; dim], // Identity normalization for stub
        }
    }

    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        // Simple RMSNorm: x / sqrt(mean(x^2)) * weight
        let norm = (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let eps_inv = 1.0 / (self.eps + norm);
        x.iter().map(|&v| v * eps_inv * self.weight[0]).collect()
    }
}

/// Stub Linear layer (minimal implementation for conformance testing)
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, in_features: usize, out_features: usize) -> Self {
        Self { weight, bias, in_features, out_features }
    }

    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        // Simple matrix-vector multiply: y = x @ W.T + b
        let mut y = vec![0.0; self.out_features];
        for (row, out_val) in y.iter_mut().enumerate() {
            let start = row * self.in_features;
            for (col, &x_val) in x.iter().enumerate() {
                *out_val += x_val * self.weight[start + col];
            }
            if let Some(ref b) = self.bias {
                *out_val += b[row];
            }
        }
        y
    }
}

impl LlamaModel {
    /// Load a stub model from GGUF file (CPU-only path).
    ///
    /// Parses GGUF header to extract model metadata without loading weights.
    pub fn load_gguf(path: &Path) -> crate::error::Result<Self> {
        let header = parse_gguf(path).map_err(|e| {
            crate::error::RunnerError::ModelLoad(format!("GGUF parse error: {}", e))
        })?;

        // Extract model dimensions from GGUF header
        let hidden_size = header.get_kv_u32("llama.embedding_length").unwrap_or(896) as usize;
        let embed_dim = hidden_size; // For most models, embed_dim = hidden_size

        let num_layers = header.block_count().unwrap_or(24) as usize;

        // Get vocab size from metadata
        let vocab_size = header
            .get_kv_u32("tokenizer.ggml.vocab_size")
            .or_else(|| {
                // Fallback: check if tokenizer.ggml.tokens exists (array of token strings)
                header.kv_pairs.iter().find(|kv| kv.key == "tokenizer.ggml.tokens").and_then(|kv| kv.value.as_u32())
            })
            .unwrap_or(32000);

        let num_heads = header.get_kv_u32("llama.attention.head_count").unwrap_or(8) as usize;
        let num_kv_heads = header.get_kv_u32("llama.attention.head_count_kv")
            .or_else(|| header.get_kv_u32("attention.head_count_kv"))
            .unwrap_or(num_heads as u32) as usize;

        let head_dim = header.get_kv_u32("llama.attention.head_size").unwrap_or(64) as usize;

        // Get rope_base from KV pairs (GgufHeader doesn't have get_kv_f32 directly)
        let rope_base = header.kv_pairs.iter()
            .find(|kv| kv.key == "llama.rope_freq_base")
            .and_then(|kv| kv.value.as_f32())
            .unwrap_or(10000.0);

        let max_seq_len = header.get_kv_u32("llama.context_length").unwrap_or(2048) as usize;

        Ok(Self {
            hidden_size,
            embed_dim,
            num_layers,
            vocab_size: vocab_size as usize,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            max_seq_len,
            final_norm: None, // Stub - no real weights loaded
            output: None,     // Stub - no real weights loaded
        })
    }

    /// Stub forward_with_dispatch (mirrors real implementation from transformer/model.rs)
    pub fn forward_with_dispatch(
        &self,
        _hidden: &[f32],
        _start_pos: usize,
    ) -> crate::error::Result<Vec<f32>> {
        // Return dummy logits for testing
        Ok(vec![0.0; self.vocab_size])
    }

    /// Stub forward_layers (mirrors real implementation from transformer/model.rs)
    pub fn forward_layers(
        &self,
        hidden: &[f32],
        _seq_len: usize,
    ) -> crate::error::Result<Vec<f32>> {
        // Pass through for conformance testing
        Ok(hidden.to_vec())
    }
}

/// Stub SamplingConfig (mirrors real SamplingConfig from transformer/sampling.rs)
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub seed: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            temperature: 1.0,
            top_k: 40,
            top_p: 0.9,
        }
    }
}

/// Stub argmax (mirrors real argmax from transformer/sampling.rs)
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Stub sample (mirrors real sample from transformer/sampling.rs)
pub fn sample(logits: &[f32], config: &SamplingConfig, _rng: &mut rand::rngs::StdRng) -> u32 {
    let sum: f32 = logits.iter().map(|&x| x.exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&x| x.exp() / sum).collect();

    // Use explicit distribution for rand 0.10+ compatibility
    let rng_seed = config.seed;
    let r = (rng_seed as f32) / u64::MAX as f32;
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ── Tokenizer Types ──────────────────────────────────────────────────────────

/// Stub tokenizer config (mirrors real GgufTokenizerConfig from transformer/tokenizer.rs)
#[derive(Debug, Clone)]
pub struct GgufTokenizerConfig {
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl GgufTokenizerConfig {
    /// Build tokenizer config from GGUF header (stub - uses defaults)
    pub fn from_gguf_header(header: &GgufHeader) -> Self {
        // Extract vocab size from metadata if available, otherwise use default
        let vocab_size = header.get_kv_u32("tokenizer.ggml.vocab_size").unwrap_or(32000) as usize;

        // Extract BOS/EOS token IDs from metadata
        let bos_id = header.get_kv_u32("tokenizer.ggml.bos_token_id");
        let eos_id = header.get_kv_u32("tokenizer.ggml.eos_token_id");

        Self {
            vocab_size,
            bos_token_id: bos_id,
            eos_token_id: eos_id,
        }
    }

    /// Convert to tokenizers::Tokenizer (stub - creates simple BPE tokenizer)
    pub fn to_tokenizer(&self) -> TokenizerType {
        use tokenizers::Tokenizer;

        // Create a minimal tokenizer for testing
        // Since we can't load from GGUF directly, create a default one
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());

        // Add special tokens if we have IDs
        if let Some(_bos) = self.bos_token_id {
            // Could add BOS token here
        }
        if let Some(_eos) = self.eos_token_id {
            // Could add EOS token here
        }

        tokenizer
    }
}

/// Stub tokenizer loader (mirrors real load_tokenizer_from_gguf from transformer/tokenizer.rs)
pub fn load_tokenizer_from_gguf(
    path: &Path,
) -> Result<(GgufTokenizerConfig, TokenizerType), crate::error::RunnerError> {
    use pesti_gguf::parser::parse_gguf;

    // Parse GGUF header to get tokenizer config
    let header =
        parse_gguf(path).map_err(|e| crate::error::RunnerError::Tokenizer(e.to_string()))?;
    let config = GgufTokenizerConfig::from_gguf_header(&header);

    // Create a simple tokenizer with the config
    let tokenizer = config.to_tokenizer();

    Ok((config, tokenizer))
}
