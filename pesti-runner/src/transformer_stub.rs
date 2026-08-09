//! Stub implementations for CPU-only builds (no CUDA).
//! 
//! This module provides minimal placeholder implementations that allow the
//! workspace to compile without CUDA dependencies, using only standard library
//! and existing workspace crates.

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
}

impl LlamaModel {
    /// Stub forward_with_dispatch (mirrors real implementation from transformer/model.rs)
    pub fn forward_with_dispatch(&self, _hidden: &[f32], _start_pos: usize) -> crate::error::Result<Vec<f32>> {
        // Return dummy logits for testing
        Ok(vec![0.0; self.vocab_size])
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
pub fn sample(logits: &[f32], config: &SamplingConfig, rng: &mut rand::rngs::StdRng) -> u32 {
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
        let vocab_size = header.get_kv_u32("tokenizer.ggml.tokens")
            .map(|_| 0) // Just check existence; we'll use default below
            .unwrap_or(32000);

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
        let mut tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        
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
pub fn load_tokenizer_from_gguf(path: &Path) -> Result<(GgufTokenizerConfig, TokenizerType), crate::error::RunnerError> {
    use pesti_gguf::parser::parse_gguf;
    
    // Parse GGUF header to get tokenizer config
    let header = parse_gguf(path).map_err(|e| crate::error::RunnerError::Tokenizer(e.to_string()))?;
    let config = GgufTokenizerConfig::from_gguf_header(&header);
    
    // Create a simple tokenizer with the config
    let tokenizer = config.to_tokenizer();
    
    Ok((config, tokenizer))
}
