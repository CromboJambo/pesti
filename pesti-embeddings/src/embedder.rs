//! Embedding model runner using PESTI's transformer primitives.

use anyhow::{anyhow, Result};
use pesti_runner::gguf_weight_loader::load_gguf_weights;
use pesti_runner::transformer::tokenizer::{
    load_tokenizer_from_gguf, PestiTokenizer, TokenizerBackend,
};
use std::path::Path;

use crate::pooling::{pool, PoolingStrategy};
use crate::similarity::l2_normalize_owned;

/// Configuration for an embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Path to GGUF or tokenizer files
    pub model_path: String,
    /// Pooling strategy (default: Mean)
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalize output embeddings
    pub normalize: bool,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            pooling: PoolingStrategy::Mean,
            normalize: true,
            max_seq_len: 512,
        }
    }
}

/// Embedding model runner.
///
/// Loads a transformer model and produces dense vector embeddings from text.
pub struct Embedder {
    config: EmbeddingConfig,
    tokenizer: PestiTokenizer,
    embed_dim: usize,
}

impl Embedder {
    /// Load an embedding model from GGUF files.
    pub fn load(path: &str) -> Result<Self> {
        let config = EmbeddingConfig {
            model_path: path.to_string(),
            ..Default::default()
        };
        Self::load_with_config(&config)
    }

    /// Load an embedding model with custom configuration.
    pub fn load_with_config(config: &EmbeddingConfig) -> Result<Self> {
        // Load GGUF weights to extract architecture info
        let weights = load_gguf_weights(Path::new(&config.model_path))?;

        // Extract embedding dimension from tensor shape
        let (embed_dim, _) = weights.tensor_shape("token_embd.weight");
        if embed_dim == 0 {
            return Err(anyhow!("Failed to determine embedding dimension"));
        }

        // Load tokenizer using pesti-runner's infrastructure
        let (_tokenizer_config, tokenizer) =
            load_tokenizer_from_gguf(Path::new(&config.model_path), TokenizerBackend::MistralRs)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            config: config.clone(),
            tokenizer,
            embed_dim,
        })
    }

    /// Embed a single text string.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self
            .tokenizer
            .encode(text)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Truncate to max sequence length
        let truncated: Vec<u32> = token_ids
            .into_iter()
            .take(self.config.max_seq_len)
            .collect();

        if truncated.is_empty() {
            return Err(anyhow!("Empty token sequence"));
        }

        // Run forward pass through transformer layers
        let hidden_states = self.forward(&truncated)?;

        // Apply pooling strategy
        let pooled = pool(&hidden_states, self.config.pooling);

        // Optionally normalize
        if self.config.normalize {
            Ok(l2_normalize_owned(&pooled))
        } else {
            Ok(pooled)
        }
    }

    /// Embed multiple texts in a batch.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text)?);
        }
        Ok(embeddings)
    }

    /// Forward pass through transformer layers.
    ///
    /// Returns sequence of hidden states (one per token).
    fn forward(&self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        // TODO: Implement actual transformer forward pass using pesti-runner primitives
        // For now, return placeholder embeddings for testing

        let seq_len = token_ids.len();
        let mut hidden_states = Vec::with_capacity(seq_len);

        for &token_id in token_ids {
            // Placeholder: use token ID to generate deterministic but fake embedding
            let mut emb = vec![0.0f32; self.embed_dim];
            for i in 0..self.embed_dim {
                emb[i] = ((token_id as f32) + (i as f32)) * 0.01;
            }
            hidden_states.push(emb);
        }

        Ok(hidden_states)
    }

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get model configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}