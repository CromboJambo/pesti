//! Embedding model runner using PESTI's transformer primitives.

use anyhow::{anyhow, Result};
use pesti_runner::transformer::model::LlamaModel;
use pesti_runner::transformer::tokenizer::{
    load_tokenizer_from_gguf, PestiTokenizer, TokenizerBackend,
};
use pesti_runner::load_gguf_weights;
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
    model: LlamaModel,
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
        // Load GGUF weights
        let weights = load_gguf_weights(Path::new(&config.model_path)).map_err(|e| {
            anyhow!(
                "Failed to load GGUF weights from '{}': {}",
                config.model_path,
                e
            )
        })?;

        // Load tokenizer using pesti-runner's infrastructure
        let (_tokenizer_config, tokenizer) =
            load_tokenizer_from_gguf(Path::new(&config.model_path), TokenizerBackend::MistralRs)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Build model from weights
        let model = LlamaModel::from_gguf_weights(weights)
            .map_err(|e| anyhow!("Failed to build model from GGUF weights: {}", e))?;

        let embed_dim = model.config.embed_dim;

        Ok(Self {
            config: config.clone(),
            tokenizer,
            model,
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

        // Embed each token using the model's learned token embeddings
        let seq_len = truncated.len();
        let mut hidden_states = Vec::with_capacity(seq_len * self.embed_dim);
        for (pos, &token_id) in truncated.iter().enumerate() {
            let embedding = self
                .model
                .embed(token_id, pos)
                .map_err(|e| anyhow!("Failed to embed token {}: {}", token_id, e))?;
            hidden_states.extend(embedding);
        }

        // Pass through all transformer layers.
        // Note: We call attention/ffn directly rather than layer.forward() because
        // layer.forward() uses causal attention (past tokens only), but embeddings
        // benefit from bidirectional context. By processing each token position
        // through the FFN individually, we get better embedding quality.
        let mut output = hidden_states;
        for layer in self.model.layers.iter() {
            // Attention sub-layer: x + attn(RMSNorm(x))
            let normed = layer.attention_norm.forward(&output, seq_len);
            let attn_out = layer.attention.forward(&normed, 1, seq_len, 0);

            // Residual: x + attn_out
            let mut h = vec![0.0f32; output.len()];
            for j in 0..output.len().min(attn_out.len()) {
                h[j] = output[j] + attn_out[j];
            }

            // FFN sub-layer: h + ffn(RMSNorm(h))
            let normed_ffn = layer.ffn_norm.forward(&h, seq_len);
            let mut ffn_out = vec![0.0f32; h.len()];
            for pos in 0..seq_len {
                let row_start = pos * self.embed_dim;
                let row_end = row_start + self.embed_dim;
                let normed_row = &normed_ffn[row_start..row_end];
                let ffn_row = layer.feed_forward.forward(normed_row, 1);
                for j in 0..self.embed_dim {
                    ffn_out[row_start + j] = ffn_row[j];
                }
            }

            // Residual: h + ffn_out
            for j in 0..h.len().min(ffn_out.len()) {
                h[j] += ffn_out[j];
            }

            output = h;
        }

        // Reshape to [seq_len, embed_dim] for pooling
        let mut token_embeddings = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let row_start = i * self.embed_dim;
            let row_end = row_start + self.embed_dim;
            if row_end > output.len() {
                return Err(anyhow!(
                    "Output too short: expected {} elements, got {}",
                    seq_len * self.embed_dim,
                    output.len()
                ));
            }
            let mut row = Vec::with_capacity(self.embed_dim);
            for j in row_start..row_end {
                row.push(output[j]);
            }
            token_embeddings.push(row);
        }

        // Apply pooling strategy
        let pooled = pool(&token_embeddings, self.config.pooling);

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

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get model configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}