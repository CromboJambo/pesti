//! CPU-only transformer implementation for pure-Rust inference.
//!
//! This module provides a complete forward pass through a Llama-style transformer model
//! using only CPU computation (no CUDA dependencies). It wires together the primitive
//! operations: RMSNorm, RoPE, attention, and SwiGLU feed-forward networks.

pub mod layer;
pub mod primitives;
pub mod rope;
pub mod swiglu;

// Re-export public API
pub use layer::{Attention, TransformerLayer};
pub use primitives::{Linear, RmsNorm};
pub use rope::RopeConfig;
pub use swiglu::SwiGLUFFN;

// Re-export argmax from transformer_stub to avoid duplicate (already exported in lib.rs)
// pub use crate::transformer_stub::argmax; // Disabled - causes duplicate export

/// CPU-only transformer model with full forward pass.
///
/// Unlike the stub version, this implementation loads real transformer layer weights
/// and performs complete autoregressive generation through all layers.
pub struct CpuTransformerModel {
    /// Model configuration extracted from GGUF header
    pub config: TransformerConfig,
    /// Token embeddings (vocab_size × embed_dim)
    pub token_embeddings: Linear,
    /// Transformer layers (one per depth)
    pub layers: Vec<TransformerLayer>,
    /// Final RMSNorm after all transformer layers
    pub final_norm: RmsNorm,
    /// Output projection (embed_dim × vocab_size)
    pub output_proj: Linear,
}

/// Configuration for a CPU transformer model.
#[derive(Debug)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub rms_norm_eps: f32,
}

impl CpuTransformerModel {
    /// Create a new CPU transformer model from loaded weights.
    ///
    /// This is a minimal implementation that loads only the necessary tensors for
    /// single-token generation (embeddings + one layer as proof of concept).
    pub fn new(
        token_embeddings: Vec<f32>,
        layers: Vec<TransformerLayer>,
        final_norm: RmsNorm,
        output_proj: Linear,
        config: TransformerConfig,
    ) -> Self {
        Self {
            token_embeddings: Linear::new(token_embeddings, None, 1, config.embed_dim),
            layers,
            final_norm,
            output_proj,
            config,
        }
    }

    /// Forward pass through all transformer layers.
    ///
    /// - Input: `[embed_dim]` (single token's hidden state)
    /// - Position: `pos` (for RoPE)
    /// - Returns: `[embed_dim]` (output logits before softmax)
    pub fn forward(&self, hidden: &[f32], pos: usize) -> Result<Vec<f32>, String> {
        let mut x = hidden.to_vec();

        // Pass through each transformer layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache_single(&x, pos)?;
            eprintln!("Layer {} complete", layer_idx);
        }

        // Apply final RMSNorm
        x = self.final_norm.forward(&x, 1);

        // Project to vocab space
        let logits = self.output_proj.forward(&x, 1);

        Ok(logits)
    }

    /// Generate next token from input tokens (autoregressive).
    ///
    /// - `token_ids`: input token IDs (single token for decode mode)
    /// - `pos`: current position in sequence
    /// - Returns: `(next_token, logits)` where logits is `[vocab_size]`
    pub fn generate(&self, token_ids: &[u32], pos: usize) -> Result<(u32, Vec<f32>), String> {
        // Embed token
        let hidden = self.embed_token(token_ids[0])?;

        // Forward through transformer
        let logits = self.forward(&hidden, pos)?;

        // Sample next token (simple argmax for now)
        let next_token = argmax(&logits);

        Ok((next_token, logits))
    }

    /// Embed a single token ID into hidden space.
    fn embed_token(&self, token_id: u32) -> Result<Vec<f32>, String> {
        if (token_id as usize) >= self.config.vocab_size {
            return Err(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, self.config.vocab_size
            ));
        }

        // Simple lookup: each row is embed_dim elements
        let start = token_id as usize * self.config.embed_dim;
        let end = start + self.config.embed_dim;

        Ok(self.token_embeddings.weight[start..end].to_vec())
    }

    /// Get model configuration.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

// Re-export argmax from transformer module to avoid duplicate
#[cfg(feature = "cuda")]
pub use crate::transformer::argmax;
#[cfg(not(feature = "cuda"))]
pub use crate::transformer_stub::argmax;

/// Softmax sampling with temperature.
pub fn sample_with_temp(logits: &[f32], temp: f32, _rng: &mut rand::rngs::StdRng) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let _probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Categorical sampling (placeholder - just return first token for now)
    0u32
}
