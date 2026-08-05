// Real CpuModel for CPU-only builds with GGUF loading support
use crate::gguf_weight_loader::{load_gguf_weights, GgufWeights};
use std::path::Path;

/// CPU model implementation for testing K-family dequantization.
///
/// Loads token_embeddings and output.weight tensors for single-token inference.
pub struct CpuModel {
    /// GGUF weights loaded from file
    pub weights: GgufWeights,
    /// Token embeddings matrix (vocab_size × hidden_size)
    pub token_embeddings: Option<Vec<f32>>,
    /// Output head weights (vocab_size × hidden_size)
    pub output_weights: Option<Vec<f32>>,
    /// Hidden size from config
    pub hidden_size: usize,
    /// Vocabulary size from config
    pub vocab_size: usize,
    /// Whether dispatch is enabled
    pub use_dispatch: bool,
}

impl CpuModel {
    /// Load minimal GGUF model for conformance testing.
    ///
    /// Only loads necessary tensors (token_embeddings + output.weight), skipping
    /// transformer layers for now since we're testing dequantization only.
    pub fn load_gguf(path: &Path) -> Result<Self> {
        let weights = load_gguf_weights(path)?;

        // Extract config values
        let hidden_size = weights.header.embedding_length()
            .map(|v| v as usize)
            .ok_or_else(|| {
                crate::error::RunnerError::Gguf(pesti_gguf::GgufError::Io(
                    "Missing llama.embedding_length".to_string(),
                ))
            })?;

        // Get vocab size from tokens array (number of tokens)
        let vocab_size = weights.header.get_kv_array("tokenizer.ggml.tokens")
            .map(|arr| arr.len())
            .unwrap_or(32000); // Default fallback

        // Load token embeddings (various naming conventions)
        let token_embeddings = match weights.tensors.get("token_embd.weight")
            .or_else(|| weights.tensors.get("tok_embeddings.weight"))
            .or_else(|| weights.tensors.get("model.embed_tokens.weight")) {
            Some(bytes) => {
                // For now, assume F16 quantization and dequantize to f32
                if bytes.len() % 2 == 0 {
                    let f16_bytes: Vec<u16> = bytes.chunks(2)
                        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    Some(f16_bytes.iter()
                        .map(|&x| half::f16::from_bits(x).to_f32())
                        .collect())
                } else {
                    // Fallback: assume f32 storage
                    Some(bytes.chunks(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .map(|x| f32::from_bits(x))
                        .collect())
                }
            },
            None => None,
        };

        // Load output head (usually named "output.weight" or "lm_head.weight")
        let output_weights = match weights.tensors.get("output.weight")
            .or_else(|| weights.tensors.get("lm_head.weight")) {
            Some(bytes) => {
                // Same dequantization logic as token embeddings
                if bytes.len() % 2 == 0 {
                    let f16_bytes: Vec<u16> = bytes.chunks(2)
                        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    Some(f16_bytes.iter()
                        .map(|&x| half::f16::from_bits(x).to_f32())
                        .collect())
                } else {
                    // Fallback: assume f32 storage
                    Some(bytes.chunks(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .map(|x| f32::from_bits(x))
                        .collect())
                }
            },
            None => None,
        };

        Ok(Self {
            weights,
            token_embeddings,
            output_weights,
            hidden_size,
            vocab_size,
            use_dispatch: false,
        })
    }

    /// Embed a single token into hidden space.
    pub fn embed(&self, token: u32, _seq_len: usize) -> Result<Vec<f32>> {
        let embeddings = self.token_embeddings.as_ref().ok_or_else(|| {
            crate::error::RunnerError::Internal("Token embeddings not loaded".to_string())
        })?;

        // Simple lookup: each row is hidden_size elements
        Ok(embeddings[token as usize * self.hidden_size..(token + 1) as usize * self.hidden_size].to_vec())
    }

    /// Apply output head to get logits.
    pub fn apply_output_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let output_weights = self.output_weights.as_ref().ok_or_else(|| {
            crate::error::RunnerError::Internal("Output weights not loaded".to_string())
        })?;

        // Simple matrix-vector multiply: logits = hidden @ output_weights.T
        let mut logits = vec![0.0f32; self.vocab_size];

        for (row_idx, row) in output_weights.chunks(self.hidden_size).enumerate() {
            let mut sum = 0.0;
            for (col_idx, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * row[col_idx];
            }
            logits[row_idx] = sum;
        }

        Ok(logits)
    }

    /// Decode token to logits (embed → output head).
    pub fn decode(&self, token: u32) -> Result<Vec<f32>> {
        let hidden = self.embed(token, 0)?;
        self.apply_output_head(&hidden)
    }

    /// Enable GPU dispatch (no-op for CPU-only builds).
    pub fn enable_dispatch(&mut self) {
        self.use_dispatch = true;
    }

    /// Check if dispatch can be used.
    pub fn can_use_dispatch(&self) -> bool {
        self.use_dispatch
    }

    /// Forward pass through layers (stub - returns input for now).
    pub fn forward_with_dispatch(&self, hidden: &[f32], _start_pos: usize) -> Result<Vec<f32>> {
        // For now, just return the hidden state (no actual layer computation)
        Ok(hidden.to_vec())
    }

    /// Forward through transformer layers.
    pub fn forward_layers(&self, _hidden: &[f32], _start_pos: usize) -> Result<Vec<f32>> {
        // Stub - returns input for testing dequantization only
        Ok(_hidden.to_vec())
    }
}
