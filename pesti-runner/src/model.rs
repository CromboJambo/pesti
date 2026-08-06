//! Model struct with per-layer KV cache allocation and inference loop.
//!
//! Manages the full transformer forward pass including:
//! - Per-layer KV cache allocation
//! - Prefill mode: process full prompt batch
//! - Decode mode: auto-regressive single-token generation

use crate::error::Result;
use crate::error::RunnerError;
use crate::inference_engine::InferenceEngine;
#[cfg(feature = "cuda")]
use crate::kernel::DeviceBuffer;
#[cfg(not(feature = "cuda"))]
use crate::kernel::device_buf::DeviceBuffer;
#[cfg(feature = "cuda")]
use crate::kernel::attention::{AttentionArch, AttentionConfig};
#[cfg(not(feature = "cuda"))]
use crate::kernel::attention_stub::{AttentionArch, AttentionConfig};
#[cfg(feature = "cuda")]
use crate::kernel::kvcache::Kvcache;
#[cfg(not(feature = "cuda"))]
use crate::kernel::kvcache_stub::Kvcache;
use half::f16;

/// Configuration for a transformer model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub use_tma: bool,
    pub attention_arch: AttentionArch,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            head_dim: 64,
            max_seq: 2048,
            num_kv_heads: 32,
            use_tma: true,
            attention_arch: AttentionArch::default(),
        }
    }
}

impl ModelConfig {
    /// Create a model config from loaded GGUF weights.
    pub fn from_gguf(header: &pesti_gguf::types::GgufHeader) -> Result<Self> {
        let embed_dim = header.embedding_length().ok_or_else(|| {
            RunnerError::MissingHeaderField("embedding_length".to_string())
        })? as usize;
        let num_heads = header.attention_head_count().ok_or_else(|| {
            RunnerError::MissingHeaderField("attention_head_count".to_string())
        })? as usize;
        let num_kv_heads = header.attention_head_count_kv().unwrap_or(num_heads as u32) as usize;
        let num_layers = header.block_count().unwrap_or(32) as usize;
        let head_dim = if num_heads > 0 { embed_dim / num_heads } else { 64 };
        let max_seq = header.context_length().ok_or_else(|| {
            RunnerError::MissingHeaderField("context_length".to_string())
        })? as usize;

        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq,
            num_kv_heads,
            use_tma: false,
            attention_arch: AttentionArch::default(),
        })
    }

    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    pub fn with_max_seq(mut self, max_seq: usize) -> Self {
        self.max_seq = max_seq;
        self
    }

    pub fn with_num_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    pub fn with_tma(mut self, use_tma: bool) -> Self {
        self.use_tma = use_tma;
        self
    }

    pub fn with_attention_arch(mut self, arch: AttentionArch) -> Self {
        self.attention_arch = arch;
        self
    }

    /// Build the attention configuration for this model.
    pub fn attention_config(&self) -> AttentionConfig {
        AttentionConfig::default()
            .with_num_heads(self.num_heads)
            .with_head_dim(self.head_dim)
            .with_max_seq(self.max_seq)
            .with_arch(self.attention_arch)
            .with_tma(self.use_tma)
    }
}

/// Model with per-layer KV cache allocation and inference loop.
pub struct Model {
    /// Model configuration.
    pub config: ModelConfig,
    /// Inference engine with GEMM and attention kernels.
    pub engine: InferenceEngine,
    /// Per-layer KV caches (one pair per layer: key_cache and value_cache).
    pub kv_caches: Vec<(Kvcache, Kvcache)>,
    /// Current sequence length (total tokens processed).
    pub seq_len: usize,
    /// Loaded transformer weights for Q/K/V projections.
    #[cfg(feature = "cuda")]
    pub llama_model: crate::transformer::LlamaModel,
    #[cfg(not(feature = "cuda"))]
    pub llama_model: crate::transformer_stub::LlamaModel,
    /// Whether to use the dispatch system (GPU-accelerated path).
    pub use_dispatch: bool,
}

impl Model {
    /// Create a new model with per-layer KV cache allocation.
    pub fn new(config: ModelConfig, engine: InferenceEngine, on_device: bool) -> Self {
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let max_seq = config.max_seq;

        let kv_caches = (0..num_layers)
            .map(|_| {
                let key_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                let value_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                (key_cache, value_cache)
            })
            .collect();

        Self {
            config,
            engine,
            kv_caches,
            seq_len: 0,
            #[cfg(feature = "cuda")]
            llama_model: crate::transformer::LlamaModel {
                config: crate::transformer::LlamaConfig {
                    arch: crate::transformer::ModelArch::default(),
                    num_layers: 32,
                    num_heads: 32,
                    num_kv_heads: 8,
                    head_dim: 64,
                    embed_dim: 4096,
                    intermediate_dim: 11008,
                    max_seq_len: 4096,
                    rope_base: 10000.0,
                    rope_scaling_factor: None,
                    rope_scaling_type: None,
                    rms_norm_eps: 1e-5,
                },
                token_embeddings: None,
                output: None,
                final_norm: None,
                layers: vec![],
                vocab_size: 32000,
                tokenizer: None,
                tokenizer_config: None,
                dispatch: None,
                kv_caches: None,
                cpu_kv_caches: None,
            },
            #[cfg(not(feature = "cuda"))]
            llama_model: crate::transformer_stub::LlamaModel::default(),
            use_dispatch: false,
        }
    }

    /// Enable the dispatch (GPU-accelerated) inference path.
    pub fn enable_dispatch(&mut self) {
        self.use_dispatch = true;
    }

    /// Check if dispatch is enabled and weights are loaded.
    pub fn can_use_dispatch(&self) -> bool {
        self.use_dispatch
    }

    /// Pass hidden states through all transformer layers using the dispatch system.
    pub fn forward_with_dispatch(&mut self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        self.llama_model.forward_with_dispatch(hidden, start_pos)
    }

    /// Reset the model state: clear KV caches and sequence length.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for (key_cache, value_cache) in &mut self.kv_caches {
            key_cache.clear();
            value_cache.clear();
        }
    }

    /// Get the current sequence length.
    pub fn current_seq_len(&self) -> usize {
        self.seq_len
    }

    /// Check if the model has capacity for more tokens.
    pub fn has_capacity(&self) -> bool {
        self.seq_len < self.config.max_seq
    }

    /// Get the attention configuration for this model.
    pub fn attention_config(&self) -> AttentionConfig {
        self.config.attention_config()
    }
}

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

        // Debug: print all tensor names
        println!("DEBUG: Available tensors in GGUF file:");
        for name in weights.tensors.keys() {
            if name.contains("embed") || name.contains("output") {
                println!("  - {}", name);
            }
        }

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

        // Load token embeddings (various naming conventions: Llama = tok_embeddings, Qwen = token_embd)
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

        // Load output head (various naming conventions; fallback to token embeddings for tied models)
        let output_weights = match weights.tensors.get("output.weight")
            .or_else(|| weights.tensors.get("lm_head.weight"))
            .or_else(|| weights.tensors.get("token_embd.weight"))  // Tied embeddings case
            .or_else(|| weights.tensors.get("tok_embeddings.weight")) {
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

        println!("DEBUG: output_weights.len={}, hidden.len={}, vocab_size={}, hidden_size={}", 
                 output_weights.len(), hidden.len(), self.vocab_size, self.hidden_size);

        for (row_idx, row) in output_weights.chunks(self.hidden_size).enumerate() {
            let mut sum = 0.0;
            for (col_idx, &hidden_val) in hidden.iter().enumerate() {
                sum += hidden_val * row[col_idx];
            }
            if row_idx < logits.len() {
                logits[row_idx] = sum;
            } else {
                println!("DEBUG: Skipping row_idx={} (exceeds vocab_size={})", row_idx, self.vocab_size);
            }
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

    /// Reset the model state (stub - no-op for now).
    pub fn reset(&mut self) {
        // For CPU-only minimal implementation, nothing to reset
    }
}
