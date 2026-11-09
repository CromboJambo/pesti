//! Model information extracted from a loaded GGUF file.

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Number of parameters (in billions).
    pub n_params: u64,
    /// Embedding dimension.
    pub n_embd: i32,
    /// Number of transformer layers.
    pub n_layer: i32,
    /// Number of attention heads.
    pub n_head: i32,
    /// Number of attention heads for KV (grouped-query attention).
    pub n_head_kv: i32,
    /// Training context length.
    pub n_ctx_train: u32,
    /// Vocabulary size.
    pub n_vocab: i32,
    /// RoPE type string.
    pub rope_type: String,
    /// Whether the model is hybrid (multiple backends).
    pub is_hybrid: bool,
    /// Whether the model is recurrent (RWKV-style).
    pub is_recurrent: bool,
    /// Vocabulary type.
    pub vocab_type: String,
}

impl ModelInfo {
    /// Approximate model size in GB (very rough estimate).
    pub fn approx_size_gb(&self) -> f64 {
        // Rough estimate: params * 2 bytes (f16) / 1e9
        self.n_params as f64 * 2.0 / 1_000_000_000.0
    }

    /// Whether this model uses grouped-query attention.
    pub fn has_gqa(&self) -> bool {
        self.n_head != self.n_head_kv
    }

    /// GQA factor (how many heads share a KV head).
    pub fn gqa_factor(&self) -> f32 {
        if self.n_head_kv > 0 {
            self.n_head as f32 / self.n_head_kv as f32
        } else {
            1.0
        }
    }

    /// Human-readable model size label.
    pub fn size_label(&self) -> String {
        let gb = self.approx_size_gb();
        if gb < 1.0 {
            format!("{}B params ({:.1} GB)", self.n_params / 1_000_000_000, gb)
        } else if gb < 10.0 {
            format!("{}B params ({:.1} GB)", self.n_params / 1_000_000_000, gb)
        } else {
            format!("{}B params ({:.0} GB)", self.n_params / 1_000_000_000, gb)
        }
    }
}
