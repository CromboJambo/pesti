//! Single transformer layer with attention and FFN.

use crate::transformer_cpu::{Linear, RmsNorm, RopeConfig};

/// Attention sub-module within a transformer layer.
pub struct Attention {
    pub wq: Linear, // Query projection
    pub wk: Linear, // Key projection
    pub wv: Linear, // Value projection
    pub wo: Linear, // Output projection
    pub rope_config: RopeConfig,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    pub fn new(
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        rope_config: RopeConfig,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let embed_dim = wq.len();

        Self {
            wq: Linear::new(wq, None, 1, embed_dim),
            wk: Linear::new(wk, None, 1, embed_dim),
            wv: Linear::new(wv, None, 1, embed_dim),
            wo: Linear::new(wo, None, embed_dim, 1),
            rope_config,
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Forward pass with KV cache (simplified for single token).
    pub fn forward_with_cache_single(&self, x: &[f32], pos: usize) -> Result<Vec<f32>, String> {
        let embed_dim = x.len();

        // Project to Q, K, V
        let q = self.wq.forward(x, 1);
        let k = self.wk.forward(x, 1);
        let v = self.wv.forward(x, 1);

        // Apply RoPE (simplified: just first head for now)
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();

        // For simplicity, apply to all dimensions
        let cos = vec![0.0f32; 1024]; // Placeholder
        let sin = vec![0.0f32; 1024]; // Placeholder

        self.rope_config.apply_single_head(&mut q_rope, pos, &cos, &sin);
        self.rope_config.apply_single_head(&mut k_rope, pos, &cos, &sin);

        // Scaled dot-product attention (single token, no cache)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; embed_dim];

        // Simplified: just do Q @ K^T for single position
        let dot: f32 = q_rope.iter().zip(k_rope.iter()).map(|(a, b)| a * b).sum();
        let attn_score = dot * scale;

        for i in 0..embed_dim {
            attn_output[i] = attn_score * v[i];
        }

        // Output projection
        Ok(self.wo.forward(&attn_output, 1))
    }
}

/// Single transformer layer with attention and SwiGLU FFN.
pub struct TransformerLayer {
    pub attention_norm: RmsNorm,
    pub attention: Attention,
    pub ffn_norm: RmsNorm,
    pub feed_forward: crate::transformer_cpu::SwiGLUFFN,
}

impl TransformerLayer {
    pub fn new(
        attention_norm: RmsNorm,
        attention: Attention,
        ffn_norm: RmsNorm,
        feed_forward: crate::transformer_cpu::SwiGLUFFN,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        }
    }

    /// Forward pass with KV cache (single token mode).
    pub fn forward_with_cache_single(&self, x: &[f32], pos: usize) -> Result<Vec<f32>, String> {
        let embed_dim = x.len();

        // Attention sub-layer with cache
        let normed = self.attention_norm.forward(x, 1);
        let attn_out = self.attention.forward_with_cache_single(&normed, pos)?;

        // Residual connection
        let mut h = vec![0.0f32; embed_dim];
        for i in 0..embed_dim {
            h[i] = x[i] + attn_out[i];
        }

        // FFN sub-layer
        let normed_ffn = self.ffn_norm.forward(&h, 1);
        let ffn_out = self.feed_forward.forward(&normed_ffn, 1);

        // Residual connection
        for i in 0..embed_dim {
            h[i] += ffn_out[i];
        }

        Ok(h)
    }
}
