//! Single transformer layer: attention + feed-forward with RMSNorm.
//!
//! Llama-style layer:
//!   x = x + attn(RMSNorm(x), Wq, Wk, Wv, Wo)
//!   x = x + ffn(RMSNorm(x), W1, W2, W3)
//!
//! FFN uses SwiGLU: gate = x @ W1^T, up = x @ W3^T,
//!   output = silu(gate) * (up @ W2^T)

use crate::transformer::kv_cache::LayerKvCache;
use crate::transformer::linear::Linear;
use crate::transformer::rms_norm::RmsNorm;
use crate::transformer::rope::RopeConfig;

/// SwiGLU activation: silu(x) * y
fn swiglu(x: &[f32], y: &[f32], size: usize) -> Vec<f32> {
    assert!(size <= x.len(), "swiglu: size={} but x.len()={}", size, x.len());
    assert!(size <= y.len(), "swiglu: size={} but y.len()={}", size, y.len());
    let mut output = vec![0.0f32; size];
    for i in 0..size {
        let sigmoid = if x[i] >= 0.0 {
            1.0 / (1.0 + (-x[i]).exp())
        } else {
            x[i] / (1.0 + x[i].exp())
        };
        output[i] = sigmoid * x[i] * y[i];
    }
    output
}

/// Attention mechanism for a single transformer layer.
pub struct Attention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub rope: RopeConfig,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize, // num_kv_heads * head_dim
}

impl Attention {
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            wq,
            wk,
            wv,
            wo,
            rope: RopeConfig::new(head_dim, 10000.0, 4096),
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
        }
    }

    /// Compute scaled dot-product attention without caching (original path).
    ///
    /// input: `[batch, embed_dim]`
    /// Returns: `[batch, embed_dim]`
    pub fn forward(
        &self,
        x: &[f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let embed_dim = self.num_heads * self.head_dim;

        // Q/K/V projections
        let q = self.wq.forward(x, batch_size);
        let k = self.wk.forward(x, batch_size);
        let v = self.wv.forward(x, batch_size);

        // Apply RoPE to Q and K separately (different head counts for GQA)
        let mut q_rope = q;
        let mut k_rope = k;
        self.rope
            .apply_single(&mut q_rope, self.num_heads, seq_len, start_pos);
        self.rope
            .apply_single(&mut k_rope, self.num_kv_heads, seq_len, start_pos);

        // Scaled dot-product attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut output = vec![0.0f32; batch_size * seq_len * embed_dim];

        for b in 0..batch_size {
            for pos in 0..seq_len {
                let q_idx = (b * seq_len + pos) * embed_dim;
                let mut attn_weights = vec![0.0f32; seq_len];

                // Q @ K^T for this position
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for h in 0..self.num_heads {
                        let q_start = q_idx + h * self.head_dim;
                        let k_start = (b * seq_len + j) * self.kv_dim
                            + h / (self.num_heads / self.num_kv_heads) * self.head_dim;
                        for d in 0..self.head_dim {
                            sum += q_rope[q_start + d] * k_rope[k_start + d];
                        }
                    }
                    attn_weights[j] = sum * scale;
                }

                // Softmax
                let max_val = attn_weights
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = attn_weights.iter().map(|w| (*w - max_val).exp()).collect();
                let exp_sum: f32 = exps.iter().sum();
                let softmax_out: Vec<f32> = if exp_sum > 0.0 {
                    exps.iter().map(|e| e / exp_sum).collect()
                } else {
                    vec![1.0 / seq_len as f32; seq_len]
                };

                // softmax_out @ V
                let mut attn_output = vec![0.0f32; self.num_heads * self.head_dim];
                for h in 0..self.num_heads {
                    let group = h / (self.num_heads / self.num_kv_heads);
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let v_start =
                                (b * seq_len + j) * self.kv_dim + group * self.head_dim + d;
                            sum += softmax_out[j] * v[v_start];
                        }
                        attn_output[h * self.head_dim + d] = sum;
                    }
                }

                // Output projection: attn_output @ wo^T
                let wo_output = self.wo.forward(&attn_output, 1);
                for i in 0..embed_dim {
                    output[(b * seq_len + pos) * embed_dim + i] = wo_output[i];
                }
            }
        }

        output
    }

    /// Compute attention with KV caching for autoregressive generation.
    ///
    /// - Projects Q, K, V from input
    /// - Applies RoPE to Q (current position) and K (current position only)
    /// - Appends K, V to the cache
    /// - Computes attention against the full cache (all previous positions + new)
    ///
    /// For batch_size=1, seq_len=1 (standard autoregressive decode):
    ///   Input: `[embed_dim]` (single token's hidden state)
    ///   Output: `[embed_dim]`
    ///
    /// The cache stores RoPE-rotated K values — RoPE is applied at append time.
    pub fn forward_with_cache(
        &self,
        x: &[f32],
        kv_cache: &mut LayerKvCache,
        pos: usize,
    ) -> Vec<f32> {
        let embed_dim = self.num_heads * self.head_dim;

        // Project Q, K, V from single token input
        // x is [embed_dim], batch_size=1
        let q_proj = self.wq.forward(x, 1); // [embed_dim]
        let k_proj = self.wk.forward(x, 1); // [kv_dim]
        let v_proj = self.wv.forward(x, 1); // [kv_dim]

        // Apply RoPE to Q (num_heads) and K (num_kv_heads) separately.
        // Must be separate because Q and K may have different head counts (GQA).
        let mut q = q_proj;
        let mut k_rotated = k_proj.clone();
        self.rope.apply_single(&mut q, self.num_heads, 1, pos);
        self.rope
            .apply_single(&mut k_rotated, self.num_kv_heads, 1, pos);

        // Append K (RoPE-rotated) and V to cache
        kv_cache.append(&k_rotated, &v_proj);

        // Scaled dot-product attention against full cache
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let cache_len = kv_cache.seq_len();

        let mut attn_output = vec![0.0f32; embed_dim];

        for h in 0..self.num_heads {
            let kv_group = h / (self.num_heads / self.num_kv_heads);

            // Q for this head: q[h * head_dim .. (h+1) * head_dim]
            let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];

            // K for this head from cache: [cache_len, head_dim]
            let k_head = kv_cache.k_head(kv_group);
            // V for this head from cache: [cache_len, head_dim]
            let v_head = kv_cache.v_head(kv_group);

            // Compute attention scores: q @ k^T for all cached positions
            let mut scores = vec![0.0f32; cache_len];
            for t in 0..cache_len {
                let k_pos = &k_head[t * self.head_dim..(t + 1) * self.head_dim];
                let dot: f32 = q_head.iter().zip(k_pos.iter()).map(|(a, b)| a * b).sum();
                scores[t] = dot * scale;
            }

            // Softmax over cached positions
            let max_val = scores
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (*s - max_val).exp()).collect();
            let exp_sum: f32 = exps.iter().sum();
            let weights: Vec<f32> = if exp_sum > 0.0 {
                exps.iter().map(|e| e / exp_sum).collect()
            } else {
                vec![1.0 / cache_len as f32; cache_len]
            };

            // Weighted sum of V
            let out_head = &mut attn_output[h * self.head_dim..(h + 1) * self.head_dim];
            for t in 0..cache_len {
                let v_pos = &v_head[t * self.head_dim..(t + 1) * self.head_dim];
                for d in 0..self.head_dim {
                    out_head[d] += weights[t] * v_pos[d];
                }
            }
        }

        // Output projection
        self.wo.forward(&attn_output, 1)
    }
}

/// Feed-forward network with SwiGLU activation.
pub struct FeedForward {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
    pub intermediate_dim: usize,
}

impl FeedForward {
    pub fn new(w1: Linear, w2: Linear, w3: Linear, intermediate_dim: usize) -> Self {
        Self {
            w1,
            w2,
            w3,
            intermediate_dim,
        }
    }

    /// Forward pass: silu(x @ W1^T) * (x @ W3^T) @ W2^T
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let gate = self.w1.forward(x, batch_size);
        let up = self.w3.forward(x, batch_size);

        eprintln!("FFN: intermediate_dim={}, gate.len={}, up.len={}, x.len={}, batch={}", self.intermediate_dim, gate.len(), up.len(), x.len(), batch_size);

        let swiglu_out = swiglu(&gate, &up, self.intermediate_dim);
        self.w2.forward(&swiglu_out, batch_size)
    }
}

/// Single transformer layer.
pub struct TransformerLayer {
    pub attention: Attention,
    pub feed_forward: FeedForward,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
}

impl TransformerLayer {
    pub fn new(
        attention: Attention,
        feed_forward: FeedForward,
        attention_norm: RmsNorm,
        ffn_norm: RmsNorm,
    ) -> Self {
        Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }

    /// Forward pass without KV cache (original path, used for prefill or non-cached inference).
    ///
    /// input: `[batch, embed_dim]`
    /// Returns: `[batch, embed_dim]` with residual connections applied
    pub fn forward(
        &self,
        x: &[f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Vec<f32> {
        let embed_dim = x.len() / batch_size;

        // Attention sub-layer: x + attn(RMSNorm(x))
        let normed = self.attention_norm.forward(x, batch_size);
        let attn_out = self
            .attention
            .forward(&normed, batch_size, seq_len, start_pos);

        // Residual: x + attn_out
        let mut h = vec![0.0f32; batch_size * embed_dim];
        for i in 0..h.len().min(x.len()).min(attn_out.len()) {
            h[i] = x[i] + attn_out[i];
        }

        // FFN sub-layer: h + ffn(RMSNorm(h))
        let normed_ffn = self.ffn_norm.forward(&h, batch_size);
        let ffn_out = self.feed_forward.forward(&normed_ffn, batch_size);

        // Residual: h + ffn_out
        for i in 0..h.len().min(ffn_out.len()) {
            h[i] += ffn_out[i];
        }

        h
    }

    /// Forward pass with KV caching (for autoregressive decode).
    ///
    /// - `x`: `[embed_dim]` — single token's hidden state
    /// - `kv_cache`: mutable reference to this layer's KV cache
    /// - `pos`: position index in the sequence (for RoPE and cache slot)
    ///
    /// Returns: `[embed_dim]` — updated hidden state
    pub fn forward_with_cache(
        &self,
        x: &[f32],
        kv_cache: &mut LayerKvCache,
        pos: usize,
    ) -> Vec<f32> {
        let embed_dim = x.len();

        // Attention sub-layer: x + attn(RMSNorm(x))
        let normed = self.attention_norm.forward(x, 1);
        let attn_out = self.attention.forward_with_cache(&normed, kv_cache, pos);

        // Residual
        let mut h = vec![0.0f32; embed_dim];
        for i in 0..embed_dim {
            h[i] = x[i] + attn_out[i];
        }

        // FFN sub-layer: h + ffn(RMSNorm(h))
        let normed_ffn = self.ffn_norm.forward(&h, 1);
        let ffn_out = self.feed_forward.forward(&normed_ffn, 1);

        // Residual
        for i in 0..embed_dim {
            h[i] += ffn_out[i];
        }

        h
    }
}
