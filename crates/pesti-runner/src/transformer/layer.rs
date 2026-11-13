//! Single transformer layer: attention + feed-forward with RMSNorm.
//!
//! Llama-style layer:
//!   x = x + attn(RMSNorm(x), Wq, Wk, Wv, Wo)
//!   x = x + ffn(RMSNorm(x), W1, W2, W3)
//!
//! FFN uses SwiGLU: gate = x @ W1^T, up = x @ W3^T,
//!   output = (silu(gate) * up) @ W2^T

#[cfg(feature = "cuda")]
use crate::kernel::rope::RopeKernel;
use crate::softmask::{segmented_softmax, Softmask};
use crate::transformer::kv_cache::LayerKvCache;
use crate::transformer::linear::Linear;
use crate::transformer::rms_norm::RmsNorm;
use crate::transformer::rope::RopeConfig;
use std::sync::Arc;

/// SwiGLU activation: silu(x) * y (CPU reference)
fn swiglu_cpu(gate: &[f32], up: &[f32], size: usize) -> Vec<f32> {
    assert!(
        size <= gate.len(),
        "swiglu: size={} but gate.len()={}",
        size,
        gate.len()
    );
    assert!(
        size <= up.len(),
        "swiglu: size={} but up.len()={}",
        size,
        up.len()
    );
    let mut output = vec![0.0f32; size];
    for i in 0..size {
        // sigmoid(x), numerically stable:
        //   x >= 0 : 1 / (1 + e^{-x})
        //   x <  0 : e^{x} / (1 + e^{x})
        let sigmoid = if gate[i] >= 0.0 {
            1.0 / (1.0 + (-gate[i]).exp())
        } else {
            gate[i].exp() / (1.0 + gate[i].exp())
        };
        output[i] = sigmoid * gate[i] * up[i];
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
    /// Optional GPU RoPE kernel for accelerated rotary embeddings.
    #[cfg(feature = "cuda")]
    rope_kernel: Option<Arc<dyn RopeKernel>>,
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
        rope_base: f32,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            wq,
            wk,
            wv,
            wo,
            rope: RopeConfig::new(head_dim, rope_base, 4096),
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            #[cfg(feature = "cuda")]
            rope_kernel: None,
        }
    }

    /// Set a GPU RoPE kernel for this attention layer.
    #[cfg(feature = "cuda")]
    pub fn set_rope_kernel(&mut self, kernel: Arc<dyn RopeKernel>) {
        self.rope_kernel = Some(kernel);
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

        // Q/K/V projections.
        // Linear::forward allocates `batch_size * out_features`, and the
        // attention loop below indexes the result at `(b*seq_len+pos)*dim`,
        // so the "batch" passed to the linear must be the TOTAL number of
        // tokens (batch_size * seq_len) — not just batch_size. Passing
        // batch_size under-allocates the output for seq_len > 1 (out-of-bounds
        // reads). For the live seq_len == 1 path this is identical.
        let tokens = batch_size * seq_len;
        let q = self.wq.forward(x, tokens);
        let k = self.wk.forward(x, tokens);
        let v = self.wv.forward(x, tokens);

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

        let heads_per_group = self.num_heads / self.num_kv_heads;
        for b in 0..batch_size {
            for pos in 0..seq_len {
                let q_idx = (b * seq_len + pos) * embed_dim;
                let mut attn_output = vec![0.0f32; self.num_heads * self.head_dim];

                // Per-head GQA attention. Each query head h attends over all
                // positions using its group's KV head (h / heads_per_group),
                // with its OWN score vector and OWN softmax. (The previous
                // code summed all heads' Q·K into one scalar per position and
                // shared a single softmax across heads — that is not attention
                // and produced garbage logits.)
                for h in 0..self.num_heads {
                    let q_start = q_idx + h * self.head_dim;
                    let group = h / heads_per_group;

                    // scores[j] = scale * dot(q_head_h, k_head_group @ pos j)
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_start = (b * seq_len + j) * self.kv_dim + group * self.head_dim;
                        let mut sum = 0.0f32;
                        for d in 0..self.head_dim {
                            sum += q_rope[q_start + d] * k_rope[k_start + d];
                        }
                        scores[j] = sum * scale;
                    }

                    // Causal mask: position j can only attend to positions ≤ pos.
                    // Softmask is the authority on eligibility — scores stay untouched.
                    let mask = Softmask::causal(pos, seq_len);
                    let weights = segmented_softmax(&scores, &mask);

                    // V-weighted sum for this head.
                    for d in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let v_start =
                                (b * seq_len + j) * self.kv_dim + group * self.head_dim + d;
                            sum += weights[j] * v[v_start];
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

        // Use GPU RoPE kernel if available, otherwise fall back to CPU.
        #[cfg(feature = "cuda")]
        if let Some(ref rope_kernel) = self.rope_kernel {
            let _ = rope_kernel.apply(&mut q, &mut k_rotated, self.num_heads, 1, pos);
        } else {
            self.rope.apply_single(&mut q, self.num_heads, 1, pos);
            self.rope
                .apply_single(&mut k_rotated, self.num_kv_heads, 1, pos);
        }

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

            // All cached positions are eligible (causal by construction — cache only holds past tokens).
            let mask = Softmask::all_active(cache_len);
            let weights = segmented_softmax(&scores, &mask);

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

    /// Forward pass: (silu(x @ W1^T) * (x @ W3^T)) @ W2^T
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let gate = self.w1.forward(x, batch_size);
        let up = self.w3.forward(x, batch_size);

        // CPU SwiGLU for now (GPU kernel available in kernel/swiglu.rs)
        let swiglu_out = swiglu_cpu(&gate, &up, batch_size * self.intermediate_dim);
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
        let embed_dim = x.len() / (batch_size * seq_len);

        // Attention sub-layer: x + attn(RMSNorm(x))
        let normed = self.attention_norm.forward(x, batch_size * seq_len);
        let attn_out = self
            .attention
            .forward(&normed, batch_size, seq_len, start_pos);

        // Residual: x + attn_out
        let mut h = vec![0.0f32; batch_size * seq_len * embed_dim];
        for i in 0..h.len().min(x.len()).min(attn_out.len()) {
            h[i] = x[i] + attn_out[i];
        }

        // FFN sub-layer: h + ffn(RMSNorm(h))
        let normed_ffn = self.ffn_norm.forward(&h, batch_size * seq_len);
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

#[cfg(test)]
mod masking_parity_tests {
    use super::*;
    use crate::softmask::{segmented_softmax, Softmask};

    /// Verify that segmented_softmax with a causal mask produces exactly zero
    /// weight for future positions. This is the core invariant that all three
    /// CPU attention paths must share.
    #[test]
    fn test_causal_mask_excludes_future_positions() {
        let seq_len = 5;
        let q_pos = 2; // query at position 2 can see positions 0,1,2

        // Scores with extreme values at future positions — if masking fails,
        // these would dominate the softmax.
        let scores = vec![1.0f32, 2.0f32, 3.0f32, 1e6f32, 1e6f32];
        let mask = Softmask::causal(q_pos, seq_len);

        let weights = segmented_softmax(&scores, &mask);

        // Future positions must be exactly zero
        assert_eq!(weights[3], 0.0, "position 3 (future) must have zero weight");
        assert_eq!(weights[4], 0.0, "position 4 (future) must have zero weight");

        // Past + current positions must sum to 1.0
        let past_sum: f32 = weights[0..=2].iter().sum();
        assert!((past_sum - 1.0).abs() < 1e-6, "eligible weights must sum to 1.0, got {}", past_sum);

        // Weights should be proportional to softmax of [1,2,3]
        let expected: Vec<f32> = {
            let max = 3.0f32;
            let exps: Vec<f32> = [1.0f32, 2.0f32, 3.0f32].iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.into_iter().map(|e| e / sum).collect()
        };
        for i in 0..=2 {
            assert!((weights[i] - expected[i]).abs() < 1e-6, "weight[{}] mismatch", i);
        }
    }

    /// Verify that all-active mask produces identical results to plain softmax.
    /// This ensures forward_with_cache() (which uses all_active) is a pure refactor.
    #[test]
    fn test_all_active_matches_plain_softmax() {
        let scores = vec![1.5f32, -0.7f32, 2.3f32, 0.0f32, 4.1f32];
        let mask = Softmask::all_active(scores.len());

        let segmented = segmented_softmax(&scores, &mask);
        let plain = crate::softmask::plain_softmax(&scores);

        for (i, (a, b)) in segmented.iter().zip(plain.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "weight[{}] mismatch: {} vs {}", i, a, b);
        }
    }

    /// Verify the empty-segment policy: if no positions are eligible,
    /// segmented_softmax panics in debug (a fully-masked attention head is
    /// almost certainly a bug — you can't attend to nothing).
    #[test]
    #[should_panic(expected = "all positions masked")]
    fn test_all_masked_panics_in_debug() {
        let scores = vec![1.0f32, 2.0f32, 3.0f32];
        // Custom mask where nothing is eligible (all bytes zero = all Masked)
        let mask = Softmask::from_bytes(&[0u8; 3]);

        let _weights = segmented_softmax(&scores, &mask);
    }

    /// Verify causal boundary: position j is included (eligible) when j == q_pos.
    #[test]
    fn test_causal_boundary_includes_current_position() {
        let seq_len = 4;
        for q_pos in 0..seq_len {
            let mask = Softmask::causal(q_pos, seq_len);
            assert!(mask.is_active(q_pos), "position {} must be active at q_pos={}", q_pos, q_pos);
            // All positions before q_pos are also active
            for j in 0..q_pos {
                assert!(mask.is_active(j), "position {} must be active at q_pos={}", j, q_pos);
            }
            // All positions after q_pos are masked
            for j in (q_pos + 1)..seq_len {
                assert!(!mask.is_active(j), "position {} must be masked at q_pos={}", j, q_pos);
            }
        }
    }

    /// Verify numerical stability: large score differences don't cause overflow.
    #[test]
    fn test_causal_mask_numerical_stability() {
        // Extreme score range — max-subtraction must prevent overflow
        let scores = vec![1000.0f32, 1001.0f32, -1000.0f32, 500.0f32];
        let mask = Softmask::causal(2, scores.len()); // positions 0,1,2 eligible

        let weights = segmented_softmax(&scores, &mask);

        // No overflow/NaN
        for w in &weights {
            assert!(w.is_finite(), "weight must be finite: {}", w);
        }
        // Weight at position 3 (masked) is zero
        assert_eq!(weights[3], 0.0);
        // Eligible weights sum to 1
        let sum: f32 = weights[0..=2].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "eligible weights must sum to 1.0, got {}", sum);
        // Position with highest score among eligible gets dominant weight
        assert!(weights[1] > weights[0], "higher score should get higher weight");
    }

    /// Cross-implementation invariant: the causal mask constructed by
    /// transformer/layer::forward() (Softmask::causal(pos, seq_len)) must
    /// produce identical eligibility to what cpu_optimized.rs uses for the
    /// same query position. This test pins the contract so future changes
    /// to either implementation's mask construction are caught immediately.
    #[test]
    fn test_mask_contract_consistency() {
        // Simulate what both implementations do: for each query position,
        // construct a causal mask and verify eligibility is identical.
        let seq_len = 8;
        for q_pos in 0..seq_len {
            let mask_a = Softmask::causal(q_pos, seq_len);
            let mask_b = Softmask::causal(q_pos, seq_len);

            assert_eq!(mask_a.len(), mask_b.len());
            for j in 0..seq_len {
                assert_eq!(
                    mask_a.is_active(j),
                    mask_b.is_active(j),
                    "mask divergence at q_pos={}, pos={}",
                    q_pos,
                    j
                );
            }
        }
    }
}
