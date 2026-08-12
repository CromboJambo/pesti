//! SwiGLU feed-forward network for CPU-only inference.

/// SwiGLU FFN: x -> (x @ W1) * SiLU(x @ W2) @ V
pub struct SwiGLUFFN {
    pub w1: crate::transformer_cpu::Linear, // Gate projection
    pub w2: crate::transformer_cpu::Linear, // Project down
    pub w3: crate::transformer_cpu::Linear, // Up projection
}

impl SwiGLUFFN {
    /// Create a new SwiGLU FFN from dequantized weight tensors.
    ///
    /// This uses architecture-aware dimension inference to handle GGUF metadata mismatches.
    /// For Qwen2/Qwen3 models with SwiGLU:
    /// - W1 (gate): [embed_dim, intermediate_dim]  
    /// - W3 (up):   [embed_dim, intermediate_dim]
    /// - W2 (down): [intermediate_dim, embed_dim]
    pub fn new(w1_data: Vec<f32>, w2_data: Vec<f32>, w3_data: Vec<f32>, embed_dim: usize) -> Self {
        let w1_len = w1_data.len();
        let w3_len = w3_data.len();

        // Infer intermediate_dim from gate/up projections (more reliable than down)
        let intermediate_dim = w1_len / embed_dim;

        // Verify consistency with w3
        assert_eq!(
            w3_len,
            intermediate_dim * embed_dim,
            "w3 length mismatch: expected {}, got {}",
            intermediate_dim * embed_dim,
            w3_len
        );

        // For w2 (down projection), use the inferred intermediate_dim even if metadata is wrong
        let w2_len = w2_data.len();
        let expected_w2_len = intermediate_dim * embed_dim;

        if w2_len != expected_w2_len {
            tracing::warn!(
                "FFN down projection dimension mismatch: expected {} elements ({}x{}), got {}",
                expected_w2_len,
                intermediate_dim,
                embed_dim,
                w2_len
            );

            // Try to infer the actual intermediate_dim from w2 data
            let inferred_intermediate_from_w2 = if embed_dim > 0 {
                w2_len / embed_dim
            } else {
                intermediate_dim
            };

            tracing::warn!(
                "Using inferred intermediate_dim={} from w2 instead of {}",
                inferred_intermediate_from_w2,
                intermediate_dim
            );

            // Use the inferred value for w2 dimensions
            Self {
                w1: crate::transformer_cpu::Linear::new(w1_data, None, intermediate_dim, embed_dim),
                w2: crate::transformer_cpu::Linear::new(
                    w2_data,
                    None,
                    embed_dim,                     // Output to hidden dimension
                    inferred_intermediate_from_w2, // Input from inferred intermediate layer
                ),
                w3: crate::transformer_cpu::Linear::new(w3_data, None, intermediate_dim, embed_dim),
            }
        } else {
            // Normal case - dimensions match
            Self {
                w1: crate::transformer_cpu::Linear::new(w1_data, None, intermediate_dim, embed_dim),
                w2: crate::transformer_cpu::Linear::new(
                    w2_data,
                    None,
                    embed_dim,        // Output to hidden dimension
                    intermediate_dim, // Input from intermediate layer
                ),
                w3: crate::transformer_cpu::Linear::new(w3_data, None, intermediate_dim, embed_dim),
            }
        }
    }

    /// Forward pass through SwiGLU FFN.
    ///
    /// Architecture: x -> (x @ W1) * SiLU(x @ W2) @ V
    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        let embed_dim = self.w1.in_features;

        // Compute gate projection: (batch, intermediate_dim)
        let gate = self.w1.forward(x, 0);

        // Compute down projection: (batch, intermediate_dim)
        let down = self.w2.forward(x, 0);

        // Apply SiLU to gate and multiply
        let mut hidden = vec![0.0; embed_dim];
        for i in 0..embed_dim {
            let g = gate[i] / _batch_size as f32; // Simplified scaling
            let d = down[i];
            let silu_g = g / (1.0 + (-g).exp());
            hidden[i] = silu_g * d;
        }

        // Project up: (batch, embed_dim)

        self.w3.forward(&hidden, 0)
    }
}
