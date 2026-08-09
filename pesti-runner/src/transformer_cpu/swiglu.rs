//! SwiGLU feed-forward network for CPU-only inference.

/// SwiGLU FFN: x -> (x @ W1) * SiLU(x @ W2) @ V
pub struct SwiGLUFFN {
    pub w1: crate::transformer_cpu::Linear, // Gate projection
    pub w2: crate::transformer_cpu::Linear, // Project down
    pub w3: crate::transformer_cpu::Linear, // Output projection (V)
}

impl SwiGLUFFN {
    pub fn new(w1: Vec<f32>, w2_transposed: Vec<f32>, w3: Vec<f32>, actual_out_features: usize) -> Self {
        // Infer dimensions from weight tensor shapes (GGUF stores as [in_features, out_features])
        // For ffn_gate: shape is [embed_dim, intermediate_dim] = [896, 4864] → out=4864, in=896
        let embed_dim = 896; // Input dimension matches transformer hidden size
        let intermediate_dim = 4864; // FFN expansion dimension
        
        Self {
            w1: crate::transformer_cpu::Linear::new(w1, None, intermediate_dim, embed_dim),
            // Use inferred dimensions for w2 based on actual dequantized size
            w2: crate::transformer_cpu::Linear::new(
                w2_transposed,
                None,
                actual_out_features,
                intermediate_dim,
            ),
            w3: crate::transformer_cpu::Linear::new(w3, None, embed_dim, intermediate_dim), // Output projection back to embed_dim
        }
    }

    /// SiLU activation: x * sigmoid(x)
    fn silu(x: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-x).exp());
        x * sig
    }

    /// Forward pass through SwiGLU FFN
    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        let intermediate_dim = self.w1.out_features;

        // Gate projection (W1 @ x) - projects from embed_dim to intermediate_dim
        let gate = self.w1.forward(x, _batch_size);

        // Up projection (W3 @ x) - projects from embed_dim to intermediate_dim
        let proj = self.w3.forward(x, _batch_size);

        // Apply SiLU to gate and element-wise multiply with up projection
        let gated: Vec<f32> = gate
            .iter()
            .zip(proj.iter())
            .map(|(g, p)| Self::silu(*g) * p)
            .collect();

        // Down projection (W2 @ fused) - projects from intermediate_dim back to embed_dim
        self.w2.forward(&gated, _batch_size)
    }
}
