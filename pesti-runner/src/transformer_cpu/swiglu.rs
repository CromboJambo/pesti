//! SwiGLU feed-forward network for CPU-only inference.

/// SwiGLU FFN: x -> (x @ W1) * SiLU(x @ W2) @ V
pub struct SwiGLUFFN {
    pub w1: crate::transformer_cpu::Linear, // Gate projection
    pub w2: crate::transformer_cpu::Linear, // Project down
    pub w3: crate::transformer_cpu::Linear, // Output projection (V)
}

impl SwiGLUFFN {
    pub fn new(w1: Vec<f32>, w2: Vec<f32>, w3: Vec<f32>) -> Self {
        // Assume all have same intermediate dimension
        let intermediate_dim = w1.len();
        
        Self {
            w1: crate::transformer_cpu::Linear::new(w1, None, 1, intermediate_dim),
            w2: crate::transformer_cpu::Linear::new(w2, None, 1, intermediate_dim),
            w3: crate::transformer_cpu::Linear::new(w3, None, intermediate_dim, 1),
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

        // Gate projection (W1 @ x)
        let gate = self.w1.forward(x, _batch_size);

        // Project down (W2 @ x)
        let proj = self.w2.forward(x, _batch_size);

        // Apply SiLU to gate
        let gated: Vec<f32> = gate.iter().map(|&g| Self::silu(g)).collect();

        // Element-wise multiply with projected values
        let fused: Vec<f32> = gated.iter().zip(proj.iter()).map(|(a, b)| a * b).collect();

        // Output projection (W3 @ fused)
        self.w3.forward(&fused, _batch_size)
    }
}
