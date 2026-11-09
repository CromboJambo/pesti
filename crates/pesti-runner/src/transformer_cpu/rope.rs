//! RoPE (Rotary Positional Embeddings) for CPU-only inference.

/// RoPE configuration matching Llama-style models.
#[derive(Debug)]
pub struct RopeConfig {
    pub dim: usize, // Half of the hidden dimension per head
    pub base: f32,  // Base frequency
    pub max_seq_len: usize,
}

impl RopeConfig {
    pub fn new(dim: usize, base: f32, max_seq_len: usize) -> Self {
        Self {
            dim,
            base,
            max_seq_len,
        }
    }

    /// Precompute the inverse frequencies (cos/sin table).
    pub fn precompute_freqs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut cos = vec![0.0f32; self.max_seq_len * self.dim];
        let mut sin = vec![0.0f32; self.max_seq_len * self.dim];

        for i in 0..self.dim {
            let freq = 1.0 / (self.base.powf((i as f32 / self.dim as f32) * 2.0));
            for pos in 0..self.max_seq_len {
                let angle = pos as f32 * freq;
                cos[pos * self.dim + i] = (angle).cos();
                sin[pos * self.dim + i] = (angle).sin();
            }
        }

        (cos, sin)
    }

    /// Apply RoPE to a single position for one head.
    pub fn apply_single_head(&self, x: &mut [f32], pos: usize, cos: &[f32], sin: &[f32]) {
        assert_eq!(x.len(), self.dim * 2); // Expect pairs

        for i in 0..self.dim / 2 {
            let c = cos[pos * (self.dim / 2) + i];
            let s = sin[pos * (self.dim / 2) + i];

            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];

            // Apply rotation: [cos -sin; sin cos]
            x[2 * i] = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}
