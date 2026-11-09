//! CPU-only transformer primitives (Linear, RMSNorm, etc.) for pure-Rust inference.
//!
//! These are standalone implementations that don't depend on CUDA kernels.

/// Simple linear layer with f32 weights.
pub struct Linear {
    /// Weight matrix: [out_features, in_features] row-major
    pub weight: Vec<f32>,
    /// Optional bias vector
    pub bias: Option<Vec<f32>>,
    pub out_features: usize,
    pub in_features: usize,
}

impl Linear {
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            out_features,
            in_features,
        }
    }

    /// Forward pass: y = x @ W^T + b
    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_features];

        for out in 0..self.out_features {
            let mut sum = 0.0f32;
            for inp in 0..self.in_features {
                // Row-major: weight[out * in_features + inp]
                sum += x[inp] * self.weight[out * self.in_features + inp];
            }
            output[out] = sum;
            if let Some(ref bias) = self.bias {
                output[out] += bias[out];
            }
        }

        output
    }
}

/// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm {
    pub eps: f32,
    pub weight: Vec<f32>,
    pub dim: usize,
}

impl RmsNorm {
    pub fn new(eps: f32, dim: usize) -> Self {
        Self {
            eps,
            weight: vec![1.0f32; dim], // Default to identity
            dim,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &[f32], _batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; self.dim];

        // Compute RMS
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / self.dim as f32 + self.eps).sqrt();

        // Scale by weight
        for i in 0..self.dim {
            output[i] = x[i] / rms * self.weight[i];
        }

        output
    }
}
