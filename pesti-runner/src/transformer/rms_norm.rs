//! RMSNorm: root mean square layer normalization.
//!
//! RMSNorm(x) = x * gain / sqrt(e^2 + variance), where variance = mean(x^2).
//! Used in Llama-style models instead of LayerNorm.

/// RMS normalization layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
    pub dim: usize,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        let dim = weight.len();
        Self { weight, eps, dim }
    }

    /// Forward pass: normalize each row of input.
    ///
    /// x: [batch_size, dim]
    /// Returns: [batch_size, dim]
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.dim];

        for b in 0..batch_size {
            let start = b * self.dim;
            let mut variance = 0.0f32;

            for i in 0..self.dim {
                variance += x[start + i] * x[start + i];
            }
            variance /= self.dim as f32;
            let rms = (variance + self.eps).sqrt();

            for i in 0..self.dim {
                output[start + i] = x[start + i] / rms * self.weight[i];
            }
        }

        output
    }
}

