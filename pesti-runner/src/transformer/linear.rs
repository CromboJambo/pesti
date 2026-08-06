//! Linear layer: y = x @ W^T + b (or without bias).
//!
//! Stores weight matrix in row-major layout: W[i][j] = weights[i * out_features + j].
//! Input x is [batch, in_features], output y is [batch, out_features].
//!

#![allow(clippy::redundant_closure)]

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn from_f16_weight(weight_f16: &[u8], bias: Option<Vec<f32>>) -> Self {
        let elements = weight_f16.len() / 2;
        let weight: Vec<f32> = weight_f16
            .chunks_exact(2)
            .map(|chunk| f16_to_f32(chunk))
            .collect();
        let (in_features, out_features) = if elements > 0 { (1, elements) } else { (0, 0) };
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Build a Linear layer from f32 bytes (used for safetensors loading).
    pub fn from_f32_weight(weight_f32: &[u8], bias: Option<Vec<f32>>) -> Self {
        let elements = weight_f32.len() / 4;
        let weight: Vec<f32> = weight_f32
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let (in_features, out_features) = if elements > 0 { (1, elements) } else { (0, 0) };
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Build a Linear layer with explicit shape (for embeddings where we know embed_dim).
    pub fn from_f32_weight_with_shape(weight_f32: &[u8], bias: Option<Vec<f32>>, in_features: usize, out_features: usize) -> Self {
        let weight: Vec<f32> = weight_f32
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = x @ W^T + bias.
    ///
    /// x: [batch_size, in_features]
    /// Returns: [batch_size, out_features]
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.out_features];

        for b in 0..batch_size {
            let x_start = b * self.in_features;
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += x[x_start + i] * self.weight[o * self.in_features + i];
                }
                if let Some(ref bias) = self.bias {
                    sum += bias[o];
                }
                output[b * self.out_features + o] = sum;
            }
        }

        output
    }

    /// Matrix multiply: C = A @ B^T, where A is [m x k] and B is [n x k].
    pub fn matmul_transpose_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[j * k + l];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}

/// Convert half-float bytes to f32.
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let f32_bits = (sign << 31) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

