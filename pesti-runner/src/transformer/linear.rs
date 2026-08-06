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

    /// Build a Linear layer from f32 bytes with explicit shape (preferred).
    ///
    /// Use this instead of `from_f32_weight` for attention/FFN weights where
    /// `in_features > 1`. The shape-less variant defaults `in_features=1`,
    /// which is only correct for 1D embedding lookups.
    pub fn from_f32_weight_with_dims(
        weight_f32: &[u8],
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
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

    /// Build a Linear layer from f32 bytes (used for safetensors loading).
    ///
    /// **Note:** Defaults `in_features=1`, which is only correct for 1D
    /// embedding tensors. For attention/FFN weights, use
    /// `from_f32_weight_with_dims` or `from_f32_weight_with_shape` instead.
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
        eprintln!("Linear::forward: in={}, out={}, batch={}, x.len={}, weight.len={}, output.len={}",
            self.in_features, self.out_features, batch_size, x.len(), self.weight.len(), output.len());

        // Matmul: output[b, o] = sum_i(x[b, i] * W[o, i])
        // Weight is [out_features, in_features] row-major.
        //
        // NOTE: gemm crate produces zero results on this system (SIMD dispatch bug).
        // Using manual matmul with rayon parallelism for correctness.
        use rayon::prelude::*;

        let k = self.in_features;
        let n = self.out_features;

        // Parallelize over batch dimension
        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(b, out_row)| {
                let x_row = &x[b * k..(b + 1) * k];
                for o in 0..n {
                    let w_row = &self.weight[o * k..(o + 1) * k];
                    let mut acc = 0.0f32;
                    // Manual dot product (auto-vectorized by LLVM)
                    for i in 0..k {
                        acc += x_row[i] * w_row[i];
                    }
                    out_row[o] = acc;
                }
            });

        // Apply bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_vs_scalar() {
        // Simple 2x3 matrix multiply: A (2x3) @ B^T (3x2) -> C (2x2)
        let weight = vec![
            1.0, 2.0, 3.0,  // row 0: W[0][0]=1, W[0][1]=2, W[0][2]=3
            4.0, 5.0, 6.0,  // row 1: W[1][0]=4, W[1][1]=5, W[1][2]=6
        ];
        let bias = Some(vec![0.1, 0.2]);

        let linear = Linear::new(weight, bias, 3, 2);

        // Input: 2x3 matrix
        let x = vec![
            1.0, 2.0, 3.0,  // batch 0
            4.0, 5.0, 6.0,  // batch 1
        ];

        let output = linear.forward(&x, 2);

        // Expected: C[b,o] = sum_i(x[b,i] * W[o,i]) + bias[o]
        // C[0,0] = 1*1 + 2*2 + 3*3 + 0.1 = 1 + 4 + 9 + 0.1 = 14.1
        // C[0,1] = 1*4 + 2*5 + 3*6 + 0.2 = 4 + 10 + 18 + 0.2 = 32.2
        // C[1,0] = 4*1 + 5*2 + 6*3 + 0.1 = 4 + 10 + 18 + 0.1 = 32.1
        // C[1,1] = 4*4 + 5*5 + 6*6 + 0.2 = 16 + 25 + 36 + 0.2 = 77.2

        let expected = vec![14.1, 32.2, 32.1, 77.2];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i, got, exp
            );
        }
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let weight = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];

        let linear = Linear::new(weight, None, 2, 3);

        let x = vec![1.0, 2.0];

        let output = linear.forward(&x, 1);

        // Expected: C[0,o] = sum_i(x[i] * W[o,i])
        // C[0,0] = 1*1 + 2*2 = 5
        // C[0,1] = 1*3 + 2*4 = 11
        // C[0,2] = 1*5 + 2*6 = 17

        let expected = vec![5.0, 11.0, 17.0];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i, got, exp
            );
        }
    }
}

