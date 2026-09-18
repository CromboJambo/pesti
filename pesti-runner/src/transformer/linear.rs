//! Linear layer: y = x @ W^T + b (or without bias).
//!
//! Stores weight matrix in row-major layout: W[i][j] = weights[i * out_features + j].
//! Input x is [batch, in_features], output y is [batch, out_features].
//!

#![allow(clippy::redundant_closure)]

use crate::kernel::gemm::{CpuGemmKernel, GemmArch, GemmKernel};
use half::f16;
use std::sync::Arc;

#[derive(Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
    /// Optional GPU GEMM kernel for accelerated matmul. When set and available,
    /// forward() uses this instead of the hand-written CPU matmul.
    gemm_kernel: Option<Arc<dyn GemmKernel + Send + Sync>>,
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
            gemm_kernel: None,
        }
    }

    /// Set a GPU GEMM kernel for this layer. Subsequent forward() calls will
    /// use the GPU path when available.
    pub fn set_gemm_kernel(&mut self, kernel: Arc<dyn GemmKernel + Send + Sync>) {
        self.gemm_kernel = Some(kernel);
    }

    /// Clear the GPU GEMM kernel (revert to CPU matmul).
    pub fn clear_gemm_kernel(&mut self) {
        self.gemm_kernel = None;
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
            gemm_kernel: None,
        }
    }

    /// Build a Linear layer from f32 bytes with explicit shape (preferred).
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
            gemm_kernel: None,
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
            gemm_kernel: None,
        }
    }

    /// Build a Linear layer with explicit shape (for embeddings where we know embed_dim).
    pub fn from_f32_weight_with_shape(
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
            gemm_kernel: None,
        }
    }

    /// Forward pass: y = x @ W^T + bias.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        // Try GPU GEMM path if kernel is available and CUDA feature is enabled.
        #[cfg(feature = "cuda")]
        if let Some(ref gemm) = self.gemm_kernel {
            if gemm.is_available() {
                return self.forward_gpu(gemm, x, m, k, n);
            }
        }

        // CPU fallback: hand-written matmul with rayon parallelism.
        self.forward_cpu(x, batch_size)
    }

    /// GPU GEMM path: F32→F16 convert → dispatch_gemm → F16→F32 convert.
    #[cfg(feature = "cuda")]
    fn forward_gpu(
        &self,
        gemm: &Arc<dyn GemmKernel + Send + Sync>,
        x: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        use crate::kernel::device_buf::DeviceBuffer;

        // Convert input to F16 for GPU GEMM.
        let x_f16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
        let a = DeviceBuffer::from_host(x_f16);

        // Transpose weights: W is [out, in] row-major; GEMM needs B as [k, n].
        let w_t: Vec<f16> = (0..k)
            .flat_map(|i| (0..n).map(move |j| f16::from_f32(self.weight[j * k + i])))
            .collect();
        let b = DeviceBuffer::from_host(w_t);

        // Allocate output buffer on device.
        let mut c = DeviceBuffer::zeros(m * n);

        // Launch GPU GEMM: C = alpha * A @ B + beta * C
        gemm.matmul(1.0, &a, &b, 0.0, &mut c, m, n, k)
            .expect("GPU GEMM matmul failed");

        // Transfer result back to host.
        let mut output = c.to_host();

        // Apply bias if present (on CPU for simplicity).
        if let Some(ref bias) = self.bias {
            for b_idx in 0..m {
                for o in 0..n {
                    output[b_idx * n + o] += bias[o];
                }
            }
        }

        output
    }

    /// CPU fallback: hand-written matmul with rayon parallelism.
    fn forward_cpu(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.out_features];

        use rayon::prelude::*;

        let k = self.in_features;
        let n = self.out_features;

        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(b, out_row)| {
                let start_idx = b * k;
                let end_idx = std::cmp::min((b + 1) * k, x.len());
                let x_row = &x[start_idx..end_idx];

                for o in 0..n {
                    let w_row = &self.weight[o * k..(o + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..std::cmp::min(k, x_row.len()) {
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
        let weight = vec![
            1.0, 2.0, 3.0, // row 0: W[0][0]=1, W[0][1]=2, W[0][2]=3
            4.0, 5.0, 6.0, // row 1: W[1][0]=4, W[1][1]=5, W[1][2]=6
        ];
        let bias = Some(vec![0.1, 0.2]);

        let linear = Linear::new(weight, bias, 3, 2);

        // Input: 2x3 matrix
        let x = vec![
            1.0, 2.0, 3.0, // batch 0
            4.0, 5.0, 6.0, // batch 1
        ];

        let output = linear.forward(&x, 2);

        // Expected: C[b,o] = sum_i(x[b,i] * W[o,i]) + bias[o]
        // C[0,0] = 1*1 + 2*2 + 3*3 + 0.1 = 14.1
        // C[0,1] = 1*4 + 2*5 + 3*6 + 0.2 = 32.2
        // C[1,0] = 4*1 + 5*2 + 6*3 + 0.1 = 32.1
        // C[1,1] = 4*4 + 5*5 + 6*6 + 0.2 = 77.2

        let expected = [14.1, 32.2, 32.1, 77.2];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let linear = Linear::new(weight, None, 2, 3);

        let x = vec![1.0, 2.0];

        let output = linear.forward(&x, 1);

        // Expected: C[0,o] = sum_i(x[i] * W[o,i])
        // C[0,0] = 1*1 + 2*2 = 5
        // C[0,1] = 1*3 + 2*4 = 11
        // C[0,2] = 1*5 + 2*6 = 17

        let expected = [5.0, 11.0, 17.0];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }
}
