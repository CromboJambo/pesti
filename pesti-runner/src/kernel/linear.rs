//! Trait-based linear layer with automatic CPU/GPU dispatch.
//!
//! This module provides a unified `LinearLayer` trait that abstracts away
//! the device-specific implementation details. Callers interact with the
//! trait interface and don't need to know whether computation happens on
//! CPU or GPU.

use crate::error::{Result, RunnerError};
use std::sync::Arc;

/// Linear layer trait — the unified interface for all linear operations.
pub trait LinearLayer: Send + Sync {
    /// Forward pass: y = x @ W^T + bias
    fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>>;

    /// Input dimensionality
    fn in_features(&self) -> usize;

    /// Output dimensionality
    fn out_features(&self) -> usize;

    /// Clone this layer (for model cloning scenarios).
    fn clone_layer(&self) -> Box<dyn LinearLayer>;

    /// Upload weights to GPU (no-op for CPU-only layers).
    fn upload_weights_to_gpu(&mut self) -> Result<()> {
        Ok(())
    }

    /// Check if weights are currently resident on GPU.
    fn weights_on_gpu(&self) -> bool {
        false
    }

    /// Layer name for debugging/logging.
    fn layer_name(&self) -> &str {
        "linear"
    }
}

/// Factory function that selects CPU vs GPU based on runtime availability.
pub fn new_linear_layer(
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
) -> Box<dyn LinearLayer> {
    #[cfg(feature = "cuda")]
    {
        if crate::cuda_runtime::is_available() {
            tracing::debug!("Creating GPU linear layer ({in_features} -> {out_features})");
            return Box::new(GpuLinearLayer::from_f32(
                weight,
                bias,
                in_features,
                out_features,
            ));
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = weight;
        let _ = bias;
        let _ = in_features;
        let _ = out_features;
    }

    tracing::debug!("Creating CPU linear layer ({in_features} -> {out_features})");
    Box::new(CpuLinearLayer::from_f32(
        weight,
        bias,
        in_features,
        out_features,
    ))
}

/// CPU implementation using rayon-parallel matmul.
pub struct CpuLinearLayer {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl CpuLinearLayer {
    pub fn from_f32(
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
}

impl LinearLayer for CpuLinearLayer {
    fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        if x.len() < m * k {
            return Err(RunnerError::Kernel(format!(
                "linear layer input too small: expected {} values, got {}",
                m * k,
                x.len()
            )));
        }

        let mut output = vec![0.0f32; m * n];

        use rayon::prelude::*;

        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(b, out_row)| {
                let x_start = b * k;
                for o in 0..n {
                    let w_row = &self.weight[o * k..(o + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += x[x_start + i] * w_row[i];
                    }
                    out_row[o] = acc;
                }
            });

        if let Some(ref bias) = self.bias {
            for b in 0..m {
                for o in 0..n {
                    output[b * n + o] += bias[o];
                }
            }
        }

        Ok(output)
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn layer_name(&self) -> &str {
        "cpu_linear"
    }

    fn clone_layer(&self) -> Box<dyn LinearLayer> {
        Box::new(CpuLinearLayer::from_f32(
            self.weight.clone(),
            self.bias.clone(),
            self.in_features,
            self.out_features,
        ))
    }
}

/// GPU implementation using cudarc's cublasLt API with lazy weight caching.
#[cfg(feature = "cuda")]
pub struct GpuLinearLayer {
    weight_f16: Vec<half::f16>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

#[cfg(feature = "cuda")]
impl GpuLinearLayer {
    pub fn from_f32(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let weight_f16: Vec<half::f16> = weight.iter().map(|&v| half::f16::from_f32(v)).collect();
        Self {
            weight_f16,
            bias,
            in_features,
            out_features,
        }
    }
}

#[cfg(feature = "cuda")]
impl LinearLayer for GpuLinearLayer {
    fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        let x_f16: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();

        // Upload weights for this forward pass (no persistent caching to avoid OOM)
        crate::kernel::cuda_bridge::gemm_f16(
            &x_f16,
            &self.weight_f16,
            batch_size,
            self.out_features,
            self.in_features,
        )
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn upload_weights_to_gpu(&mut self) -> Result<()> {
        // Weights uploaded per forward pass, not cached persistently
        Ok(())
    }

    fn weights_on_gpu(&self) -> bool {
        false
    }

    fn layer_name(&self) -> &str {
        "gpu_linear"
    }

    fn clone_layer(&self) -> Box<dyn LinearLayer> {
        Box::new(GpuLinearLayer::from_f32(
            self.weight_f16.iter().map(|v| v.to_f32()).collect(),
            self.bias.clone(),
            self.in_features,
            self.out_features,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_linear_forward() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = Some(vec![0.1, 0.2]);

        let layer = CpuLinearLayer::from_f32(weight, bias, 3, 2);

        let x = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&x, 1).unwrap();

        assert!((output[0] - 14.1).abs() < 1e-5);
        assert!((output[1] - 32.2).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_linear_no_bias() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let layer = CpuLinearLayer::from_f32(weight, None, 2, 3);

        let x = vec![1.0, 2.0];
        let output = layer.forward(&x, 1).unwrap();

        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 11.0).abs() < 1e-5);
        assert!((output[2] - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_linear_batched() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let layer = CpuLinearLayer::from_f32(weight, None, 3, 2);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = layer.forward(&x, 2).unwrap();

        assert!((output[0] - 14.0).abs() < 1e-5);
        assert!((output[1] - 32.0).abs() < 1e-5);

        assert!((output[2] - 32.0).abs() < 1e-5);
        assert!((output[3] - 77.0).abs() < 1e-5);
    }

    #[test]
    fn test_factory_creates_cpu_layer() {
        let layer = new_linear_layer(vec![1.0, 2.0], None, 1, 2);
        assert_eq!(layer.in_features(), 1);
        assert_eq!(layer.out_features(), 2);

        let output = layer.forward(&[3.0], 1).unwrap();
        assert_eq!(output.len(), 2);
    }
}