//! GPU weight cache: upload weights once, reuse across forward passes.
//!
//! Eliminates per-call H2D transfers for weight matrices, which were
//! identified as 23.2% of kernel time in Week 26 profiling.

use crate::kernel::candle_bridge;
use candle_core::{Device, Tensor};
use half::f16;
use std::collections::HashMap;
use tracing::info;

/// Cache of GPU-resident weight tensors keyed by layer name.
pub struct WeightCache {
    tensors: HashMap<String, Tensor>,
}

impl WeightCache {
    pub fn new() -> Self {
        info!("WeightCache initialized");
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Get or create the transposed weight tensor for a layer.
    ///
    /// Weights are stored as [out_features, in_features] f16 on host.
    /// For GEMM we need [in_features, out_features], so transpose and upload once.
    pub fn get_or_create(
        &mut self,
        name: &str,
        weights: &[f16],
        in_features: usize,
        out_features: usize,
    ) -> Option<&Tensor> {
        // Check if already cached
        if self.tensors.contains_key(name) {
            return Some(self.tensors.get(name).unwrap());
        }

        info!(
            layer = name,
            in_features, out_features, "Uploading weights to GPU"
        );

        // Transpose: W is [out, in], need [in, out] for GEMM
        let w_t: Vec<f16> = {
            let mut out = Vec::with_capacity(in_features * out_features);
            for i in 0..in_features {
                for j in 0..out_features {
                    out.push(weights[j * in_features + i]);
                }
            }
            out
        };

        // Upload to GPU as f16 tensor [in_features, out_features]
        let device = candle_bridge::bridge_device();
        match Tensor::from_vec(w_t, (in_features, out_features), &device) {
            Ok(t) => {
                self.tensors.insert(name.to_string(), t);
                // Return reference to the newly inserted tensor
                Some(self.tensors.get(name).unwrap())
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to upload weights to GPU");
                None
            }
        }
    }

    /// Check if a layer's weights are cached.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Clear the cache (useful for model reload).
    pub fn clear(&mut self) {
        let count = self.tensors.len();
        self.tensors.clear();
        info!(count, "WeightCache cleared");
    }

    /// Number of cached weight tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
}

impl Default for WeightCache {
    fn default() -> Self {
        Self::new()
    }
}
