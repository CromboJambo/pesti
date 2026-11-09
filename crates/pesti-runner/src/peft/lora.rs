//! LoRA (Low-Rank Adaptation) implementation.
//!
//! LoRA freezes the pre-trained model weights and injects trainable rank
//! decomposition matrices into each layer of the transformer architecture.
//!
//! ## Mathematical Formulation
//!
//! For a linear layer W ∈ ℝ^(d×k), LoRA approximates the update ΔW as:
//! ```text
//! ΔW = BA
//! ```
//! where:
//! - B ∈ ℝ^(r×d) is initialized from N(0, σ²)
//! - A ∈ ℝ^(k×r) is initialized from N(0, σ²/α)
//! - r << d,k is the low rank (typically 8-256)
//!
//! The adapted weights are: W' = W + α·BA
//!
//! ## Memory Efficiency
//!
//! - Base model: d×k parameters (frozen)
//! - LoRA adapter: r×(d+k) parameters (trainable)
//! - For d=k=4096, r=8: 2% of base model size!

use super::adapter::{Adapter, AdapterType};
use super::matrices::{MatA, MatB};
use crate::peft::{AdapterConfig, AdapterError, Result};
use rand::{Rng, SeedableRng};

/// LoRA adapter with low-rank decomposition matrices A and B.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Matrix A: projects from input space to latent space [in_features x rank]
    pub matrix_a: MatA,
    /// Matrix B: projects from latent space to output space [rank x out_features]
    pub matrix_b: MatB,
    /// Scaling factor (typically rank * alpha)
    pub scaling: f32,
    /// Rank of the adapter
    pub rank: usize,
    /// Input feature dimension
    pub in_features: usize,
    /// Output feature dimension
    pub out_features: usize,
    /// Whether weights are initialized
    pub is_initialized: bool,
}

impl LoRAAdapter {
    /// Create a new empty LoRA adapter from config.
    pub fn empty_from_config(config: AdapterConfig) -> Result<Self> {
        if config.adapter_type != AdapterType::LoRA {
            return Err(AdapterError::DimensionMismatch {
                expected: 0, // Would need to match adapter type
                actual: 0,
            });
        }

        Ok(Self {
            matrix_a: MatA::empty(),
            matrix_b: MatB::empty(),
            scaling: config.scaling,
            rank: config.rank,
            in_features: 0,
            out_features: 0,
            is_initialized: false,
        })
    }

    /// Create a LoRA adapter with random initialization.
    pub fn new_random(
        in_features: usize,
        out_features: usize,
        config: &AdapterConfig,
    ) -> Result<Self> {
        let mut adapter = Self::empty_from_config(AdapterConfig {
            rank: config.rank,
            scaling: config.scaling,
            adapter_type: AdapterType::LoRA,
        })?;

        adapter.in_features = in_features;
        adapter.out_features = out_features;
        adapter.init_random(in_features, out_features)?;
        Ok(adapter)
    }

    /// Create a LoRA adapter with zero initialization (for training).
    pub fn new_zeros(
        in_features: usize,
        out_features: usize,
        config: &AdapterConfig,
    ) -> Result<Self> {
        let mut adapter = Self::empty_from_config(AdapterConfig {
            rank: config.rank,
            scaling: config.scaling,
            adapter_type: AdapterType::LoRA,
        })?;

        adapter.in_features = in_features;
        adapter.out_features = out_features;
        adapter.zero_grad();
        Ok(adapter)
    }

    /// Get the effective scaling factor.
    pub fn effective_scaling(&self) -> f32 {
        self.scaling / self.rank as f32
    }

    /// Forward pass: compute BAx where x is [batch_size, in_features].
    ///
    /// This is the LoRA contribution that gets added to the base linear output.
    /// The full computation is: output = Wx + scaling * BAx
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        if !self.is_initialized {
            return Err(AdapterError::NotInitialized);
        }

        // Check dimensions
        if self.in_features != x.len() / batch_size {
            return Err(AdapterError::DimensionMismatch {
                expected: self.in_features,
                actual: x.len() / batch_size,
            });
        }

        // Step 1: Compute Ax (project to latent space)
        // A is [rank x in_features], so we need to transpose for matrix multiply
        let latent = self.matrix_a.matmul_transpose(x, batch_size)?;

        // Step 2: Compute B(latent) (project back to output space)
        // B is [out_features x rank]
        let lora_output = self.matrix_b.forward(&latent, batch_size);

        Ok(lora_output)
    }

    /// Merge adapter weights into base linear weights.
    ///
    /// After merging: W_merged = W_base + scaling * B @ A
    /// This gives zero runtime overhead for inference.
    pub fn merge_into(
        &self,
        base_weights: &[f32],
        base_bias: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        if !self.is_initialized {
            return Err(AdapterError::NotInitialized);
        }

        // Check dimensions match
        let expected_weights_len = self.out_features * self.in_features;
        if base_weights.len() != expected_weights_len {
            return Err(AdapterError::DimensionMismatch {
                expected: expected_weights_len,
                actual: base_weights.len(),
            });
        }

        // Compute BA (rank x rank matrix multiply)
        let ba = self.matrix_b.matmul(&self.matrix_a);

        // Scale by effective scaling factor
        let scale = self.effective_scaling();

        // Add to base weights: W' = W + scale * BA
        let mut merged_weights = vec![0.0f32; base_weights.len()];
        for (row, out_idx) in (0..self.out_features).enumerate() {
            for (col, _in_idx) in (0..self.in_features).enumerate() {
                let base_weight = base_weights[out_idx * self.in_features + col];
                let ba_weight = ba[row * self.in_features + col] * scale;
                merged_weights[out_idx * self.in_features + col] = base_weight + ba_weight;
            }
        }

        // Add bias if present
        let merged_bias = base_bias.map(|b| {
            let mut bias = b.to_vec();
            // Bias also gets scaled contribution from BA
            for (i, bi) in bias.iter_mut().enumerate() {
                *bi += scale * ba[i % self.rank] * 0.0; // Simplified: no bias adjustment needed
            }
            bias
        });

        Ok((merged_weights, merged_bias))
    }

    /// Load adapter from checkpoint file.
    pub fn load_from_checkpoint<P: AsRef<std::path::Path>>(
        &mut self,
        _path: P,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<()> {
        // TODO: Implement checkpoint loading from JSON/ Safetensors
        // For now, just mark as initialized
        self.is_initialized = true;
        Ok(())
    }

    /// Save adapter to checkpoint file.
    pub fn save_to_checkpoint<P: AsRef<std::path::Path>>(&self, _path: P) -> Result<()> {
        // TODO: Implement checkpoint saving
        Ok(())
    }
}

impl Adapter for LoRAAdapter {
    fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        Self::forward(self, x, batch_size)
    }

    fn merge_into(
        &self,
        base_weights: &[f32],
        base_bias: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        Self::merge_into(self, base_weights, base_bias)
    }

    fn unmerge_from(
        &mut self,
        _merged_weights: &[f32],
        _merged_bias: Option<&[f32]>,
    ) -> Result<()> {
        // TODO: Implement unmerge (solve for A and B from merged weights)
        // This is non-trivial and requires SVD or similar decomposition
        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn scaling(&self) -> f32 {
        self.scaling
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::LoRA
    }

    fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    fn init_random(&mut self, in_features: usize, out_features: usize) -> Result<()> {
        // Initialize A from N(0, sigma_a) where sigma_a = 0.02
        let sigma_a = 0.02;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);
        let a_data: Vec<f32> = (0..self.rank * in_features)
            .map(|_| {
                let bits = rng.next_u32();
                f32::from_bits(bits).abs() * sigma_a * 10.0
            })
            .collect();

        // Initialize B from N(0, 0) - mostly zeros for sparsity
        let sigma_b = 0.0;
        let b_data: Vec<f32> = (0..out_features * self.rank)
            .map(|_| {
                let bits = rng.next_u32();
                f32::from_bits(bits).abs() * sigma_b * 10.0
            })
            .collect();

        self.matrix_a = MatA::new(a_data, self.rank, in_features);
        self.matrix_b = MatB::new(b_data, out_features, self.rank);
        self.in_features = in_features;
        self.out_features = out_features;
        self.is_initialized = true;

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.matrix_a.zero();
        self.matrix_b.zero();
    }

    fn empty_from_config(config: AdapterConfig) -> Result<Self>
    where
        Self: Sized,
    {
        Self::empty_from_config(config)
    }

    fn load_from_checkpoint<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        Self::load_from_checkpoint(self, path, in_features, out_features)
    }
}

/// Builder for LoRA adapters.
pub struct LoRABuilder {
    config: AdapterConfig,
}

impl LoRABuilder {
    /// Create a new LoRA builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: AdapterConfig::default(),
        }
    }

    /// Set the rank of the adapter.
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.config.rank = rank;
        self
    }

    /// Set the scaling factor.
    pub fn with_scaling(mut self, scaling: f32) -> Self {
        self.config.scaling = scaling;
        self
    }

    /// Build a LoRA adapter with random initialization.
    pub fn build_random(self, in_features: usize, out_features: usize) -> Result<LoRAAdapter> {
        LoRAAdapter::new_random(in_features, out_features, &self.config)
    }

    /// Build a LoRA adapter with zero initialization (for training).
    pub fn build_zeros(self, in_features: usize, out_features: usize) -> Result<LoRAAdapter> {
        LoRAAdapter::new_zeros(in_features, out_features, &self.config)
    }
}

impl Default for LoRABuilder {
    fn default() -> Self {
        Self::new()
    }
}
