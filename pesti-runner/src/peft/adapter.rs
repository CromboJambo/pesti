//! Base adapter trait and type definitions.

use super::{AdapterConfig, AdapterError, Result};

/// Trait that all adapters must implement.
///
/// This defines the core interface for parameter-efficient fine-tuning adapters.
/// Adapters are lightweight modifications to base linear layers that allow
/// fine-tuning with minimal parameter overhead.
pub trait Adapter: Send + Sync {
    /// Apply the adapter to an input tensor.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, in_features]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, out_features]
    fn forward(&self, x: &[f32], batch_size: usize) -> Result<Vec<f32>>;

    /// Merge adapter weights into base linear weights.
    ///
    /// This is useful for deployment where you want zero runtime overhead
    /// from the adapter. The merge is typically done once after training.
    ///
    /// # Arguments
    /// * `base_weights` - Base linear layer weights [out_features, in_features]
    /// * `base_bias` - Optional base bias
    ///
    /// # Returns
    /// Merged weights and bias
    fn merge_into(
        &self,
        base_weights: &[f32],
        base_bias: Option<&[f32]>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)>;

    /// Unmerge adapter from weights (reverse of merge_into).
    ///
    /// Not all adapters need this, but it's useful for checkpointing
    /// or switching between adapter variants.
    fn unmerge_from(&mut self, merged_weights: &[f32], merged_bias: Option<&[f32]>) -> Result<()>;

    /// Get the rank of this adapter.
    fn rank(&self) -> usize;

    /// Get the scaling factor.
    fn scaling(&self) -> f32;

    /// Get the type of adapter.
    fn adapter_type(&self) -> AdapterType;

    /// Check if adapter is initialized (weights loaded).
    fn is_initialized(&self) -> bool;

    /// Initialize adapter with random weights.
    fn init_random(&mut self, in_features: usize, out_features: usize) -> Result<()>;

    /// Zero out adapter weights (for training from scratch).
    fn zero_grad(&mut self);

    /// Create an empty adapter from config (no weights loaded yet).
    fn empty_from_config(config: AdapterConfig) -> Result<Self>
    where
        Self: Sized;

    /// Load adapter from checkpoint file.
    fn load_from_checkpoint<P: AsRef<std::path::Path>>(
        &mut self,
        _path: P,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<()>;
}

/// Type of adapter being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterType {
    /// Low-Rank Adaptation: W' = W + BA where A is (in_features x r) and B is (r x out_features)
    #[default]
    LoRA,
    /// Quantized LoRA: LoRA on top of quantized base weights
    QLoRA,
    /// Adapter Block: Small FFN inserted before/after main layer
    AdapterBlock,
    /// Prefix Tuning: Learnable prefix tokens prepended to input
    PrefixTuning,
    /// Prompt Tuning: Learnable prompt embeddings
    PromptTuning,
}

/// Builder for creating adapters with common configurations.
pub struct AdapterBuilder<T> {
    config: AdapterConfig,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Adapter> AdapterBuilder<T> {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: AdapterConfig::default(),
            _marker: std::marker::PhantomData,
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

    /// Build the adapter with random initialization.
    pub fn build_random(self, in_features: usize, out_features: usize) -> Result<T>
    where
        T: Adapter + Sized,
    {
        let mut adapter = Self::build_empty(self)?;
        adapter.init_random(in_features, out_features)?;
        Ok(adapter)
    }

    /// Build an empty adapter (weights will be loaded from checkpoint).
    pub fn build_empty(self) -> Result<T>
    where
        T: Adapter + Sized,
    {
        if self.config.rank == 0 {
            return Err(AdapterError::InvalidRank);
        }
        if self.config.scaling <= 0.0 {
            return Err(AdapterError::InvalidScaling);
        }
        T::empty_from_config(self.config)
    }

    /// Build from checkpoint (weights loaded from file).
    pub fn build_from_checkpoint<P: AsRef<std::path::Path>>(
        self,
        _path: P,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<T>
    where
        T: Adapter + Sized,
    {
        let mut adapter = Self::build_empty(self)?;
        adapter.load_from_checkpoint(_path, _in_features, _out_features)?;
        Ok(adapter)
    }
}

impl<T: Adapter> Default for AdapterBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}
