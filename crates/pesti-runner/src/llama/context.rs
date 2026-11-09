//! Context configuration for llama.cpp inference.
#[allow(unused_imports)]
use crate::llama::sampler::KvCacheType;

/// Configuration for a [`LlamaContext`](llama_cpp_2::context::LlamaContext).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextConfig {
    /// Number of tokens in the KV cache.
    pub n_ctx: u32,
    /// Batch size for token processing.
    pub n_batch: u32,
    /// Ubatch size (micro-batch for compute).
    pub n_ubatch: u32,
    /// Number of threads. 0 = auto.
    pub n_threads: i32,
    /// Random seed.
    pub seed: u32,
    /// KV cache type.
    pub kv_cache_type: KvCacheType,
    /// Attention type (auto, causal, etc.).
    pub attention_type: AttentionType,
    /// Pooling type for embeddings.
    pub pooling_type: PoolingType,
    /// RoPE scaling type.
    pub rope_scaling_type: RopeScalingType,
    /// RoPE frequency scaling factor.
    pub rope_freq_scale: f32,
    /// Number of layers to offload to GPU.
    pub n_gpu_layers: i32,
}

/// Attention type.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub enum AttentionType {
    #[default]
    Auto,
    Causal,
    NonCausal,
    Pool,
}

/// Pooling type for embeddings.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub enum PoolingType {
    #[default]
    None,
    Mean,
    Cls,
}

/// RoPE scaling type.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub enum RopeScalingType {
    #[default]
    None,
    Linear,
    Yarn,
    LongContext,
    FlashAttn,
    Any,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: 0,
            seed: 42,
            kv_cache_type: KvCacheType::F32,
            attention_type: AttentionType::Auto,
            pooling_type: PoolingType::None,
            rope_scaling_type: RopeScalingType::None,
            rope_freq_scale: 1.0,
            n_gpu_layers: -1,
        }
    }
}
