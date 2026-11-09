//! Sampling configuration for llama.cpp inference.
//!
//! Maps to llama-cpp-2's sampler chain components:
//! - Temperature
//! - Top-k
//! - Top-p (nucleus sampling)
//! - Min-p
//! - Tail-free sampling (TFS)
//! - Typical p
//! - Mirostat
//! - Repetition penalty
//! - Loopy penalty (alpha/beta)

/// Configuration for the sampling pipeline.
///
/// These map directly to llama.cpp's sampler chain. The default config
/// uses greedy sampling (temperature = 0).
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Context size in tokens.
    pub n_ctx: u32,
    /// Batch size for token processing.
    pub n_batch: u32,
    /// Number of threads to use.
    pub n_threads: i32,
    /// Number of layers to offload to GPU (-1 = auto, 0 = CPU only).
    pub n_gpu_layers: i32,
    /// KV cache type (f32 or f16).
    pub kv_cache_type: KvCacheType,

    // ── Sampling parameters ──
    /// Sampling temperature. Higher = more random. 0 = greedy (deterministic).
    pub temperature: f64,

    /// Top-k sampling: consider only the top k tokens. 0 = disabled.
    pub top_k: i32,

    /// Top-p (nucleus) sampling: consider tokens with cumulative probability >= p.
    /// 0 = disabled.
    pub top_p: f64,

    /// Min-p sampling: consider tokens with probability >= min_p * max_probability.
    /// 0 = disabled.
    pub min_p: f64,

    /// Tail-free sampling parameter. 1 = disabled, 0 = aggressive.
    pub tfs: f64,

    /// Typical p sampling parameter. 1 = disabled, 0 = aggressive.
    pub typical_p: f64,

    /// Whether to use Mirostat sampling.
    pub mirostat: bool,

    /// Mirostat target entropy (controls perplexity).
    pub mirostat_tau: f64,

    /// Mirostat learning rate (controls adaptation speed).
    pub mirostat_eta: f64,

    /// Repetition penalty. 1.0 = no penalty. > 1.0 penalizes repeated tokens.
    pub repetition_penalty: f64,

    /// Number of recent tokens to consider for repetition penalty.
    pub repeat_last_n: i32,

    /// Loopy penalty alpha (contextual penality). 0 = disabled.
    pub penalty_alpha: f64,

    /// Loopy penalty beta.
    pub penalty_beta: f64,

    /// Maximum number of tokens to generate.
    pub max_tokens: u32,

    /// Random seed for reproducibility.
    pub seed: u32,
}

/// KV cache type.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub enum KvCacheType {
    #[default]
    F32,
    F16,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 0,     // auto
            n_gpu_layers: -1, // auto
            kv_cache_type: KvCacheType::F32,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            tfs: 1.0,
            typical_p: 1.0,
            mirostat: false,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            penalty_alpha: 0.0,
            penalty_beta: 0.0,
            max_tokens: 1024,
            seed: 42,
        }
    }
}

impl SamplingConfig {
    /// Greedy sampling (deterministic, no randomness).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.0,
            min_p: 0.0,
            tfs: 1.0,
            typical_p: 1.0,
            mirostat: false,
            repetition_penalty: 1.0,
            penalty_alpha: 0.0,
            penalty_beta: 0.0,
            ..Default::default()
        }
    }

    /// Creative sampling (high temperature, high top-p).
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_k: 50,
            top_p: 0.99,
            min_p: 0.05,
            tfs: 0.5,
            typical_p: 0.5,
            ..Default::default()
        }
    }

    /// Balanced sampling (good for most use cases).
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            tfs: 1.0,
            typical_p: 1.0,
            ..Default::default()
        }
    }

    /// Precise sampling (low temperature, low top-p).
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 20,
            top_p: 0.9,
            min_p: 0.1,
            tfs: 1.0,
            typical_p: 1.0,
            repetition_penalty: 1.1,
            repeat_last_n: 128,
            ..Default::default()
        }
    }
}
