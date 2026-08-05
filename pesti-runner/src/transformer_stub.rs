//! Stub transformer module for CPU-only builds.
//!
//! Provides stub implementations matching the real transformer API
//! to allow compilation without CUDA dependencies.

use rand::distributions::Uniform;
use rand::Rng;
use std::path::Path;

/// Stub model architecture (mirrors real ModelArch)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelArch {
    #[default]
    Llama,
    Gemma,
    Qwen2,
    Qwen3,
    Phi3,
    Mixtral,
    Starcoder2,
}

/// Stub model config (mirrors real LlamaConfig)
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
}

impl LlamaConfig {
    pub fn from_gguf_header(_header: &pesti_gguf::types::GgufHeader) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            arch: ModelArch::default(),
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            embed_dim: 4096,
            intermediate_dim: 11008,
            max_seq_len: 4096,
        })
    }

    pub fn from_safetensors_metadata(_meta: &std::collections::HashMap<String, String>) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            arch: ModelArch::default(),
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            embed_dim: 4096,
            intermediate_dim: 11008,
            max_seq_len: 4096,
        })
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            arch: ModelArch::default(),
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            embed_dim: 4096,
            intermediate_dim: 11008,
            max_seq_len: 4096,
        }
    }
}

/// Stub transformer layer
#[derive(Debug, Clone)]
pub struct TransformerLayer {}

impl TransformerLayer {
    pub fn new() -> Self {
        Self {}
    }
}

/// Stub linear layer (mirrors real Linear)
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Vec<f32>,
}

impl Linear {
    pub fn from_f32_weight(_weight: &[f32], _bias: Option<&[f32]>) -> Self {
        Self {
            weight: _weight.to_vec(),
        }
    }

    pub fn from_f32_weight_with_shape(
        _weight: &[f32],
        _bias: Option<&[f32]>,
        _in_features: usize,
        _out_features: usize,
    ) -> Self {
        Self {
            weight: _weight.to_vec(),
        }
    }

    pub fn forward(&self, _input: &[f32], _batch_size: usize) -> Vec<f32> {
        vec![0.0; self.weight.len()]
    }
}

/// Stub RMS norm (mirrors real RmsNorm)
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(_weight: Vec<f32>, _eps: f32) -> Self {
        Self { weight: _weight, eps: _eps }
    }

    pub fn forward(&self, _input: &[f32], _batch_size: usize) -> Vec<f32> {
        vec![0.0; _input.len()]
    }
}

/// Stub rope config (mirrors real RopeConfig)
#[derive(Debug, Clone)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub rope_base: f32,
    pub max_seq_len: usize,
}

impl RopeConfig {
    pub fn new(_head_dim: usize, _rope_base: f32, _max_seq_len: usize) -> Self {
        Self {
            head_dim: _head_dim,
            rope_base: _rope_base,
            max_seq_len: _max_seq_len,
        }
    }
}

/// Stub SamplingConfig (mirrors real SamplingConfig from sampling.rs)
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { seed: 42 }
    }
}

/// Stub argmax (mirrors real argmax from sampling.rs)
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Stub sample (mirrors real sample from sampling.rs)
pub fn sample(logits: &[f32], _config: &SamplingConfig, rng: &mut rand::rngs::StdRng) -> u32 {
    let sum: f32 = logits.iter().map(|&x| x.exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&x| (x.exp() / sum)).collect();

    // Use explicit distribution for rand 0.10+ compatibility
    let dist = Uniform::from(0.0..1.0);
    let mut r = rng.sample(dist);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Stub tokenizer config (mirrors real GgufTokenizerConfig)
#[derive(Debug, Clone)]
pub struct GgufTokenizerConfig {
    pub vocab_size: usize,
    pub eos_token_id: u32,
}

/// Stub tokenizer loader (mirrors real load_tokenizer_from_gguf)
pub fn load_tokenizer_from_gguf(_path: &Path) -> Result<super::GgufTokenizer, crate::error::RunnerError> {
    use tokenizers::Tokenizer;
    Ok(Tokenizer::from_file(_path).map_err(|e| crate::error::RunnerError::Tokenizer(e.to_string()))?)
}

/// Stub Llama model (mirrors real LlamaModel from transformer/model.rs)
#[derive(Debug, Clone)]
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub token_embeddings: Option<Linear>,
    pub output: Option<Linear>,
    pub final_norm: Option<RmsNorm>,
    pub layers: Vec<TransformerLayer>,
    pub vocab_size: u32,
}

impl LlamaModel {
    pub fn load_gguf(_path: &Path) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            config: LlamaConfig::default(),
            token_embeddings: None,
            output: None,
            final_norm: None,
            layers: vec![TransformerLayer::new(); 32],
            vocab_size: 32000,
        })
    }

    pub fn from_gguf_weights(_weights: crate::gguf_weight_loader::GgufWeights) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            config: LlamaConfig::default(),
            token_embeddings: None,
            output: None,
            final_norm: None,
            layers: vec![TransformerLayer::new(); 32],
            vocab_size: 32000,
        })
    }

    pub fn from_safetensors_weights(_weights: crate::safetensors_weight_loader::SafetensorsWeights, _config: LlamaConfig) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            config: _config,
            token_embeddings: None,
            output: None,
            final_norm: None,
            layers: vec![TransformerLayer::new(); 32],
            vocab_size: 32000,
        })
    }

    pub fn load_safetensors(_path: &Path, _config: LlamaConfig) -> Result<Self, crate::error::RunnerError> {
        Ok(Self {
            config: _config,
            token_embeddings: None,
            output: None,
            final_norm: None,
            layers: vec![TransformerLayer::new(); 32],
            vocab_size: 32000,
        })
    }

    pub fn forward_with_dispatch(&self, _hidden: &[f32], _start_pos: usize) -> Result<Vec<f32>, crate::error::RunnerError> {
        Ok(vec![0.0; _hidden.len()])
    }

    pub fn embed(&self, _token: u32, _seq_len: usize) -> Result<Vec<f32>, crate::error::RunnerError> {
        Ok(vec![0.0; 4096])
    }

    pub fn forward_layers(&self, _hidden: &[f32], _seq_len: usize) -> Result<Vec<f32>, crate::error::RunnerError> {
        Ok(vec![0.0; _hidden.len()])
    }

    pub fn apply_output_head(&self, _hidden: &[f32]) -> Result<Vec<f32>, crate::error::RunnerError> {
        Ok(vec![0.0; 32000])
    }

    pub fn is_some(&self) -> bool {
        true
    }
}

impl Default for LlamaModel {
    fn default() -> Self {
        Self {
            config: LlamaConfig::default(),
            token_embeddings: None,
            output: None,
            final_norm: None,
            layers: vec![TransformerLayer::new(); 32],
            vocab_size: 32000,
        }
    }
}
