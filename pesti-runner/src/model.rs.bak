//! Model struct with per-layer KV cache allocation and inference loop.
//!
//! Manages the full transformer forward pass including:
//! - Per-layer KV cache allocation
//! - Prefill mode: process full prompt batch
//! - Decode mode: auto-regressive single-token generation
//!
//! ## Architecture
//!
//! ```text
//! Model
//!   ├── config: ModelConfig (num_layers, num_heads, head_dim, max_seq)
//!   ├── engine: InferenceEngine (GEMM + attention kernels)
//!   ├── kv_caches: Vec<Kvcache> (one per transformer layer)
//!   ├── seq_len: usize (current sequence length)
//!   └── prefill() / decode() → inference loop
//! ```
//!
//! ## Inference Loop
//!
//! 1. **Prefill**: Process full prompt batch, compute attention, append KV to cache
//! 2. **Decode**: Auto-regressive loop — generate one token at a time
//!    - Extract last token's query
//!    - Append new KV pair
//!    - Compute attention over full cache (box_y=1 for decode)
//!    - Sample next token from output logits

use crate::error::Result;
use crate::error::RunnerError;
use crate::inference_engine::InferenceEngine;
#[cfg(feature = "cuda")]
use crate::kernel::DeviceBuffer;
#[cfg(not(feature = "cuda"))]
use crate::kernel::device_buf::DeviceBuffer;
#[cfg(feature = "cuda")]
use crate::kernel::attention::{AttentionArch, AttentionConfig};
#[cfg(not(feature = "cuda"))]
#[cfg(not(feature = "cuda"))]
use crate::kernel::attention_stub::{AttentionArch, AttentionConfig};
#[cfg(feature = "cuda")]
use crate::kernel::kvcache::Kvcache;
#[cfg(not(feature = "cuda"))]
use crate::kernel::kvcache_stub::Kvcache;
#[cfg(feature = "cuda")]
use crate::kernel::dispatch::DispatchContext;
#[cfg(feature = "cuda")]
use crate::transformer::{LlamaConfig, LlamaModel, ModelArch};
#[cfg(not(feature = "cuda"))]
#[cfg(not(feature = "cuda"))]
use crate::transformer_stub::LlamaModel;
use half::f16;

/// Configuration for a transformer model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub num_kv_heads: usize,
    pub use_tma: bool,
    pub attention_arch: AttentionArch,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_heads: 32,
            head_dim: 64,
            max_seq: 2048,
            num_kv_heads: 32,
            use_tma: true,
            attention_arch: AttentionArch::default(),
        }
    }
}

impl ModelConfig {
    /// Create a model config from loaded GGUF weights.
    pub fn from_gguf(header: &pesti_gguf::types::GgufHeader) -> Result<Self> {
        let embed_dim = header.embedding_length().ok_or_else(|| {
            RunnerError::MissingHeaderField("embedding_length".to_string())
        })? as usize;
        let num_heads = header.attention_head_count().ok_or_else(|| {
            RunnerError::MissingHeaderField("attention_head_count".to_string())
        })? as usize;
        let num_kv_heads = header.attention_head_count_kv().unwrap_or(num_heads as u32) as usize;
        let num_layers = header.block_count().unwrap_or(32) as usize;
        let head_dim = if num_heads > 0 { embed_dim / num_heads } else { 64 };
        let max_seq = header.context_length().ok_or_else(|| {
            RunnerError::MissingHeaderField("context_length".to_string())
        })? as usize;

        Ok(Self {
            num_layers,
            num_heads,
            head_dim,
            max_seq,
            num_kv_heads,
            use_tma: false,
            attention_arch: AttentionArch::default(),
        })
    }

    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_head_dim(mut self, head_dim: usize) -> Self {
        self.head_dim = head_dim;
        self
    }

    pub fn with_max_seq(mut self, max_seq: usize) -> Self {
        self.max_seq = max_seq;
        self
    }

    pub fn with_num_kv_heads(mut self, num_kv_heads: usize) -> Self {
        self.num_kv_heads = num_kv_heads;
        self
    }

    pub fn with_tma(mut self, use_tma: bool) -> Self {
        self.use_tma = use_tma;
        self
    }

    pub fn with_attention_arch(mut self, arch: AttentionArch) -> Self {
        self.attention_arch = arch;
        self
    }

    /// Build the attention configuration for this model.
    pub fn attention_config(&self) -> AttentionConfig {
        AttentionConfig::default()
            .with_num_heads(self.num_heads)
            .with_head_dim(self.head_dim)
            .with_max_seq(self.max_seq)
            .with_arch(self.attention_arch)
            .with_tma(self.use_tma)
    }
}

/// Model with per-layer KV cache allocation and inference loop.
pub struct Model {
    /// Model configuration.
    pub config: ModelConfig,
    /// Inference engine with GEMM and attention kernels.
    pub engine: InferenceEngine,
    /// Per-layer KV caches (one pair per layer: key_cache and value_cache).
    pub kv_caches: Vec<(Kvcache, Kvcache)>,
    /// Current sequence length (total tokens processed).
    pub seq_len: usize,
    /// Loaded transformer weights for Q/K/V projections.
    #[cfg(feature = "cuda")]
    pub llama_model: crate::transformer::LlamaModel,
    #[cfg(not(feature = "cuda"))]
    pub llama_model: crate::transformer_stub::LlamaModel,
    /// Whether to use the dispatch system (GPU-accelerated path).
    pub use_dispatch: bool,
}

impl Model {
    /// Create a new model with per-layer KV cache allocation.
    pub fn new(config: ModelConfig, engine: InferenceEngine, on_device: bool) -> Self {
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let max_seq = config.max_seq;

        let kv_caches = (0..num_layers)
            .map(|_| {
                let key_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                let value_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                (key_cache, value_cache)
            })
            .collect();

        Self {
            config,
            engine,
            kv_caches,
            seq_len: 0,
            #[cfg(feature = "cuda")]
            llama_model: LlamaModel {
                config: LlamaConfig {
                    arch: ModelArch::Llama,
                    num_layers: 32,
                    num_heads: 32,
                    num_kv_heads: 8,
                    head_dim: 128,
                    embed_dim: 4096,
                    intermediate_dim: 11008,
                    max_seq_len: 4096,
                    rope_base: 10000.0,
                    rope_scaling_factor: None,
                    rope_scaling_type: None,
                    rms_norm_eps: 1e-5,
                },
                token_embeddings: None,
                output: None,
                final_norm: None,
                layers: vec![],
                vocab_size: 32000,
                tokenizer: None,
                tokenizer_config: None,
                dispatch: Some(DispatchContext::new()),
                kv_caches: None,
            },
            #[cfg(not(feature = "cuda"))]
            llama_model: crate::transformer_stub::LlamaModel::default(),
            use_dispatch: false,
        }
    }

    /// Create a model with loaded transformer weights for proper Q/K/V projections.
    pub fn with_llama_model(
        config: ModelConfig,
        engine: InferenceEngine,
        llama_model: LlamaModel, // Now uses real type
        on_device: bool,
    ) -> Self {
        let num_layers = config.num_layers;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let max_seq = config.max_seq;

        let kv_caches = (0..num_layers)
            .map(|_| {
                let key_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                let value_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, on_device);
                (key_cache, value_cache)
            })
            .collect();

        Self {
            config,
            engine,
            kv_caches,
            seq_len: 0,
            llama_model, // Store the real model directly (not wrapped in Option)
            use_dispatch: false,
        }
    }

    /// Enable the dispatch (GPU-accelerated) inference path.
    pub fn enable_dispatch(&mut self) {
        self.use_dispatch = true;
    }

    /// Check if dispatch is enabled and weights are loaded.
    pub fn can_use_dispatch(&self) -> bool {
        self.use_dispatch // Always true now - no need to check Option
    }

    /// Pass hidden states through all transformer layers using the dispatch system.
    pub fn forward_with_dispatch(&mut self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        self.llama_model.forward_with_dispatch(hidden, start_pos)
    }

    /// Create a model with a specific GEMM kernel.
    pub fn with_gemm(
        config: ModelConfig,
        gemm: Box<dyn crate::kernel::GemmKernel>,
        on_device: bool,
    ) -> Self {
        let engine =
            InferenceEngine::with_gemm(candle_core::Device::Cpu, candle_core::DType::F32, gemm);
        Self::new(config, engine, on_device)
    }

    /// Reset the model state: clear KV caches and sequence length.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for (key_cache, value_cache) in &mut self.kv_caches {
            key_cache.clear();
            value_cache.clear();
        }
    }

    /// Get the current sequence length.
    pub fn current_seq_len(&self) -> usize {
        self.seq_len
    }

    /// Check if the model has capacity for more tokens.
    pub fn has_capacity(&self) -> bool {
        self.seq_len < self.config.max_seq
    }

    /// Get the attention configuration for this model.
    pub fn attention_config(&self) -> AttentionConfig {
        self.config.attention_config()
    }

    /// Process a batch of tokens in prefill mode.
    pub fn prefill(&mut self, query: DeviceBuffer<f16>) -> Result<Vec<DeviceBuffer<f32>>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let out_dim = num_heads * head_dim;
        let batch_size = query.len().checked_div(out_dim).unwrap_or(0);

        if batch_size == 0 {
            return Ok(vec![]);
        }

        let config = self.attention_config();
        let mut outputs = Vec::with_capacity(self.kv_caches.len());

        // Always use layer weights (no stub mode anymore)
        for (layer_idx, (key_cache, value_cache)) in self.kv_caches.iter_mut().enumerate() {
            let layer = &self.llama_model.layers[layer_idx];

            // Extract Q, K, V from the input using layer weights
            let query_host: Vec<f32> = query
                .as_slice()
                .unwrap_or(&[])
                .iter()
                .map(|&x| x.to_f32())
                .collect();
            let q = layer.attention.wq.forward(&query_host, batch_size);
            let k = layer.attention.wk.forward(&query_host, batch_size);
            let v = layer.attention.wv.forward(&query_host, batch_size);

            // Convert to f16 for the engine
            let q_f16 = DeviceBuffer::from_host(q.iter().map(|&x| f16::from_f32(x)).collect());
            let k_f16 = DeviceBuffer::from_host(k.iter().map(|&x| f16::from_f32(x)).collect());
            let v_f16 = DeviceBuffer::from_host(v.iter().map(|&x| f16::from_f32(x)).collect());

            // Compute attention using proper Q, K, V
            let output = if key_cache.seq_len() == 0 {
                // First prefill step: no KV cache yet, use Q as output (identity projection)
                DeviceBuffer::from_host(
                    q_f16
                        .as_slice()
                        .unwrap_or(&[])
                        .iter()
                        .map(|&x| x.to_f32())
                        .collect(),
                )
            } else {
                self.engine
                    .attention(&q_f16, key_cache, value_cache, None, &config)?
            };

            // Append the last row of K and V as new KV for this layer
            let last_row_size = head_dim;
            let last_k: Vec<f16> = if let Some(slice) = k_f16.as_slice() {
                let start = (batch_size - 1) * last_row_size;
                let end = start + last_row_size;
                if end <= slice.len() {
                    slice[start..end].to_vec()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

            let last_v: Vec<f16> = if let Some(slice) = v_f16.as_slice() {
                let start = (batch_size - 1) * last_row_size;
                let end = start + last_row_size;
                if end <= slice.len() {
                    slice[start..end].to_vec()
                } else {
                    vec![]
                }
            } else {
                vec![]
            };

            outputs.push(output);

            if !last_k.is_empty() && !last_v.is_empty() {
                key_cache
                    .append(&last_k, &last_v)
                    .map_err(|e| {
                        RunnerError::Tensor(format!(
                            "Layer {layer_idx} KV append failed: {e}"
                        ))
                    })?;
            }
        }

        self.seq_len += batch_size;
        Ok(outputs)
    }

    /// Generate a single token in decode mode.
    pub fn decode(&mut self, query: DeviceBuffer<f16>) -> Result<DeviceBuffer<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let out_dim = num_heads * head_dim;
        let batch_size = 1;

        let config = self.attention_config();
        let mut last_output = DeviceBuffer::from_host(vec![0.0f32; out_dim]);

        // Always use layer weights (no stub mode anymore)
        for (layer_idx, (key_cache, value_cache)) in self.kv_caches.iter_mut().enumerate() {
            let layer = &self.llama_model.layers[layer_idx];

            // Extract Q, K, V from the input using layer weights
            let query_host: Vec<f32> = query
                .as_slice()
                .unwrap_or(&[])
                .iter()
                .map(|&x| x.to_f32())
                .collect();
            let q = layer.attention.wq.forward(&query_host, batch_size);
            let k = layer.attention.wk.forward(&query_host, batch_size);
            let v = layer.attention.wv.forward(&query_host, batch_size);

            // Convert to f16 for the engine
            let q_f16 = DeviceBuffer::from_host(q.iter().map(|&x| f16::from_f32(x)).collect());
            let k_f16 = DeviceBuffer::from_host(k.iter().map(|&x| f16::from_f32(x)).collect());
            let v_f16 = DeviceBuffer::from_host(v.iter().map(|&x| f16::from_f32(x)).collect());

            // Compute attention using proper Q, K, V
            let output = if key_cache.seq_len() == 0 {
                DeviceBuffer::from_host(
                    q_f16
                        .as_slice()
                        .unwrap_or(&[])
                        .iter()
                        .map(|&x| x.to_f32())
                        .collect(),
                )
            } else {
                self.engine
                    .attention(&q_f16, key_cache, value_cache, None, &config)?
            };

            last_output = output;

            // Append the last row of K and V as new KV for this layer
            let last_k: Vec<f16> = k_f16.as_slice().unwrap_or(&[]).to_vec();
            let last_v: Vec<f16> = v_f16.as_slice().unwrap_or(&[]).to_vec();

            if !last_k.is_empty() && !last_v.is_empty() {
                key_cache.append(&last_k, &last_v).map_err(|e| {
                    RunnerError::Tensor(format!(
                        "Layer {layer_idx} KV append failed: {e}"
                    ))
                })?;
            }
        }

        self.seq_len += 1;
        Ok(last_output)
    }
}
// Stub CpuModel for now - will be implemented later when GGUF loading is complete
// Stub CpuModel for CPU-only builds
pub struct CpuModel {
    pub llama_model: (), // Stub type - will be real LlamaModel when implemented
    pub config: crate::model::ModelConfig,
    pub kv_caches: Vec<(crate::kernel::Kvcache, crate::kernel::Kvcache)>,
    pub seq_len: usize,
    pub use_dispatch: bool,
}

impl CpuModel {
    pub fn load_gguf(_path: &std::path::Path) -> crate::model::Result<Self> {
        todo!("CPU-only model loading not yet implemented")
    }

    pub fn from_llama_model(_llama_model: ()) -> crate::model::Result<Self> {
        todo!("CPU-only model loading not yet implemented")
    }

    pub fn can_use_dispatch(&self) -> bool {
        self.use_dispatch
    }

    pub fn forward_with_dispatch(
        &mut self,
        _hidden: &[f32],
        _start_pos: usize,
    ) -> crate::model::Result<Vec<f32>> {
        todo!("CPU-only model loading not yet implemented")
    }

    pub fn embed(&self, _token: u32, _seq_len: usize) -> crate::model::Result<Vec<f32>> {
        todo!("CPU-only model loading not yet implemented")
    }

    pub fn forward_layers(
        &self,
        _hidden: &[f32],
        _start_pos: usize,
    ) -> crate::model::Result<Vec<f32>> {
        todo!("CPU-only model loading not yet implemented")
    }

    pub fn apply_output_head(&self, _hidden: &[f32]) -> crate::model::Result<Vec<f32>> {
        todo!("CPU-only model loading not yet implemented")
    }
}
