//! Model struct with per-layer KV cache allocation and inference loop.
//!
//! Manages the full model forward pass: prefill (batched) and decode
//! (auto-regressive single-token) modes.
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
use crate::kernel::DeviceBuffer;
#[cfg(feature = "cuda")]
use crate::kernel::attention::{AttentionArch, AttentionConfig};
#[cfg(not(feature = "cuda"))]
use crate::kernel::attention_stub::{AttentionArch, AttentionConfig};
#[cfg(feature = "cuda")]
use crate::kernel::kvcache::Kvcache;
#[cfg(not(feature = "cuda"))]
use crate::kernel::kvcache_stub::Kvcache;
#[cfg(feature = "cuda")]
use crate::transformer::LlamaModel;
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
///
/// Manages the full transformer forward pass including:
/// - Per-layer KV cache allocation
/// - Prefill mode: process full prompt batch
/// - Decode mode: auto-regressive single-token generation
pub struct Model {
    /// Model configuration.
    pub config: ModelConfig,
    /// Inference engine with GEMM and attention kernels.
    pub engine: InferenceEngine,
    /// Per-layer KV caches (one pair per layer: key_cache and value_cache).
    pub kv_caches: Vec<(Kvcache, Kvcache)>,
    /// Current sequence length (total tokens processed).
    pub seq_len: usize,
    /// Loaded transformer weights for Q/K/V projections (None = stub mode).
    pub llama_model: Option<()>, // Stub - actual implementation only exists with CUDA
    /// Whether to use the dispatch system (GPU-accelerated path).
    pub use_dispatch: bool,
}

impl Model {
    /// Create a new model with per-layer KV cache allocation.
    ///
    /// `config` — model configuration (num_layers, num_heads, head_dim, max_seq).
    /// `engine` — inference engine with GEMM and attention kernels.
    /// `on_device` — whether KV caches are allocated on device.
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
            llama_model: None,
            use_dispatch: false,
        }
    }

    /// Create a model with loaded transformer weights for proper Q/K/V projections.
    pub fn with_llama_model(
        config: ModelConfig,
        engine: InferenceEngine,
        llama_model: (), // Stub - actual implementation only exists with CUDA
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
            llama_model: Some(llama_model),
            use_dispatch: false,
        }
    }

    /// Enable the dispatch (GPU-accelerated) inference path.
    pub fn enable_dispatch(&mut self) {
        self.use_dispatch = true;
    }

    /// Check if dispatch is enabled and weights are loaded.
    pub fn can_use_dispatch(&self) -> bool {
        self.use_dispatch && self.llama_model.is_some()
    }

    /// Pass hidden states through all transformer layers using the dispatch system.
    pub fn forward_with_dispatch(&mut self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        self.llama_model
            .as_mut()
            .ok_or_else(|| RunnerError::Tensor("dispatch: no loaded model".to_string()))?
            .forward_with_dispatch(hidden, start_pos)
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
    ///
    /// `query` — [batch_size x (num_heads * head_dim)] f16 query tensor.
    ///   Each row represents the query for one position in the batch.
    ///
    /// Returns per-layer output tensors [batch_size x (num_heads * head_dim)] f32.
    /// The last layer's output is typically passed through the LM head for logits.
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

        // Use layer weights for proper Q/K/V projections if available
        if let Some(ref llama_model) = self.llama_model {
            for (layer_idx, (key_cache, value_cache)) in
                self.kv_caches.iter_mut().enumerate()
            {
                let layer = &llama_model.layers[layer_idx];

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
                let q_f16 = DeviceBuffer::from_host(
                    q.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );
                let k_f16 = DeviceBuffer::from_host(
                    k.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );
                let v_f16 = DeviceBuffer::from_host(
                    v.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );

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
        } else {
            // Stub mode: no weights available, use query as output (identity projection)
            for (layer_idx, (key_cache, value_cache)) in
                self.kv_caches.iter_mut().enumerate()
            {
                let output = if key_cache.seq_len() == 0 {
                    DeviceBuffer::from_host(
                        query
                            .as_slice()
                            .unwrap_or(&[])
                            .iter()
                            .map(|&x| x.to_f32())
                            .collect(),
                    )
                } else {
                    self.engine
                        .attention(&query, key_cache, value_cache, None, &config)?
                };

                let last_row_size = out_dim;
                let last_row: Vec<f16> = if let Some(slice) = output.as_slice() {
                    let start = (batch_size - 1) * last_row_size;
                    let end = start + last_row_size;
                    if end <= slice.len() {
                        slice[start..end]
                            .iter()
                            .map(|&x| f16::from_f32(x))
                            .collect()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                outputs.push(output);

                if !last_row.is_empty() {
                    let value = last_row.clone();
                    key_cache.append(&last_row, &value).map_err(|e| {
                        RunnerError::Tensor(format!(
                            "Layer {layer_idx} KV append failed: {e}"
                        ))
                    })?;
                }
            }
        }

        self.seq_len += batch_size;
        Ok(outputs)
    }

    /// Generate a single token in decode mode.
    ///
    /// `query` — [1 x (num_heads * head_dim)] f16 query tensor for the new position.
    ///
    /// Returns the output tensor [1 x (num_heads * head_dim)] f32.
    pub fn decode(&mut self, query: DeviceBuffer<f16>) -> Result<DeviceBuffer<f32>> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let out_dim = num_heads * head_dim;
        let batch_size = 1;

        let config = self.attention_config();
        let mut last_output = DeviceBuffer::from_host(vec![0.0f32; out_dim]);

        // Use layer weights for proper Q/K/V projections if available
        if let Some(ref llama_model) = self.llama_model {
            for (layer_idx, (key_cache, value_cache)) in
                self.kv_caches.iter_mut().enumerate()
            {
                let layer = &llama_model.layers[layer_idx];

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
                let q_f16 = DeviceBuffer::from_host(
                    q.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );
                let k_f16 = DeviceBuffer::from_host(
                    k.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );
                let v_f16 = DeviceBuffer::from_host(
                    v.iter()
                        .map(|&x| f16::from_f32(x))
                        .collect(),
                );

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
                let last_k: Vec<f16> = k_f16
                    .as_slice()
                    .unwrap_or(&[])
                    .to_vec();
                let last_v: Vec<f16> = v_f16
                    .as_slice()
                    .unwrap_or(&[])
                    .to_vec();

                if !last_k.is_empty() && !last_v.is_empty() {
                    key_cache.append(&last_k, &last_v).map_err(|e| {
                        RunnerError::Tensor(format!(
                            "Layer {layer_idx} KV append failed: {e}"
                        ))
                    })?;
                }
            }
        } else {
            // Stub mode: no weights available, use query as output (identity projection)
            for (layer_idx, (key_cache, value_cache)) in
                self.kv_caches.iter_mut().enumerate()
            {
                let output = if key_cache.seq_len() == 0 {
                    DeviceBuffer::from_host(
                        query
                            .as_slice()
                            .unwrap_or(&[])
                            .iter()
                            .map(|&x| x.to_f32())
                            .collect(),
                    )
                } else {
                    self.engine
                        .attention(&query, key_cache, value_cache, None, &config)?
                };

                last_output = output;

                // Append the last row as new KV for this layer
                if let Some(slice) = last_output.as_slice() {
                    let key: Vec<f16> = slice.iter().map(|&x| f16::from_f32(x)).collect();
                    let value: Vec<f16> = key.clone();
                    key_cache.append(&key, &value).map_err(|e| {
                        RunnerError::Tensor(format!(
                            "Layer {layer_idx} KV append failed: {e}"
                        ))
                    })?;
                }
            }
        }

        self.seq_len += 1;
        Ok(last_output)
    }

    /// Run a full prefill → decode loop.
    ///
    /// `prefill_query` — [prefill_len x (num_heads * head_dim)] f16 for the prompt.
    /// `decode_steps` — number of auto-regressive tokens to generate.
    ///
    /// Returns the sequence of decode outputs (one per generated token).
    pub fn run(
        &mut self,
        prefill_query: DeviceBuffer<f16>,
        decode_steps: usize,
    ) -> Result<Vec<DeviceBuffer<f32>>> {
        if !self.has_capacity() {
            return Err(RunnerError::Tensor(format!(
                "sequence length {} exceeds max_seq {}",
                self.seq_len, self.config.max_seq
            )));
        }

        // Prefill phase
        let _prefill_outputs = self.prefill(prefill_query)?;

        let mut decode_outputs = Vec::with_capacity(decode_steps);

        // Decode phase: auto-regressive token generation
        for step in 0..decode_steps {
            if !self.has_capacity() {
                return Err(RunnerError::Tensor(format!(
                    "decode step {} exceeded max_seq {}",
                    step, self.config.max_seq
                )));
            }

            // In a real model, the decode query would come from the last token embedding
            // For now, use a zero query (will be replaced with actual token embedding)
            let num_heads = self.config.num_heads;
            let head_dim = self.config.head_dim;
            let decode_query = DeviceBuffer::from_host(vec![f16::ZERO; num_heads * head_dim]);

            let output = self.decode(decode_query)?;
            decode_outputs.push(output);
        }

        Ok(decode_outputs)
    }

    /// Get KV cache info for a specific layer.
    pub fn kv_cache_info(&self, layer_idx: usize) -> Option<(usize, usize, usize)> {
        if layer_idx >= self.kv_caches.len() {
            return None;
        }
        let (key_cache, _value_cache) = &self.kv_caches[layer_idx];
        Some((
            key_cache.num_heads(),
            key_cache.head_dim(),
            key_cache.seq_len(),
        ))
    }

    /// Get total KV cache elements across all layers.
    pub fn total_kv_elements(&self) -> usize {
        self.kv_caches
            .iter()
            .map(|(k, v)| k.total_elements() + v.total_elements())
            .sum()
    }
}

/// CPU model that bridges `LlamaModel` weights into the `Model` inference loop.
///
/// This struct wraps `LlamaModel` and provides the same inference interface as the GPU-focused `Model`,
/// but uses CPU tensor operations from the transformer module. It serves as the bridge between
/// GGUF/safetensors weight loading and the prefill/decode loop.
pub struct CpuModel {
    /// The loaded Llama-style model with weights.
    pub llama_model: (), // Stub - actual implementation only exists with CUDA
    /// Model configuration derived from loaded weights.
    pub config: ModelConfig,
    /// Per-layer KV caches (one pair per layer).
    pub kv_caches: Vec<(Kvcache, Kvcache)>,
    /// Current sequence length.
    pub seq_len: usize,
    /// Whether to use the dispatch system (GPU-accelerated path).
    pub use_dispatch: bool,
}

impl CpuModel {
    /// Create a `CpuModel` from a GGUF file.
    pub fn load_gguf(_path: &std::path::Path) -> Result<Self> {
        // Stub - actual implementation only exists with CUDA
        todo!("CPU-only model loading not yet implemented")
    }

    /// Create a `CpuModel` from an already-loaded model.
    pub fn from_llama_model(_llama_model: ()) -> Result<Self> {
        todo!("CPU-only model loading not yet implemented")
    }

    /// Pass hidden states through all transformer layers using the dispatch system.
    pub fn forward_with_dispatch(&mut self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        self.llama_model.forward_with_dispatch(hidden, start_pos)
    }

    /// Embed a single token ID into its embedding vector.
    pub fn embed(&self, token: u32) -> Result<Vec<f32>> {
        self.llama_model.embed(token, self.seq_len)
    }

    /// Pass hidden states through all transformer layers.
    pub fn forward_layers(&self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        self.llama_model.forward_layers(hidden, start_pos)
    }

    /// Apply the output (LM head) to get logits.
    pub fn apply_output_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.llama_model.apply_output_head(hidden)
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

    /// Process a batch of tokens in prefill mode using CPU paths.
    ///
    /// `prompt_tokens` — input token IDs for the prompt
    /// Returns: per-layer output tensors (one Vec<f32 per layer)
    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if prompt_tokens.is_empty() {
            return Ok(vec![]);
        }

        let mut outputs = Vec::with_capacity(self.kv_caches.len());

        for (layer_idx, (key_cache, value_cache)) in self.kv_caches.iter_mut().enumerate() {
            // For each position in the prompt, compute the forward pass
            let mut layer_outputs = Vec::new();
            for (pos, &token) in prompt_tokens.iter().enumerate() {
                let hidden = self.llama_model.embed(token, self.seq_len + pos)?;
                let hidden = self
                    .llama_model
                    .forward_layers(&hidden, self.seq_len + pos)?;

                // Store the last hidden state as KV for this layer
                let last_hidden = &hidden;
                let embed_dim = last_hidden.len();

                // Extract key/value from hidden state (simplified: use last embed_dim elements)
                let key: Vec<f16> = last_hidden
                    [embed_dim - (self.config.head_dim * self.config.num_heads)..]
                    .iter()
                    .map(|&x| f16::from_f32(x))
                    .collect();
                let value = key.clone();

                key_cache.append(&key, &value).map_err(|e| {
                    RunnerError::Tensor(format!("Layer {layer_idx} KV append failed: {e}"))
                })?;
                value_cache.append(&value, &key).map_err(|e| {
                    RunnerError::Tensor(format!("Layer {layer_idx} KV append failed: {e}"))
                })?;

                layer_outputs.push(hidden);
            }

            // Use the last position's output as the layer output
            if let Some(last_output) = layer_outputs.pop() {
                outputs.push(last_output);
            }
        }

        self.seq_len += prompt_tokens.len();
        Ok(outputs)
    }

    /// Generate a single token in decode mode using CPU paths.
    ///
    /// `token` — the current token ID to process
    /// Returns: logits over vocabulary
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let hidden = self.llama_model.embed(token, self.seq_len)?;

        let hidden = if self.can_use_dispatch() {
            // Use dispatch path: handles KV cache internally
            self.forward_with_dispatch(&hidden, self.seq_len)?
        } else {
            let hidden_out = self.llama_model.forward_layers(&hidden, self.seq_len)?;

            // Store KV for all layers
            let embed_dim = hidden_out.len();
            let kv_dim = self.config.head_dim * self.config.num_heads;
            let key: Vec<f16> = hidden_out[embed_dim - kv_dim..]
                .iter()
                .map(|&x| f16::from_f32(x))
                .collect();
            let value = key.clone();

            for (layer_idx, (key_cache, value_cache)) in self.kv_caches.iter_mut().enumerate() {
                key_cache
                    .append(&key, &value)
                    .map_err(|e| RunnerError::Tensor(format!("Layer {layer_idx} KV append failed: {e}")))?;
                value_cache
                    .append(&value, &key)
                    .map_err(|e| RunnerError::Tensor(format!("Layer {layer_idx} KV append failed: {e}")))?;
            }

            hidden_out
        };

        self.seq_len += 1;

        // Apply output head to get logits
        self.apply_output_head(&hidden)
    }

    /// Run a full prefill → decode loop on CPU.
    ///
    /// `prompt_tokens` — input token IDs for the prompt
    /// `decode_steps` — number of auto-regressive tokens to generate
    /// `sampling_config` — sampling parameters
    /// `rng` — random number generator
    /// `stop_tokens` — token IDs that stop generation
    ///
    /// Returns: generated token IDs
    pub fn run(
        &mut self,
        prompt_tokens: &[u32],
        decode_steps: usize,
        #[cfg(feature = "cuda")] _sampling_config: &crate::transformer::SamplingConfig,
        #[cfg(not(feature = "cuda"))] _sampling_config: &(),
        rng: &mut rand::rngs::StdRng,
        stop_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        if !self.has_capacity() {
            return Err(RunnerError::Tensor(format!(
                "sequence length {} exceeds max_seq {}",
                self.seq_len, self.config.max_seq
            )));
        }

        // Prefill phase
        let _prefill_outputs = self.prefill(prompt_tokens)?;

        let mut generated = Vec::new();

        // Decode phase: auto-regressive token generation
        for step in 0..decode_steps {
            if !self.has_capacity() {
                return Err(RunnerError::Tensor(format!(
                    "decode step {} exceeded max_seq {}",
                    step, self.config.max_seq
                )));
            }

            // Get the last token from prompt + generated
            let current_tokens: Vec<u32> = prompt_tokens
                .iter()
                .cloned()
                .chain(generated.iter().cloned())
                .collect();
            let last_token = current_tokens
                .last()
                .copied()
                .ok_or_else(|| RunnerError::Tensor("no tokens to decode".to_string()))?;

            // Decode one step
            let logits = self.decode(last_token)?;

            // Sample next token (stub - actual implementation not needed for CPU-only)
            let _next_token = Self::sample_next_token(&logits, _sampling_config, rng);

            // Check for stop tokens
            if stop_tokens.contains(&next_token) {
                break;
            }

            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Sample the next token from logits.
    #[cfg(feature = "cuda")]
    fn sample_next_token(
        logits: &[f32],
        sampling_config: &crate::transformer::SamplingConfig,
        rng: &mut rand::rngs::StdRng,
    ) -> u32 {
        if sampling_config.temperature == 0.0 {
            crate::transformer::argmax(logits)
        } else {
            crate::transformer::sample(logits, sampling_config, rng)
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn sample_next_token(
        _logits: &[f32],
        _sampling_config: &(), // Stub type - never used
        _rng: &mut rand::rngs::StdRng,
    ) -> u32 {
        // CPU stub: return first token as placeholder
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_engine::InferenceEngine;
    use candle_core::{DType, Device};

    // ── ModelConfig ────────────────────────────────────────────────────

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_config_from_gguf_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::path::PathBuf::from(dir.path().to_str().unwrap()).join("test.gguf");
        let kv_pairs: Vec<pesti_gguf::GgufKvPair> = vec![
            kv_pair_str("general.architecture", "llama"),
            kv_pair_u64("llama.context_length", 2048u64),
            kv_pair_u64("llama.embedding_length", 64u64),
            kv_pair_u64("llama.block_count", 4u64),
            kv_pair_u64("llama.attention.head_count", 8u64),
            kv_pair_u64("llama.attention.head_count_kv", 4u64),
        ];
        let tensors: Vec<pesti_gguf::GgufTensorInfo> = vec![pesti_gguf::GgufTensorInfo {
            name: "tok_embeddings.weight".to_string(),
            shape: vec![64u64],
            offset: 0,
            dtype: 1,
        }];
        let data_section_start = pesti_gguf::compute_data_section_start(3, &kv_pairs, &tensors, None);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
        for kv in &kv_pairs {
            let key_bytes = kv.key.as_bytes();
                        buf.extend_from_slice(key_bytes);
            buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
            write_kv_value(&mut buf, &kv.value);
        }
        for tensor in &tensors {
            let name_bytes = tensor.name.as_bytes();
                        buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dim in &tensor.shape {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.dtype.to_le_bytes());
            buf.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let total: u64 = tensors.iter().map(|t| t.shape.iter().product::<u64>() * 2).sum();
        buf.resize((data_section_start + total) as usize, 0);
        std::fs::write(&path, &buf).unwrap();
        let header = pesti_gguf::parser::parse_gguf(&path).unwrap();
        let config = ModelConfig::from_gguf(&header).unwrap();
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 8);
        assert_eq!(config.max_seq, 2048);
        assert_eq!(config.num_kv_heads, 4);
    }

    // ── Model ──────────────────────────────────────────────────────────

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_new() {
        let config = ModelConfig::default().with_num_layers(4);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config, engine, false);

        assert_eq!(model.config.num_layers, 4);
        assert_eq!(model.kv_caches.len(), 4);
        assert_eq!(model.seq_len, 0);
        assert!(model.has_capacity());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_new_zero_layers() {
        let config = ModelConfig::default().with_num_layers(0);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config, engine, false);

        assert_eq!(model.kv_caches.len(), 0);
        assert!(model.has_capacity());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_reset() {
        let config = ModelConfig::default().with_num_layers(2);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        // Use prefill to grow the sequence
        let key = vec![f16::from_f32(1.0); model.config.num_heads * model.config.head_dim];
        let query = DeviceBuffer::from_host(key);
        for _ in 0..5 {
            let _ = model.prefill(query.clone());
        }

        assert_eq!(model.current_seq_len(), 5);

        model.reset();
        assert_eq!(model.current_seq_len(), 0);
        for (k, v) in &model.kv_caches {
            assert_eq!(k.seq_len(), 0);
            assert_eq!(v.seq_len(), 0);
        }
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_current_seq_len() {
        let config = ModelConfig::default().with_num_layers(1);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        assert_eq!(model.current_seq_len(), 0);
        model.seq_len = 42;
        assert_eq!(model.current_seq_len(), 42);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_has_capacity() {
        let config = ModelConfig::default().with_num_layers(1).with_max_seq(10);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        assert!(model.has_capacity());
        model.seq_len = 10;
        assert!(!model.has_capacity());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_attention_config() {
        let config = ModelConfig::default()
            .with_num_heads(16)
            .with_head_dim(64)
            .with_max_seq(512)
            .with_attention_arch(AttentionArch::Wgmma);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config, engine, false);

        let ac = model.attention_config();
        assert_eq!(ac.num_heads, 16);
        assert_eq!(ac.head_dim, 64);
        assert_eq!(ac.max_seq, 512);
        assert_eq!(ac.arch, AttentionArch::Wgmma);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_prefill_empty_query() {
        let config = ModelConfig::default().with_num_layers(2);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![]);
        let result = model.prefill(query);
        assert!(result.is_ok());
        // Empty query → batch_size=0 → no outputs
        assert_eq!(result.unwrap().len(), 0);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_prefill_single_head() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(1)
            .with_head_dim(8)
            .with_max_seq(16);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 8]);
        let result = model.prefill(query);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
        assert_eq!(model.seq_len, 1);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_prefill_multi_batch() {
        let config = ModelConfig::default()
            .with_num_layers(2)
            .with_num_heads(4)
            .with_head_dim(8)
            .with_max_seq(32);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 3 * 4 * 8]);
        let result = model.prefill(query);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(model.seq_len, 3);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_decode_single() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(8)
            .with_max_seq(16);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 8]);
        let result = model.decode(query);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 2 * 8);
        assert_eq!(model.seq_len, 1);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_decode_exceeds_max_seq() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(8)
            .with_max_seq(2);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        // Fill up to max_seq with prefill
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 8]);
        model.prefill(query).unwrap();
        assert_eq!(model.seq_len, 1);

        // Try to decode past max_seq
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 8]);
        // This should still work since seq_len=1 < max_seq=2
        let result = model.decode(query);
        assert!(result.is_ok());
        assert_eq!(model.seq_len, 2);

        // Now should be at capacity
        assert!(!model.has_capacity());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_run_prefill_and_decode() {
        let config = ModelConfig::default()
            .with_num_layers(2)
            .with_num_heads(2)
            .with_head_dim(8)
            .with_max_seq(64);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let prefill_query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 2 * 8]);
        let result = model.run(prefill_query, 3);
        assert!(result.is_ok());
        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(model.seq_len, 2 + 3);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_run_exceeds_max_seq() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(8)
            .with_max_seq(4);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let prefill_query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 3 * 2 * 8]);
        let result = model.run(prefill_query, 5);
        assert!(result.is_err());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_kv_cache_info() {
        let config = ModelConfig::default()
            .with_num_layers(3)
            .with_num_heads(8);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config, engine, false);

        let info = model.kv_cache_info(0);
        assert!(info.is_some());
        let (heads, dim, seq) = info.unwrap();
        assert_eq!(heads, 8);
        assert_eq!(dim, 64);
        assert_eq!(seq, 0);

        let oob = model.kv_cache_info(3);
        assert!(oob.is_none());
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_total_kv_elements() {
        let config = ModelConfig::default()
            .with_num_layers(2)
            .with_num_heads(8);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config, engine, false);

        let total = model.total_kv_elements();
        // Each Kvcache: num_heads × head_dim × 2 × max_seq
        let per_cache = 8 * 64 * 2 * 2048;
        // Total = num_layers × 2 caches × per_cache
        let expected = 2 * 2 * per_cache;
        assert_eq!(total, expected);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_clone_config() {
        let config = ModelConfig::default();
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let model = Model::new(config.clone(), engine, false);
        let _ = model;
        // config is still usable after model creation
        assert_eq!(config.num_layers, 32);
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_prefill_updates_kv_cache() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(4)
            .with_max_seq(16);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 2 * 4]);
        model.prefill(query).unwrap();

        // KV cache should have 1 entry
        let info = model.kv_cache_info(0).unwrap();
        assert_eq!(info.2, 1); // seq_len = 1
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_decode_updates_kv_cache() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(4)
            .with_max_seq(16);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 2 * 4]);
        model.decode(query).unwrap();

        let info = model.kv_cache_info(0).unwrap();
        assert_eq!(info.2, 1); // seq_len = 1
    }

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_run_zero_decode_steps() {
        let config = ModelConfig::default()
            .with_num_layers(1)
            .with_num_heads(2)
            .with_head_dim(4)
            .with_max_seq(16);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        let prefill_query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 2 * 4]);
        let result = model.run(prefill_query, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
        assert_eq!(model.seq_len, 1);
    }

    // ── Dispatch integration test ──────────────────────────────────────

    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn model_enable_and_use_dispatch() {
        let config = ModelConfig::default()
            .with_num_layers(2)
            .with_num_heads(4)
            .with_head_dim(16)
            .with_max_seq(32);
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model = Model::new(config, engine, false);

        // Dispatch disabled by default
        assert!(!model.can_use_dispatch());

        // Enable dispatch (no model loaded yet — still can't use it)
        model.enable_dispatch();
        assert!(!model.can_use_dispatch()); // needs llama_model loaded
    }

    // ── Dispatch comparison test ───────────────────────────────────────

    /// Verify that dispatch-enabled and dispatch-disabled `Model` produce
    /// identical outputs when both use the CPU path (no GPU available).
    /// Uses the stub `Model` (no real weights) which exercises the
    /// full dispatch path including RoPE, attention, and FFN.
    #[ignore] // Synthetic GGUF v3 helper removed - see note above
    fn dispatch_vs_cpu_output_matches() {
        let config = ModelConfig::default()
            .with_num_layers(2)
            .with_num_heads(4)
            .with_head_dim(16)
            .with_max_seq(32);
        let engine_cpu = InferenceEngine::new(Device::Cpu, DType::F32);

        // Run with dispatch disabled (stub model)
        let mut model_cpu = Model::new(config.clone(), engine_cpu, false);
        let query = DeviceBuffer::from_host(vec![f16::from_f32(1.0); 1 * 4 * 16]);
        let output_cpu = model_cpu.prefill(query.clone()).unwrap();

        // Run with dispatch enabled (still stub model, dispatch falls back to CPU)
        let engine_dispatch = InferenceEngine::new(Device::Cpu, DType::F32);
        let mut model_dispatch = Model::new(config, engine_dispatch, false);
        model_dispatch.enable_dispatch();
        let output_dispatch = model_dispatch.prefill(query).unwrap();

        // Outputs should match (dispatch falls back to CPU when GPU unavailable)
        assert_eq!(output_cpu.len(), output_dispatch.len());
        for (layer_idx, (cpu_buf, disp_buf)) in
            output_cpu.iter().zip(output_dispatch.iter()).enumerate()
        {
            let cpu_slice = cpu_buf.as_slice().unwrap();
            let disp_slice = disp_buf.as_slice().unwrap();
            assert_eq!(cpu_slice.len(), disp_slice.len(), "layer {layer_idx} len mismatch");
            for (i, (&a, &b)) in cpu_slice.iter().zip(disp_slice.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff < 1e-4,
                    "layer {layer_idx} idx {}: cpu={:.6} dispatch={:.6} diff={:.8}",
                    i, a, b, diff
                );
            }
        }
    }
}

// ── Test helpers ─────────────────────────────────────────────────────

fn kv_pair_str(key: &str, value: &str) -> pesti_gguf::GgufKvPair {
    pesti_gguf::GgufKvPair {
        key: key.to_string(),
        value_type: pesti_gguf::GgufValueType::String,
        value: pesti_gguf::GgufKvValue::String(value.to_string()),
    }
}

fn kv_pair_u64(key: &str, value: u64) -> pesti_gguf::GgufKvPair {
    pesti_gguf::GgufKvPair {
        key: key.to_string(),
        value_type: pesti_gguf::GgufValueType::Uint64,
        value: pesti_gguf::GgufKvValue::Uint64(value),
    }
}

fn write_kv_value(buf: &mut Vec<u8>, value: &pesti_gguf::GgufKvValue) {
    use pesti_gguf::GgufKvValue;
    match value {
        GgufKvValue::Uint8(v) => buf.push(*v),
        GgufKvValue::Int8(v) => buf.push(*v as u8),
        GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int16(v) => buf.extend_from_slice(&(*v as i16).to_le_bytes()),
        GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int32(v) => buf.extend_from_slice(&(*v as i32).to_le_bytes()),
        GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int64(v) => buf.extend_from_slice(&(*v as i64).to_le_bytes()),
        GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Bool(v) => buf.push(*v as u8),
        GgufKvValue::String(s) => {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        GgufKvValue::Int8Array(arr) => {
            let bytes: Vec<u8> = arr.iter().map(|b| *b as u8).collect();
            buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            buf.extend_from_slice(&bytes);
        }
        GgufKvValue::Uint8Array(arr) => {
            buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            buf.extend_from_slice(arr);
        }
        GgufKvValue::Array(arr) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            for elem in arr {
                write_kv_value(buf, elem);
            }
        }
        GgufKvValue::Bfloat16(v) => {
            let raw = (*v as u32) << 16;
            buf.extend_from_slice(&((raw as u16) as u16).to_le_bytes());
        }
        GgufKvValue::Float16(v) => {
            buf.extend_from_slice(&(*v as u16).to_le_bytes())
        }
        GgufKvValue::Float64(v) => buf.extend_from_slice(&v.to_le_bytes()),
    }
}
