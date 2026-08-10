//! Llama-style model: loads GGUF weights, wires transformer layers.
//!
//! Supports the llama architecture family (llama, mistral, phi3, etc.)
//! with standard tensor naming conventions.

use std::path::Path;

use pesti_gguf::types::GgufHeader;
use tracing::debug;

use crate::error::{Result, RunnerError};
use crate::gguf_weight_loader::{GgufWeights, load_gguf_weights};
use crate::kernel::dispatch::DispatchContext;
use crate::kernel::kvcache::Kvcache;
use crate::model_loader::GgufHeaderExt;
use crate::safetensors_weight_loader::SafetensorsWeights;
use crate::transformer::GgufTokenizer;
use crate::transformer::layer::{Attention, FeedForward, TransformerLayer};
use crate::transformer::linear::Linear;
use crate::transformer::rms_norm::RmsNorm;
use crate::transformer::rope::RopeConfig;
use crate::transformer::tokenizer::{GgufTokenizerConfig, load_tokenizer_from_gguf};

/// Model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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

/// Llama-style model configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlamaConfig {
    pub arch: ModelArch,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_type: Option<String>,
    pub rms_norm_eps: f32,
}

impl LlamaConfig {
    /// Build config from a GGUF header.
    pub fn from_gguf_header(header: &GgufHeader) -> Result<Self> {
        let arch_str = header.architecture().unwrap_or("llama");
        let arch = match arch_str {
            "gemma" => ModelArch::Gemma,
            "qwen2" => ModelArch::Qwen2,
            "qwen3" => ModelArch::Qwen3,
            "phi3" => ModelArch::Phi3,
            "mixtral" => ModelArch::Mixtral,
            "starcoder2" => ModelArch::Starcoder2,
            _ => ModelArch::Llama,
        };

        let embed_dim = header
            .embedding_length()
            .ok_or_else(|| RunnerError::MissingHeaderField("embedding_length".to_string()))?
            as usize;
        let num_heads = header.get_kv_u32("attention.head_count").unwrap_or(32) as usize;

        let num_kv_heads = match arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => header
                .get_kv_u32(&format!("{arch_str}.num_key_value_heads"))
                .unwrap_or(8) as usize,
            _ => header.attention_head_count_kv().unwrap_or(num_heads as u32) as usize,
        };

        let num_layers = header.block_count().unwrap_or(32) as usize;
        let mut head_dim = if num_heads > 0 {
            embed_dim / num_heads
        } else {
            64
        };
        // For GQA models (num_kv_heads < num_heads), the correct head_dim
        // comes from the KV weight shape, not embed_dim / num_heads.
        // Try to infer from K weight tensor: shape is [embed_dim, kv_dim].
        if num_kv_heads < num_heads {
            let k_name = match arch {
                ModelArch::Qwen2 | ModelArch::Qwen3 => "blk.0.attn_k.weight".to_string(),
                _ => format!("layers.0.attention.wk.weight"),
            };
            // Use gguf_model_loader's get_tensor_byte_range helper
            if let Some(tensor_info) = header.tensors.iter().find(|t| t.name == k_name) {
                if tensor_info.shape.len() >= 2 {
                    let kv_dim = tensor_info.shape[1] as usize;
                    let inferred = kv_dim / num_kv_heads;
                    if inferred > 0 && inferred != head_dim {
                        tracing::info!(
                            head_dim_before = head_dim,
                            head_dim_after = inferred,
                            kv_dim,
                            num_kv_heads,
                            "Corrected head_dim from KV weight shape"
                        );
                        head_dim = inferred;
                    }
                }
            }
        }
        let intermediate_dim = match arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => header
                .get_kv_u32(&format!("{arch_str}.feed_forward_length"))
                .unwrap_or(11008) as usize,
            _ => header
                .get_kv_u32("llama.feed_forward_length")
                .or_else(|| header.get_kv_u32("general.architecture.ffn_size"))
                .unwrap_or(11008) as usize,
        };
        let max_seq_len = header.context_length().unwrap_or(4096) as usize;
        let rope_base = 10000.0;
        let rope_dim = header.rope_dimension_count().unwrap_or(head_dim as i32) as usize;
        let rms_norm_eps = header.normalization_epsilon().unwrap_or(1e-5);

        let actual_head_dim = if rope_dim > 0 { rope_dim } else { head_dim };

        Ok(Self {
            arch,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim: actual_head_dim,
            embed_dim,
            intermediate_dim,
            max_seq_len,
            rope_base,
            rope_scaling_factor: None,
            rope_scaling_type: None,
            rms_norm_eps,
        })
    }

    /// Build config from safetensors metadata (the HashMap stored in the file header).
    ///
    /// Safetensors files embed a JSON metadata object. Common keys include:
    /// - `model_type` / `architectures` — architecture name
    /// - `hidden_size` / `dim` — embedding dimension
    /// - `num_hidden_layers` / `n_layers` — number of layers
    /// - `num_attention_heads` / `n_heads` — number of heads
    /// - `num_key_value_heads` — KV heads (optional)
    /// - `intermediate_size` / `ffn_dim` — FFN intermediate dim
    /// - `max_position_embeddings` / `context_length` — max seq len
    /// - `rope_theta` — RoPE base
    /// - `rms_norm_eps` / `layer_norm_epsilon` — normalization epsilon
    pub fn from_safetensors_metadata(
        meta: &std::collections::HashMap<String, String>,
    ) -> Result<Self> {
        // Architecture
        let arch = meta
            .get("model_type")
            .or(meta.get("architectures"))
            .map(|s| s.trim_matches('"').to_lowercase())
            .and_then(|s| match s.as_str() {
                "gemma" | "google/gemma" => Some(ModelArch::Gemma),
                "qwen2" | "qwen2vl" => Some(ModelArch::Qwen2),
                "qwen3" => Some(ModelArch::Qwen3),
                "phi3" | "microsoft/phi-3" => Some(ModelArch::Phi3),
                "mixtral" | "mistral" | "mistralai" => Some(ModelArch::Mixtral),
                "starcoder2" => Some(ModelArch::Starcoder2),
                _ => None,
            })
            .unwrap_or(ModelArch::Llama);

        // Helper: get a u64 from metadata, trying multiple key names
        let get_u64 = |keys: &[&str]| -> Option<u64> {
            for &k in keys {
                if let Some(v) = meta.get(k) {
                    if let Ok(n) = v.trim_matches('"').parse::<u64>() {
                        return Some(n);
                    }
                }
            }
            None
        };

        let get_f32 = |keys: &[&str]| -> Option<f32> {
            for &k in keys {
                if let Some(v) = meta.get(k) {
                    if let Ok(n) = v.trim_matches('"').parse::<f32>() {
                        return Some(n);
                    }
                }
            }
            None
        };

        let embed_dim = get_u64(&["hidden_size", "dim", "d_model"])
            .map(|v| v as usize)
            .ok_or_else(|| {
                RunnerError::ModelLoad("safetensors metadata missing hidden_size/dim".to_string())
            })?;

        let num_heads = get_u64(&["num_attention_heads", "n_heads", "num_heads"])
            .map(|v| v as usize)
            .unwrap_or(32);
        let num_kv_heads = get_u64(&["num_key_value_heads"])
            .map(|v| v as usize)
            .unwrap_or(num_heads);
        let num_layers = get_u64(&["num_hidden_layers", "n_layers", "num_layers"])
            .map(|v| v as usize)
            .unwrap_or(32);
        let head_dim = if num_heads > 0 {
            embed_dim / num_heads
        } else {
            64
        };
        let intermediate_dim = get_u64(&["intermediate_size", "ffn_dim", "feed_forward_length"])
            .map(|v| v as usize)
            .unwrap_or(11008);
        let max_seq_len = get_u64(&["max_position_embeddings", "context_length", "seq_length"])
            .map(|v| v as usize)
            .unwrap_or(4096);
        let rope_base = get_f32(&["rope_theta", "rope_scaling_factor"]).unwrap_or(10000.0);
        let rms_norm_eps =
            get_f32(&["rms_norm_eps", "layer_norm_epsilon", "layer_norm_epsilon"]).unwrap_or(1e-5);

        // Try to get rope dimension from metadata
        let rope_dim = get_u64(&[
            "rope_dim",
            "rope_dimension_count",
            "rope_scaling_rope_dimension",
        ])
        .map(|v| v as usize)
        .unwrap_or(head_dim);
        let actual_head_dim = if rope_dim > 0 { rope_dim } else { head_dim };

        // Rope scaling
        let rope_scaling = meta.get("rope_scaling");
        let rope_scaling_factor = rope_scaling
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s.trim_matches('"')).ok())
            .and_then(|v| v.get("factor").and_then(|fv| fv.as_f64()).map(|f| f as f32));
        let rope_scaling_type = rope_scaling
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s.trim_matches('"')).ok())
            .and_then(|v| {
                v.get("type")
                    .and_then(|tv| tv.as_str())
                    .map(|s| s.to_string())
            });

        Ok(Self {
            arch,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim: actual_head_dim,
            embed_dim,
            intermediate_dim,
            max_seq_len,
            rope_base,
            rope_scaling_factor,
            rope_scaling_type,
            rms_norm_eps,
        })
    }

    /// Get the layer prefix for this architecture.
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        match self.arch {
            ModelArch::Gemma => format!("model.layers.{layer_idx}."),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("blk.{layer_idx}."),
            _ => format!("layers.{layer_idx}."),
        }
    }

    /// Check if this architecture uses `gate_proj` / `up_proj` / `down_proj` naming.
    pub fn uses_gate_up_down(&self) -> bool {
        matches!(self.arch, ModelArch::Qwen2 | ModelArch::Qwen3)
    }

    /// Check if this architecture uses `q_proj` / `k_proj` / `v_proj` / `o_proj` naming.
    pub fn uses_proj_naming(&self) -> bool {
        matches!(
            self.arch,
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3
        )
    }

    /// Get the attention weight suffix for this architecture.
    pub fn attn_weight_suffix(&self) -> &str {
        match self.arch {
            ModelArch::Gemma | ModelArch::Qwen2 | ModelArch::Qwen3 => "proj.weight",
            _ => ".weight",
        }
    }

    /// Get the embedding tensor name for this architecture.
    pub fn embedding_name(&self) -> &str {
        match self.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => "token_embd.weight",
            ModelArch::Gemma => "model.embed_tokens.weight",
            _ => "tok_embeddings.weight",
        }
    }

    /// Get the output/LM-head tensor name for this architecture.
    pub fn output_name(&self) -> &str {
        match self.arch {
            ModelArch::Gemma => "lm_head.weight",
            ModelArch::Qwen2 | ModelArch::Qwen3 => "output.weight", // Qwen2 uses "output.weight" not "lm_head.weight"
            _ => "output.weight",
        }
    }

    /// Get the final norm tensor name for this architecture (if any).
    pub fn final_norm_name(&self) -> Option<&str> {
        match self.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => Some("output_norm.weight"),
            _ => None,
        }
    }
}

/// A loaded Llama-style model ready for inference.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub token_embeddings: Option<Linear>,
    pub output: Option<Linear>,
    pub final_norm: Option<RmsNorm>,
    pub layers: Vec<TransformerLayer>,
    pub vocab_size: u32,
    pub tokenizer: Option<GgufTokenizer>,
    pub tokenizer_config: Option<GgufTokenizerConfig>,
    /// GPU dispatch context (None = CPU-only mode).
    pub dispatch: Option<DispatchContext>,
    /// KV caches per layer (used when dispatch is enabled).
    pub kv_caches: Option<(Vec<Kvcache>, Vec<Kvcache>)>,
    /// CPU-side KV caches for the pure-Rust transformer path.
    /// One `LayerKvCache` per transformer layer. Initialized on first
    /// `forward_layers_with_cache()` call.
    pub cpu_kv_caches: Option<Vec<crate::transformer::kv_cache::LayerKvCache>>,
}

impl LlamaModel {
    /// Load a Llama-style model from a GGUF file.
    pub fn load_gguf(path: &Path) -> Result<Self> {
        let _header = pesti_gguf::parser::parse_gguf(path)
            .map_err(|e| RunnerError::ModelLoad(e.to_string()))?;
        let weights = load_gguf_weights(path).map_err(|e| RunnerError::ModelLoad(e.to_string()))?;
        let mut model = Self::from_gguf_weights(weights)?;

        // Load tokenizer from GGUF file
        if let Ok((tokenizer_config, tokenizer)) = load_tokenizer_from_gguf(path) {
            model.tokenizer_config = Some(tokenizer_config);
            model.tokenizer = Some(tokenizer);
            debug!(path = %path.display(), "Loaded model with tokenizer");
        } else {
            debug!(path = %path.display(), "No tokenizer found in GGUF file");
        }

        Ok(model)
    }

    /// Build a model from already-loaded GGUF weights.
    pub fn from_gguf_weights(weights: GgufWeights) -> Result<Self> {
        let header = &weights.header;
        let config = LlamaConfig::from_gguf_header(header)?;

        let vocab_size = header.vocab_size();
        let rope_config = RopeConfig::new(config.head_dim, config.rope_base, config.max_seq_len);

        // Load token embeddings — architecture-dependent name
        let embedding_name = config.embedding_name();
        let token_embeddings = weights.tensors.get(embedding_name).map(|tensor_data| {
            // For Qwen2, the embedding tensor shape is [embed_dim, vocab_size]
            // We need to set in_features = embed_dim for correct row lookup
            if matches!(config.arch, ModelArch::Qwen2 | ModelArch::Qwen3) {
                let embed_dim = config.embed_dim;
                Linear::from_f32_weight_with_shape(
                    tensor_data,
                    None,
                    embed_dim as usize,
                    vocab_size as usize * embed_dim as usize,
                )
            } else {
                Linear::from_f32_weight(tensor_data, None)
            }
        });

        // Load output (LM head) — architecture-dependent name
        let output_name = config.output_name();
        let output = weights.tensors.get(output_name).map(|tensor_data| {
            // For Qwen2, output weight shape is [embed_dim, vocab_size] (transposed)
            // Linear layer expects [vocab_size, embed_dim], so we need to transpose
            // or interpret correctly. Here we set in_features=embed_dim, out_features=vocab_size
            if matches!(config.arch, ModelArch::Qwen2 | ModelArch::Qwen3) {
                let embed_dim = config.embed_dim;
                let vocab = vocab_size as usize;
                Linear::from_f32_weight_with_shape(tensor_data, None, embed_dim as usize, vocab)
            } else {
                Linear::from_f32_weight(tensor_data, None)
            }
        });

        // Build transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_layer(&weights, layer_idx, &config, &rope_config)?;
            layers.push(layer);
        }

        // Load final norm for architectures that have it (qwen2/qwen3)
        let final_norm = if let Some(norm_name) = config.final_norm_name() {
            weights.tensors.get(norm_name).map(|tensor_data| {
                let weight: Vec<f32> = tensor_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                RmsNorm::new(weight, config.rms_norm_eps)
            })
        } else {
            None
        };

        Ok(Self {
            config,
            token_embeddings,
            output,
            final_norm,
            layers,
            vocab_size,
            tokenizer: None,
            tokenizer_config: None,
            dispatch: Some(DispatchContext::new()),
            kv_caches: None,
            cpu_kv_caches: None,
        })
    }

    /// Build a model from already-loaded safetensors weights.
    ///
    /// Unlike GGUF, safetensors doesn't embed model config — the caller must
    /// provide `LlamaConfig` (e.g., from a companion `config.json` file).
    /// All tensor data is already in f32 format (the loader converted f16/bf16).
    pub fn from_safetensors_weights(
        weights: SafetensorsWeights,
        config: LlamaConfig,
    ) -> Result<Self> {
        let rope_config = RopeConfig::new(config.head_dim, config.rope_base, config.max_seq_len);
        let vocab_size = config.embed_dim as u32; // default if not in metadata

        // Load token embeddings — architecture-dependent name
        let embedding_name = config.embedding_name();
        let token_embeddings = weights
            .tensors
            .get(embedding_name)
            .map(|tensor_data| Linear::from_f32_weight(tensor_data, None));

        // Load output (LM head) — architecture-dependent name
        let output_name = config.output_name();
        let output = weights.tensors.get(output_name).map(|tensor_data| {
            // For Qwen2, output weight shape is [embed_dim, vocab_size] (transposed)
            // Linear layer expects [vocab_size, embed_dim], so we need to transpose
            // or interpret correctly. Here we set in_features=embed_dim, out_features=vocab_size
            if matches!(config.arch, ModelArch::Qwen2 | ModelArch::Qwen3) {
                let embed_dim = config.embed_dim;
                let vocab = vocab_size as usize;
                Linear::from_f32_weight_with_shape(tensor_data, None, embed_dim as usize, vocab)
            } else {
                Linear::from_f32_weight(tensor_data, None)
            }
        });

        // Build transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer =
                Self::load_layer_from_safetensors(&weights, layer_idx, &config, &rope_config)?;
            layers.push(layer);
        }

        // Load final norm for architectures that have it (qwen2/qwen3)
        let final_norm = if let Some(norm_name) = config.final_norm_name() {
            weights
                .tensors
                .get(norm_name)
                .map(|tensor_data| RmsNorm::new(f32_bytes_to_f32(tensor_data), config.rms_norm_eps))
        } else {
            None
        };

        Ok(Self {
            config,
            token_embeddings,
            output,
            final_norm,
            layers,
            vocab_size,
            tokenizer: None,
            tokenizer_config: None,
            dispatch: Some(DispatchContext::new()),
            kv_caches: None,
            cpu_kv_caches: None,
        })
    }

    /// Load a Llama-style model from a safetensors file.
    ///
    /// The safetensors file must be accompanied by a `config.json` in the same
    /// directory (or you can construct `LlamaConfig` manually).
    pub fn load_safetensors(path: &Path, config: LlamaConfig) -> Result<Self> {
        let weights = crate::safetensors_weight_loader::load_safetensors_weights(path)
            .map_err(|e| RunnerError::ModelLoad(format!("Safetensors load failed: {e}")))?;
        Self::from_safetensors_weights(weights, config)
    }

    /// Load a single transformer layer from GGUF weights.
    fn load_layer(
        weights: &GgufWeights,
        layer_idx: usize,
        config: &LlamaConfig,
        _rope: &RopeConfig,
    ) -> Result<TransformerLayer> {
        let prefix = config.layer_prefix(layer_idx);

        // RMSNorm weights — architecture-dependent names (with fallbacks)
        let attention_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}input_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_norm.weight"),
            _ => format!("{prefix}attention_norm.weight"),
        };

        // Try primary name first, then fallback for Qwen (which uses attn_norm)
        let attention_norm_data = weights
            .tensors
            .get(&attention_norm_name)
            .or_else(|| {
                if matches!(config.arch, ModelArch::Qwen2 | ModelArch::Qwen3) {
                    weights.tensors.get(&format!("{prefix}attn_norm.weight"))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                RunnerError::ModelLoad(format!(
                    "missing attention norm (tried: {})",
                    attention_norm_name
                ))
            })?;
        // Data is already f32 (gguf_weight_loader dequantized F16→f32)
        let norm_weight: Vec<f32> = attention_norm_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let attention_norm = RmsNorm::new(norm_weight, config.rms_norm_eps);

        let ffn_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}post_attention_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_norm.weight"),
            _ => format!("{prefix}ffn_norm.weight"),
        };
        let ffn_norm_data = weights
            .tensors
            .get(&ffn_norm_name)
            .ok_or_else(|| RunnerError::ModelLoad(format!("missing {ffn_norm_name}")))?;
        let ffn_norm_weight: Vec<f32> = ffn_norm_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ffn_norm = RmsNorm::new(ffn_norm_weight, config.rms_norm_eps);

        // Attention weights — architecture-dependent naming
        let suffix = config.attn_weight_suffix();
        let wq_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}q_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing q_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_q.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_q.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wq.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wq.weight"))
                }),
        }?;
        let wk_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}k_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing k_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_k.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_k.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wk.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wk.weight"))
                }),
        }?;
        let wv_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}v_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing v_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_v.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_v.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wv.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wv.weight"))
                }),
        }?;
        let wo_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}o_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing o_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_output.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_output.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wo.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wo.weight"))
                }),
        }?;

        // Use tensor shapes for correct in_features/out_features.
        // Shape is [out_features, in_features] in both GGUF and safetensors.
        let wq_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}q_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_q.weight"),
            _ => format!("{prefix}attention.wq.weight"),
        };
        let (wq_in, wq_out) = weights.tensor_shape(&wq_name);
        let wq = Linear::from_f32_weight_with_dims(wq_data, None, wq_in, wq_out);

        let wk_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}k_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_k.weight"),
            _ => format!("{prefix}attention.wk.weight"),
        };
        let (wk_in, wk_out) = weights.tensor_shape(&wk_name);
        let wk = Linear::from_f32_weight_with_dims(wk_data, None, wk_in, wk_out);

        let wv_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}v_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_v.weight"),
            _ => format!("{prefix}attention.wv.weight"),
        };
        let (wv_in, wv_out) = weights.tensor_shape(&wv_name);
        let wv = Linear::from_f32_weight_with_dims(wv_data, None, wv_in, wv_out);

        let wo_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}o_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_output.weight"),
            _ => format!("{prefix}attention.wo.weight"),
        };
        let (wo_in, wo_out) = weights.tensor_shape(&wo_name);
        let wo = Linear::from_f32_weight_with_dims(wo_data, None, wo_in, wo_out);

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.head_dim,
            config.num_heads,
            config.num_kv_heads,
        );

        // FFN weights — architecture-dependent naming
        let (w1_data, w2_data, w3_data) = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_gate.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_gate.weight".to_string()))?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_down.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_down.weight".to_string()))?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_up.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_up.weight".to_string()))?;
                (w1, w2, w3)
            }
            _ => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w1.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w1.weight"))
                    })?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w2.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w2.weight"))
                    })?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w3.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w3.weight"))
                    })?;
                (w1, w2, w3)
            }
        };

        // Use tensor shapes for FFN weights
        let w1_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_gate.weight"),
            _ => format!("{prefix}feed_forward.w1.weight"),
        };
        let (w1_in, w1_out) = weights.tensor_shape(&w1_name);
        let w1 = Linear::from_f32_weight_with_dims(w1_data, None, w1_in, w1_out);

        let w2_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_down.weight"),
            _ => format!("{prefix}feed_forward.w2.weight"),
        };
        let (w2_in, w2_out) = weights.tensor_shape(&w2_name);
        let w2 = Linear::from_f32_weight_with_dims(w2_data, None, w2_in, w2_out);

        let w3_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_up.weight"),
            _ => format!("{prefix}feed_forward.w3.weight"),
        };
        let (w3_in, w3_out) = weights.tensor_shape(&w3_name);
        let w3 = Linear::from_f32_weight_with_dims(w3_data, None, w3_in, w3_out);

        let feed_forward = FeedForward::new(w1, w2, w3, config.intermediate_dim);

        Ok(TransformerLayer::new(
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        ))
    }

    /// Load a single transformer layer from safetensors weights.
    ///
    /// Mirrors `load_layer` but uses f32 bytes directly (safetensors tensors
    /// are already in f32 format after loading).
    fn load_layer_from_safetensors(
        weights: &SafetensorsWeights,
        layer_idx: usize,
        config: &LlamaConfig,
        _rope: &RopeConfig,
    ) -> Result<TransformerLayer> {
        let prefix = config.layer_prefix(layer_idx);

        // RMSNorm weights — architecture-dependent names (with fallbacks)
        let attention_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}input_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_norm.weight"),
            _ => format!("{prefix}attention_norm.weight"),
        };

        // Try primary name first, then fallback for Qwen (which uses attn_norm)
        let attention_norm_data = weights
            .tensors
            .get(&attention_norm_name)
            .or_else(|| {
                if matches!(config.arch, ModelArch::Qwen2 | ModelArch::Qwen3) {
                    weights.tensors.get(&format!("{prefix}attn_norm.weight"))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                RunnerError::ModelLoad(format!(
                    "missing attention norm (tried: {})",
                    attention_norm_name
                ))
            })?;
        let attention_norm =
            RmsNorm::new(f32_bytes_to_f32(attention_norm_data), config.rms_norm_eps);

        let ffn_norm_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}post_attention_layernorm.weight"),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_norm.weight"),
            _ => format!("{prefix}ffn_norm.weight"),
        };
        let ffn_norm_data = weights
            .tensors
            .get(&ffn_norm_name)
            .ok_or_else(|| RunnerError::ModelLoad(format!("missing {ffn_norm_name}")))?;
        let ffn_norm = RmsNorm::new(f32_bytes_to_f32(ffn_norm_data), config.rms_norm_eps);

        // Attention weights — architecture-dependent naming
        let suffix = config.attn_weight_suffix();
        let wq_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}q_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing q_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_q.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_q.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wq.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wq.weight"))
                }),
        }?;
        let wk_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}k_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing k_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_k.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_k.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wk.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wk.weight"))
                }),
        }?;
        let wv_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}v_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing v_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_v.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_v.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wv.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wv.weight"))
                }),
        }?;
        let wo_data = match config.arch {
            ModelArch::Gemma => weights
                .tensors
                .get(&format!("{prefix}o_proj{suffix}"))
                .ok_or_else(|| RunnerError::ModelLoad("missing o_proj weight".to_string())),
            ModelArch::Qwen2 | ModelArch::Qwen3 => weights
                .tensors
                .get(&format!("{prefix}attn_output.weight"))
                .ok_or_else(|| RunnerError::ModelLoad("missing attn_output.weight".to_string())),
            _ => weights
                .tensors
                .get(&format!("{prefix}attention.wo.weight"))
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("missing {prefix}attention.wo.weight"))
                }),
        }?;

        // Use tensor shapes for correct in_features/out_features.
        // Shape is [out_features, in_features] in both GGUF and safetensors.
        let wq_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}q_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_q.weight"),
            _ => format!("{prefix}attention.wq.weight"),
        };
        let (wq_in, wq_out) = weights.tensor_shape(&wq_name);
        let wq = Linear::from_f32_weight_with_dims(wq_data, None, wq_in, wq_out);

        let wk_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}k_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_k.weight"),
            _ => format!("{prefix}attention.wk.weight"),
        };
        let (wk_in, wk_out) = weights.tensor_shape(&wk_name);
        let wk = Linear::from_f32_weight_with_dims(wk_data, None, wk_in, wk_out);

        let wv_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}v_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_v.weight"),
            _ => format!("{prefix}attention.wv.weight"),
        };
        let (wv_in, wv_out) = weights.tensor_shape(&wv_name);
        let wv = Linear::from_f32_weight_with_dims(wv_data, None, wv_in, wv_out);

        let wo_name = match config.arch {
            ModelArch::Gemma => format!("{prefix}o_proj{}", config.attn_weight_suffix()),
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}attn_output.weight"),
            _ => format!("{prefix}attention.wo.weight"),
        };
        let (wo_in, wo_out) = weights.tensor_shape(&wo_name);
        let wo = Linear::from_f32_weight_with_dims(wo_data, None, wo_in, wo_out);

        let attention = Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.head_dim,
            config.num_heads,
            config.num_kv_heads,
        );

        // FFN weights — architecture-dependent naming
        let (w1_data, w2_data, w3_data) = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_gate.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_gate.weight".to_string()))?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_down.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_down.weight".to_string()))?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}ffn_up.weight"))
                    .ok_or_else(|| RunnerError::ModelLoad("missing ffn_up.weight".to_string()))?;
                (w1, w2, w3)
            }
            _ => {
                let w1 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w1.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w1.weight"))
                    })?;
                let w2 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w2.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w2.weight"))
                    })?;
                let w3 = weights
                    .tensors
                    .get(&format!("{prefix}feed_forward.w3.weight"))
                    .ok_or_else(|| {
                        RunnerError::ModelLoad(format!("missing {prefix}feed_forward.w3.weight"))
                    })?;
                (w1, w2, w3)
            }
        };

        // Use tensor shapes for FFN weights
        let w1_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_gate.weight"),
            _ => format!("{prefix}feed_forward.w1.weight"),
        };
        let (w1_in, w1_out) = weights.tensor_shape(&w1_name);
        let w1 = Linear::from_f32_weight_with_dims(w1_data, None, w1_in, w1_out);

        let w2_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_down.weight"),
            _ => format!("{prefix}feed_forward.w2.weight"),
        };
        let (w2_in, w2_out) = weights.tensor_shape(&w2_name);
        let w2 = Linear::from_f32_weight_with_dims(w2_data, None, w2_in, w2_out);

        let w3_name = match config.arch {
            ModelArch::Qwen2 | ModelArch::Qwen3 => format!("{prefix}ffn_up.weight"),
            _ => format!("{prefix}feed_forward.w3.weight"),
        };
        let (w3_in, w3_out) = weights.tensor_shape(&w3_name);
        let w3 = Linear::from_f32_weight_with_dims(w3_data, None, w3_in, w3_out);

        let feed_forward = FeedForward::new(w1, w2, w3, config.intermediate_dim);

        Ok(TransformerLayer::new(
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        ))
    }

    /// Run the model on a single token input.
    ///
    /// `token` — input token ID
    /// `start_pos` — position in the sequence (for RoPE)
    /// Returns: logits over vocabulary [vocab_size]
    pub fn forward(&self, token: u32, start_pos: usize) -> Result<Vec<f32>> {
        let logits = self.embed(token, start_pos)?;
        self.apply_output_head(&logits)
    }

    /// Embed a single token ID into its embedding vector.
    pub fn embed(&self, token: u32, _start_pos: usize) -> Result<Vec<f32>> {
        let emb = self
            .token_embeddings
            .as_ref()
            .ok_or_else(|| RunnerError::ModelLoad("missing token embeddings".to_string()))?;

        let token_idx = token as usize;
        // For 1D embedding tensors (shape [N]), the embedding dimension is the tensor length.
        // For 2D tensors (shape [vocab_size, embed_dim]), use in_features.
        let emb_dim = if emb.in_features == 1 && emb.out_features > 1 {
            // 1D tensor stored as [1, N] — the actual dim is out_features
            emb.out_features
        } else {
            emb.in_features
        };
        let start = token_idx * emb_dim;
        let x = emb.weight[start..start + emb_dim].to_vec();
        Ok(x)
    }

    /// Apply the output (LM head) to get logits from hidden states.
    pub fn apply_output_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| RunnerError::ModelLoad("missing output layer".to_string()))?;

        let logits = output.forward(hidden, 1);
        Ok(logits)
    }

    /// Pass hidden states through all transformer layers.
    pub fn forward_layers(&self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        let _embed_dim = hidden.len();
        let mut h = hidden.to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, 1, 1, start_pos); // Fixed: was start_pos + layer_idx
        }

        // Apply final norm for architectures that have it (qwen2/qwen3)
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h, 1);
        }

        Ok(h)
    }

    /// Pass hidden states through all transformer layers with KV caching.
    ///
    /// This is the efficient autoregressive decode path. Each layer's KV cache
    /// stores previously computed keys and values, so attention only computes
    /// over the new position rather than recomputing the entire sequence.
    ///
    /// - `hidden`: `[embed_dim]` — single token's hidden state
    /// - `start_pos`: position in the sequence (for RoPE and cache slot)
    ///
    /// Returns: `[embed_dim]` — updated hidden state after all layers.
    pub fn forward_layers_with_cache(
        &mut self,
        hidden: &[f32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        // Initialize CPU KV caches on first call
        if self.cpu_kv_caches.is_none() {
            let caches: Vec<crate::transformer::kv_cache::LayerKvCache> = self
                .layers
                .iter()
                .map(|layer| {
                    crate::transformer::kv_cache::LayerKvCache::new(
                        layer.attention.num_kv_heads,
                        layer.attention.head_dim,
                        self.config.max_seq_len,
                    )
                })
                .collect();
            self.cpu_kv_caches = Some(caches);
        }

        let caches = self.cpu_kv_caches.as_mut().unwrap();
        let mut h = hidden.to_vec();

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward_with_cache(&h, &mut caches[layer_idx], start_pos);
        }

        // Apply final norm for architectures that have it (qwen2/qwen3)
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h, 1);
        }

        Ok(h)
    }

    /// Reset all CPU KV caches (call before starting a new generation sequence).
    pub fn reset_cpu_kv_caches(&mut self) {
        if let Some(ref mut caches) = self.cpu_kv_caches {
            for cache in caches.iter_mut() {
                cache.clear();
            }
        }
    }

    /// Pass hidden states through all layers using GPU dispatch (if available).
    ///
    /// Builds `LayerDispatch` from the model's weights and runs through each
    /// layer using the dispatch context. Falls back to CPU if GPU is unavailable.
    ///
    /// KV caches are initialized on first call and persist across calls.
    pub fn forward_with_dispatch(&mut self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
        let ctx = self
            .dispatch
            .as_ref()
            .ok_or_else(|| RunnerError::Tensor("dispatch context not initialized".into()))?;

        // Initialize KV caches on first call
        if self.kv_caches.is_none() {
            let mut key_caches = Vec::with_capacity(self.config.num_layers);
            let mut value_caches = Vec::with_capacity(self.config.num_layers);
            for _ in 0..self.config.num_layers {
                let key_cache = Kvcache::new(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.max_seq_len,
                    if ctx.gpu_available() { true } else { false },
                );
                let value_cache = Kvcache::new(
                    self.config.num_heads,
                    self.config.num_kv_heads,
                    self.config.head_dim,
                    self.config.max_seq_len,
                    if ctx.gpu_available() { true } else { false },
                );
                key_caches.push(key_cache);
                value_caches.push(value_cache);
            }
            self.kv_caches = Some((key_caches, value_caches));
        }

        let (key_caches, value_caches) = self
            .kv_caches
            .as_mut()
            .ok_or_else(|| RunnerError::Tensor("kv caches not initialized".into()))?;

        let mut h = hidden.to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Build LayerDispatch from this layer's weights
            let attention_dispatch = crate::kernel::dispatch::AttentionDispatch {
                wq: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.attention.wq.weight),
                    layer.attention.wq.weight.clone(),
                    layer.attention.wq.bias.clone(),
                    layer.attention.wq.in_features,
                    layer.attention.wq.out_features,
                ),
                wk: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.attention.wk.weight),
                    layer.attention.wk.weight.clone(),
                    layer.attention.wk.bias.clone(),
                    layer.attention.wk.in_features,
                    layer.attention.wk.out_features,
                ),
                wv: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.attention.wv.weight),
                    layer.attention.wv.weight.clone(),
                    layer.attention.wv.bias.clone(),
                    layer.attention.wv.in_features,
                    layer.attention.wv.out_features,
                ),
                wo: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.attention.wo.weight),
                    layer.attention.wo.weight.clone(),
                    layer.attention.wo.bias.clone(),
                    layer.attention.wo.in_features,
                    layer.attention.wo.out_features,
                ),
                num_heads: layer.attention.num_heads,
                num_kv_heads: layer.attention.num_kv_heads,
                head_dim: layer.attention.head_dim,
                kv_dim: layer.attention.kv_dim,
            };

            let feed_forward_dispatch = crate::kernel::dispatch::FeedForwardDispatch {
                w1: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.feed_forward.w1.weight),
                    layer.feed_forward.w1.weight.clone(),
                    layer.feed_forward.w1.bias.clone(),
                    layer.feed_forward.w1.in_features,
                    layer.feed_forward.w1.out_features,
                ),
                w2: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.feed_forward.w2.weight),
                    layer.feed_forward.w2.weight.clone(),
                    layer.feed_forward.w2.bias.clone(),
                    layer.feed_forward.w2.in_features,
                    layer.feed_forward.w2.out_features,
                ),
                w3: crate::kernel::dispatch::LinearDispatch::new(
                    f32_to_f16(&layer.feed_forward.w3.weight),
                    layer.feed_forward.w3.weight.clone(),
                    layer.feed_forward.w3.bias.clone(),
                    layer.feed_forward.w3.in_features,
                    layer.feed_forward.w3.out_features,
                ),
                intermediate_dim: layer.feed_forward.intermediate_dim,
            };

            let attention_norm = crate::kernel::dispatch::RmsNormDispatch::new(
                layer.attention_norm.weight.clone(),
                layer.attention_norm.eps,
            );

            let ffn_norm = crate::kernel::dispatch::RmsNormDispatch::new(
                layer.ffn_norm.weight.clone(),
                layer.ffn_norm.eps,
            );

            let mut layer_dispatch = crate::kernel::dispatch::LayerDispatch {
                attention: attention_dispatch,
                feed_forward: feed_forward_dispatch,
                attention_norm,
                ffn_norm,
            };

            let layer_start_pos = start_pos + layer_idx;
            h = layer_dispatch.forward(
                ctx,
                &h,
                1, // batch_size
                1, // seq_len
                layer_start_pos,
                &mut key_caches[layer_idx],
                &mut value_caches[layer_idx],
            )?;
        }

        // Apply final norm for architectures that have it (qwen2/qwen3)
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h, 1);
        }

        // Output head projection: hidden → logits via GEMM
        let output = self.output.as_ref().ok_or_else(|| {
            RunnerError::ModelLoad("missing output layer for GPU forward".to_string())
        })?;

        // Use dispatch's GEMM kernel to compute: logits = h @ output.weight.T
        // A: [1, hidden] (single token), B: [hidden, vocab_size], C: [1, vocab_size]
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Convert weights to f16 for dispatch_gemm_cpu fallback (or GPU if available)
        let output_f16: Vec<half::f16> = output.weight.iter().map(|&v| half::f16::from_f32(v)).collect();

        // Dispatch GEMM: C[1×vocab] = alpha * A[1×hidden] @ B[hidden×vocab] + beta*C
        let logits_vec = ctx.dispatch_gemm_cpu(
            &h.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<half::f16>>(),
            &output_f16,
            None,
            1, // m: output batch (1 token)
            self.vocab_size as usize, // n: vocab_size logits
            self.config.embed_dim,   // k: hidden dimension
            alpha,
            beta,
        )?;

        Ok(logits_vec)
    }

    /// Get the model architecture string from GGUF header.
    pub fn architecture(header: &GgufHeader) -> Option<&str> {
        header.architecture()
    }

    /// Check if this model supports the given architecture.
    pub fn is_supported_architecture(arch: &str) -> bool {
        matches!(
            arch,
            "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2" | "qwen3" | "starcoder2"
        )
    }

    /// Sample a token from logits using the configured sampling strategy.
    pub fn sample_from_logits(
        logits: &[f32],
        config: &crate::transformer::SamplingConfig,
        rng: &mut rand::rngs::StdRng,
    ) -> u32 {
        crate::transformer::sample(logits, config, rng)
    }

    /// Greedy decode: argmax over logits.
    pub fn argmax_from_logits(logits: &[f32]) -> u32 {
        crate::transformer::argmax(logits)
    }

    /// Generate tokens autoregressively.
    ///
    /// `prompt` — input token IDs
    /// `max_tokens` — maximum tokens to generate
    /// `sampling_config` — sampling parameters (temperature, top-p, top-k)
    /// `rng` — random number generator
    /// `stop_tokens` — token IDs that stop generation
    ///
    /// Returns: generated token IDs (excluding prompt)
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        sampling_config: &crate::transformer::SamplingConfig,
        rng: &mut rand::rngs::StdRng,
        stop_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();

        // Process prompt: for each token, run forward and update position
        let mut context = prompt.to_vec();
        let mut pos = 0;

        // Use the last token for each forward pass (autoregressive)
        for _ in 0..max_tokens {
            let last_token = *context
                .last()
                .ok_or_else(|| RunnerError::ModelLoad("empty context".to_string()))?;

            // Get logits for the last token
            let logits = self.forward(last_token, pos)?;

            // Sample next token
            let next_token = if sampling_config.temperature == 0.0 {
                Self::argmax_from_logits(&logits)
            } else {
                Self::sample_from_logits(&logits, sampling_config, rng)
            };

            // Check for stop tokens
            if stop_tokens.contains(&next_token) {
                break;
            }

            generated.push(next_token);
            context.push(next_token);
            pos += 1;

            if pos >= self.config.max_seq_len {
                break;
            }
        }

        Ok(generated)
    }
}

/// Convert f16 tensor bytes to f32 Vec.
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
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
        })
        .collect()
}

/// Convert f32 bytes to f32 Vec.
fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Convert f32 slice to f16 Vec.
fn f32_to_f16(data: &[f32]) -> Vec<half::f16> {
    data.iter().map(|&v| half::f16::from_f32(v)).collect()
}
