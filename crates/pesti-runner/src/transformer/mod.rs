//! Transformer module: Llama-style model implementation for CPU inference.
//!
//! ## Components
//!
//! - `model` — `LlamaModel` loads GGUF weights and wires transformer layers
//! - `layer` — `TransformerLayer` with attention, FFN, RMSNorm
//! - `linear` — `Linear` layer for matrix multiplication
//! - `rms_norm` — RMS normalization
//! - `rope` — Rotary positional embeddings
//! - `sampling` — Token sampling (temperature, top-p, top-k)
//! - `tokenizer` — GGUF tokenizer integration with multi-backend support
//! - `kv_cache` — KV cache for autoregressive generation
//!
//! ## Inference Flow
//!
//! ```text
//! GGUF file → LlamaModel::load_gguf() → forward(token, pos) → logits → sample() → next_token
//! ```

pub mod kv_cache;
pub mod layer;
pub mod linear;
pub mod model;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
pub mod tokenizer;

pub use kv_cache::LayerKvCache;
pub use model::{LlamaConfig, LlamaModel, ModelArch};
pub use sampling::{SamplingConfig, argmax, sample};
pub use tokenizer::{GgufTokenizerConfig, TokenizerBackend, load_tokenizer_from_gguf};
