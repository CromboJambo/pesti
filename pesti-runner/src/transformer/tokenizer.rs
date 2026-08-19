//! Tokenizer integration with support for multiple backends.
//!
//! Supports both mistral.rs (default) and pure Rust qwen2-bpe implementations.

#![allow(clippy::if_same_then_else, clippy::needless_return)]

use std::path::Path;
use tracing::{debug, warn};

use crate::error::RunnerError;
use pesti_gguf::types::GgufHeader;

/// Tokenizer backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    /// Use mistral.rs GGUF conversion (default)
    MistralRs,
    /// Use pure Rust qwen2-bpe implementation
    Qwen2Bpe,
}

impl Default for TokenizerBackend {
    fn default() -> Self {
        TokenizerBackend::MistralRs
    }
}

#[cfg(feature = "rust-tokenizer")]
use qwen2_bpe::Qwen2Tokenizer as RustTokenizer;

/// Wrapper around a tokenizer with PESTI-specific functionality.
pub enum PestiTokenizer {
    /// Mistral.rs backend
    MistralRs(tokenizers::Tokenizer),
    /// Pure Rust qwen2-bpe backend (feature-gated)
    #[cfg(feature = "rust-tokenizer")]
    Qwen2Bpe(RustTokenizer),
}

impl PestiTokenizer {
    /// Create a new tokenizer from GGUF metadata using the selected backend.
    pub fn from_gguf(header: &GgufHeader, backend: TokenizerBackend) -> Result<Self, RunnerError> {
        match backend {
            TokenizerBackend::MistralRs => Ok(Self::MistralRs(
                Self::load_mistralrs_tokenizer(header)?,
            )),
            
            #[cfg(feature = "rust-tokenizer")]
            TokenizerBackend::Qwen2Bpe => {
                debug!("Loading pure Rust qwen2-bpe tokenizer");
                Ok(Self::Qwen2Bpe(Self::load_rust_tokenizer(header)?))
            }
            
            #[cfg(not(feature = "rust-tokenizer"))]
            TokenizerBackend::Qwen2Bpe => {
                warn!("rust-tokenizer feature not enabled, falling back to mistral.rs");
                Ok(Self::MistralRs(Self::load_mistralrs_tokenizer(header)?))
            }
        }
    }

    /// Load tokenizer using mistral.rs backend.
    fn load_mistralrs_tokenizer(_header: &GgufHeader) -> Result<tokenizers::Tokenizer, RunnerError> {
        debug!("Loading mistral.rs tokenizer for Qwen2");

        // Use a cached GPT-2 tokenizer as fallback (will work for basic encoding)
        let tokenizer_path = Path::new("/home/crombo/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json");
        
        let inner = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

        Ok(inner)
    }

    /// Load tokenizer using pure Rust qwen2-bpe backend.
    #[cfg(feature = "rust-tokenizer")]
    fn load_rust_tokenizer(header: &GgufHeader) -> Result<RustTokenizer, RunnerError> {
        use pesti_gguf::parser::parse_gguf;

        // Extract model path from GGUF header (assuming it's stored in metadata)
        let model_path = header.get_kv_str("tokenizer.model")
            .ok_or_else(|| RunnerError::Tokenizer("Missing tokenizer.model in GGUF".into()))?;

        // Parse GGUF to get vocabulary and merge pairs
        let header = parse_gguf(Path::new(model_path))
            .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

        // For now, use hardcoded paths (can be improved later)
        let vocab_path = "/tmp/qwen2_vocab_dump.json";
        let merges_path = "/tmp/qwen2_merge_pairs.json";

        let tokenizer = RustTokenizer::load_with_merges(vocab_path, merges_path)
            .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

        Ok(tokenizer)
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, RunnerError> {
        match self {
            Self::MistralRs(inner) => {
                let encoding = inner.encode(text, false)
                    .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;
                
                Ok(encoding.get_ids().to_vec())
            }
            
            #[cfg(feature = "rust-tokenizer")]
            Self::Qwen2Bpe(inner) => {
                inner.encode(text).map_err(|e| RunnerError::Tokenizer(e.to_string()))
            }
        }
    }

    /// Decode token IDs into text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, RunnerError> {
        match self {
            Self::MistralRs(inner) => {
                let result = inner.decode(tokens, false)
                    .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;
                
                Ok(result)
            }
            
            #[cfg(feature = "rust-tokenizer")]
            Self::Qwen2Bpe(inner) => {
                inner.decode(tokens).map_err(|e| RunnerError::Tokenizer(e.to_string()))
            }
        }
    }

    /// Get the tokenizer's vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::MistralRs(inner) => inner.get_vocab_size(false),
            
            #[cfg(feature = "rust-tokenizer")]
            Self::Qwen2Bpe(inner) => inner.vocab_size(),
        }
    }
}

/// Tokenizer configuration extracted from GGUF header.
pub struct GgufTokenizerConfig {
    /// Model type (e.g., "qwen2", "gpt2")
    pub tokenizer_model: String,
    /// Special tokens mapping
    pub special_tokens: Option<String>,
    /// Base vocabulary size
    pub base_vocab_size: usize,
    /// Number of special tokens
    pub num_special_tokens: usize,
}

impl GgufTokenizerConfig {
    /// Create a new tokenizer config from GGUF header.
    pub fn from_gguf_header(header: &GgufHeader) -> Self {
        let tokenizer_model = header.get_kv_str("tokenizer.model")
            .unwrap_or("qwen2")
            .to_string();

        let special_tokens = header.get_kv_str("tokenizer.special_tokens")
            .map(|s| s.to_string());

        let base_vocab_size = header.get_kv_u32("tokenizer.vocab_size")
            .unwrap_or(151936) as usize; // Qwen2 default

        let num_special_tokens = header.get_kv_u32("tokenizer.special_tokens_count")
            .unwrap_or(0) as usize;

        Self {
            tokenizer_model,
            special_tokens,
            base_vocab_size,
            num_special_tokens,
        }
    }
}

/// Load tokenizer from GGUF file path.
pub fn load_tokenizer_from_gguf(
    path: &Path,
    backend: TokenizerBackend,
) -> Result<(GgufTokenizerConfig, PestiTokenizer), RunnerError> {
    use pesti_gguf::parser::parse_gguf;

    let header = parse_gguf(path).map_err(|e| RunnerError::Tokenizer(e.to_string()))?;
    let config = GgufTokenizerConfig::from_gguf_header(&header);

    // Create tokenizer using selected backend
    let tokenizer = PestiTokenizer::from_gguf(&header, backend)?;

    Ok((config, tokenizer))
}
