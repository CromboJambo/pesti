//! Tokenizer integration with support for multiple backends.
//!
//! The default (MistralRs) backend now builds the real `tokenizers::Tokenizer`
//! **directly from the GGUF-embedded tokenizer arrays** (`tokenizer.ggml.*`),
//! instead of loading a hardcoded external tokenizer.json. This makes encoding
//! fully self-contained: a Qwen2 GGUF carries its full BPE vocab + merges, and
//! we reconstruct the exact HF-compatible tokenizer (BPE model + Qwen2
//! pre-tokenizer + ByteLevel decoder + NFC normalizer).
//!
//! Validated against the HF reference: the fox-sentence prompt encodes to
//! `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]` with both the
//! HF `tokenizer.json` and the GGUF-extracted rebuild.

#![allow(clippy::if_same_then_else, clippy::needless_return)]

use std::path::Path;
use tracing::{debug, warn};

use crate::error::RunnerError;
use pesti_gguf::types::{GgufHeader, GgufKvValue};

/// Tokenizer backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerBackend {
    /// Build the real tokenizer from GGUF-embedded arrays (default)
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

/// Qwen2 pre-tokenizer regex (from the HF `tokenizer.json`). Requires the
/// `fancy-regex` feature of the `tokenizers` crate for `\p{L}`/`\p{N}` classes
/// and the `(?!...)` lookahead.
const QWEN2_PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Canonical Qwen2 special tokens (registered as atomic/special so the
/// pre-tokenizer never splits them). Content-only; IDs come from the vocab.
const QWEN2_SPECIAL_TOKENS: &[&str] = &[
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
];

/// Wrapper around a tokenizer with PESTI-specific functionality.
pub enum PestiTokenizer {
    /// GGUF-extracted `tokenizers` backend
    MistralRs(tokenizers::Tokenizer),
    /// Pure Rust qwen2-bpe backend (feature-gated)
    #[cfg(feature = "rust-tokenizer")]
    Qwen2Bpe(RustTokenizer),
}

impl PestiTokenizer {
    /// Create a new tokenizer from GGUF metadata using the selected backend.
    pub fn from_gguf(header: &GgufHeader, backend: TokenizerBackend) -> Result<Self, RunnerError> {
        match backend {
            TokenizerBackend::MistralRs => {
                Ok(Self::MistralRs(Self::load_mistralrs_tokenizer(header)?))
            }

            #[cfg(feature = "rust-tokenizer")]
            TokenizerBackend::Qwen2Bpe => {
                debug!("Loading pure Rust qwen2-bpe tokenizer");
                Ok(Self::Qwen2Bpe(Self::load_rust_tokenizer(header)?))
            }

            #[cfg(not(feature = "rust-tokenizer"))]
            TokenizerBackend::Qwen2Bpe => {
                warn!(
                    "rust-tokenizer feature not enabled, falling back to GGUF-extracted tokenizer"
                );
                Ok(Self::MistralRs(Self::load_mistralrs_tokenizer(header)?))
            }
        }
    }

    /// Build the real tokenizer from the GGUF-embedded `tokenizer.ggml.*` arrays.
    fn load_mistralrs_tokenizer(header: &GgufHeader) -> Result<tokenizers::Tokenizer, RunnerError> {
        use tokenizers::AddedToken;
        use tokenizers::models::bpe::{BPE, Vocab};
        use tokenizers::normalizers::NFC;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::Split;
        use tokenizers::tokenizer::normalizer::SplitDelimiterBehavior;

        // 1. Extract vocab (token -> id) and merges from the GGUF arrays.
        let tokens = Self::string_array(header, "tokenizer.ggml.tokens").ok_or_else(|| {
            RunnerError::Tokenizer("missing tokenizer.ggml.tokens array in GGUF".into())
        })?;
        let merge_strs = Self::string_array(header, "tokenizer.ggml.merges").ok_or_else(|| {
            RunnerError::Tokenizer("missing tokenizer.ggml.merges array in GGUF".into())
        })?;

        let vocab: Vocab = tokens
            .into_iter()
            .enumerate()
            .map(|(id, tok)| (tok, id as u32))
            .collect();

        let merges: Vec<(String, String)> = merge_strs
            .iter()
            .filter_map(|pair| {
                let mut it = pair.splitn(2, ' ');
                let left = it.next()?.to_string();
                let right = it.next()?.to_string();
                Some((left, right))
            })
            .collect();

        debug!(
            vocab = vocab.len(),
            merges = merges.len(),
            "Built BPE vocab/merges from GGUF arrays"
        );

        // 2. BPE model (byte_fallback=false, no UNK — matches HF Qwen2).
        let model = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .byte_fallback(false)
            .build()
            .map_err(|e| RunnerError::Tokenizer(format!("BPE build failed: {e}")))?;

        // 3. Qwen2 pre-tokenizer: Sequence[Split(Regex, Isolated), ByteLevel].
        let split = Split::new(
            QWEN2_PRETOKENIZE_REGEX,
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| RunnerError::Tokenizer(format!("pre-tokenizer regex failed: {e}")))?;
        let byte_level = ByteLevel::new(false, false, false);
        let pre_tokenizer = Sequence::new(vec![split.into(), byte_level.into()]);

        // 4. Assemble the tokenizer (Tokenizer DerefMuts to TokenizerImpl for with_*).
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_normalizer(Some(NFC));
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));

        // 5. Register canonical Qwen2 special tokens (atomic, non-normalized).
        let added: Vec<AddedToken> = QWEN2_SPECIAL_TOKENS
            .iter()
            .filter(|tok| tokenizer.token_to_id(tok).is_some())
            .map(|tok| AddedToken::from(*tok, true))
            .collect();
        let n_added = tokenizer.add_special_tokens(&added);
        debug!(added = n_added, "Registered special tokens");

        Ok(tokenizer)
    }

    /// Pull a `tokenizer.ggml.*` string array out of the header KV pairs.
    fn string_array(header: &GgufHeader, key: &str) -> Option<Vec<String>> {
        header
            .kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| match &kv.value {
                GgufKvValue::Array(items) => Some(
                    items
                        .iter()
                        .filter_map(|v| match v {
                            GgufKvValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
    }

    /// Load tokenizer using pure Rust qwen2-bpe backend.
    #[cfg(feature = "rust-tokenizer")]
    fn load_rust_tokenizer(header: &GgufHeader) -> Result<RustTokenizer, RunnerError> {
        use pesti_gguf::parser::parse_gguf;

        // Extract model path from GGUF header (assuming it's stored in metadata)
        let model_path = header
            .get_kv_str("tokenizer.model")
            .ok_or_else(|| RunnerError::Tokenizer("Missing tokenizer.model in GGUF".into()))?;

        // Parse GGUF to get vocabulary and merge pairs
        let header =
            parse_gguf(Path::new(model_path)).map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

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
                let encoding = inner
                    .encode(text, false)
                    .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

                Ok(encoding.get_ids().to_vec())
            }

            #[cfg(feature = "rust-tokenizer")]
            Self::Qwen2Bpe(inner) => inner
                .encode(text)
                .map_err(|e| RunnerError::Tokenizer(e.to_string())),
        }
    }

    /// Decode token IDs into text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, RunnerError> {
        match self {
            Self::MistralRs(inner) => {
                let result = inner
                    .decode(tokens, false)
                    .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

                Ok(result)
            }

            #[cfg(feature = "rust-tokenizer")]
            Self::Qwen2Bpe(inner) => inner
                .decode(tokens)
                .map_err(|e| RunnerError::Tokenizer(e.to_string())),
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

/// Tokenizer configuration extracted from GGUF header (real `tokenizer.ggml.*` keys).
#[derive(Debug, Clone)]
pub struct GgufTokenizerConfig {
    /// Total vocabulary size (base + special), from `tokenizer.ggml.tokens`.
    pub vocab_size: usize,
    /// BOS token ID (`tokenizer.ggml.bos_token_id`), if present.
    pub bos_token_id: Option<u32>,
    /// EOS token ID (`tokenizer.ggml.eos_token_id`), if present.
    pub eos_token_id: Option<u32>,
}

impl GgufTokenizerConfig {
    /// Create a new tokenizer config from GGUF header (real `tokenizer.ggml.*` keys).
    pub fn from_gguf_header(header: &GgufHeader) -> Self {
        // Vocab size = length of the tokens array (falls back to a scalar key).
        let vocab_size = PestiTokenizer::string_array(header, "tokenizer.ggml.tokens")
            .map(|t| t.len())
            .or_else(|| {
                header
                    .get_kv_u32("tokenizer.ggml.vocab_size")
                    .map(|v| v as usize)
            })
            .unwrap_or(0);

        let bos_token_id = header.get_kv_u32("tokenizer.ggml.bos_token_id");
        let eos_token_id = header.get_kv_u32("tokenizer.ggml.eos_token_id");

        Self {
            vocab_size,
            bos_token_id,
            eos_token_id,
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
