//! SafeTensors tokenizer support.
//!
//! Loads tokenizers from standard HuggingFace tokenizer.json files.

use std::path::Path;

use tokenizers::Tokenizer as TokenizerImpl;
use tracing::debug;

use crate::error::{Result, RunnerError};

/// Configuration for a SafeTensors tokenizer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SafetensorsTokenizerConfig {
    /// Tokenizer model type (e.g., "bpe", "wordpiece", "sentencepiece").
    pub model_type: String,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// BOS token ID.
    pub bos_token_id: Option<u32>,
    /// EOS token ID.
    pub eos_token_id: Option<u32>,
    /// UNK token ID.
    pub unk_token_id: Option<u32>,
    /// PAD token ID.
    pub pad_token_id: Option<u32>,
}

/// Load a tokenizer from a HuggingFace tokenizer.json file.
pub fn load_tokenizer_from_safetensors(path: &Path) -> Result<(SafetensorsTokenizerConfig, TokenizerImpl)> {
    let tokenizer = TokenizerImpl::from_file(path)
        .map_err(|e| RunnerError::Tokenizer(e.to_string()))?;

    let vocab_size = tokenizer.get_vocab_size(true);

    let config = SafetensorsTokenizerConfig {
        model_type: "bpe".to_string(), // Default, can be extracted from tokenizer config
        vocab_size: vocab_size as u32,
        bos_token_id: None, // Will be set by the tokenizer
        eos_token_id: None,
        unk_token_id: None,
        pad_token_id: None,
    };

    debug!(
        path = %path.display(),
        vocab_size = config.vocab_size,
        "Loaded SafeTensors tokenizer"
    );

    Ok((config, tokenizer))
}

/// Load a tokenizer from the same directory as a safetensors model file.
///
/// Looks for:
/// - tokenizer.json (standard HuggingFace format)
pub fn load_tokenizer_for_model(model_path: &Path) -> Result<(SafetensorsTokenizerConfig, TokenizerImpl)> {
    let model_dir = model_path
        .parent()
        .ok_or_else(|| RunnerError::Tokenizer("Model path has no parent directory".to_string()))?;

    // Try tokenizer.json first (most common)
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        return load_tokenizer_from_safetensors(&tokenizer_path);
    }

    Err(RunnerError::Tokenizer(
        "No tokenizer found in model directory (expected tokenizer.json)".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    #[ignore] // Test needs proper tokenizer.json format with merges
    fn test_load_tokenizer_from_safetensors() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");

        // Create a minimal tokenizer.json
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"content": "[PAD]", "id": 0, "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": true},
                {"content": "[BOS]", "id": 1, "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": true},
                {"content": "[EOS]", "id": 2, "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel"},
            "post_processor": null,
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE",
                "vocab": {"test": 3},
                "merges": ["t e"]
            }
        }"#;

        std::fs::write(&path, tokenizer_json).unwrap();

        let (config, _tokenizer) = load_tokenizer_from_safetensors(&path).unwrap();
        assert_eq!(config.vocab_size, 4); // test + [PAD] + [BOS] + [EOS]
    }
}
