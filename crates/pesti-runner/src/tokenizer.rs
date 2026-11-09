use crate::error::RunnerError;
use tiktoken_rs::bpe_for_model;
use tokenizers::Tokenizer as TokenizerImpl;
use tracing::debug;

/// Tokenizer for prompt encoding.
///
/// supports BPE models (GPT-2, GPT-3) and tokenizers library.
pub struct Tokenizer {
    pub model: String,
    pub tokenizer: Option<TokenizerImpl>,
}

impl Tokenizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            tokenizer: None,
        }
    }

    /// Initialize BPE from model name.
    pub fn init_bpe(&mut self) -> Result<(), RunnerError> {
        let _bpe = bpe_for_model(&self.model)
            .map_err(|e: anyhow::Error| RunnerError::Tokenizer(e.to_string()))?;
        debug!(model = %self.model, "Tokenizer: BPE initialized");
        Ok(())
    }

    /// Encode prompt to token IDs.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, RunnerError> {
        if let Some(ref tok) = self.tokenizer {
            tok.encode(prompt, false)
                .map_err(|e: Box<dyn std::error::Error + Send + Sync>| {
                    RunnerError::Tokenizer(e.to_string())
                })
                .map(|e| e.get_ids().to_vec())
        } else {
            Err(RunnerError::Tokenizer(
                "tokenizer not initialized".to_string(),
            ))
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, RunnerError> {
        if let Some(ref tok) = self.tokenizer {
            tok.decode(tokens, false)
                .map_err(|e: Box<dyn std::error::Error + Send + Sync>| {
                    RunnerError::Tokenizer(e.to_string())
                })
        } else {
            Err(RunnerError::Tokenizer(
                "tokenizer not initialized".to_string(),
            ))
        }
    }

    /// Get token count.
    pub fn token_count(&self, prompt: &str) -> Result<usize, RunnerError> {
        if let Some(ref tok) = self.tokenizer {
            Ok(tok
                .encode(prompt, false)
                .map_err(|e: Box<dyn std::error::Error + Send + Sync>| {
                    RunnerError::Tokenizer(e.to_string())
                })?
                .get_ids()
                .len())
        } else {
            Err(RunnerError::Tokenizer(
                "tokenizer not initialized".to_string(),
            ))
        }
    }
}
