//! Dataset loader for crabjar decision training data.
//!
//! Loads JSONL files from crabjar's dump directory and converts them into
//! tokenized training samples for QLoRA fine-tuning.

use crate::error::{DatasetError, Result};
use qwen2_bpe::Qwen2Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A single decision training example from crabjar's dump.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionExample {
    pub scenario: String,
    #[serde(default)]
    pub context: serde_json::Value,
    pub decision: String,
    pub confidence: u8,
    pub reasoning: String,
    #[serde(default)]
    pub risk_factors: Vec<String>,
    #[serde(default)]
    pub next_steps: Vec<String>,
}

/// A tokenized training sample ready for the model.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Tokenized input (prompt)
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u8>,
    /// Target token IDs (-100 for ignored positions)
    pub labels: Vec<i64>,
}

/// Dataset loader that reads crabjar JSONL and produces training samples.
pub struct DatasetLoader {
    tokenizer: Qwen2Tokenizer,
    max_seq_length: usize,
}

impl DatasetLoader {
    /// Create a new dataset loader.
    pub const fn new(tokenizer: Qwen2Tokenizer, max_seq_length: usize) -> Self {
        Self {
            tokenizer,
            max_seq_length,
        }
    }

    /// Load and tokenize a JSONL training file.
    pub fn load_jsonl<P: AsRef<Path>>(&self, path: P) -> Result<Vec<TrainingSample>> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| DatasetError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut samples = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let example: DecisionExample = serde_json::from_str(line).map_err(|e| {
                DatasetError::Json {
                    path: path.to_path_buf(),
                    line: line_num + 1,
                    source: e,
                }
            })?;

            let sample = self.format_decision_sample(example);
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Format a decision example as an instruction-following training sample.
    fn format_decision_sample(&self, example: DecisionExample) -> TrainingSample {
        // Build the prompt using Qwen2 chat format
        let system = "You are the Kev-9B decision engine for crabjar agent orchestration.";
        let user_prompt = format!(
            "Scenario: {}\nContext: {}\n\nProvide your decision as JSON with fields: decision, confidence, reasoning, risk_factors, next_steps.",
            example.scenario,
            serde_json::to_string(&example.context).unwrap_or_default()
        );

        // Build the assistant response (target)
        let target = serde_json::json!({
            "decision": example.decision,
            "confidence": example.confidence,
            "reasoning": example.reasoning,
            "risk_factors": example.risk_factors,
            "next_steps": example.next_steps
        });

        // Format as Qwen2 chat template: system + user -> assistant
        let full_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user_prompt
        );

        // Tokenize the prompt
        let prompt_tokens = self.tokenizer.encode(&full_prompt).unwrap_or_default();

        // Tokenize the target response
        let target_text = serde_json::to_string(&target).unwrap_or_default();
        let target_tokens = self
            .tokenizer
            .encode(&format!("{}<|im_end|>", target_text))
            .unwrap_or_default();

        // Combine: input is prompt, labels are target with -100 for prompt positions
        let total_len = prompt_tokens.len() + target_tokens.len();

        // Truncate if needed
        let seq_len = self.max_seq_length.min(total_len);

        let mut input_ids = vec![0u32; seq_len];
        let mut attention_mask = vec![0u8; seq_len];
        let mut labels = vec![-100i64; seq_len];

        // Copy prompt tokens
        let prompt_len = prompt_tokens.len().min(seq_len);
        input_ids[..prompt_len].copy_from_slice(&prompt_tokens[..prompt_len]);
        attention_mask[..prompt_len].fill(1);

        // Copy target tokens and set labels
        let remaining = seq_len - prompt_len;
        if !target_tokens.is_empty() && remaining > 0 {
            let copy_len = target_tokens.len().min(remaining);
            input_ids[prompt_len..prompt_len + copy_len]
                .copy_from_slice(&target_tokens[..copy_len]);
            attention_mask[prompt_len..prompt_len + copy_len].fill(1);

            // Labels: target tokens (shifted for next-token prediction)
            for i in 0..copy_len {
                labels[prompt_len + i] = target_tokens[i] as i64;
            }
        }

        TrainingSample {
            input_ids,
            attention_mask,
            labels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_parse_decision_example() {
        let json = r#"{"scenario":"Test scenario","context":{"node":"ftw3"},"decision":"approve","confidence":95,"reasoning":"Test reasoning","risk_factors":[],"next_steps":["verify"]}"#;
        let example: DecisionExample = serde_json::from_str(json).unwrap();
        assert_eq!(example.decision, "approve");
        assert_eq!(example.confidence, 95);
    }

    #[test]
    fn test_format_sample_has_labels() {
        // Create a minimal loader (tokenizer config doesn't matter for this test)
        let config = qwen2_bpe::Qwen2Config {
            vocab: HashMap::new(),
            merges: vec![],
            special_tokens: None,
        };
        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        let loader = DatasetLoader::new(tokenizer, 512);

        let example = DecisionExample {
            scenario: "Test".to_string(),
            context: serde_json::json!({"node": "test"}),
            decision: "approve".to_string(),
            confidence: 95,
            reasoning: "Test".to_string(),
            risk_factors: vec![],
            next_steps: vec!["verify".to_string()],
        };

        let sample = loader.format_decision_sample(example);
        assert!(!sample.input_ids.is_empty());
        // Should have some non-ignored labels
        assert!(sample.labels.iter().any(|&l| l != -100));
    }
}
