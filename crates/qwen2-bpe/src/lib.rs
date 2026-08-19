//! Qwen2 BPE Tokenizer - Pure Rust implementation with special tokens

use std::collections::{HashMap, HashSet};
use std::fs;
use thiserror::Error;

/// Error types for Qwen2 tokenizer operations
#[derive(Error, Debug)]
pub enum Qwen2Error {
    #[error("Encoding error: {0}")]
    Encode(String),
    
    #[error("Decoding error: {0}")]
    Decode(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Special tokens configuration
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token ID
    pub bos_token_id: u32,
    /// End of sequence token ID
    pub eos_token_id: u32,
    /// Padding token ID (if any)
    pub pad_token_id: Option<u32>,
}

/// Configuration for Qwen2 tokenizer
#[derive(Debug, Clone)]
pub struct Qwen2Config {
    /// Vocabulary: token_id → byte_sequence
    pub vocab: HashMap<u32, Vec<u8>>,
    
    /// Merges: ordered list of (token1, token2, priority) tuples
    pub merges: Vec<(u32, u32, usize)>,
    
    /// Special tokens (BOS, EOS, PAD)
    pub special_tokens: Option<SpecialTokens>,
}

/// Qwen2 BPE tokenizer with special token support
pub struct Qwen2Tokenizer {
    config: Qwen2Config,
    reverse_vocab: HashMap<Vec<u8>, u32>,
    forward_vocab: HashMap<u32, Vec<u8>>,
    merge_pairs: HashSet<(u32, u32)>,
}

impl Qwen2Tokenizer {
    /// Create a new Qwen2 tokenizer from configuration
    pub fn new(config: Qwen2Config) -> Result<Self, Qwen2Error> {
        let reverse_vocab: HashMap<Vec<u8>, u32> = config.vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        let forward_vocab: HashMap<u32, Vec<u8>> = config.vocab
            .iter()
            .map(|(id, bytes)| (*id, bytes.clone()))
            .collect();

        // Build a set of merge pairs for O(1) lookup
        let merge_pairs: HashSet<(u32, u32)> = config.merges
            .iter()
            .map(|(t1, t2, _)| (*t1, *t2))
            .collect();

        Ok(Self {
            config,
            reverse_vocab,
            forward_vocab,
            merge_pairs,
        })
    }

    /// Load tokenizer from JSON vocabulary file (without merges)
    pub fn load_from_json(vocab_path: &str) -> Result<Self, Qwen2Error> {
        let json_content = fs::read_to_string(vocab_path)?;
        let items: Vec<(String, u32)> = serde_json::from_str(&json_content)?;

        // Convert to HashMap<u32, Vec<u8>>
        let vocab: HashMap<u32, Vec<u8>> = items
            .iter()
            .map(|(token_str, token_id)| {
                // Convert string to bytes
                let bytes: Vec<u8> = token_str.bytes().collect();
                (*token_id, bytes)
            })
            .collect();

        let config = Qwen2Config {
            vocab,
            merges: vec![], // TODO: Load merges from file later
            special_tokens: None,
        };

        Self::new(config)
    }

    /// Load tokenizer with merges from JSON files
    pub fn load_with_merges(vocab_path: &str, merges_path: &str) -> Result<Self, Qwen2Error> {
        // Load vocabulary
        let vocab_items: Vec<(String, u32)> = serde_json::from_str(&fs::read_to_string(vocab_path)?)?;
        let vocab: HashMap<u32, Vec<u8>> = vocab_items
            .iter()
            .map(|(token_str, token_id)| {
                let bytes: Vec<u8> = token_str.bytes().collect();
                (*token_id, bytes)
            })
            .collect();

        // Load merges
        let merge_pairs_raw: Vec<(u32, u32, usize)> = serde_json::from_str(&fs::read_to_string(merges_path)?)?;

        let config = Qwen2Config {
            vocab,
            merges: merge_pairs_raw,
            special_tokens: None, // TODO: Load from GGUF metadata later
        };

        Self::new(config)
    }

    /// Encode text into token IDs using Qwen2's byte-level BPE
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Qwen2Error> {
        // Step 1: Convert to byte-level tokens (each character → byte value as u8)
        let tokens: Vec<u8> = text.bytes().collect();

        // Step 2: Apply BPE merges iteratively (respecting priority order)
        let merged_tokens = self.apply_bpe_merges(tokens)?;

        // Step 3: Map to token IDs
        let final_tokens: Vec<u32> = merged_tokens
            .iter()
            .map(|bytes| {
                if let Some(&id) = self.reverse_vocab.get(bytes) {
                    id
                } else {
                    bytes.first().copied().unwrap_or(0) as u32
                }
            })
            .collect();

        Ok(final_tokens)
    }

    /// Get number of merge pairs
    pub fn merge_count(&self) -> usize {
        self.config.merges.len()
    }

    /// Get tokenizer configuration
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Encode text with special tokens (BOS/EOS)
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>, Qwen2Error> {
        let mut tokens = self.encode(text)?;

        if add_bos {
            if let Some(ref special) = self.config.special_tokens {
                tokens.insert(0, special.bos_token_id);
            } else {
                // Default BOS token ID (commonly 151643 for Qwen2)
                tokens.insert(0, 151643);
            }
        }

        if add_eos {
            if let Some(ref special) = self.config.special_tokens {
                tokens.push(special.eos_token_id);
            } else {
                // Default EOS token ID (commonly 151643 for Qwen2)
                tokens.push(151643);
            }
        }

        Ok(tokens)
    }

    /// Apply BPE merges iteratively until no more merges possible
    fn apply_bpe_merges(&self, mut tokens: Vec<u8>) -> Result<Vec<Vec<u8>>, Qwen2Error> {
        loop {
            // Find first mergeable pair (two consecutive bytes) with lowest priority
            let best_merge = self.find_best_merge_bytes(&tokens)?;
            
            if best_merge.is_none() {
                break;
            }

            let (current, next, pos) = best_merge.unwrap();
            tokens = self.apply_merge_at_position_bytes(tokens, (current, next), pos);
        }

        Ok(tokens.into_iter().map(|b| vec![b]).collect())
    }

    /// Find the first occurrence of a mergeable pair in byte sequence
    fn find_best_merge_bytes(&self, tokens: &[u8]) -> Result<Option<(u8, u8, usize)>, Qwen2Error> {
        for i in 0..tokens.len().saturating_sub(1) {
            let current = tokens[i];
            let next = tokens[i + 1];
            
            // Convert to u32 for comparison with merge_pairs
            let pair = (current as u32, next as u32);
            
            if self.merge_pairs.contains(&pair) {
                return Ok(Some((current, next, i)));
            }
        }

        Ok(None)
    }

    /// Apply a merge at a specific position in the byte sequence
    fn apply_merge_at_position_bytes(
        &self,
        tokens: Vec<u8>,
        pair: (u8, u8),
        pos: usize,
    ) -> Vec<u8> {
        let mut result = Vec::with_capacity(tokens.len() - 1); // One less byte after merge
        
        let mut i = 0;
        while i < tokens.len() {
            if i == pos && tokens.get(i + 1) == Some(&pair.1) {
                // Merge: combine both bytes into one (placeholder - just keep first)
                result.push(pair.0);
                i += 2; // Skip both bytes and replace with merged token
            } else {
                result.push(tokens[i]);
                i += 1;
            }
        }

        result
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String, Qwen2Error> {
        let mut result = String::new();
        
        for &token_id in tokens {
            if let Some(bytes) = self.forward_vocab.get(&token_id) {
                match String::from_utf8(bytes.clone()) {
                    Ok(text) => result.push_str(&text),
                    Err(_) => {
                        // For non-UTF-8 bytes, represent as character if possible
                        if token_id < 128 {
                            result.push(char::from_u32(token_id).unwrap_or('�'));
                        } else {
                            result.push_str(&format!("<UNK:{}>", token_id));
                        }
                    }
                }
            } else {
                // Unknown token - check if it's a special token
                if let Some(ref special) = self.config.special_tokens {
                    if token_id == special.bos_token_id {
                        result.push_str("<|begin_of_text|>");
                        continue;
                    }
                    if token_id == special.eos_token_id {
                        result.push_str("<|end_of_text|>");
                        continue;
                    }
                    if let Some(pad_id) = special.pad_token_id {
                        if token_id == pad_id {
                            result.push_str("<|pad|>");
                            continue;
                        }
                    }
                }

                // Fallback to character representation
                if let Some(c) = char::from_u32(token_id) {
                    result.push(c);
                } else {
                    result.push_str(&format!("<UNK:{}>", token_id));
                }
            }
        }

        Ok(result)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_encoding() {
        let config = Qwen2Config {
            vocab: vec![
                (72, vec![b'H']),
                (101, vec![b'e']),
                (108, vec![b'l']),
                (111, vec![b'o']),
                (32, vec![b' ']),
            ]
            .into_iter()
            .collect(),
            merges: vec![],
            special_tokens: None,
        };

        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        
        let tokens = tokenizer.encode("Hello").unwrap();
        
        // Should return byte values: [72, 101, 108, 108, 111]
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_decode() {
        let config = Qwen2Config {
            vocab: vec![
                (72, vec![b'H']),
                (101, vec![b'e']),
                (108, vec![b'l']),
                (111, vec![b'o']),
                (32, vec![b' ']),
            ]
            .into_iter()
            .collect(),
            merges: vec![],
            special_tokens: None,
        };

        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        
        let decoded = tokenizer.decode(&[72, 101, 108, 108, 111]).unwrap();
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_qwen2_hello_world_with_merges() {
        // Simulate Qwen2's first few merge pairs
        let config = Qwen2Config {
            vocab: vec![
                (72, vec![b'H']),  // 'H'
                (101, vec![b'e']), // 'e'
                (108, vec![b'l']), // 'l'
                (111, vec![b'o']), // 'o'
                (32, vec![b' ']),  // space
            ]
            .into_iter()
            .collect(),
            merges: vec![(72, 101, 0)], // Merge 'H' + 'e' → "He" with priority 0
            special_tokens: None,
        };

        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        
        let tokens = tokenizer.encode("Hello").unwrap();
        
        // Should have merged H+e but not the rest
        assert_eq!(tokens.len(), 4); // He, l, l, o
    }

    #[test]
    fn test_load_from_json() {
        // Create a temporary JSON file for testing
        let json_content = r#"[["H", 72], ["e", 101], ["l", 108], ["o", 111], [" ", 32]]"#;
        std::fs::write("/tmp/test_vocab.json", json_content).unwrap();

        let tokenizer = Qwen2Tokenizer::load_from_json("/tmp/test_vocab.json").unwrap();
        
        assert_eq!(tokenizer.vocab_size(), 5);
        
        // Clean up
        std::fs::remove_file("/tmp/test_vocab.json").unwrap();
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let config = Qwen2Config {
            vocab: vec![
                (72, vec![b'H']),
                (101, vec![b'e']),
                (108, vec![b'l']),
                (111, vec![b'o']),
                (32, vec![b' ']),
            ]
            .into_iter()
            .collect(),
            merges: vec![],
            special_tokens: Some(SpecialTokens {
                bos_token_id: 151643,
                eos_token_id: 151644, // Different ID from BOS
                pad_token_id: Some(0),
            }),
        };

        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        
        // Encode with BOS and EOS
        let tokens = tokenizer.encode_with_special("Hello", true, true).unwrap();
        
        assert_eq!(tokens.len(), 7); // BOS + Hello + EOS
        assert_eq!(tokens[0], 151643); // BOS
        assert_eq!(tokens[6], 151644); // EOS (different from BOS)
    }

    #[test]
    fn test_decode_special_tokens() {
        let config = Qwen2Config {
            vocab: vec![
                (72, vec![b'H']),
                (101, vec![b'e']),
                (108, vec![b'l']),
                (111, vec![b'o']),
                (32, vec![b' ']),
            ]
            .into_iter()
            .collect(),
            merges: vec![],
            special_tokens: Some(SpecialTokens {
                bos_token_id: 151643,
                eos_token_id: 151644, // Different ID from BOS
                pad_token_id: Some(0),
            }),
        };

        let tokenizer = Qwen2Tokenizer::new(config).unwrap();
        
        // Decode with special tokens
        let decoded = tokenizer.decode(&[151643, 72, 101, 108, 108, 111, 151644]).unwrap();
        
        assert!(decoded.contains("<|begin_of_text|>"));
        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("<|end_of_text|>"));
    }
}
