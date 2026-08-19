//! Debug tokenizer loading with pre-tokenizer hint inspection.

use pesti_runner::transformer::{load_tokenizer_from_gguf, GgufTokenizerConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    println!("=== Loading GGUF tokenizer ===");
    let (config, _tokenizer) = load_tokenizer_from_gguf(model_path)?;

    println!("\n=== Config Details ===");
    println!("Model type: {}", config.model_type);
    println!("Pre-tokenizer hint: {:?}", config.pre_tokenizer_hint);
    
    // Print first 10 base tokens to see if they're bytes or strings
    println!("\nFirst 10 base tokens (by ID):");
    let mut sorted_ids: Vec<_> = config.base_vocab.keys().cloned().collect();
    sorted_ids.sort();
    for id in sorted_ids.iter().take(10) {
        if let Some(token) = config.base_vocab.get(id) {
            println!("  ID {}: {:?}", id, token);
        }
    }

    // Check if first tokens are single bytes (ASCII 0-255)
    println!("\nChecking byte-level tokens:");
    for id in sorted_ids.iter().take(256) {
        if let Some(token) = config.base_vocab.get(id) {
            if token.len() == 1 {
                println!("  ID {}: byte '{}' (u8: {})", id, token, token.as_bytes()[0]);
            } else {
                println!("  ID {}: string '{}'", id, token);
                break; // Found non-byte token
            }
        }
    }

    Ok(())
}
