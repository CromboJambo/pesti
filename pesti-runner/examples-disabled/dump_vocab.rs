//! Debug tokenizer loading and encoding with mistral.rs + dump vocabulary.

use pesti_runner::transformer::tokenizer::{load_tokenizer_from_gguf, GgufTokenizerConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    println!("Loading tokenizer from GGUF...");
    
    let (config, tokenizer) = load_tokenizer_from_gguf(model_path)?;
    
    println!("✅ Tokenizer loaded successfully!");
    println!("Model type: {}", config.tokenizer_model);
    println!("Base vocab size: {} ✅", config.base_vocab_size);
    println!("Special tokens: {} ✅", config.num_special_tokens);
    
    // Get full vocabulary
    let vocab = tokenizer.get_vocab(true);  // with_added_tokens
    let mut items: Vec<(&String, &u32)> = vocab.iter().collect();
    items.sort_by_key(|(_, id)| *id);
    
    println!("\n📊 Vocabulary Statistics:");
    println!("Total tokens: {}", items.len());
    
    // Show first 100 tokens (including byte-level)
    println!("\nFirst 100 tokens (sorted by ID):");
    for (token, id) in items.iter().take(100) {
        let display_token = if token.chars().all(|c| c.is_ascii() && !c.is_control()) {
            format!("'{}'", token)
        } else {
            format!("{:?}", token)
        };
        println!("  {:6}: {}", *id, display_token);
    }
    
    // Show byte-level tokens (0-255 range)
    println!("\n📦 Byte-level tokens (IDs 0-255):");
    for (token, id) in items.iter().filter(|(_, id)| **id <= 255) {
        let display_token = if token.chars().all(|c| c.is_ascii() && !c.is_control()) {
            format!("'{}'", token)
        } else {
            format!("{:?}", token)
        };
        println!("  {:6}: {}", *id, display_token);
    }
    
    // Save to JSON
    let json_output = serde_json::to_string_pretty(&items)?;
    std::fs::write("/tmp/qwen2_vocab_dump.json", &json_output)?;
    println!("\n💾 Saved full vocabulary to /tmp/qwen2_vocab_dump.json");
    
    // Test encoding
    let test_text = "Hello, world!";
    let encoding = tokenizer.encode(test_text, false)?;
    let tokens = encoding.get_ids().to_vec();
    
    println!("\n🔤 Encoding test: '{}'", test_text);
    println!("  Tokens: {:?}", tokens);
    
    // Decode back
    let decoded = tokenizer.decode(&tokens, false)?;
    println!("  Decoded: {}", decoded);
    
    Ok(())
}
