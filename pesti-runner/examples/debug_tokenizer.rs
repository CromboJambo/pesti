//! Debug tokenizer loading and encoding with mistral.rs.

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

    // Test encoding - use the raw tokenizer directly
    let test_text = "Hello, world!";
    let encoding = tokenizer.encode(test_text, false)?;
    let tokens = encoding.get_ids().to_vec();
    
    println!("\n⚠️ WARNING: {} tokens for '{}'", tokens.len(), test_text);
    println!("✅ Token count: {} ✅", tokens.len());
    println!("Tokens: {:?}", tokens);

    // Test decoding
    let decoded = tokenizer.decode(&tokens, false)?;
    println!("Decoded: {}", decoded);

    Ok(())
}
