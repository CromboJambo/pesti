//! Debug tokenizer loading and encoding with mistral.rs.

use pesti_runner::transformer::tokenizer::{load_tokenizer_from_gguf, GgufTokenizerConfig, TokenizerBackend};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    println!("Loading tokenizer from GGUF...");
    
    let (config, tokenizer) = load_tokenizer_from_gguf(model_path, TokenizerBackend::MistralRs)?;

    println!("✅ Tokenizer loaded successfully!");
    println!("Vocab size: {} ✅", config.vocab_size);
    println!("BOS token ID: {:?}", config.bos_token_id);
    println!("EOS token ID: {:?}", config.eos_token_id);

    // Test encoding - use the raw tokenizer directly
    let test_text = "Hello, world!";
    let tokens = tokenizer.encode(test_text)?;
    
    println!("\n⚠️ WARNING: {} tokens for '{}'", tokens.len(), test_text);
    println!("✅ Token count: {} ✅", tokens.len());
    println!("Tokens: {:?}", tokens);

    // Test decoding
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: {}", decoded);

    Ok(())
}
