//! Test tokenizer with Week 16 prompt

use pesti_runner::transformer::tokenizer::{load_tokenizer_from_gguf, TokenizerBackend};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");

    println!("Loading tokenizer from GGUF...");
    
    let (_, tokenizer) = load_tokenizer_from_gguf(model_path, TokenizerBackend::MistralRs)?;

    // Test with the Week 16 prompt
    let test_text = "The quick brown fox jumps over the lazy dog.";
    let tokens = tokenizer.encode(test_text)?;
    
    println!("\nPrompt: '{}'", test_text);
    println!("Token count: {}", tokens.len());
    println!("Tokens: {:?}", tokens);

    // Decode back to verify
    let decoded = tokenizer.decode(&tokens)?;
    println!("Decoded: '{}'", decoded);

    Ok(())
}
