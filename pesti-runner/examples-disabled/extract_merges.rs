//! Extract merge pairs from GGUF tokenizer JSON

use pesti_runner::transformer::tokenizer::{load_tokenizer_from_gguf, GgufTokenizerConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    
    println!("Loading tokenizer from GGUF...");
    let (_, tokenizer) = load_tokenizer_from_gguf(model_path)?;
    
    // Get vocabulary
    let vocab = tokenizer.get_vocab(true);
    println!("Vocabulary size: {}", vocab.len());
    
    // For now, just dump a sample of the tokenizer to see what's available
    // We'll manually inspect the JSON structure
    
    Ok(())
}
