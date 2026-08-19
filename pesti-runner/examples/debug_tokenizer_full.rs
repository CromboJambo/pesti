//! Debug tokenizer loading and encoding with full introspection.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("=== Debug GGUF Tokenizer Loading ===");
    println!("Model: {}", model_path);
    
    // Load tokenizer config from GGUF header
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);
    
    println!("\n=== Tokenizer Config ===");
    println!("Model type: {}", config.model_type);
    println!("Base vocab size: {}", config.base_vocab.len());
    println!("Special tokens: {}", config.special_tokens.len());
    println!("Merges count: {}", config.merges.len());
    println!("BOS token ID: {:?}", config.bos_token_id);
    println!("EOS token ID: {:?}", config.eos_token_id);

    // Print first 5 base tokens
    println!("\n=== First 5 base tokens ===");
    for (id, tok) in config.base_vocab.iter().take(5) {
        println!("  id={}: {:?}", id, tok);
    }

    // Print first 3 merges (as IDs)
    println!("\n=== First 3 merges (ID pairs) ===");
    for (left_id, right_id) in config.merges.iter().take(3) {
        let left = config.base_vocab.get(left_id).or(config.special_tokens.get(left_id));
        let right = config.base_vocab.get(right_id).or(config.special_tokens.get(right_id));
        println!("  ({}, {}) -> ({:?}, {:?})", left_id, right_id, left, right);
    }

    // Build tokenizer and test
    println!("\n=== Building Tokenizer ===");
    let tokenizer = config.to_tokenizer();
    
    println!("\n=== Tokenizer Test ===");
    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("Prompt: {}", prompt);
    
    match tokenizer.encode(prompt, true) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            println!("✅ Encoding successful!");
            println!("Token count: {}", ids.len());
            if !ids.is_empty() {
                println!("First 10 token IDs: {:?}", &ids[..ids.len().min(10)]);
                
                // Try decoding
                match tokenizer.decode(ids, true) {
                    Ok(decoded) => {
                        println!("\n✅ Decoding successful!");
                        println!("Decoded text: {}", decoded);
                    }
                    Err(e) => {
                        println!("\n❌ Decode error: {}", e);
                    }
                }
            } else {
                println!("⚠️ WARNING: 0 tokens generated — tokenizer may be empty or broken");
                
                // Try to see what the config looks like
                let mut added_tokens: Vec<_> = config.special_tokens.iter()
                    .map(|(id, tok)| format!("id={} content={}", id, tok))
                    .collect();
                added_tokens.sort();
                println!("Special tokens (first 10):");
                for s in added_tokens.iter().take(10) {
                    println!("  {}", s);
                }
            }
        }
        Err(e) => {
            println!("❌ Encoding error: {}", e);
        }
    }
    
    Ok(())
}
