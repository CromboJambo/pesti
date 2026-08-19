//! Debug tokenizer loading from file.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);

    println!("Model type: {}", config.model_type);
    println!("Base vocab size: {}", config.base_vocab.len());
    println!("Special tokens: {}", config.special_tokens.len());
    println!("Merges loaded: {}", config.merges.len());

    let tokenizer = config.to_tokenizer();

    println!("\n✅ Tokenizer loaded successfully!");
    
    let prompt = "The quick brown fox jumps over the lazy dog.";
    match tokenizer.encode(prompt, true) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            println!("Encoding result: {} tokens", ids.len());
            if !ids.is_empty() {
                println!("First 10 token IDs: {:?}", &ids[..ids.len().min(10)]);
                
                match tokenizer.decode(ids, true) {
                    Ok(decoded) => {
                        println!("Decoded text: {}", decoded);
                    }
                    Err(e) => println!("Decode error: {}", e),
                }
            } else {
                println!("⚠️ WARNING: 0 tokens");
            }
        }
        Err(e) => println!("Encoding error: {}", e),
    }

    Ok(())
}
