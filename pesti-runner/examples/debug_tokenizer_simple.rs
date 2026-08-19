//! Test with ByteLevel pre-tokenizer like Qwen2/GPT-2.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);

    println!("Model type: {}", config.model_type);
    println!("Base vocab size: {}", config.base_vocab.len());
    println!("Special tokens: {}", config.special_tokens.len());
    println!("Merges loaded: {}", config.merges.len());

    // Try building with ByteLevel BPE + ByteLevel pre-tokenizer
    use tokenizers::models::bpe::{BPE, BPEOptions};
    use tokenizers::tokenizer::{PreTokenizer, Tokenizer};
    
    // Build vocab: token -> id (like your current code)
    let mut vocab: std::collections::HashMap<String, u32> = config.base_vocab.iter()
        .map(|(id, tok)| (tok.clone(), *id))
        .chain(config.special_tokens.iter().map(|(id, tok)| (tok.clone(), *id)))
        .collect();

    // Build merges: Vec<(String, String)>
    let mut merges: Vec<(String, String)> = Vec::new();
    let id_to_token: std::collections::HashMap<u32, &String> = config.base_vocab.iter()
        .map(|(id, tok)| (*id, tok))
        .chain(config.special_tokens.iter().map(|(id, tok)| (*id, tok)))
        .collect();

    for (left_id, right_id) in &config.merges {
        if let (Some(left), Some(right)) = (id_to_token.get(left_id), id_to_token.get(right_id)) {
            merges.push((left.clone(), right.clone()));
        }
    }

    println!("Merges count: {}", merges.len());

    // Create BPE tokenizer
    let bpe = BPE::new(vocab, merges, BPEOptions::default());
    let mut tokenizer = Tokenizer::new(bpe);

    // Set ByteLevel pre-tokenizer (Qwen2/GPT-2 style)
    // Note: tokenizers crate API may vary by version
    println!("\n✅ Built BPE tokenizer with {} vocab, {} merges", 
             config.base_vocab.len() + config.special_tokens.len(), 
             merges.len());

    let prompt = "The quick brown fox";
    match tokenizer.encode(prompt, true) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            println!("Encoding result: {} tokens", ids.len());
            if !ids.is_empty() {
                println!("First 10 token IDs: {:?}", &ids[..ids.len().min(10)]);
                
                match tokenizer.decode(ids, true) {
                    Ok(decoded) => println!("Decoded: {}", decoded),
                    Err(e) => println!("Decode error: {}", e),
                }
            } else {
                println!("⚠️  Still 0 tokens - checking pre-tokenizer...");
            }
        }
        Err(e) => println!("Encoding error: {}", e),
    }

    Ok(())
}
