//! Test with explicit BPE config matching HuggingFace format.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);

    println!("Model type: {}", config.model_type);
    println!("Base vocab size: {}", config.base_vocab.len());
    println!("Special tokens: {}", config.special_tokens.len());
    println!("Merges loaded: {}", config.merges.len());

    // Try building with explicit BPE type and whitespace pre-tokenizer
    use tokenizers::tokenizer::{Tokenizer, PreTokenizer};
    use tokenizers::models::bpe::{BPE, BPEOptions};
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    
    // Build vocab: token -> id
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

    // Create BPE tokenizer with explicit options
    let bpe = BPE::new(
        vocab,
        merges,
        BPEOptions {
            unk_token: None,
            end_of_word_suffix: None,
            fuse_unk: false,
        },
    );

    let mut tokenizer = Tokenizer::new(bpe);
    
    // Set pre-tokenizer to whitespace (Qwen2/GPT-2 style)
    tokenizer.set_pre_tokenizer(PreTokenizer::custom(Whitespace));

    println!("\n✅ BPE tokenizer created with explicit config!");
    
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
                println!("⚠️ WARNING: 0 tokens - checking if prompt contains special chars...");
                println!("Prompt bytes: {:?}", prompt.as_bytes());
            }
        }
        Err(e) => println!("Encoding error: {}", e),
    }

    Ok(())
}
