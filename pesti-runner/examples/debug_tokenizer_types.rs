//! Test different tokenizer types for Qwen2.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);

    println!("Model type: {}", config.model_type);
    println!("Merges loaded: {}", config.merges.len());

    // Try 1: Regular BPE (current approach)
    {
        let tokenizer = pesti_runner::transformer::tokenizer::load_tokenizer_from_gguf(model_path)?;
        let ids = tokenizer.encode("The quick brown fox", true)?.get_ids();
        println!("\n📦 Current BPE: {} tokens", ids.len());
    }

    // Try 2: ByteLevel BPE (Qwen2/GPT-2 style)
    {
        use tokenizers::tokenizer::{Tokenizer, PreTokenizer};
        use tokenizers::models::bpe::{BPE, BPEOptions};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        
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

        // Create ByteLevel BPE tokenizer
        let bpe = BPE::new(vocab, merges, BPEOptions::default());
        let mut tokenizer = Tokenizer::new(bpe);
        
        // Use ByteLevel pre-tokenizer (like GPT-2/Qwen2)
        tokenizer.set_pre_tokenizer(PreTokenizer::custom(ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
        }));

        let prompt = "The quick brown fox";
        match tokenizer.encode(prompt, true) {
            Ok(encoding) => {
                let ids = encoding.get_ids();
                println!("🔥 ByteLevel BPE: {} tokens ✅", ids.len());
                if !ids.is_empty() {
                    println!("   First 5 IDs: {:?}", &ids[..ids.len().min(5)]);
                }
            }
            Err(e) => println!("❌ ByteLevel BPE error: {}", e),
        }
    }

    // Try 3: Check if Qwen2 uses WordPiece or other type
    {
        let tokenizer_type = config.model_type.clone();
        println!("\n📋 Model claims to be: {}", tokenizer_type);
        
        // Qwen2 is actually GPT-2 style (ByteLevel BPE)
        // Check if we can detect this from GGUF
        if let Some(model_arch) = header.string_kv("tokenizer.model") {
            println!("   tokenizer.model: {}", model_arch);
        }
    }

    Ok(())
}
