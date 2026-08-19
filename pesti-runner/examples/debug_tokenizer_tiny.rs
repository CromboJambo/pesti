//! Debug tokenizer JSON config output - very minimal version.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);

    // Build the exact JSON that to_tokenizer() builds - but only first 10 vocab entries
    let mut vocab: std::collections::HashMap<String, u32> = config.base_vocab.iter()
        .map(|(id, tok)| (tok.clone(), *id))
        .chain(config.special_tokens.iter().map(|(id, tok)| (tok.clone(), *id)))
        .collect();

    // Build reverse lookup: id → token string
    let id_to_token: std::collections::HashMap<u32, &String> = config.base_vocab.iter()
        .map(|(id, tok)| (*id, tok))
        .chain(config.special_tokens.iter().map(|(id, tok)| (*id, tok)))
        .collect();

    // Build merges list - only first 5 for testing
    let mut merges_json: Vec<serde_json::Value> = config.merges.iter()
        .filter_map(|(left_id, right_id)| {
            if let (Some(left), Some(right)) = (id_to_token.get(left_id), id_to_token.get(right_id)) {
                Some(serde_json::json!([left.as_str(), right.as_str()]))
            } else {
                None
            }
        })
        .take(5) // Only first 5 merges for testing
        .collect();

    println!("Merges in JSON: {}", merges_json.len());
    println!("First merge as string: {}", merges_json[0]);

    // Build added_tokens - only first 3
    let mut added_tokens_json: Vec<serde_json::Value> = config.special_tokens.iter()
        .take(3)
        .map(|(id, tok)| {
            serde_json::json!({
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
                "id": *id
            })
        })
        .collect();

    let config_json = serde_json::json!({
        "version": "1.0",
        "type": "BPE",
        "dropout": null,
        "unk_token": null,
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "vocab": vocab,
        "merges": merges_json,
        "added_tokens": serde_json::Value::Array(added_tokens_json)
    });

    println!("\n=== Full JSON config ===");
    let json_str = config_json.to_string();
    println!("{}", json_str);

    // Try to load tokenizer from this JSON
    println!("\n=== Loading tokenizer from JSON ===");
    match tokenizers::Tokenizer::from_bytes(json_str.as_bytes()) {
        Ok(t) => {
            println!("✅ Tokenizer loaded successfully!");
            
            let prompt = "The quick brown fox jumps over the lazy dog.";
            match t.encode(prompt, true) {
                Ok(encoding) => {
                    let ids = encoding.get_ids();
                    println!("Encoding result: {} tokens", ids.len());
                    if !ids.is_empty() {
                        println!("First 10 token IDs: {:?}", &ids[..ids.len().min(10)]);
                        
                        match t.decode(ids, true) {
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
        }
        Err(e) => {
            println!("❌ Failed to load tokenizer: {}", e);
            
            // Print first 1KB of JSON for debugging
            if json_str.len() > 1024 {
                println!("\n=== First 1KB of JSON ===");
                println!("{}", &json_str[..1024]);
            }
        }
    }

    Ok(())
}
