//! Debug tokenizer JSON config output.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;
    let config = pesti_runner::transformer::tokenizer::tokenizer_config_from_header(&header);
    
    println!("=== Tokenizer Config ===");
    println!("Base vocab size: {}", config.base_vocab.len());
    println!("Special tokens: {}", config.special_tokens.len());
    println!("Merges count: {}", config.merges.len());

    // Build the exact JSON that to_tokenizer() builds
    let mut vocab: std::collections::HashMap<String, u32> = config.base_vocab.iter()
        .map(|(id, tok)| (tok.clone(), *id))
        .chain(config.special_tokens.iter().map(|(id, tok)| (tok.clone(), *id)))
        .collect();

    // Build reverse lookup: id → token string
    let id_to_token: std::collections::HashMap<u32, &String> = config.base_vocab.iter()
        .map(|(id, tok)| (*id, tok))
        .chain(config.special_tokens.iter().map(|(id, tok)| (*id, tok)))
        .collect();

    // Build merges list
    let mut merges_json: Vec<serde_json::Value> = config.merges.iter()
        .filter_map(|(left_id, right_id)| {
            if let (Some(left), Some(right)) = (id_to_token.get(left_id), id_to_token.get(right_id)) {
                Some(serde_json::json!([left.as_str(), right.as_str()]))
            } else {
                None
            }
        })
        .collect();

    println!("Merges in JSON: {}", merges_json.len());

    // Print first 5 merges as strings
    println!("\n=== First 5 merge pairs (as strings) ===");
    for m in merges_json.iter().take(5) {
        if let serde_json::Value::Array(arr) = m {
            if let (Some(a), Some(b)) = (arr.get(0), arr.get(1)) {
                println!("  '{}' + '{}' -> merged", a, b);
            }
        }
    }

    // Print first 5 vocab entries
    println!("\n=== First 5 vocab entries ===");
    let mut sorted_vocab: Vec<_> = vocab.iter().collect();
    sorted_vocab.sort_by_key(|(_, id)| *id);
    for (tok, id) in sorted_vocab.iter().take(5) {
        println!("  id={}: {:?}", id, tok);
    }

    // Build added_tokens with required fields
    let mut added_tokens_json: Vec<serde_json::Value> = config.special_tokens.iter()
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

    // Add BOS/EOS if not present
    if let Some(bos_id) = config.bos_token_id {
        if let Some(bos_tok) = config.special_tokens.get(&bos_id) {
            if !added_tokens_json.iter().any(|t| t["id"] == serde_json::json!(bos_id)) {
                added_tokens_json.push(serde_json::json!({
                    "content": bos_tok,
                    "special": true,
                    "id": bos_id
                }));
            }
        }
    }

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

    println!("\n=== Full JSON config (first 2KB) ===");
    let json_str = config_json.to_string();
    if json_str.len() > 2048 {
        println!("{}", &json_str[..2048]);
        println!("... ({} more bytes)", json_str.len() - 2048);
    } else {
        println!("{}", json_str);
    }

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
        }
    }

    Ok(())
}
