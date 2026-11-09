//! Test correct tokenizer config format

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== Test Correct Tokenizer Config ===");

    // Load weights to get tokenizer metadata
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let header = &weights.header;

    // Extract vocab tokens from GGUF
    let mut added_tokens: Vec<(u32, String)> = Vec::new();
    let token_count = header
        .get_kv_u32("tokenizer.ggml.tokens")
        .or_else(|| header.get_kv_u32("tokenizer.ggml.length"))
        .unwrap_or(0);

    println!("Token count from GGUF: {}", token_count);

    for id in 0..token_count.min(100) {
        // Only first 100 for testing
        let key = format!("tokenizer.ggml.tokens.{}", id);
        if let Some(value) = header.get_kv_str(&key) {
            added_tokens.push((id, value.to_string()));
        }
    }

    println!(
        "\nFirst 5 tokens: {:?}",
        &added_tokens[..5.min(added_tokens.len())]
    );

    // Try building tokenizer with CORRECT format (map instead of array)
    let vocab_map: std::collections::HashMap<String, u32> = added_tokens
        .iter()
        .map(|(id, token)| (token.clone(), *id))
        .collect();

    println!("\nVocab map size: {}", vocab_map.len());

    // Build config in correct format for GPT-2 tokenizer
    let config_json = serde_json::json!({
        "version": "1.0",
        "type": "BPE",
        "vocab": vocab_map,
        "merges": vec![vec![String::new(), String::new()]], // Empty merges for word-level tokenizer
        "added_tokens": vec![
            serde_json::json!({"content": "<|endoftext|>", "special": true}),
        ]
    });

    println!(
        "\nConfig JSON (truncated): {}",
        serde_json::to_string(&config_json)
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>()
    );

    match tokenizers::Tokenizer::from_bytes(config_json.to_string().as_bytes()) {
        Ok(tokenizer) => {
            let prompt = "The quick brown fox jumps over the lazy dog.";
            match tokenizer.encode(prompt, true) {
                Ok(encoding) => {
                    println!("\n✅ Success with map format!");
                    println!("Tokens: {}", encoding.get_ids().len());
                    if !encoding.get_ids().is_empty() {
                        println!(
                            "First 10 IDs: {:?}",
                            &encoding.get_ids()[..10.min(encoding.get_ids().len())]
                        );
                    }
                }
                Err(e) => println!("\n❌ Encode error: {}", e),
            }
        }
        Err(e) => println!("\n❌ Config parse error: {}", e),
    }

    Ok(())
}
