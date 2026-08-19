//! Check GGUF header for tokenizer info.

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let header = pesti_gguf::parser::parse_gguf(Path::new(model_path))?;

    println!("=== GGUF Tokenizer KV Pairs ===");
    for (key, value) in &header.kv {
        if key.contains("tokenizer") || key.contains("vocab") || key.contains("merge") {
            match value {
                pesti_gguf::types::GgufValue::String(s) => {
                    println!("{} = {}", key, s);
                }
                pesti_gguf::types::GgufValue::Array(pesti_gguf::types::GgufArrayType::String, arr) => {
                    println!("{} (array of {} strings)", key, arr.len());
                    if arr.len() <= 5 {
                        for s in arr.iter().take(3) {
                            println!("  → {}", s);
                        }
                    }
                }
                pesti_gguf::types::GgufValue::Array(pesti_gguf::types::GgufArrayType::U32, arr) => {
                    println!("{} (array of {} u32)", key, arr.len());
                }
                _ => {
                    println!("{} = {:?}", key, value);
                }
            }
        }
    }

    Ok(())
}
