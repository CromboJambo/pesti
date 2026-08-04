
use pesti_gguf::parser::parse_gguf;
use std::path::Path;

fn main() {
    let path = "/mnt/data/state/ai/lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    println!("Parsing: {}", path);
    
    match parse_gguf(Path::new(path)) {
        Ok(header) => {
            println!("Success! Version={}, KV pairs={}, Tensors={}", header.version, header.kv_pairs.len(), header.tensors.len());
            
            // Print first few KV pairs
            for (i, kv) in header.kv_pairs.iter().enumerate() {
                if i < 5 {
                    println!("KV {}: key='{}' type={:?}", i, kv.key, kv.value_type);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
