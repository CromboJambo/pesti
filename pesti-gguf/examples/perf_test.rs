//! Simple performance test for pesti-gguf
//! Run with: cargo run --bin perf_test

use pesti_gguf::parse_gguf;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), pesti_gguf::GgufError> {
    println!("=== pesti-gguf Performance Test ===\n");
    
    let models = [
        ("0.5B Q4_K_M", "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        ("3B Q4_K_M", "conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf"),
    ];
    
    for (name, path) in &models {
        let path = Path::new(path);
        
        if !path.exists() {
            println!("⚠️  Skipping {} (file not found)", name);
            continue;
        }
        
        let start = Instant::now();
        let header = parse_gguf(path)?;
        let duration = start.elapsed();
        
        println!("{}:", name);
        println!("  ⏱️  Parse time: {:?}", duration);
        println!("  📦 File size: {} MB", 
            std::fs::metadata(path)?.len() / (1024 * 1024));
        println!("  🔑 KV pairs: {}", header.kv_pairs.len());
        println!("  🧠 Tensors: {}", header.tensors.len());
        println!();
    }
    
    Ok(())
}
