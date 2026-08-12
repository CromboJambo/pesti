//! Debug: Check FFN shapes across multiple layers

use pesti_gguf::parser::parse_gguf;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("PESTI_MODEL")
        .unwrap_or_else(|_| "conformance-corpus/Qwen2.5-0.5B-Q4_K_M.gguf".to_string());
    
    println!("Model: {}", model_path);
    
    let header = parse_gguf(&PathBuf::from(&model_path))?;
    
    println!("\n=== FFN Tensors (layers 0-2) ===");
    for layer in 0..3 {
        println!("\n--- Layer {} ---", layer);
        for tensor in &header.tensors {
            if tensor.name.contains(&format!("blk.{}.ffn", layer)) {
                let shape_str = if tensor.shape.len() >= 2 {
                    format!("[{}, {}]", tensor.shape[0], tensor.shape[1])
                } else {
                    format!("[{}]", tensor.shape[0])
                };
                
                let expected_elements: usize = if tensor.shape.len() >= 2 {
                    tensor.shape[0] as usize * tensor.shape[1] as usize
                } else {
                    tensor.shape[0] as usize
                };
                
                let stored_bytes = tensor.stored_size()?;
                println!("{}: {}", tensor.name, shape_str);
                println!("  Expected elements: {}", expected_elements);
                println!("  Stored bytes: {}", stored_bytes);
            }
        }
    }
    
    Ok(())
}
