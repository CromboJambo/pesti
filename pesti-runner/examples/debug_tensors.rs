//! Debug: List all tensors in a GGUF file

use pesti_gguf::parser::parse_gguf;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("PESTI_MODEL")
        .unwrap_or_else(|_| "conformance-corpus/Qwen2.5-0.5B-Q4_K_M.gguf".to_string());
    
    println!("Model: {}", model_path);
    
    let header = parse_gguf(&PathBuf::from(&model_path))?;
    
    println!("\n=== GGUF Header ===");
    println!("Version: {}", header.version);
    println!("Data section start: {} bytes", header.data_section_start);
    
    println!("\n=== Config Info ===");
    if let Some(vocab_size) = header.tensors.iter().find(|t| t.name == "tokenizer.ggml.tokens") {
        println!("Vocab size: {}", vocab_size.shape[2]);
    }
    
    println!("\n=== FFN Tensors (first layer) ===");
    for tensor in &header.tensors {
        if tensor.name.contains("blk.0.ffn") {
            let shape_str = if tensor.shape.len() >= 2 {
                format!("[{}, {}]", tensor.shape[0], tensor.shape[1])
            } else {
                format!("[{}]", tensor.shape[0])
            };
            println!("{}: {}", tensor.name, shape_str);
        }
    }
    
    Ok(())
}
