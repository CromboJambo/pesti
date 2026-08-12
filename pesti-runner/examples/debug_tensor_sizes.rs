//! Debug: Check actual tensor sizes vs declared shapes

use pesti_gguf::parser::parse_gguf;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("PESTI_MODEL")
        .unwrap_or_else(|_| "conformance-corpus/Qwen2.5-0.5B-Q4_K_M.gguf".to_string());
    
    println!("Model: {}", model_path);
    
    let header = parse_gguf(&PathBuf::from(&model_path))?;
    
    println!("\n=== FFN Tensors (first layer) ===");
    for tensor in &header.tensors {
        if tensor.name.contains("blk.0.ffn") {
            let shape_str = if tensor.shape.len() >= 2 {
                format!("[{}, {}]", tensor.shape[0], tensor.shape[1])
            } else {
                format!("[{}]", tensor.shape[0])
            };
            
            // Calculate expected element count
            let expected_elements: usize = if tensor.shape.len() >= 2 {
                tensor.shape[0] as usize * tensor.shape[1] as usize
            } else {
                tensor.shape[0] as usize
            };
            
            println!("{}: {}", tensor.name, shape_str);
            println!("  Expected elements: {} (shape product)", expected_elements);
            println!("  Stored size: {} bytes", tensor.stored_size()?);
            println!("  Dtype: {:?}", pesti_gguf::types::GgufDtype::from_u32(tensor.dtype));
        }
    }
    
    Ok(())
}
