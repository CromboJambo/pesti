//! List all tensors in a GGUF file with shapes and dtypes.
use pesti_gguf::parser::parse_gguf;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <gguf_file>", args[0]);
        return;
    }
    let path = Path::new(&args[1]);
    match parse_gguf(path) {
        Ok(header) => {
            println!("Architecture: {:?}", header.architecture);
            println!("Tensor count: {}", header.tensors.len());
            println!("\nAll tensors:");
            for (name, tensor) in &header.tensors {
                let total_elements = tensor.shape.iter().product::<usize>();
                let dtype_size = match tensor.dtype {
                    pesti_gguf::parser::DType::F32 => 4,
                    pesti_gguf::parser::DType::F16 => 2,
                    pesti_gguf::parser::DType::Q4_K => 2,
                    _ => 0,
                };
                let size_mb = if dtype_size > 0 {
                    (total_elements as f64 * dtype_size as f64) / (1024.0 * 1024.0)
                } else {
                    0.0
                };
                println!("  {:?}: shape={:?} dtype={:?}", name, tensor.shape, tensor.dtype);
            }

            // Check for embedding tensors specifically
            println!("\nEmbedding-related tensors:");
            let embed_names = ["tok_embeddings.weight", "token_embd.weight", "embed_tokens.weight"];
            for name in &embed_names {
                if header.tensors.contains_key(name) {
                    let tensor = header.tensors.get(name).unwrap();
                    println!("  FOUND: {} - shape={:?} dtype={:?}", name, tensor.shape, tensor.dtype);
                } else {
                    println!("  MISSING: {}", name);
                }
            }
        }
        Err(e) => {
            eprintln!("Error parsing GGUF: {:?}", e);
        }
    }
}
