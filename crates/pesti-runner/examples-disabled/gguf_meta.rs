//! Print GGUF metadata fields.
use pesti_gguf::parser::parse_gguf;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }
    let path = Path::new(&args[1]);
    let header = parse_gguf(path).expect("parse");
    
    // Print architecture-related fields
    if let Some(arch) = header.architecture() {
        println!("architecture: {}", arch);
    }
    if let Some(vocab_size) = header.vocab_size() {
        println!("vocab_size: {}", vocab_size);
    }
    if let Some(embed_len) = header.embedding_length() {
        println!("embedding_length: {}", embed_len);
    }
    if let Some(block_count) = header.block_count() {
        println!("block_count: {}", block_count);
    }
}
