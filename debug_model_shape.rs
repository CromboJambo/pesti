use pesti_gguf::parser;
use std::path::Path;

fn main() {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    let header = parser::parse_gguf(model_path).expect("Failed to parse GGUF");
    
    println!("=== Model Config ===");
    println!("Architecture: {:?}", header.architecture());
    println!("Embedding length: {:?}", header.embedding_length());
    println!("Vocab size: {:?}", header.vocab_size());
    println!("Context length: {:?}", header.context_length());
    println!("Num layers: {:?}", header.block_count());
    println!("Attention heads: {:?}", header.attention_head_count());
    println!("KV heads: {:?}", header.attention_head_count_kv());
    
    println!("\n=== Tensor Shapes ===");
    for tensor in &header.tensors {
        let name = &tensor.name;
        let shape = tensor.shape.as_slice();
        
        if name.contains("token_embd") || name.contains("tok_embeddings") {
            println!("token_embeddings: shape={:?}, dtype={}", shape, tensor.dtype);
        }
        if name.contains("output.weight") {
            println!("output (LM head): shape={:?}, dtype={}", shape, tensor.dtype);
        }
    }
}
