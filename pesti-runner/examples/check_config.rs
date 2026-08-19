//! Check model config and KV cache allocation

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let header = pesti_runner::model_loader::load_gguf_header(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )?;
    
    println!("Model Config:");
    println!("  max_seq_len: {:?}", header.context_length());
    println!("  vocab_size: {:?}", header.vocab_size());
    println!("  hidden_size: {:?}", header.hidden_size());
    println!("  num_layers: {:?}", header.num_layers());
    println!("  num_heads: {:?}", header.num_attention_heads());
    println!("  num_kv_heads: {:?}", header.num_kv_heads());
    
    Ok(())
}
