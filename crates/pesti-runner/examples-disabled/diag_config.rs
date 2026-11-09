//! Diagnostic: print the LlamaConfig derived from the GGUF header.

use pesti_runner::model_loader::GgufHeaderExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    let header = pesti_gguf::parser::parse_gguf(path)?;

    println!("arch: {:?}", header.architecture());
    println!("embedding_length: {:?}", header.embedding_length());
    println!("block_count: {:?}", header.block_count());
    println!("context_length: {:?}", header.context_length());
    println!("vocab_size: {:?}", header.vocab_size());
    println!(
        "rope.dimension_count: {:?}",
        header.get_kv_u32("rope.dimension_count")
    );
    println!(
        "attention.head_count: {:?}",
        header.get_kv_u32("attention.head_count")
    );
    println!(
        "qwen2.attention.head_count: {:?}",
        header.get_kv_u32("qwen2.attention.head_count")
    );
    println!(
        "qwen2.attention.head_count_kv: {:?}",
        header.get_kv_u32("qwen2.attention.head_count_kv")
    );
    println!(
        "qwen2.attention.head_count: {:?}",
        header.get_kv_u32("qwen2.attention.head_count")
    );
    println!(
        "qwen2.num_key_value_heads: {:?}",
        header.get_kv_u32("qwen2.num_key_value_heads")
    );
    println!(
        "qwen2.feed_forward_length: {:?}",
        header.get_kv_u32("qwen2.feed_forward_length")
    );
    println!(
        "qwen2.context_length: {:?}",
        header.get_kv_u32("qwen2.context_length")
    );
    println!(
        "qwen2.attention.layer_norm_eps: {:?}",
        header.get_kv_f32("qwen2.attention.layer_norm_eps")
    );
    println!(
        "qwen2.attention.rope.freq_base: {:?}",
        header.get_kv_f32("qwen2.attention.rope.freq_base")
    );

    // All KV keys for visibility
    println!("\nAll architecture KV keys:");
    for p in &header.kv_pairs {
        if p.key.starts_with("qwen2") || p.key.starts_with("rope") || p.key.starts_with("general") {
            println!("  {} = {:?}", p.key, p.value);
        }
    }

    // Now the config the model actually uses
    let config = pesti_runner::transformer::LlamaConfig::from_gguf_header(&header)?;
    println!("\n=== LlamaConfig (what the model uses) ===");
    println!("arch: {:?}", config.arch);
    println!("num_layers: {}", config.num_layers);
    println!("num_heads: {}", config.num_heads);
    println!("num_kv_heads: {}", config.num_kv_heads);
    println!("head_dim: {}", config.head_dim);
    println!("embed_dim: {}", config.embed_dim);
    println!("intermediate_dim: {}", config.intermediate_dim);
    println!("max_seq_len: {}", config.max_seq_len);
    println!("rope_base: {}", config.rope_base);
    println!(
        "num_heads * head_dim = {}",
        config.num_heads * config.head_dim
    );
    println!(
        "num_kv_heads * head_dim = {}",
        config.num_kv_heads * config.head_dim
    );

    // Check tensor shapes in header
    let k = header
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_k.weight");
    if let Some(t) = k {
        println!("\nblk.0.attn_k.weight shape: {:?}", t.shape);
    }
    let q = header
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight");
    if let Some(t) = q {
        println!("blk.0.attn_q.weight shape: {:?}", t.shape);
    }

    Ok(())
}
