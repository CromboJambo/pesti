//! Probe: print the raw GGUF metadata + inferred config + layer attention dims.
use pesti_runner::model_loader::ModelLoader;
use pesti_runner::transformer::{LlamaConfig, LlamaModel};

fn main() {
    let path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    let header = ModelLoader::load_gguf_header(path).expect("header");

    println!("=== RAW GGUF KV (arch-relevant) ===");
    for k in [
        "general.architecture",
        "attention.head_count",
        "attention.head_count_kv",
        "rope.dimension_count",
        "embedding.length",
        "block_count",
        "attention.layer_norm_epsilon",
        "context.length",
    ] {
        let v = header
            .kv_pairs
            .iter()
            .find(|kv| kv.key == k)
            .map(|kv| format!("{:?}", kv.value));
        println!("  {:32} = {:?}", k, v);
    }

    println!("\n=== ALL KV KEYS (total {}) ===", header.kv_pairs.len());
    for kv in &header.kv_pairs {
        let v = format!("{:?}", kv.value);
        let v = if v.len() > 50 { format!("{}...", &v[..50]) } else { v };
        println!("  {:42} = {}", kv.key, v);
    }

    // tensor shapes for blk.0 attention
    println!("\n=== blk.0 attention tensor shapes ===");
    for t in &header.tensors {
        if t.name.starts_with("blk.0.attn") {
            println!("  {:28} ndim={} shape={:?}", t.name, t.shape.len(), t.shape);
        }
    }

    println!("\n=== INFERRED CONFIG ===");
    let config = LlamaConfig::from_gguf_header(&header).expect("config");
    println!("  arch            = {:?}", config.arch);
    println!("  embed_dim       = {}", config.embed_dim);
    println!("  num_heads       = {}", config.num_heads);
    println!("  num_kv_heads    = {}", config.num_kv_heads);
    println!("  head_dim        = {}   (embed/heads = {})", config.head_dim, config.embed_dim / config.num_heads);
    println!("  num_layers      = {}", config.num_layers);
    println!("  intermediate    = {}", config.intermediate_dim);
    println!("  max_seq_len     = {}", config.max_seq_len);
    println!("  num_heads*head_dim = {}", config.num_heads * config.head_dim);

    println!("\n=== LAYER 0 ATTENTION (loaded weights) ===");
    let model = LlamaModel::load_gguf(path).expect("load");
    let l0 = &model.layers[0];
    println!("  attn.num_heads  = {}", l0.attention.num_heads);
    println!("  attn.num_kv_heads = {}", l0.attention.num_kv_heads);
    println!("  attn.head_dim   = {}", l0.attention.head_dim);
    println!("  attn.kv_dim     = {}", l0.attention.kv_dim);
    println!("  wq.out_features = {}", l0.attention.wq.out_features);
    println!("  wk.out_features = {}", l0.attention.wk.out_features);
    println!("  wo.in_features  = {}", l0.attention.wo.in_features);
    println!("  wo.out_features = {}", l0.attention.wo.out_features);
    println!("  rope.head_dim   = {}", l0.attention.rope.head_dim);
}
