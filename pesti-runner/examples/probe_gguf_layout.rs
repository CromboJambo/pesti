//! Semantic gauge: print how pesti-gguf *interprets* a GGUF's config + key
//! tensor shapes. Cross-check against an independent parse (gguf_shapes.py /
//! triangulate.py) to see whether the runner's model of the file matches
//! reality. Written against pesti-gguf 0.2.4 (`parse_gguf`).
use pesti_gguf::{GgufKvValue, parse_gguf};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: probe_gguf_layout <file.gguf>");
    let header = parse_gguf(std::path::Path::new(&path)).expect("parse gguf");

    println!("=== CONFIG (as pesti-gguf sees it) ===");
    println!("architecture       = {:?}", header.architecture());
    println!("embedding_length   = {:?}", header.embedding_length());
    println!("block_count        = {:?}", header.block_count());
    println!("context_length     = {:?}", header.context_length());
    // head counts + vocab are arch-specific keys; read them raw so we see
    // exactly what the parser normalized (and what it did not).
    for key in [
        "head_count",
        "kv_head_count",
        "vocab_size",
        "ffn_length",
        "rope_freq_base",
        "rms_norm_eps",
        "use_parallel_residual",
        "final_layernorm",
    ] {
        // try arch-prefixed then bare
        let arch = header.architecture().unwrap_or("llama");
        let akey = format!("{arch}.{key}");
        let v = header
            .get_kv_str(&akey)
            .map(|s| s.to_string())
            .or_else(|| header.get_kv_u32(&akey).map(|n| n.to_string()))
            .or_else(|| header.get_kv_u32(key).map(|n| n.to_string()));
        println!("{:<20} = {:?}", key, v);
    }

    println!("\n=== KEY TENSOR SHAPES (dims in file order) ===");
    let want = [
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_q",
        "blk.0.attn_k",
        "blk.0.attn_v",
        "blk.0.attn_output",
        "blk.0.ffn_gate",
        "blk.0.ffn_down",
    ];
    for t in &header.tensors {
        let hit = want.iter().any(|w| t.name == *w || t.name.starts_with(w));
        if hit {
            let n: u64 = t.shape.iter().copied().product();
            println!("  {:<28} shape={:?} n={}", t.name, t.shape, n);
        }
    }

    // Raw arch-prefixed KV sample: what did the parser keep verbatim?
    println!("\n=== ARCH KV (sample) ===");
    let arch = header.architecture().unwrap_or("llama");
    for pair in &header.kv_pairs {
        if pair.key.starts_with(&arch) {
            println!("  {} = {}", pair.key, kv_str(&pair.value));
        }
    }
}

fn kv_str(v: &GgufKvValue) -> String {
    match v {
        GgufKvValue::String(s) => s.clone(),
        GgufKvValue::Uint8(n) => n.to_string(),
        GgufKvValue::Int8(n) => n.to_string(),
        GgufKvValue::Uint16(n) => n.to_string(),
        GgufKvValue::Int16(n) => n.to_string(),
        GgufKvValue::Uint32(n) => n.to_string(),
        GgufKvValue::Int32(n) => n.to_string(),
        GgufKvValue::Uint64(n) => n.to_string(),
        GgufKvValue::Int64(n) => n.to_string(),
        GgufKvValue::Float32(f) => f.to_string(),
        GgufKvValue::Float64(f) => f.to_string(),
        GgufKvValue::Bool(b) => b.to_string(),
        GgufKvValue::Bfloat16(f) => f.to_string(),
        GgufKvValue::Float16(n) => n.to_string(),
        GgufKvValue::Array(items) => format!("[{} items]", items.len()),
        GgufKvValue::Int8Array(items) => format!("[{} items]", items.len()),
        GgufKvValue::Uint8Array(items) => format!("[{} items]", items.len()),
    }
}
