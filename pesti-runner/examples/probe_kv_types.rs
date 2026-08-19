//! Probe: dump the raw KV types of tokenizer.* keys in a GGUF file.
//!
//! Goal: determine what pesti_gguf actually parsed for `tokenizer.ggml.tokens`
//! in the Qwen2.5-0.5B file (Array of Strings? count? something else?).

use std::path::Path;

fn main() {
    let model_path = Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );

    let header = pesti_gguf::parser::parse_gguf(model_path)
        .expect("failed to parse GGUF");

    println!("=== All tokenizer.* KV pairs (raw types) ===");
    let mut count = 0;
    for kv in &header.kv_pairs {
        if kv.key.starts_with("tokenizer.") {
            count += 1;
            match &kv.value {
                pesti_gguf::GgufKvValue::Array(arr) => {
                    let first: Vec<String> = arr
                        .iter()
                        .take(5)
                        .map(|v| format!("{:?}", v))
                        .collect();
                    println!(
                        "  {:<45} ARRAY len={} first5={:?}",
                        kv.key,
                        arr.len(),
                        first
                    );
                }
                other => {
                    println!("  {:<45} {:?}", kv.key, other);
                }
            }
        }
    }
    println!("total tokenizer.* pairs: {}", count);

    println!("\n=== Non-tokenizer KV types (first 20) ===");
    for kv in header.kv_pairs.iter().take(20) {
        if !kv.key.starts_with("tokenizer.") {
            println!("  {:<45} {:?}", kv.key, kv.value);
        }
    }
}
