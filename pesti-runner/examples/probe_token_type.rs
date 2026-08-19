//! Probe token_type distribution + merges parsing to design the real tokenizer.
use std::path::Path;

fn main() {
    let p = Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );
    let header = pesti_gguf::parser::parse_gguf(p).expect("parse");

    let get_arr = |key: &str| -> Option<Vec<pesti_gguf::GgufKvValue>> {
        header
            .kv_pairs
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| match &kv.value {
                pesti_gguf::GgufKvValue::Array(a) => Some(a.clone()),
                _ => None,
            })
    };

    let tokens = get_arr("tokenizer.ggml.tokens").expect("tokens");
    let types = get_arr("tokenizer.ggml.token_type").expect("types");
    let merges = get_arr("tokenizer.ggml.merges").expect("merges");

    println!("tokens={} types={} merges={}", tokens.len(), types.len(), merges.len());

    // token_type histogram
    let mut hist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for t in &types {
        if let pesti_gguf::GgufKvValue::Int32(v) = t {
            *hist.entry(*v).or_insert(0) += 1;
        }
    }
    println!("\ntoken_type histogram: {:?}", hist);

    // Show the special (added) tokens: type != 1
    println!("\n=== Non-normal tokens (type != 1), first 40 ===");
    let mut shown = 0;
    for (id, tok) in tokens.iter().enumerate() {
        let ty = match &types[id] {
            pesti_gguf::GgufKvValue::Int32(v) => *v,
            _ => 1,
        };
        if ty != 1 && shown < 40 {
            let s = tok.as_str().unwrap_or("?");
            println!("  id={:<7} type={} {:?}", id, ty, s);
            shown += 1;
        }
    }

    // Verify merges parse as "a b"
    let mut bad = 0;
    let mut parsed_ok = 0;
    for m in &merges {
        if let pesti_gguf::GgufKvValue::String(s) = m {
            match s.split_once(' ') {
                Some((a, b)) if !a.is_empty() && !b.is_empty() => {
                    parsed_ok += 1;
                }
                _ => {
                    bad += 1;
                    if bad < 5 {
                        println!("  BAD MERGE: {:?}", s);
                    }
                }
            }
        }
    }
    println!("\nmerges parsed_ok={} bad={}", parsed_ok, bad);

    // Check base vocab = tokens with type==1; do any base tokens contain a raw space?
    let mut base_with_space = 0;
    for (id, tok) in tokens.iter().enumerate() {
        let ty = match &types[id] {
            pesti_gguf::GgufKvValue::Int32(v) => *v,
            _ => 1,
        };
        if ty == 1 {
            if let Some(s) = tok.as_str() {
                if s.contains(' ') {
                    base_with_space += 1;
                    if base_with_space < 5 {
                        println!("  BASE TOKEN WITH RAW SPACE id={}: {:?}", id, s);
                    }
                }
            }
        }
    }
    println!("base tokens containing raw space: {}", base_with_space);
}
