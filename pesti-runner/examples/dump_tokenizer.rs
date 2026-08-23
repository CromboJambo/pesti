//! Dump the tokenizer arrays embedded in a GGUF file to JSON, for building
//! an HF-format tokenizer.json. Uses the pesti_gguf parser (same one PESTI
//! uses to load weights).
use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufKvValue;
use std::path::Path;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string());
    let out_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/qwen2_tok".to_string());
    std::fs::create_dir_all(&out_dir).unwrap();

    let header = parse_gguf(Path::new(&path)).expect("parse failed");
    println!("arch: {:?}", header.architecture());

    let mut out = std::collections::BTreeMap::new();
    for kv in &header.kv_pairs {
        if !kv.key.starts_with("tokenizer.") {
            continue;
        }
        match &kv.value {
            GgufKvValue::String(s) => {
                println!(
                    "  {} = {:?} (len {})",
                    kv.key,
                    &s[..s.len().min(60)],
                    s.len()
                );
                out.insert(kv.key.clone(), serde_json::Value::String(s.clone()));
            }
            GgufKvValue::Array(arr) => {
                let et = arr
                    .first()
                    .map(|v| format!("{:?}", v.value_type()))
                    .unwrap_or_default();
                let n = arr.len();
                println!("  {} = Array[{}] elem={}", kv.key, n, et);
                // Dump as JSON array
                let items: Vec<serde_json::Value> = arr
                    .iter()
                    .map(|v| match v {
                        GgufKvValue::String(s) => serde_json::Value::String(s.clone()),
                        GgufKvValue::Uint32(u) => serde_json::Value::from(*u),
                        GgufKvValue::Uint64(u) => serde_json::Value::from(*u),
                        GgufKvValue::Float32(f) => serde_json::Value::from(*f),
                        GgufKvValue::Float64(f) => serde_json::Value::from(*f),
                        other => {
                            eprintln!(
                                "  ! unexpected elem in {}: {:?}",
                                kv.key,
                                other.value_type()
                            );
                            serde_json::Value::Null
                        }
                    })
                    .collect();
                out.insert(kv.key.clone(), serde_json::Value::Array(items));
            }
            other => {
                println!("  {} = {:?} (scalar)", kv.key, other.value_type());
                let v = match other {
                    GgufKvValue::Uint32(u) => serde_json::Value::from(*u),
                    GgufKvValue::Uint64(u) => serde_json::Value::from(*u),
                    GgufKvValue::Int32(i) => serde_json::Value::from(*i),
                    GgufKvValue::Int64(i) => serde_json::Value::from(*i),
                    GgufKvValue::Float32(f) => serde_json::Value::from(*f),
                    GgufKvValue::Float64(f) => serde_json::Value::from(*f),
                    GgufKvValue::Bool(b) => serde_json::Value::from(*b),
                    GgufKvValue::String(s) => serde_json::Value::String(s.clone()),
                    _ => serde_json::Value::Null,
                };
                out.insert(kv.key.clone(), v);
            }
        }
    }

    let json = serde_json::to_string_pretty(&out).unwrap();
    let path_out = format!("{}/tokenizer_kv.json", out_dir);
    std::fs::write(&path_out, &json).unwrap();
    println!("\nwrote {} ({} bytes)", path_out, json.len());
}
