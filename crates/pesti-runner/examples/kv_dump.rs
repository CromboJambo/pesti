use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufKvValue;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let h = parse_gguf(std::path::Path::new(&path)).unwrap();
    println!("=== ALL KV PAIRS ===");
    for kv in &h.kv_pairs {
        let v = match &kv.value {
            GgufKvValue::Uint32(x) => format!("u32={x}"),
            GgufKvValue::Int32(x) => format!("i32={x}"),
            GgufKvValue::Uint64(x) => format!("u64={x}"),
            GgufKvValue::Int64(x) => format!("i64={x}"),
            GgufKvValue::Float32(x) => format!("f32={x}"),
            GgufKvValue::Float64(x) => format!("f64={x}"),
            GgufKvValue::Bool(x) => format!("bool={x}"),
            GgufKvValue::String(s) => format!("str={s}"),
            GgufKvValue::Array(a) => format!("array len={}", a.len()),
            other => format!("{other:?}"),
        };
        println!("{:<50} {}", kv.key, v);
    }
}
