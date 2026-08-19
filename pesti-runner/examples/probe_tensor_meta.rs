//! Dump GGUF tensor offsets and compute ACTUAL per-tensor byte sizes
//! (delta to next tensor) vs pesti-gguf's stored_size() formula.
use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufDtype;

fn main() {
    let arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string());
    let path = std::path::Path::new(arg.as_str());
    let header = parse_gguf(path).expect("parse gguf");
    let file_len = std::fs::metadata(path).unwrap().len();
    println!("file_len = {file_len}, data_section_start = {}", header.data_section_start);
    println!("tensor count = {}", header.tensors.len());
    println!();
    println!("{:55} {:>12} {:>12} {:>12} {:>12}  dtype", "name", "offset", "stored_sz", "actual_sz", "n_elems");
    let mut prev: Option<&pesti_gguf::types::GgufTensorInfo> = None;
    for t in &header.tensors {
        let dtype = GgufDtype::from_u32(t.dtype);
        let stored = t.stored_size().unwrap_or(0);
        // Show all of blk.0 + embd + output; sample blk.12 and blk.23
        let show = t.name.starts_with("blk.0.")
            || t.name.starts_with("blk.12.")
            || t.name.starts_with("blk.23.")
            || t.name.starts_with("token_embd")
            || t.name.starts_with("output");
        if show {
            let actual = prev.map(|p| t.offset.saturating_sub(p.offset)).unwrap_or(0);
            println!(
                "{:55} {:>12} {:>12} {:>12} {:>12}  {:?}",
                t.name, t.offset, stored, actual, t.element_count(), dtype
            );
        }
        prev = Some(t);
    }
}
