//! Print dtype + shape for named tensors using pesti's own parser.
use pesti_gguf::parser::parse_gguf;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = Path::new(&args[1]);
    let header = parse_gguf(path).expect("parse");
    for t in &header.tensors {
        if args[2..].iter().any(|a| a == &t.name) {
            println!("{} dtype={} shape={:?}", t.name, t.dtype, t.shape);
        }
    }
}
