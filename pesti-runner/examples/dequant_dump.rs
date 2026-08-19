//! Dump raw + dequantized bytes for a tensor so it can be compared against
//! llama.cpp's reference dequantizer (gold standard conformance).
//!
//! Usage: dequant_dump <gguf_path> <tensor_name> <out_prefix>
//!   writes: <out_prefix>.raw   (raw quantized bytes)
//!           <out_prefix>.f32   (pesti-dequantized f32, little-endian)
//!           prints element count to stdout

use pesti_runner::gguf_weight_loader::load_gguf_weights;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dequant_dump <gguf_path> <tensor_name> <out_prefix>");
        std::process::exit(2);
    }
    let gguf_path = Path::new(&args[1]);
    let tensor_name = args[2].as_str();
    let out_prefix = &args[3];

    // Load all weights: gives us both raw quantized bytes and dequantized f32.
    let weights = load_gguf_weights(gguf_path).expect("load weights");
    let raw = weights
        .raw_tensors
        .get(tensor_name)
        .unwrap_or_else(|| panic!("tensor {} not in raw map", tensor_name));
    let deq = weights
        .tensors
        .get(tensor_name)
        .unwrap_or_else(|| panic!("tensor {} not in dequantized map", tensor_name));

    let element_count = deq.len() / 4;

    std::fs::write(format!("{}.raw", out_prefix), &raw).expect("write raw");
    std::fs::write(format!("{}.f32", out_prefix), deq).expect("write f32");

    println!("{}", element_count);
}
