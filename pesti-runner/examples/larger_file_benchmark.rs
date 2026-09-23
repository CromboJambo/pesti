//! Larger file benchmark: lib.rs (~10KB) with timing measurements

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use std::fs;
use tiktoken_rs::cl100k_base;

fn main() {
    let source = fs::read_to_string("pesti-runner/src/lib.rs").expect("failed to read lib.rs");
    println!("Testing with pesti-runner/src/lib.rs ({} bytes, {} lines)", source.len(), source.lines().count());

    // BPE baseline: count tokens only (no generation)
    let enc = cl100k_base();
    let bpe_tokens = enc.encode_ordinary(&source).len();
    println!("BPE token count: {}", bpe_tokens);

    // Structural: count tokens only
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
    let struct_tokens = tokenizer.tokenize(&source).expect("failed to tokenize");
    println!("Structural token count: {}", struct_tokens.len());
    println!("Compression ratio: {:.2}x", bpe_tokens as f64 / struct_tokens.len() as f64);
}
