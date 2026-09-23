use pesti_runner::transformer::tokenizer::{PestiTokenizer, TokenizerBackend};
use std::env;

fn main() {
    // Use the structural tokenizer directly via its crate API.
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();

    println!("Loaded structural Rust tokenizer");

    // Test encoding real Rust code
    let rust_code = r#"fn fibonacci(n: u64) -> u64 {
    if n <= 1 { return n; }
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n - 1 {
        let temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}"#;

    println!("\nEncoding {} bytes of Rust code...", rust_code.len());
    let tokens = tokenizer.tokenize(rust_code).expect("Tokenization failed");
    println!("Encoded to {} structural tokens", tokens.len());
    for (i, tok) in tokens.iter().take(10).enumerate() {
        println!("  [{}] {:?} -> '{}'", i, tok.kind, &tok.text[..std::cmp::min(tok.text.len(), 20)]);
    }

    // Map to model IDs via pesti-runner integration
    let ids: Vec<u32> = tokens.iter()
        .map(|t| pesti_runner::transformer::tokenizer::PestiTokenizer::token_kind_to_id(&t.kind))
        .collect();
    println!("\nModel token IDs: {:?}", &ids[..std::cmp::min(10, ids.len())]);
}
