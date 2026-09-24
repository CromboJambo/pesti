use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;

fn main() {
    let path = "/home/crombo/projects/ripgrep/crates/cli/src/human.rs";
    let src = fs::read_to_string(path).unwrap();

    println!("File: {}", path);
    println!("Source length: {} bytes", src.len());
    println!();

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&src).unwrap();

    println!("Structural tokens: {}\n", tokens.len());

    for (i, token) in tokens.iter().enumerate() {
        // Truncate long values for readability
        let val = if token.text.len() > 60 {
            format!("{}...", &token.text[..57])
        } else {
            token.text.clone()
        };
        println!("[{:4}] {:?} {}", i, token.kind, val);
    }
}
