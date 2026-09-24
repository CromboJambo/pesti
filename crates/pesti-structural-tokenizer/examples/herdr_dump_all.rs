use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;

fn main() {
    let path = "/tmp/herdr/src/workspace.rs";
    let content = fs::read_to_string(path).expect("failed to read");

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content).expect("tokenization failed");

    println!("=== herdr workspace.rs structural tokens ===");
    println!("File: {}", path);
    println!("Original size: {} bytes", content.len());
    println!("Structural tokens: {}\n", tokens.len());

    for (i, tok) in tokens.iter().enumerate() {
        let preview = if tok.text.chars().count() > 80 {
            format!("{}...", tok.text.chars().take(77).collect::<String>())
        } else {
            tok.text.clone()
        };
        println!("[{:>3}] {:?}: {}", i, tok.kind, preview);
    }
}
