use pesti_structural_tokenizer::StructuralTokenizer;

fn main() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn add(a: i32, b: i32) -> i32 {\n    return a + b;\n}";
    let tokens = tokenizer.tokenize(code).unwrap();

    println!("Tokens with span info:");
    for tok in &tokens {
        if let Some(loc) = &tok.location {
            println!(
                "  {} (line {}, col {}, byte {}): {}",
                tok.kind, loc.line, loc.column, loc.start_byte, tok.text
            );
        } else {
            println!("  {} (no location): {}", tok.kind, tok.text);
        }
    }
}
