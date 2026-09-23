use pesti_structural_tokenizer::StructuralTokenizer;

fn main() {
    // Test doc comments with code examples
    let src = "/// Doc comment with `code`\n/// and special chars: <, >, &, |\npub fn foo() {}\n";
    let tok = StructuralTokenizer::new();
    println!("Testing doc comments...");
    let tokens = tok.tokenize(src).expect("tokenize failed");
    println!("Tokens: {}", tokens.len());
    
    // Test raw strings
    let src2 = r#"let s = "hello";"#;
    println!("Testing string literals...");
    let tokens2 = tok.tokenize(src2).expect("tokenize failed");
    println!("Tokens: {}", tokens2.len());
}
