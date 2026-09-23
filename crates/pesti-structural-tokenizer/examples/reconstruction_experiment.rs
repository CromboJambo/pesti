//! Reconstruction experiment: Can a model generate correct Rust from
//! structural tokens alone? Tests generative power, not just description.

use pesti_structural_tokenizer::StructuralTokenizer;

fn main() {
    // Source program to tokenize and reconstruct
    let source = r#"fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn multiply(a: i32, b: i32) -> i32 {
    return a * b;
}

fn main() {
    let result = add(5, 3);
    if result > 10 {
        print("big");
    } else {
        print("small");
    }
}"#;

    println!("=== RECONSTRUCTION EXPERIMENT ===\n");
    println!("Original source ({} bytes):", source.len());
    println!("{}", source);
    println!("\n---\n");

    // Tokenize
    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(source).unwrap();

    println!("Structural token sequence ({} tokens):", tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        println!("  [{}] {:?}: {:?}", i, tok.kind, tok.text);
    }

    println!("\n--- RECONSTRUCTION PROMPT ---\n");
    // This is what we'd feed to an LLM: just the structural token sequence
    // with kind names and text content. No original source code.
    let mut prompt =
        String::from("Reconstruct a Rust program from this structural representation:\n\n");
    for tok in &tokens {
        match &tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => {
                // FnDecl text is "name(args) -> ret " — extract name and signature
                let sig = tok.text.trim();
                prompt.push_str(&format!("FUNCTION_DECL: fn {} \n", sig));
            }
            pesti_structural_tokenizer::TokenKind::BlockStart => {
                prompt.push_str("  BLOCK_START\n");
            }
            pesti_structural_tokenizer::TokenKind::BlockEnd => {
                prompt.push_str("  BLOCK_END\n");
            }
            pesti_structural_tokenizer::TokenKind::IfElse => {
                prompt.push_str(&format!("  IF: {}\n", tok.text));
            }
            pesti_structural_tokenizer::TokenKind::LetStmt => {
                prompt.push_str(&format!("  LET: {}\n", tok.text));
            }
            pesti_structural_tokenizer::TokenKind::ExprStmt => {
                prompt.push_str(&format!("  EXPR_STMT: {}\n", tok.text));
            }
            pesti_structural_tokenizer::TokenKind::Ident(name) => {
                prompt.push_str(&format!("  IDENT: {}\n", name));
            }
            _ => {
                prompt.push_str(&format!("  {:?}: {}\n", tok.kind, tok.text));
            }
        }
    }

    println!("{}", prompt);
    println!("---\n");
    println!("Model should reconstruct: two functions (add, multiply) and main with if/else.");
    println!("Success criteria: model produces compilable Rust with correct structure.");
}
