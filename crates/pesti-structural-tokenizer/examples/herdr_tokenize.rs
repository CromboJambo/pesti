use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;

fn main() {
    let path = "/tmp/herdr/src/workspace.rs";
    let content = fs::read_to_string(path).expect("failed to read");

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content).expect("tokenization failed");

    println!("File: {}", path);
    println!("Original size: {} bytes", content.len());
    println!("Structural tokens: {}\n", tokens.len());

    // Show first 20 tokens as examples
    for (i, tok) in tokens.iter().take(20).enumerate() {
        let preview = if tok.text.len() > 60 {
            format!("{}...", &tok.text[..60])
        } else {
            tok.text.clone()
        };
        println!(
            "[{:3}] {:?} ({}..{}): {}",
            i, tok.kind, tok.span.0, tok.span.1, preview
        );
    }

    if tokens.len() > 20 {
        println!("... and {} more\n", tokens.len() - 20);
    }

    // Count by kind (manual since TokenKind doesn't derive Clone/Hash)
    let mut fn_decl = 0;
    let mut let_stmt = 0;
    let mut struct_decl = 0;
    let mut enum_decl = 0;
    let mut impl_block = 0;
    let mut other = 0;

    for tok in &tokens {
        match tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => fn_decl += 1,
            pesti_structural_tokenizer::TokenKind::LetStmt => let_stmt += 1,
            pesti_structural_tokenizer::TokenKind::StructDecl => struct_decl += 1,
            pesti_structural_tokenizer::TokenKind::EnumDecl => enum_decl += 1,
            pesti_structural_tokenizer::TokenKind::ImplBlock => impl_block += 1,
            _ => other += 1,
        }
    }

    println!("Token distribution:");
    println!("  FnDecl: {}", fn_decl);
    println!("  LetStmt: {}", let_stmt);
    println!("  StructDecl: {}", struct_decl);
    println!("  EnumDecl: {}", enum_decl);
    println!("  ImplBlock: {}", impl_block);
    println!("  Other: {}", other);
}
