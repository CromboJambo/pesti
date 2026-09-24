use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::collections::HashMap;
use std::fs;

fn main() {
    let path = "/tmp/herdr/src/workspace.rs";
    let content = fs::read_to_string(path).expect("failed to read");

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content).expect("tokenization failed");

    println!("=== herdr workspace.rs structural coverage report ===");
    println!("File: {}", path);
    println!("Original size: {} bytes", content.len());
    println!("Structural tokens: {}\n", tokens.len());

    // Detailed distribution of ALL token kinds
    let mut kind_counts: HashMap<String, usize> = HashMap::new();
    for tok in &tokens {
        let name = format!("{:?}", tok.kind);
        *kind_counts.entry(name).or_insert(0) += 1;
    }

    println!("Token distribution (all kinds):");
    let mut entries: Vec<_> = kind_counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    for (kind, count) in &entries {
        println!("  {:<20} {}", kind, count);
    }

    // Coverage analysis: which bytes are owned by tokens?
    let mut covered_bytes = 0;
    let total_chars = content.chars().count();

    for tok in &tokens {
        let span_len = tok.span.1 - tok.span.0;
        covered_bytes += span_len;
    }

    println!("\nCoverage analysis:");
    println!("  Total source characters: {}", total_chars);
    println!("  Characters covered by tokens: {}", covered_bytes);
    println!(
        "  Coverage: {:.2}%",
        (covered_bytes as f64 / total_chars as f64) * 100.0
    );

    // Check for gaps between consecutive tokens
    let mut prev_end = 0;
    let mut gaps = 0;
    let mut max_gap = 0;
    for tok in &tokens {
        if tok.span.0 > prev_end {
            gaps += 1;
            let gap_size = tok.span.0 - prev_end;
            if gap_size > max_gap {
                max_gap = gap_size;
            }
        }
        prev_end = tok.span.1;
    }
    println!("  Gaps between tokens: {}", gaps);
    println!("  Maximum gap size: {} chars", max_gap);

    // What constructs did we encounter?
    let mut construct_counts: HashMap<String, usize> = HashMap::new();
    for tok in &tokens {
        match tok.kind {
            TokenKind::FnDecl => *construct_counts.entry("function".to_string()).or_insert(0) += 1,
            TokenKind::LetStmt => *construct_counts.entry("let binding".to_string()).or_insert(0) += 1,
            TokenKind::StructDecl => *construct_counts.entry("struct".to_string()).or_insert(0) += 1,
            TokenKind::EnumDecl => *construct_counts.entry("enum".to_string()).or_insert(0) += 1,
            TokenKind::ImplBlock => *construct_counts.entry("impl block".to_string()).or_insert(0) += 1,
            TokenKind::UseStmt => *construct_counts.entry("use statement".to_string()).or_insert(0) += 1,
            TokenKind::ModDecl => *construct_counts.entry("module declaration".to_string()).or_insert(0) += 1,
            TokenKind::IfElse => *construct_counts.entry("if/else".to_string()).or_insert(0) += 1,
            TokenKind::ForLoop => *construct_counts.entry("for loop".to_string()).or_insert(0) += 1,
            TokenKind::WhileLoop => *construct_counts.entry("while loop".to_string()).or_insert(0) += 1,
            TokenKind::Attribute => *construct_counts.entry("attribute".to_string()).or_insert(0) += 1,
            TokenKind::ExprStmt => {
                *construct_counts.entry("expression statement".to_string()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    println!("\nConstructs encountered:");
    let mut sorted_constructs: Vec<_> = construct_counts.iter().collect();
    sorted_constructs.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in sorted_constructs {
        println!("  {}: {}", name, count);
    }
}
