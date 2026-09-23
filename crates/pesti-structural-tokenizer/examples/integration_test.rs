//! Integration test: Compare BPE vs Structural tokenization on real Rust code
//! Demonstrates compression benefit for code-specialized LLM workloads

use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::fs;

fn main() {
    // Load a real Rust file to tokenize
    let test_code = fs::read_to_string("/home/crombo/projects/pesti/crates/qwen2-bpe/src/lib.rs")
        .expect("Failed to read qwen2-bpe source");

    println!("=== PESTI Tokenization Comparison ===\n");
    println!(
        "Test file: crates/qwen2-bpe/src/lib.rs ({} bytes, {} lines)",
        test_code.len(),
        test_code.lines().count()
    );

    // Structural tokenization
    let tokenizer = StructuralTokenizer::new();
    match tokenizer.tokenize(&test_code) {
        Ok(structural_tokens) => {
            println!("Structural tokenization: {} tokens", structural_tokens.len());

            // Estimate BPE (rough heuristic: ~4 bytes per token for code)
            let bpe_estimate = test_code.len() / 4;
            let ratio = bpe_estimate as f64 / structural_tokens.len() as f64;

            println!("\nCompression vs estimated BPE (~{} tokens): {:.1}x fewer", bpe_estimate, ratio);
            println!("Estimated tokens saved: {}", bpe_estimate - structural_tokens.len());

            // Token distribution by kind
            let mut fn_count = 0;
            let mut let_count = 0;
            let mut ident_count = 0;
            for token in &structural_tokens {
                match &token.kind {
                    TokenKind::FnDecl => fn_count += 1,
                    TokenKind::LetStmt => let_count += 1,
                    TokenKind::Ident(_) => ident_count += 1,
                    _ => {}
                }
            }

            println!("\nStructural token distribution:");
            println!("  fn_decl: {}", fn_count);
            println!("  let_stmt: {}", let_count);
            println!("  identifiers: {}", ident_count);
        }
        Err(e) => {
            eprintln!("Structural tokenization error: {}", e);
        }
    }
}