//! Benchmark: BPE (cl100k_base) vs PESTI structural tokenizer
//! on real Rust source files. Measures compression ratio, processing time,
//! and information density.

use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn count_lines(content: &str) -> usize {
    content.lines().count()
}

fn category_name(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::StructDecl => "struct",
        TokenKind::EnumDecl => "enum",
        TokenKind::FnDecl => "fn_decl",
        TokenKind::ImplBlock => "impl",
        TokenKind::IfElse => "if_else",
        TokenKind::ForLoop => "for_loop",
        TokenKind::WhileLoop => "while_loop",
        TokenKind::MatchExpr => "match",
        TokenKind::LetStmt => "let_stmt",
        TokenKind::ExprStmt => "expr_stmt",
        TokenKind::BlockStart => "{",
        TokenKind::BlockEnd => "}",
        TokenKind::ParenOpen => "(",
        TokenKind::ParenClose => ")",
        TokenKind::Ident(_) => "ident",
        TokenKind::StrLit => "string",
        TokenKind::IntLit => "number",
        _ => "other",
    }
}

fn benchmark_file(path: &Path, tokenizer: &StructuralTokenizer) {
    let filename = path.file_name().unwrap().to_string_lossy();

    match fs::read_to_string(path) {
        Ok(content) => {
            let bytes = content.len();
            let lines = count_lines(&content);

            // Structural tokenizer
            let start = Instant::now();
            let struct_tokens = tokenizer.tokenize(&content).unwrap();
            let struct_time = start.elapsed().as_secs_f64() * 1000.0;

            // BPE tokenizer (tiktoken cl100k_base)
            let start = Instant::now();
            let bpe_tokenizer = tiktoken_rs::cl100k_base().unwrap();
            let bpe_tokens = bpe_tokenizer.encode_ordinary(&content);
            let bpe_time = start.elapsed().as_secs_f64() * 1000.0;

            println!("=== {} ({}, {} lines) ===", filename, bytes, lines);
            println!(
                "BPE (cl100k_base):   {} tokens in {:.2}ms",
                bpe_tokens.len(),
                bpe_time
            );
            println!(
                "Structural:          {} tokens in {:.2}ms",
                struct_tokens.len(),
                struct_time
            );
            println!(
                "Compression ratio:   {:.2}x",
                bytes as f64 / struct_tokens.len() as f64
            );

            // Structural category distribution
            let mut categories = HashMap::new();
            for t in &struct_tokens {
                *categories.entry(category_name(&t.kind)).or_insert(0usize) += 1;
            }
            println!("Structural categories:");
            for (name, count) in categories.iter().take(10) {
                println!("  {}: {}", name, count);
            }

            // Max nesting depth
            let mut depth = 0;
            let mut max_depth = 0;
            for t in &struct_tokens {
                match &t.kind {
                    TokenKind::BlockStart => {
                        depth += 1;
                        if depth > max_depth {
                            max_depth = depth;
                        }
                    }
                    TokenKind::BlockEnd => {
                        if depth > 0 {
                            depth -= 1;
                        }
                    }
                    _ => {}
                }
            }
            println!("Max nesting depth:   {}", max_depth);
            println!();
        }
        Err(e) => {
            eprintln!("Error reading {}: {}", filename, e);
        }
    }
}

fn main() {
    let tokenizer = StructuralTokenizer::new();

    // Test files: mix of pesti and crabjar crates
    let test_files = vec![
        "pesti-runner/src/llama.rs",
        "pesti-runner/src/lib.rs",
        "pesti-runner/src/transformer/model.rs",
        "pesti-runner/src/transformer/tokenizer.rs",
        "crates/pesti-structural-tokenizer/src/lib.rs",
    ];

    for path in test_files {
        if Path::new(path).exists() {
            benchmark_file(Path::new(path), &tokenizer);
        } else {
            println!("=== {} (not found) ===", path);
        }
    }

    println!("Benchmark complete.");
}
