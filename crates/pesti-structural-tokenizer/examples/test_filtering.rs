//! Test structural filtering on real code from user's projects

use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::fs;

fn main() {
    // Find a file with actual function declarations
    let test_files = [
        "/home/crombo/projects/pesti/crates/pesti-runner/src/main.rs",
        "/home/crombo/projects/pesti/crates/pesti-structural-tokenizer/examples/benchmark_real_files.rs",
    ];

    for path in &test_files {
        if let Ok(src) = fs::read_to_string(path) {
            println!("Testing: {}", path);
            println!("Source: {} bytes, {} lines", src.len(), src.lines().count());

            let tokenizer = StructuralTokenizer::new();
            match tokenizer.tokenize(&src) {
                Ok(tokens) => {
                    println!("Total structural tokens: {}", tokens.len());

                    // Count by type
                    let mut counts = std::collections::HashMap::new();
                    for tok in &tokens {
                        *counts.entry(format!("{:?}", tok.kind)).or_insert(0) += 1;
                    }
                    for (kind, count) in &counts {
                        println!("  {}: {}", kind, count);
                    }

                    // Filter: just function declarations
                    let fns: Vec<_> = tokens.iter()
                        .filter(|t| matches!(t.kind, TokenKind::FnDecl))
                        .collect();

                    println!("\nFunction declarations found: {}", fns.len());
                    if !fns.is_empty() {
                        println!("First function preview:");
                        let preview = &fns[0].text[..std::cmp::min(200, fns[0].text.len())];
                        println!("{}", preview);
                    }

                    // Filter: just error handling (try/catch patterns)
                    let error_handling: Vec<_> = tokens.iter()
                        .filter(|t| {
                            matches!(t.kind, TokenKind::MatchExpr) ||
                            t.text.contains("Err(") ||
                            t.text.contains("panic!") ||
                            t.text.contains("?")
                        })
                        .collect();

                    println!("\nError-handling patterns found: {}", error_handling.len());
                    for (i, tok) in error_handling.iter().take(3).enumerate() {
                        let preview = &tok.text[..std::cmp::min(80, tok.text.len())];
                        println!("  [{}] {}...", i, preview.replace("\n", " "));
                    }

                    // Size comparison: full vs filtered
                    let full_size: usize = tokens.iter().map(|t| t.text.len()).sum();
                    let fn_size: usize = fns.iter().map(|t| t.text.len()).sum();
                    let err_size: usize = error_handling.iter().map(|t| t.text.len()).sum();

                    println!("\nSize comparison:");
                    println!("  Full file tokens: {} chars", full_size);
                    if !fns.is_empty() {
                        println!("  Function declarations only: {} chars ({:.1}x smaller)", fn_size, full_size as f64 / fn_size as f64);
                    }
                    if !error_handling.is_empty() {
                        println!("  Error handling only: {} chars ({:.1}x smaller)", err_size, full_size as f64 / err_size as f64);
                    }
                }
                Err(e) => {
                    println!("Error tokenizing: {:?}", e);
                }
            }
            break; // Just test one file for now
        }
    }
}