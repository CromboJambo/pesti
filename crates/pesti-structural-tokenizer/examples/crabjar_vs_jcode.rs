use pesti_structural_tokenizer::StructuralTokenizer;
use std::collections::HashMap;
use std::fs;

fn analyze_file(path: &str, label: &str) -> Option<(usize, usize, usize, HashMap<String, usize>)> {
    println!("=== {} ({}) ===", label, path);

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("  ERROR reading: {}", e);
            return None;
        }
    };

    let lines = content.lines().count();
    let bytes = content.len();

    let tokenizer = StructuralTokenizer::new();
    let tokens = match tokenizer.tokenize(&content) {
        Ok(t) => t,
        Err(e) => {
            println!("  ERROR tokenizing: {}", e);
            return None;
        }
    };

    let token_count = tokens.len();
    let compression_ratio = bytes as f64 / (token_count as f64 * 5.0); // approximate bytes per structural token
    println!(
        "  Lines: {} | Bytes: {} | Structural tokens: {} | Compression: {:.1}x",
        lines, bytes, token_count, compression_ratio
    );

    // Count by kind
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tok in &tokens {
        let kind_name = match tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => "FnDecl",
            pesti_structural_tokenizer::TokenKind::LetStmt => "LetStmt",
            pesti_structural_tokenizer::TokenKind::StructDecl => "StructDecl",
            pesti_structural_tokenizer::TokenKind::EnumDecl => "EnumDecl",
            pesti_structural_tokenizer::TokenKind::ImplBlock => "ImplBlock",
            pesti_structural_tokenizer::TokenKind::IfElse => "IfElse",
            pesti_structural_tokenizer::TokenKind::ForLoop => "ForLoop",
            pesti_structural_tokenizer::TokenKind::WhileLoop => "WhileLoop",
            pesti_structural_tokenizer::TokenKind::MatchExpr => "MatchExpr",
            pesti_structural_tokenizer::TokenKind::Closure => "Closure",
            pesti_structural_tokenizer::TokenKind::TraitDecl => "TraitDecl",
            pesti_structural_tokenizer::TokenKind::Attribute => "Attribute",
            pesti_structural_tokenizer::TokenKind::UseStmt => "UseStmt",
            pesti_structural_tokenizer::TokenKind::ModDecl => "ModDecl",
            _ => continue,
        };
        *counts.entry(kind_name.to_string()).or_insert(0) += 1;
    }

    println!("  Top structural elements:");
    let mut entries: Vec<_> = counts.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    for (kind, count) in entries.iter().take(8) {
        println!("    {}: {}", kind, count);
    }
    println!();

    Some((lines, bytes, token_count, counts))
}

fn main() {
    let crabjar_base = "/home/crombo/projects/active/crabjar";
    let jcode_base = "/tmp/jcode";

    println!("========================================");
    println!("CRABJAR vs JCODE: Structural Comparison");
    println!("Using pesti-structural-tokenizer v0.2");
    println!("========================================\n");

    // Crabjar core implementation files
    analyze_file(
        &format!("{}/orchestrator/src/lm_studio_client/client.rs", crabjar_base),
        "crabjar/orchestrator client",
    );
    analyze_file(
        &format!("{}/guard/src/lib.rs", crabjar_base),
        "crabjar/guard",
    );

    println!("--- JCODE ---\n");

    // Jcode core implementation files
    analyze_file(
        &format!("{}/crates/jcode-agent-runtime/src/lib.rs", jcode_base),
        "jcode/agent-runtime",
    );
    analyze_file(
        &format!("{}/crates/jcode-core/src/util.rs", jcode_base),
        "jcode/core util",
    );

    println!("========================================");
    println!("Analysis complete.");
    println!("========================================");
}