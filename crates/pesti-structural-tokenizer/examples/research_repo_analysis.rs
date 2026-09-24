use pesti_structural_tokenizer::StructuralTokenizer;
use std::collections::HashMap;
use std::fs;

fn analyze_file(path: &str, label: &str) {
    println!("=== {} ({}) ===", label, path);

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("  ERROR reading: {}", e);
            return;
        }
    };

    let tokenizer = StructuralTokenizer::new();
    let tokens = match tokenizer.tokenize(&content) {
        Ok(t) => t,
        Err(e) => {
            println!("  ERROR tokenizing: {}", e);
            return;
        }
    };

    println!("  Lines: {}", content.lines().count());
    println!("  Bytes: {}", content.len());
    println!("  Structural tokens: {}\n", tokens.len());

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
    let mut entries: Vec<_> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    for (kind, count) in entries.iter().take(8) {
        println!("    {}: {}", kind, count);
    }
    println!();
}

fn main() {
    let base = "/home/crombo/projects/docs/GitCloneResearch";

    // Test on core files from different research repos
    analyze_file(&format!("{}/dora/core/src/lib.rs", base), "dora/core");
    analyze_file(
        &format!("{}/mistral.rs/mistralrs-core/src/lib.rs", base),
        "mistral.rs",
    );
    analyze_file(
        &format!("{}/ironclaw/crates/ironclaw_agent_loop/src/lib.rs", base),
        "ironclaw agent loop",
    );
    analyze_file(&format!("{}/ratty/src/main.rs", base), "ratty");
    analyze_file(
        &format!("{}/oxc/crates/oxc_linter/src/lib.rs", base),
        "oxc linter",
    );
}
