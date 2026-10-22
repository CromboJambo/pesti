use std::fs;
use std::path::PathBuf;
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

fn walk_dir(path: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
                if name != "target" && name != ".venv" && name != ".git" {
                    results.extend(walk_dir(&p.to_string_lossy()));
                }
            } else if p.extension().map_or(false, |e| e == "rs") {
                results.push(p);
            }
        }
    }
    results
}

fn analyze_file(path: &PathBuf) -> (u32, u32, u32) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return (0, 0, 0),
    };
    
    let tokenizer = StructuralTokenizer::new();
    let tokens = match tokenizer.tokenize(&content) {
        Ok(t) => t,
        Err(_) => return (0, 0, 0),
    };
    
    let mut fns = 0;
    let mut types = 0;
    let mut total = 0;
    
    for tok in &tokens {
        total += 1;
        match tok.kind {
            TokenKind::FnDecl => fns += 1,
            TokenKind::StructDecl | TokenKind::EnumDecl => types += 1,
            _ => {}
        }
    }
    
    (fns, types, total)
}

fn main() {
    let root = "/home/crombo/projects/active/pesti";
    let files = walk_dir(root);
    
    println!("Found {} Rust source files", files.len());
    
    let mut total_fns = 0;
    let mut total_types = 0;
    let mut total_tokens = 0;
    let mut file_count = 0;
    
    for f in &files {
        let (fns, types, tokens) = analyze_file(f);
        if tokens > 0 {
            total_fns += fns;
            total_types += types;
            total_tokens += tokens;
            file_count += 1;
        }
    }
    
    println!("Successfully analyzed {} files", file_count);
    println!("Total structural tokens: {}", total_tokens);
    println!("Function declarations (FnDecl): {}", total_fns);
    println!("Type declarations (Struct/Enum): {}", total_types);
}
