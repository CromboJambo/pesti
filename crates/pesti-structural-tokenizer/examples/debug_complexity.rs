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

fn main() {
    let root = "/home/crombo/projects/active/pesti";
    let files = walk_dir(root);
    
    println!("Found {} Rust source files", files.len());
    
    let tokenizer = StructuralTokenizer::new();
    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_fn_count = 0;
    
    for f in &files {
        let content = match fs::read_to_string(f) {
            Ok(c) => c,
            Err(_) => continue,
        };
        
        match tokenizer.tokenize(&content) {
            Ok(tokens) => {
                success_count += 1;
                for tok in &tokens {
                    if let TokenKind::FnDecl = tok.kind {
                        total_fn_count += 1;
                    }
                }
            }
            Err(_) => {
                fail_count += 1;
            }
        }
    }
    
    println!("Tokenization success: {}", success_count);
    println!("Tokenization failures: {}", fail_count);
    println!("Total FnDecl tokens found: {}", total_fn_count);
}
