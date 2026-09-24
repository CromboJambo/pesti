//! Benchmark structural tokenizer on ripgrep codebase with detailed error reporting

use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;
use std::path::{Path, PathBuf};

fn find_rust_files(root: &str) -> Vec<PathBuf> {
    let mut files = vec![];
    fn walk(path: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(path) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let skip = ["target", ".git"];
                let should_skip = skip.iter().any(|s| {
                    p.file_name()
                        .map(|n| n.to_string_lossy() == *s)
                        .unwrap_or(false)
                });
                if !should_skip {
                    walk(&p, out);
                }
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    walk(Path::new(root), &mut files);
    files.sort();
    files
}

fn main() {
    let root = "/home/crombo/projects/ripgrep";
    let files = find_rust_files(root);
    
    println!("Ripgrep structural tokenizer benchmark");
    println!("=======================================");
    println!("Files found: {}", files.len());
    println!();
    
    let tokenizer = StructuralTokenizer::new();
    
    let mut total_bytes = 0;
    let mut total_tokens = 0;
    let mut errors = vec![];
    
    for file in &files {
        match fs::read_to_string(file) {
            Ok(src) => {
                total_bytes += src.len();
                match tokenizer.tokenize(&src) {
                    Ok(tokens) => total_tokens += tokens.len(),
                    Err(e) => errors.push((file.clone(), src.len(), e.to_string())),
                }
            },
            Err(_) => {}
        }
    }
    
    let success = files.len() - errors.len();
    println!("Parsed successfully: {}/{}", success, files.len());
    println!("Total source bytes: ~{} MB", total_bytes / 1024 / 1024);
    println!("Total structural tokens: {}", total_tokens);
    
    if !errors.is_empty() {
        println!("\n=== PARSE ERRORS (detailed) ===");
        for (file, size, err) in &errors {
            let rel = file.strip_prefix(root).unwrap_or(file.as_path());
            println!("\nFile: {}", rel.display());
            println!("Size: {} bytes", size);
            println!("Error: {}", err);
            
            // Show a snippet of the problematic file
            if let Ok(src) = fs::read_to_string(file) {
                println!("First 200 chars:");
                let snippet = src.chars().take(200).collect::<String>();
                for line in snippet.lines() {
                    println!("  | {}", line);
                }
            }
        }
    }
    
    if success > 0 {
        let ratio = total_bytes as f64 / total_tokens as f64;
        println!("\nAvg bytes per structural token: {:.0}", ratio);
    }
}