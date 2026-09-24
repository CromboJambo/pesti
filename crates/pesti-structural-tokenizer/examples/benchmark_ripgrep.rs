//! Benchmark structural tokenizer on ripgrep codebase

use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;
use std::path::{Path, PathBuf};

fn find_rust_files(root: &str) -> Vec<PathBuf> {
    let mut files = vec![];
    fn walk(path: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(path) {
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
                } else if p.extension() == Some(std::ffi::OsStr::new("rs")) {
                    out.push(p);
                }
            }
        }
    }
    walk(Path::new(root), &mut files);
    files
}

fn main() {
    let root = "/home/crombo/projects/ripgrep";
    println!("=== Ripgrep Structural Tokenizer Benchmark ===");
    println!();

    let files = find_rust_files(root);
    println!("Found {} Rust files", files.len());

    let tokenizer = StructuralTokenizer::new();

    let mut total_bytes = 0;
    let mut total_structural = 0;
    let mut errors = 0;

    for file in &files {
        match fs::read_to_string(file) {
            Ok(src) => {
                let structural = tokenizer.tokenize(&src);
                match structural {
                    Ok(structural_tokens) => {
                        total_structural += structural_tokens.len();
                        total_bytes += src.len();
                    }
                    Err(_) => {
                        errors += 1;
                    }
                }
            }
            Err(_) => {}
        }
    }

    println!();
    println!("Results:");
    println!("--------");
    println!("Files processed: {}", files.len() - errors);
    println!("Parse errors: {}", errors);
    println!("Total bytes: {}", total_bytes);
    println!("Structural tokens: {}", total_structural);
}