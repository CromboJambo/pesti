//! Benchmark structural tokenizer on real pesti source files.
//! Compare BPE vs structural token counts across the codebase.

use std::path::PathBuf;
use std::process::Command;

fn find_rust_files(root: &str) -> Vec<PathBuf> {
    let output = Command::new("find")
        .arg(root)
        .arg("-name")
        .arg("*.rs")
        .arg("-type")
        .arg("f")
        .output()
        .expect("failed to run find");

    let files: Vec<PathBuf> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect();

    files
}

fn main() {
    println!("=== PESTI CODEBASE TOKENIZATION BENCHMARK ===\n");

    let root = "/home/crombo/projects/pesti";
    let files = find_rust_files(root);

    println!("Found {} Rust files in pesti workspace\n", files.len());

    let mut total_bpe: usize = 0;
    let mut total_structural: usize = 0;
    let mut file_count: usize = 0;

    for path in &files {
        if !path.exists() {
            continue;
        }

        match std::fs::read_to_string(path) {
            Ok(src) => {
                // BPE baseline (using tiktoken-rs cl100k_base)
                let bpe = tiktoken_rs::cl100k_base().unwrap();
                let bpe_count = bpe.encode_ordinary(&src).len();
                drop(bpe);

                // Structural tokens
                let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
                match tokenizer.tokenize(&src) {
                    Ok(struct_tokens) => {
                        let struct_count = struct_tokens.len();
                        let ratio = bpe_count as f64 / struct_count as f64;

                        println!(
                            "{:50} {:>7}BPE {:>6}Struct {:>5.1}x",
                            path.file_name().unwrap_or_default().to_string_lossy(),
                            bpe_count,
                            struct_count,
                            ratio
                        );

                        total_bpe += bpe_count;
                        total_structural += struct_count;
                        file_count += 1;
                    }
                    Err(e) => {
                        println!("{}: ERROR - {}", path.display(), e);
                    }
                }
            }
            Err(_) => {}
        }
    }

    println!("\n=== TOTALS ===");
    println!("Files processed: {}", file_count);
    println!("Total BPE tokens: {}", total_bpe);
    println!("Total structural tokens: {}", total_structural);
    if total_structural > 0 {
        println!(
            "Overall compression ratio: {:.1}x",
            total_bpe as f64 / total_structural as f64
        );
    }
}
