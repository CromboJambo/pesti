//! Comprehensive benchmark: tokenize all Rust files across ~/projects/
//! Reports compression ratios, failure modes, and edge cases.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct FileResult {
    path: String,
    bytes: usize,
    bpe_tokens: usize,
    structural_tokens: usize,
    ratio: f64,
    status: &'static str,
}

fn find_rust_files(root: &str) -> Vec<PathBuf> {
    let output = Command::new("find")
        .arg(root)
        .arg("-name")
        .arg("*.rs")
        .arg("-type")
        .arg("f")
        .output()
        .expect("failed to run find");

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn main() {
    let root = "/home/crombo/projects";
    println!("=== FULL ~/projects/ TOKENIZATION BENCHMARK ===");
    println!("Root: {}", root);
    println!();

    let files = find_rust_files(root);
    println!("Found {} Rust files across all projects\n", files.len());

    let mut results: Vec<FileResult> = Vec::new();
    let mut by_project: HashMap<String, Vec<usize>> = HashMap::new();

    for path in &files {
        if !path.exists() {
            continue;
        }

        // Skip generated/build artifacts
        if path.to_string_lossy().contains("/target/")
            || path.to_string_lossy().contains("/build/")
            || path.to_string_lossy().contains(".git/")
        {
            continue;
        }

        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Skip empty files
        if src.trim().is_empty() {
            continue;
        }

        let bytes = src.len();

        // BPE baseline
        let bpe = tiktoken_rs::cl100k_base().unwrap();
        let bpe_count = bpe.encode_ordinary(&src).len();
        drop(bpe);

        // Structural tokens
        let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
        match tokenizer.tokenize(&src) {
            Ok(struct_tokens) => {
                let struct_count = struct_tokens.len();
                let ratio = if struct_count > 0 {
                    bpe_count as f64 / struct_count as f64
                } else {
                    0.0
                };

                // Detect suspicious results (likely tokenizer bugs)
                let status = if ratio > 100.0 && bytes > 1000 {
                    "SUSPICIOUS_HIGH"
                } else if struct_count == 0 && bytes > 10 {
                    "ZERO_STRUCTURAL"
                } else {
                    "OK"
                };

                results.push(FileResult {
                    path: path.to_string_lossy().to_string(),
                    bytes,
                    bpe_tokens: bpe_count,
                    structural_tokens: struct_count,
                    ratio,
                    status,
                });

                // Track by project
                let project = path
                    .components()
                    .nth(3)
                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                by_project.entry(project).or_default().push(struct_count);
            }
            Err(e) => {
                results.push(FileResult {
                    path: path.to_string_lossy().to_string(),
                    bytes,
                    bpe_tokens: bpe_count,
                    structural_tokens: 0,
                    ratio: 0.0,
                    status: "PARSE_ERROR",
                });
            }
        }
    }

    // Overall stats
    let total_bpe: usize = results.iter().map(|r| r.bpe_tokens).sum();
    let total_structural: usize = results.iter().map(|r| r.structural_tokens).sum();
    let overall_ratio = if total_structural > 0 {
        total_bpe as f64 / total_structural as f64
    } else {
        0.0
    };

    println!("=== OVERALL RESULTS ===");
    println!("Files processed: {}", results.len());
    println!("Total BPE tokens: {}", total_bpe);
    println!("Total structural tokens: {}", total_structural);
    println!("Overall compression ratio: {:.1}x\n", overall_ratio);

    // Per-project breakdown
    println!("=== BY PROJECT ===");
    for (project, counts) in by_project.iter() {
        let project_total: usize = counts.iter().sum();
        println!(
            "{:30} {:5} files {:8} structural tokens",
            project,
            counts.len(),
            project_total
        );
    }

    // Suspicious results (potential tokenizer bugs)
    let suspicious: Vec<&FileResult> = results
        .iter()
        .filter(|r| r.status == "SUSPICIOUS_HIGH" || r.status == "ZERO_STRUCTURAL")
        .collect();

    if !suspicious.is_empty() {
        println!("\n=== SUSPICIOUS RESULTS (potential tokenizer bugs) ===");
        for r in &suspicious {
            println!(
                "{:40} {:>6}BPE {:>5}Struct {:>8.1}x [{}]",
                r.path.split('/').last().unwrap_or("unknown"),
                r.bpe_tokens,
                r.structural_tokens,
                r.ratio,
                r.status
            );
        }
    }

    // Parse errors
    let errors: Vec<&FileResult> = results
        .iter()
        .filter(|r| r.status == "PARSE_ERROR")
        .collect();
    if !errors.is_empty() {
        println!("\n=== PARSE ERRORS ===");
        for r in &errors {
            println!(
                "{:40} {:>6}BPE {} bytes",
                r.path.split('/').last().unwrap_or("unknown"),
                r.bpe_tokens,
                r.bytes
            );
        }
    }

    // Best and worst compression (excluding suspicious)
    let good_results: Vec<&FileResult> = results
        .iter()
        .filter(|r| r.status == "OK" && r.structural_tokens > 0)
        .collect();

    if !good_results.is_empty() {
        println!("\n=== BEST COMPRESSION (top 10) ===");
        let mut sorted: Vec<&FileResult> = good_results.clone();
        sorted.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap());
        for r in sorted.iter().take(10) {
            println!(
                "{:40} {:>6}BPE {:>5}Struct {:>8.1}x",
                r.path.split('/').last().unwrap_or("unknown"),
                r.bpe_tokens,
                r.structural_tokens,
                r.ratio
            );
        }

        println!("\n=== WORST COMPRESSION (bottom 10) ===");
        sorted.sort_by(|a, b| a.ratio.partial_cmp(&b.ratio).unwrap());
        for r in sorted.iter().take(10) {
            println!(
                "{:40} {:>6}BPE {:>5}Struct {:>8.1}x",
                r.path.split('/').last().unwrap_or("unknown"),
                r.bpe_tokens,
                r.structural_tokens,
                r.ratio
            );
        }
    }
}
