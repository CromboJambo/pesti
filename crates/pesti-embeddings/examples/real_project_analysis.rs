//! Real-project structural analysis for pesti-embeddings.
//!
//! Parses actual Rust source files from a project directory using the PEG grammar,
//! extracts structural features, and reports on coverage and quality.
//!
//! Usage:
//!   cargo run -p pesti-embeddings --example real_project_analysis <path> [--recursive]
//!
//! Examples:
//!   cargo run -p pesti-embeddings --example real_project_analysis /path/to/project/src
//!   cargo run -p pesti-embeddings --example real_project_analysis /path/to/project --recursive

use anyhow::Result;
use pesti_embeddings::{StructuralConfig, StructuralEmbedder};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

struct FileStats {
    path: String,
    lines: usize,
    parsed_ok: bool,
    error: Option<String>,
    feature_count: usize,
    roles: HashMap<String, usize>,
}

fn find_source_files(dir: &Path, recursive: bool, language: &str) -> Result<Vec<PathBuf>> {
    let ext = match language {
        "python" | "py" => "py",
        "typescript" | "ts" => "ts",
        _ => "rs",
    };
    let mut files = Vec::new();
    collect_files(dir, &mut files, recursive, ext)?;
    files.sort();
    Ok(files)
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>, recursive: bool, ext: &str) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Skip target/ and hidden directories
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "target" || name.starts_with('.') {
                continue;
            }
            if recursive {
                collect_files(&path, out, recursive, ext)?;
            }
        } else if path.extension().map(|e| e == ext).unwrap_or(false) {
            out.push(path);
        }
    }
    Ok(())
}

fn analyze_file(embedder: &mut StructuralEmbedder, path: &Path) -> FileStats {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            return FileStats {
                path: path.display().to_string(),
                lines: 0,
                parsed_ok: false,
                error: Some(format!("read error: {}", e)),
                feature_count: 0,
                roles: HashMap::new(),
            };
        }
    };

    let lines = content.lines().count();

    match embedder.extract_features(&content) {
        Ok(features) => {
            let mut roles = HashMap::new();
            for f in &features {
                *roles.entry(f.role.clone()).or_insert(0) += 1;
            }
            FileStats {
                path: path.display().to_string(),
                lines,
                parsed_ok: true,
                error: None,
                feature_count: features.len(),
                roles,
            }
        }
        Err(e) => FileStats {
            path: path.display().to_string(),
            lines,
            parsed_ok: false,
            error: Some(e.to_string()),
            feature_count: 0,
            roles: HashMap::new(),
        },
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path> [--recursive] [--language rust|python|typescript]", args[0]);
        std::process::exit(1);
    }

    let target_path = PathBuf::from(&args[1]);
    let recursive = args.iter().any(|a| a == "--recursive");
    let language = if let Some(pos) = args.iter().position(|a| a == "--language") {
        args.get(pos + 1).cloned().unwrap_or_else(|| "rust".to_string())
    } else {
        "rust".to_string()
    };

    if !target_path.exists() {
        anyhow::bail!("Path does not exist: {}", target_path.display());
    }

    // Initialize embedder with language-specific grammar
    let config = StructuralConfig {
        grammar_path: format!("crates/pesti-structural/grammars/{}.peg", language),
        language: Some(language.clone()),
        ..Default::default()
    };
    let mut embedder = StructuralEmbedder::new(config);

    // Find source files for the specified language
    let files = if target_path.is_file() {
        vec![target_path.clone()]
    } else {
        find_source_files(&target_path, recursive, &language)?
    };

    println!("=== Real Project Structural Analysis ===");
    println!("Target: {}", target_path.display());
    println!("Files found: {}\n", files.len());

    // Analyze each file
    let mut all_stats = Vec::new();
    for file in &files {
        let stats = analyze_file(&mut embedder, file);
        all_stats.push(stats);
    }

    // Summary
    let total_files = all_stats.len();
    let parsed_ok: Vec<_> = all_stats.iter().filter(|s| s.parsed_ok).collect();
    let parse_failures: Vec<_> = all_stats.iter().filter(|s| !s.parsed_ok).collect();
    let total_features: usize = parsed_ok.iter().map(|s| s.feature_count).sum();
    let total_lines: usize = all_stats.iter().map(|s| s.lines).sum();

    println!("=== Summary ===");
    println!("Total files: {}", total_files);
    println!(
        "Parsed OK: {} ({:.1}%)",
        parsed_ok.len(),
        parsed_ok.len() as f64 / total_files.max(1) as f64 * 100.0
    );
    println!("Parse failures: {}", parse_failures.len());
    println!("Total lines analyzed: {}", total_lines);
    println!("Total features extracted: {}", total_features);
    if !parsed_ok.is_empty() {
        let avg_features = total_features as f64 / parsed_ok.len() as f64;
        println!("Avg features per file: {:.1}", avg_features);
    }

    // Role distribution
    let mut role_totals: HashMap<String, usize> = HashMap::new();
    for stats in &parsed_ok {
        for (role, count) in &stats.roles {
            *role_totals.entry(role.clone()).or_insert(0) += count;
        }
    }
    if !role_totals.is_empty() {
        println!("\n=== Role Distribution ===");
        let mut roles_sorted: Vec<_> = role_totals.iter().collect();
        roles_sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (role, count) in &roles_sorted {
            println!("  {:20} {}", role, count);
        }
    }

    // Per-file breakdown (top files by features)
    println!("\n=== Files by Feature Count ===");
    let mut files_by_features: Vec<_> = parsed_ok.iter().collect();
    files_by_features.sort_by(|a, b| b.feature_count.cmp(&a.feature_count));
    for stats in files_by_features.iter().take(15) {
        println!(
            "  {:4} features | {:4} lines | {}",
            stats.feature_count,
            stats.lines,
            stats
                .path
                .strip_prefix(target_path.to_str().unwrap_or(""))
                .unwrap_or(&stats.path)
        );
    }

    // Parse failures
    if !parse_failures.is_empty() {
        println!("\n=== Parse Failures ===");
        for stats in &parse_failures {
            let err = stats.error.as_deref().unwrap_or("unknown");
            // Truncate long error messages
            let err_short = if err.len() > 80 {
                format!("{}...", &err[..77])
            } else {
                err.to_string()
            };
            println!(
                "  FAIL {:4} lines | {} | {}",
                stats.lines,
                stats
                    .path
                    .strip_prefix(target_path.to_str().unwrap_or(""))
                    .unwrap_or(&stats.path),
                err_short
            );
        }

        // Error pattern analysis
        println!("\n=== Common Parse Errors ===");
        let mut error_patterns: HashMap<String, usize> = HashMap::new();
        for stats in &parse_failures {
            if let Some(err) = &stats.error {
                // Extract the first meaningful part of the error
                let pattern = err
                    .lines()
                    .next()
                    .unwrap_or("unknown")
                    .chars()
                    .take(60)
                    .collect::<String>();
                *error_patterns.entry(pattern).or_insert(0) += 1;
            }
        }
        let mut patterns_sorted: Vec<_> = error_patterns.iter().collect();
        patterns_sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (pattern, count) in &patterns_sorted {
            println!("  [{}] {}", count, pattern);
        }
    }

    // Sample feature extraction from a successful file
    if let Some(sample) = files_by_features.first() {
        println!("\n=== Sample Features ({}) ===", sample.path);
        let content = fs::read_to_string(&PathBuf::from(&sample.path))?;
        if let Ok(features) = embedder.extract_features(&content) {
            for (i, f) in features.iter().take(10).enumerate() {
                println!(
                    "  [{:2}] role={:<16} class={:<14} {}",
                    i,
                    f.role,
                    f.class,
                    truncate(&f.label, 50)
                );
            }
            if features.len() > 10 {
                println!("  ... and {} more", features.len() - 10);
            }
        }
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
