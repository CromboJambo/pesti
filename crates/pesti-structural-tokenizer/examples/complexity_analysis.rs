use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

#[derive(Debug)]
struct FunctionInfo {
    name: String,
    file: String,
    cyclomatic: u32,
    max_depth: u32,
    control_flow: u32,
}

fn analyze_function_complexity(tokens: &[pesti_structural_tokenizer::StructuralToken]) -> (u32, u32, u32) {
    let mut nesting_depth = 0;
    let mut max_depth = 0;
    let mut control_flow_count = 0;

    for tok in tokens {
        match tok.kind {
            TokenKind::BlockStart | TokenKind::ParenOpen | TokenKind::BraceOpen => {
                nesting_depth += 1;
                if nesting_depth > max_depth {
                    max_depth = nesting_depth;
                }
            }
            TokenKind::BlockEnd | TokenKind::ParenClose | TokenKind::BraceClose => {
                if nesting_depth > 0 {
                    nesting_depth -= 1;
                }
            }
            TokenKind::IfElse | TokenKind::WhileLoop | TokenKind::ForLoop
            | TokenKind::MatchExpr | TokenKind::ReturnExpr | TokenKind::BreakExpr
            | TokenKind::ContinueExpr => {
                control_flow_count += 1;
            }
            _ => {}
        }
    }

    let cyclomatic = control_flow_count + 1;
    (cyclomatic, max_depth, control_flow_count)
}

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

    println!("Analyzing {} Rust source files for complexity...", files.len());

    let tokenizer = StructuralTokenizer::new();
    let mut all_functions = Vec::new();
    let mut file_count = 0;

    for f in &files {
        let content = match fs::read_to_string(f) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let tokens = match tokenizer.tokenize(&content) {
            Ok(t) => t,
            Err(_) => continue,
        };

        file_count += 1;

        // Find all function declarations and their bodies
        for (i, tok) in tokens.iter().enumerate() {
            if let TokenKind::FnDecl = tok.kind {
                // Find the function name
                let mut name = "unknown".to_string();
                if i + 1 < tokens.len() {
                    if let TokenKind::Ident(n) = &tokens[i + 1].kind {
                        name = n.clone();
                    }
                }

                // Analyze complexity from this point to the next FnDecl or end
                let body_end = tokens.iter().skip(i + 1).position(|t| {
                    matches!(t.kind, TokenKind::FnDecl | TokenKind::StructDecl | TokenKind::EnumDecl)
                }).map(|p| p + i + 1).unwrap_or(tokens.len());

                let fn_tokens = &tokens[i..body_end];
                let (cyclomatic, max_depth, control_flow) = analyze_function_complexity(fn_tokens);

                all_functions.push(FunctionInfo {
                    name,
                    file: f.to_string_lossy().to_string(),
                    cyclomatic,
                    max_depth,
                    control_flow,
                });
            }
        }
    }

    println!("Analyzed {} files", file_count);
    println!("Found {} function declarations", all_functions.len());

    // Sort by cyclomatic complexity (highest first)
    all_functions.sort_by(|a, b| b.cyclomatic.cmp(&a.cyclomatic));

    println!("\n=== Top 30 Most Complex Functions ===");
    println!("{:<50} {:<20} {:>10} {:>8} {:>12}", "File", "Function", "Cyclomatic", "Max Depth", "Control Flow");
    println!("{}", "-".repeat(100));

    for f in all_functions.iter().take(30) {
        let short_file = if f.file.starts_with("/home/crombo/projects/active/pesti/") {
            f.file.replace("/home/crombo/projects/active/pesti/", "")
        } else {
            f.file.clone()
        };

        println!("{:<50} {:<20} {:>10} {:>8} {:>12}", short_file, f.name, f.cyclomatic, f.max_depth, f.control_flow);
    }

    // Summary statistics
    let avg_cyclomatic: f64 = all_functions.iter().map(|f| f.cyclomatic as f64).sum::<f64>() / all_functions.len() as f64;
    let high_complexity = all_functions.iter().filter(|f| f.cyclomatic > 10).count();

    println!("\n=== Summary ===");
    println!("Average cyclomatic complexity: {:.1}", avg_cyclomatic);
    println!("Functions with cyclomatic complexity > 10: {} ({:.1}%)",
             high_complexity, (high_complexity as f64 / all_functions.len() as f64) * 100.0);
}
