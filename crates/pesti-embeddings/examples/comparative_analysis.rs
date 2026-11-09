//! Comparative analysis of llama.cpp, mistral.rs, and pesti using universal structure tokens.
//!
//! This demonstrates the value of our structural tokenizer tools by analyzing three
//! different LLM inference implementations and comparing their architectural patterns.

use anyhow::Result;
use std::fs;
use std::path::Path;

use universal_structure::adapters::{CppAdapter, RustAdapter};
use universal_structure::{LanguageAdapter, SpanKind};

fn main() -> Result<()> {
    println!("=== Comparative Analysis: llama.cpp vs mistral.rs vs pesti ===");
    println!("\nUsing universal structure tokens to compare architectural patterns across three LLM inference implementations.");
    println!();

    let projects = vec![
        ("llama.cpp", "/home/crombo/llama.cpp"),
        ("mistral.rs", "/home/crombo/projects/experiments/GitCloneResearch/mistral.rs"),
        ("pesti", "/home/crombo/projects/active/pesti"),
    ];

    let mut results = Vec::new();

    for (name, path) in &projects {
        println!("Analyzing {}...", name);
        let result = analyze_project(name, *path)?;
        results.push(result);
    }

    // Compare architectural complexity
    println!("\n=== Architectural Complexity Comparison ===");
    println!("{:<15} {:>10} {:>10} {:>10}", "Project", "Functions", "Classes", "Complexity");
    for r in &results {
        println!(
            "{:<15} {:>10} {:>10} {:>10.2}",
            r.name, r.function_count, r.class_count, r.complexity_score
        );
    }

    // Identify common architectural patterns
    println!("\n=== Common Architectural Patterns ===");
    let all_tokens: Vec<String> = results.iter().flat_map(|r| r.tokens.clone()).collect();
    if !all_tokens.is_empty() {
        let token_counts: std::collections::HashMap<String, usize> = all_tokens.iter().fold(
            std::collections::HashMap::new(),
            |mut map, token| {
                *map.entry(token.clone()).or_insert(0) += 1;
                map
            },
        );

        let mut sorted: Vec<_> = token_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        println!("Top structural patterns across all projects:");
        for (token, count) in sorted.iter().take(10) {
            println!("  {:<30} x{}", token, count);
        }
    }

    // Analyze codebase health metrics
    println!("\n=== Codebase Health Metrics ===");
    let mut total_functions = 0;
    let mut total_classes = 0;
    let mut total_complexity = 0.0;

    for r in &results {
        total_functions += r.function_count;
        total_classes += r.class_count;
        total_complexity += r.complexity_score;
    }

    println!("Total functions analyzed: {}", total_functions);
    println!("Total classes/interfaces: {}", total_classes);
    println!("Average complexity score: {:.2}", total_complexity / results.len() as f64);

    // Identify architectural differences
    println!("\n=== Architectural Differences ===");
    for r in &results {
        println!("\n{} unique patterns:", r.name);
        let project_tokens: std::collections::HashSet<String> = r.tokens.iter().cloned().collect();
        let other_tokens: Vec<std::collections::HashSet<String>> = results
            .iter()
            .filter(|other| other.name != r.name)
            .map(|other| other.tokens.iter().cloned().collect())
            .collect();

        for token in &r.tokens {
            let unique = other_tokens.iter().all(|set| !set.contains(token));
            if unique {
                println!("  - {}", token);
            }
        }
    }

    Ok(())
}

struct ProjectAnalysis {
    name: String,
    function_count: usize,
    class_count: usize,
    complexity_score: f64,
    tokens: Vec<String>,
}

fn analyze_project(name: &str, path: &str) -> Result<ProjectAnalysis> {
    let mut function_count = 0;
    let mut class_count = 0;
    let mut total_complexity = 0.0;
    let mut all_tokens = Vec::new();

    // Determine project type and select appropriate adapter
    let is_rust = name == "mistral.rs" || name == "pesti";

    // Analyze key architectural files
    let key_files = get_key_files(name, path);

    for file in &key_files {
        if !Path::new(file).exists() {
            continue;
        }

        let content = fs::read_to_string(file)?;

        if is_rust {
            let adapter = RustAdapter::new();
            let span_tree = adapter.tokenize_and_adapt(&content);
            for span in &span_tree.spans {
                all_tokens.push(span.kind.as_str().to_string());
                match span.kind {
                    SpanKind::FnDecl => {
                        function_count += 1;
                        total_complexity += estimate_function_complexity(span_tree.source.clone(), span);
                    }
                    SpanKind::ClassDecl | SpanKind::MethodDecl | SpanKind::TraitItem => {
                        class_count += 1;
                    }
                    _ => {}
                }
            }
        } else {
            // C/C++ project (llama.cpp)
            let adapter = CppAdapter::new();
            let span_tree = adapter.tokenize_and_adapt(&content);
            for span in &span_tree.spans {
                all_tokens.push(span.kind.as_str().to_string());
                match span.kind {
                    SpanKind::FnDecl => {
                        function_count += 1;
                        total_complexity += estimate_function_complexity(span_tree.source.clone(), span);
                    }
                    SpanKind::ClassDecl | SpanKind::MethodDecl | SpanKind::TraitItem => {
                        class_count += 1;
                    }
                    _ => {}
                }
            }
        }
    }

    let avg_complexity = if function_count > 0 {
        total_complexity / function_count as f64
    } else {
        0.0
    };

    Ok(ProjectAnalysis {
        name: name.to_string(),
        function_count,
        class_count,
        complexity_score: avg_complexity,
        tokens: all_tokens,
    })
}

fn get_key_files(name: &str, path: &str) -> Vec<String> {
    let mut files = Vec::new();

    match name {
        "llama.cpp" => {
            // Core inference and model architecture files
            files.push(format!("{}/src/llama-arch.cpp", path));
            files.push(format!("{}/src/llama-graph.cpp", path));
            files.push(format!("{}/src/llama-context.cpp", path));
            files.push(format!("{}/src/llama-kv-cache.cpp", path));
            files.push(format!("{}/ggml/src/ggml-quants.c", path));
        }
        "mistral.rs" => {
            // Core model and inference files
            files.push(format!("{}/mistralrs/src/model.rs", path));
            files.push(format!("{}/mistralrs/src/agent.rs", path));
            files.push(format!("{}/mistralrs/src/gguf.rs", path));
        }
        "pesti" => {
            // Core inference and model files
            files.push(format!("{}/pesti-runner/src/inference_engine.rs", path));
            files.push(format!("{}/pesti-runner/src/transformer.rs", path));
            files.push(format!("{}/pesti-runner/src/kernel/kvcache.rs", path));
            files.push(format!("{}/pesti-runner/src/gguf_weight_loader.rs", path));
        }
        _ => {}
    }

    files
}

fn estimate_function_complexity(_content: String, _span: &universal_structure::Span) -> f64 {
    // Simplified complexity estimation based on span type
    1.0
}