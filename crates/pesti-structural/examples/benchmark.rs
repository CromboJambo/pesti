//! Benchmark for the structural PEG pipeline.
//!
//! Measures:
//! 1. Parse performance (files/s, features/file) on real code
//! 2. Structural similarity: structurally similar functions should produce
//!    closer feature sets than structurally different ones.
//!
//! Usage:
//!   cargo run -p pesti-structural --example benchmark [--path <dir>]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pesti_structural::evaluator::PegEvaluator;
use pesti_structural::lexer::tokenize_rust;
use pesti_structural::parser::parse_peg_file;

/// A feature signature: (role, class) pairs from a parse.
type FeatureSig = Vec<(String, String)>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_path = args
        .iter()
        .position(|a| a == "--path")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("crates"));

    println!("=== Structural PEG Benchmark ===\n");
    println!("Target: {}\n", target_path.display());

    // Load grammar once.
    let grammar_path = "crates/pesti-structural/grammars/rust.peg";
    let grammar = parse_peg_file(Path::new(grammar_path)).expect("Failed to load rust.peg");
    println!(
        "Grammar: {} ({} rules)\n",
        grammar.name,
        grammar.productions.len()
    );

    // Collect Rust files.
    let mut files = Vec::new();
    collect_rust_files(&target_path, &mut files);
    println!("Rust files found: {}\n", files.len());

    if files.is_empty() {
        eprintln!("No Rust files found at {}", target_path.display());
        return;
    }

    // === Part 1: Parse Performance ===
    let mut total_features = 0usize;
    let mut parse_times = Vec::new();
    let mut feature_counts = Vec::new();
    let mut failed_files = Vec::new();

    let start_all = Instant::now();

    for file in &files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };

        if source.trim().is_empty() {
            continue;
        }

        let start = Instant::now();

        let (tokens, _lex_errors) = tokenize_rust(&source);
        if tokens.is_empty() {
            continue;
        }

        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse_with_recovery();

        let elapsed = start.elapsed();
        parse_times.push(elapsed);

        // Count features (annotated nodes).
        if let Some(tree) = &result.tree {
            let count = count_annotated_nodes(tree, &grammar);
            feature_counts.push(count);
            total_features += count;
        } else {
            failed_files.push(file.clone());
        }
    }

    let total_time = start_all.elapsed();
    let total_lines: usize = files.iter().filter_map(|f| fs::read_to_string(f).ok()).map(|s| s.lines().count()).sum();

    println!("=== Parse Performance ===");
    println!("Files parsed:     {}/{}", files.len() - failed_files.len(), files.len());
    println!("Total time:       {:.3}s", total_time.as_secs_f64());
    println!("Avg time/file:    {:.2}ms", parse_times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / parse_times.len() as f64 * 1000.0);

    let mut sorted_times = parse_times.clone();
    sorted_times.sort_by_key(|t| t.as_nanos());
    if !sorted_times.is_empty() {
        println!("Median time/file: {:.2}ms", sorted_times[sorted_times.len() / 2].as_secs_f64() * 1000.0);
        println!("P95 time/file:    {:.2}ms", sorted_times[(sorted_times.len() as f64 * 0.95) as usize].as_secs_f64() * 1000.0);
        println!("Max time/file:    {:.2}ms", sorted_times.last().unwrap().as_secs_f64() * 1000.0);
    }

    println!("Total features:   {}", total_features);
    if !feature_counts.is_empty() {
        println!("Avg features/file: {:.1}", feature_counts.iter().sum::<usize>() as f64 / feature_counts.len() as f64);
    }
    println!("Lines processed:  {}", total_lines);
    println!("Throughput:       {:.0} files/s, {:.0} lines/s", files.len() as f64 / total_time.as_secs_f64(), total_lines as f64 / total_time.as_secs_f64());

    if !failed_files.is_empty() {
        println!("\nFailed files ({}):", failed_files.len());
        for f in failed_files.iter().take(5) {
            println!("  - {}", f.display());
        }
    }

    // === Part 2: Structural Similarity ===
    println!("\n=== Structural Similarity Test ===");
    test_structural_similarity(&grammar);

    // === Part 3: Feature Distribution ===
    println!("\n=== Feature Distribution (sample) ===");
    let sample_files: Vec<&PathBuf> = files.iter().take(50).collect();
    let mut role_counts: HashMap<String, usize> = HashMap::new();
    for file in &sample_files {
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if source.trim().is_empty() {
            continue;
        }
        let (tokens, _) = tokenize_rust(&source);
        if tokens.is_empty() {
            continue;
        }
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse_with_recovery();
        if let Some(tree) = &result.tree {
            count_roles(tree, &grammar, &mut role_counts);
        }
    }
    let mut role_vec: Vec<_> = role_counts.into_iter().collect();
    role_vec.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (role, count) in role_vec {
        println!("  {:>16}: {}", role, count);
    }
}

fn collect_rust_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().map_or(false, |e| e == "rs") {
            out.push(path.to_path_buf());
        }
        return;
    }
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                // Skip target/ and .git/
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name == "target" || name == ".git" {
                        continue;
                    }
                }
                collect_rust_files(&p, out);
            } else if p.extension().map_or(false, |e| e == "rs") {
                out.push(p);
            }
        }
    }
}

fn count_annotated_nodes(node: &pesti_structural::tree::TreeNode, grammar: &pesti_structural::grammar::StructuralGrammar) -> usize {
    let mut count = 0;
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if grammar.annotations.contains_key(&pesti_structural::grammar::Nonterminal(rule_name.to_string())) {
        count += 1;
    }
    for child in &node.children {
        count += count_annotated_nodes(child, grammar);
    }
    for capture in node.captures.values() {
        count += count_annotated_nodes(capture, grammar);
    }
    count
}

fn count_roles(node: &pesti_structural::tree::TreeNode, grammar: &pesti_structural::grammar::StructuralGrammar, out: &mut HashMap<String, usize>) {
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if let Some(ann) = grammar.annotations.get(&pesti_structural::grammar::Nonterminal(rule_name.to_string())) {
        *out.entry(ann.role.clone()).or_insert(0) += 1;
    }
    for child in &node.children {
        count_roles(child, grammar, out);
    }
    for capture in node.captures.values() {
        count_roles(capture, grammar, out);
    }
}

/// Test structural similarity: structurally similar functions should produce
/// closer feature sets than structurally different ones.
fn test_structural_similarity(grammar: &pesti_structural::grammar::StructuralGrammar) {
    // Pair 1: Two functions with identical structure (if + loop), different names/vars
    let func_a = r#"
fn process_items(items: Vec<String>) -> Vec<String> {
    let mut results = Vec::new();
    for item in items {
        if item.len() > 5 {
            results.push(item.to_uppercase());
        }
    }
    return results;
}"#;

    let func_b = r#"
fn filter_words(words: Vec<String>) -> Vec<String> {
    let mut output = Vec::new();
    for word in words {
        if word.len() > 3 {
            output.push(word.to_lowercase());
        }
    }
    return output;
}"#;

    // Pair 2: Function with very different structure (match + recursion)
    let func_c = r#"
fn count_nodes(node: &Option<Box<TreeNode>>) -> usize {
    match node {
        None => 0,
        Some(n) => 1 + count_nodes(&n.left) + count_nodes(&n.right),
    }
}"#;

    // Pair 3: Another if+loop function (similar to A and B)
    let func_d = r#"
fn transform_data(data: Vec<i32>) -> Vec<i32> {
    let mut transformed = Vec::new();
    for value in data {
        if value > 0 {
            transformed.push(value * 2);
        }
    }
    return transformed;
}"#;

    let features_a = extract_feature_sig(func_a, grammar);
    let features_b = extract_feature_sig(func_b, grammar);
    let features_c = extract_feature_sig(func_c, grammar);
    let features_d = extract_feature_sig(func_d, grammar);

    println!("Features A (if+loop): {:?}", features_a);
    println!("Features B (if+loop): {:?}", features_b);
    println!("Features C (match+recursion): {:?}", features_c);
    println!("Features D (if+loop): {:?}", features_d);

    let sim_ab = feature_similarity(&features_a, &features_b);
    let sim_ac = feature_similarity(&features_a, &features_c);
    let sim_ad = feature_similarity(&features_a, &features_d);
    let sim_bc = feature_similarity(&features_b, &features_c);
    let sim_bd = feature_similarity(&features_b, &features_d);
    let sim_cd = feature_similarity(&features_c, &features_d);

    println!("\nSimilarity scores (Jaccard on role+class sets):");
    println!("  A vs B (similar structure): {:.3}", sim_ab);
    println!("  A vs C (different structure): {:.3}", sim_ac);
    println!("  A vs D (similar structure): {:.3}", sim_ad);
    println!("  B vs C (different structure): {:.3}", sim_bc);
    println!("  B vs D (similar structure): {:.3}", sim_bd);
    println!("  C vs D (different structure): {:.3}", sim_cd);

    // The key assertion: similar structures should score higher than dissimilar ones.
    let similar_avg = (sim_ab + sim_ad + sim_bd) / 3.0;
    let dissimilar_avg = (sim_ac + sim_bc + sim_cd) / 3.0;

    println!("\nAvg similarity (same structure): {:.3}", similar_avg);
    println!("Avg similarity (diff structure):  {:.3}", dissimilar_avg);

    if similar_avg > dissimilar_avg {
        println!("\n✓ PASS: Structurally similar functions score closer ({:.3} > {:.3})", similar_avg, dissimilar_avg);
    } else {
        println!("\n✗ FAIL: Structural similarity not captured ({:.3} <= {:.3})", similar_avg, dissimilar_avg);
    }
}

/// Extract a feature signature from source code.
fn extract_feature_sig(source: &str, grammar: &pesti_structural::grammar::StructuralGrammar) -> FeatureSig {
    let (tokens, _) = tokenize_rust(source);
    if tokens.is_empty() {
        return vec![];
    }
    let evaluator = PegEvaluator::new(grammar, &tokens);
    let result = evaluator.parse_with_recovery();
    let mut sig = Vec::new();
    if let Some(tree) = &result.tree {
        collect_features(tree, grammar, &mut sig);
    }
    sig
}

fn collect_features(node: &pesti_structural::tree::TreeNode, grammar: &pesti_structural::grammar::StructuralGrammar, out: &mut FeatureSig) {
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if let Some(ann) = grammar.annotations.get(&pesti_structural::grammar::Nonterminal(rule_name.to_string())) {
        out.push((ann.role.clone(), ann.similarity_class.clone().unwrap_or_default()));
    }
    for child in &node.children {
        collect_features(child, grammar, out);
    }
    for capture in node.captures.values() {
        collect_features(capture, grammar, out);
    }
}

/// Jaccard similarity on feature signatures (multiset-aware).
fn feature_similarity(a: &FeatureSig, b: &FeatureSig) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Count occurrences of each feature in both sets.
    let mut counts_a: HashMap<&(String, String), usize> = HashMap::new();
    for f in a {
        *counts_a.entry(f).or_insert(0) += 1;
    }
    let mut counts_b: HashMap<&(String, String), usize> = HashMap::new();
    for f in b {
        *counts_b.entry(f).or_insert(0) += 1;
    }

    // Intersection: min count for each feature.
    let intersection: usize = counts_a.iter().map(|(k, v)| (*v).min(*counts_b.get(k).unwrap_or(&0))).sum();
    let union = a.len() + b.len() - intersection;

    if union == 0 {
        return 1.0;
    }
    intersection as f64 / union as f64
}
