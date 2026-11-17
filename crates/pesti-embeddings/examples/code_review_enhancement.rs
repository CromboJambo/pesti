//! Code review enhancement demo using structural PEG embeddings.
//!
//! Demonstrates what our new PEG-based structural analysis adds over the
//! current code-audit-mcp (which uses the old flat tokenizer):
//!
//! 1. **Duplicate detection** — find structurally similar functions across files
//! 2. **Better complexity context** — complexity scores with structural breakdown
//! 3. **Accurate locations** — span-based line numbers for each finding
//!
//! Usage:
//!   cargo run -p pesti-embeddings --example code_review_enhancement <path> [--recursive]

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process;

use pesti_structural::evaluator::PegEvaluator;
use pesti_structural::grammar::{Nonterminal, StructuralGrammar};
use pesti_structural::lexer::tokenize_for_language;
use pesti_structural::parser::parse_peg_file;
use pesti_structural::tree::{Span, TreeNode};

/// A function-level profile extracted from the parse tree.
struct FunctionProfile {
    file: String,
    line_start: usize,
    line_end: usize,
    name: String,
    span: Span,
    feature_count: usize,
    roles: Vec<String>,
    classes: Vec<String>,
}

impl FunctionProfile {
    fn complexity_score(&self) -> usize {
        let role_diversity = self.roles.iter().map(|s| s.as_str()).collect::<HashSet<&str>>().len();
        let class_diversity = self.classes.iter().map(|s| s.as_str()).collect::<HashSet<&str>>().len();
        let impl_weight = self
            .roles
            .iter()
            .filter(|r| r.as_str() == "implementation" || r.as_str() == "expression")
            .count();
        role_diversity + class_diversity + impl_weight / 2
    }

    fn structural_similarity(&self, other: &FunctionProfile) -> f32 {
        let roles_a: HashSet<&str> = self.roles.iter().map(|s| s.as_str()).collect();
        let roles_b: HashSet<&str> = other.roles.iter().map(|s| s.as_str()).collect();
        let role_sim = jaccard(&roles_a, &roles_b);

        let classes_a: HashSet<&str> = self.classes.iter().map(|s| s.as_str()).collect();
        let classes_b: HashSet<&str> = other.classes.iter().map(|s| s.as_str()).collect();
        let class_sim = jaccard(&classes_a, &classes_b);

        0.5 * role_sim + 0.5 * class_sim
    }
}

fn jaccard(a: &HashSet<&str>, b: &HashSet<&str>) -> f32 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Convert a byte offset to a line number (1-indexed).
fn byte_offset_to_line(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())].matches('\n').count() + 1
}

/// Collect all Rust files in a directory.
fn collect_rust_files(path: &Path, recursive: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if path.is_file() && path.extension().map_or(false, |e| e == "rs") {
        files.push(path.to_path_buf());
        return files;
    }

    if !path.is_dir() {
        return files;
    }

    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return files,
    };

    for entry in entries.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if recursive && p.is_dir() {
            files.extend(collect_rust_files(&p, true));
        } else if p.extension().map_or(false, |e| e == "rs") {
            files.push(p);
        }
    }

    files.sort();
    files
}

/// Find all FuncDef nodes in the parse tree (pre-order traversal).
fn find_func_defs<'a>(node: &'a TreeNode, out: &mut Vec<&'a TreeNode>) {
    if node.label == "FuncDef" {
        out.push(node);
    }
    for child in &node.children {
        find_func_defs(child, out);
    }
}

/// Extract features from a subtree (roles/classes of annotated nodes).
fn extract_subtree_features(
    node: &TreeNode,
    grammar: &StructuralGrammar,
    roles: &mut Vec<String>,
    classes: &mut Vec<String>,
) {
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if let Some(ann) = grammar.annotations.get(&Nonterminal(rule_name.to_string())) {
        roles.push(ann.role.clone());
        if let Some(class) = &ann.similarity_class {
            classes.push(class.clone());
        }
    }

    for child in &node.children {
        extract_subtree_features(child, grammar, roles, classes);
    }
    for capture in node.captures.values() {
        extract_subtree_features(capture, grammar, roles, classes);
    }
}

/// Extract function profiles from a source file using span-based grouping.
fn extract_function_profiles(
    grammar: &StructuralGrammar,
    source: &str,
    file: &str,
) -> Vec<FunctionProfile> {
    let (tokens, _) = tokenize_for_language(source, "rust");
    if tokens.is_empty() {
        return Vec::new();
    }

    let evaluator = PegEvaluator::new(grammar, &tokens);
    let result = evaluator.parse_with_recovery();

    let tree = match result.tree {
        Some(t) => t,
        None => return Vec::new(),
    };

    // Find all FuncDef nodes.
    let mut func_nodes: Vec<&TreeNode> = Vec::new();
    find_func_defs(&tree, &mut func_nodes);

    let mut profiles = Vec::new();

    for func_node in func_nodes {
        // Extract the function name from captures.
        let name = if let Some(capture) = func_node.captures.get("name") {
            capture.span.text(source).to_string()
        } else {
            "<unnamed>".to_string()
        };

        // Extract features from this function's subtree.
        let mut roles = Vec::new();
        let mut classes = Vec::new();
        extract_subtree_features(func_node, grammar, &mut roles, &mut classes);

        let line_start = byte_offset_to_line(source, func_node.span.start);
        let line_end = byte_offset_to_line(source, func_node.span.end.min(source.len()));

        profiles.push(FunctionProfile {
            file: file.to_string(),
            line_start,
            line_end,
            name,
            span: func_node.span,
            feature_count: roles.len() + classes.len(),
            roles,
            classes,
        });
    }

    profiles
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path> [--recursive]", args[0]);
        process::exit(1);
    }

    let path = Path::new(&args[1]);
    let recursive = args.iter().any(|a| a == "--recursive");

    // Resolve grammar path.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let grammar_path = workspace_root
        .join("crates")
        .join("pesti-structural")
        .join("grammars")
        .join("rust.peg");

    if !grammar_path.exists() {
        eprintln!("Grammar not found at {}", grammar_path.display());
        process::exit(1);
    }

    let grammar = match parse_peg_file(&grammar_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to parse grammar: {}", e);
            process::exit(1);
        }
    };

    // Collect files.
    let files = collect_rust_files(path, recursive);
    if files.is_empty() {
        eprintln!("No Rust files found in {}", path.display());
        process::exit(1);
    }

    println!("=== Structural Code Review Enhancement ===");
    println!("Scanning {} files...\n", files.len());

    // Extract profiles from all files.
    let mut all_profiles: Vec<FunctionProfile> = Vec::new();

    for file in &files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let rel_path = file
            .strip_prefix(workspace_root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        let profiles = extract_function_profiles(&grammar, &content, &rel_path);
        all_profiles.extend(profiles);
    }

    println!(
        "Extracted {} function profiles from {} files\n",
        all_profiles.len(),
        files.len()
    );

    // === Analysis 1: High structural complexity ===
    println!("=== HIGH STRUCTURAL COMPLEXITY (>8) ===");
    let mut complex: Vec<&FunctionProfile> = all_profiles
        .iter()
        .filter(|p| p.complexity_score() > 8)
        .collect();
    complex.sort_by(|a, b| b.complexity_score().cmp(&a.complexity_score()));

    for (i, p) in complex.iter().take(10).enumerate() {
        println!(
            "  {}. {}:{}-{} ({}) — score={}, roles={}",
            i + 1,
            p.file,
            p.line_start,
            p.line_end,
            p.name,
            p.complexity_score(),
            p.roles.iter().map(|r| r.as_str()).collect::<HashSet<_>>().len()
        );
    }

    // === Analysis 2: Structural duplicates (similarity > 0.85) ===
    println!("\n=== STRUCTURAL DUPLICATES (similarity > 0.85, different files) ===");
    let mut duplicate_pairs: Vec<(usize, usize, f32)> = Vec::new();

    // Compare profiles across files only, with a window to keep it fast.
    for i in 0..all_profiles.len() {
        for j in (i + 1)..all_profiles.len().min(i + 100) {
            let pi = &all_profiles[i];
            let pj = &all_profiles[j];
            if pi.file == pj.file {
                continue;
            }
            // Only compare functions with similar feature counts.
            if pi.feature_count.abs_diff(pj.feature_count) > 5 {
                continue;
            }
            let sim = pi.structural_similarity(pj);
            if sim > 0.85 {
                duplicate_pairs.push((i, j, sim));
            }
        }
    }

    duplicate_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (i, (a, b, sim)) in duplicate_pairs.iter().take(5).enumerate() {
        let pa = &all_profiles[*a];
        let pb = &all_profiles[*b];
        println!("  {}. similarity={:.2}", i + 1, sim);
        println!("     A: {}:{} ({})", pa.file, pa.line_start, pa.name);
        println!("     B: {}:{} ({})", pb.file, pb.line_start, pb.name);
    }

    if duplicate_pairs.is_empty() {
        println!("  (no cross-file structural duplicates found)");
    }

    // === Analysis 3: Feature class distribution ===
    println!("\n=== FEATURE CLASS DISTRIBUTION ===");
    let mut class_counts: BTreeMap<String, usize> = BTreeMap::new();
    for p in &all_profiles {
        for c in &p.classes {
            *class_counts.entry(c.clone()).or_insert(0) += 1;
        }
    }

    for (class, count) in class_counts.iter().take(10) {
        println!("  {:<20} {}", class, count);
    }

    // === Summary ===
    println!("\n=== SUMMARY ===");
    println!("Files scanned: {}", files.len());
    println!("Function profiles: {}", all_profiles.len());
    println!("High complexity (>8): {}", complex.len());
    println!("Cross-file duplicates: {}", duplicate_pairs.len());
    println!("\nComparison with current code-audit-mcp:");
    println!("  - Current: flat token counts, no structural context");
    println!(
        "  - This: role/class-based profiles, cross-file similarity, accurate line numbers"
    );
}
