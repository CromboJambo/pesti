//! Structural parsing demo: Rust source → tokens → annotated tree → embedding features.
//!
//! Usage: cargo run -p pesti-structural --example structural_demo [file.rs]
//! If no file given, parses a built-in sample.

use pesti_structural::tree::TreeNode;
use pesti_structural::{Nonterminal, PegEvaluator, parse_peg, tokenize_rust, validate};
use std::collections::BTreeMap;
use std::fs;

fn main() {
    let source = if let Some(path) = std::env::args().nth(1) {
        fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", path, e);
            std::process::exit(1);
        })
    } else {
        // Built-in sample: structurally interesting Rust code.
        // Exercises: use, struct, impl with methods, main with calls and let bindings.
        r#"use std::fmt;

pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

fn main() {
    let a = Point::new(1.0, 2.0);
    println!("{}", a);
}"#
        .to_string()
    };

    // Load and validate the grammar.
    let grammar_src = include_str!("../grammars/rust.peg");
    let grammar = parse_peg(grammar_src).expect("rust.peg should parse");
    let result = validate(&grammar);
    if !result.valid {
        eprintln!("Grammar validation failed: {:?}", result.errors);
        std::process::exit(1);
    }

    // Tokenize.
    let (tokens, lex_errors) = tokenize_rust(&source);
    if !lex_errors.is_empty() {
        for err in &lex_errors {
            eprintln!("Lex error at {:?}: {}", err.span, err.message);
        }
    }

    // Parse.
    let evaluator = PegEvaluator::new(&grammar, &tokens);
    let parse_result = evaluator.parse();

    if !parse_result.errors.is_empty() {
        eprintln!("Parse failed:");
        for err in &parse_result.errors {
            eprintln!("  - {}", err);
        }
        std::process::exit(1);
    }

    let tree = match parse_result.tree {
        Some(t) => t,
        None => {
            eprintln!("Parse produced no tree");
            std::process::exit(1);
        }
    };

    println!("=== Parse succeeded ===");
    println!("Tree depth: {}, nodes: {}", tree.depth(), tree.node_count());
    println!("\n=== Annotated Tree (first 40 lines) ===\n");
    print_tree_limited(&tree, 0, &source, &mut 40);
    println!();

    // Extract embedding features from annotations.
    let features = extract_embedding_features(&tree, &grammar.annotations);
    println!("=== Embedding Features ({}) ===", features.len());
    for (i, f) in features.iter().enumerate() {
        println!("[{:>3}] {:<14} {:<14} {}", i + 1, f.role, f.class, f.label);
    }

    // Show similarity groups (nodes that should embed near each other).
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in features.iter().enumerate() {
        groups.entry(f.class.as_str()).or_default().push(i + 1);
    }
    println!("\n=== Similarity Groups ===");
    for (class, members) in &groups {
        let ids: Vec<String> = members.iter().map(|m| m.to_string()).collect();
        println!("  {:<14} → [{}]", class, ids.join(", "));
    }

    // Show how this differs from flat tokenization.
    println!("\n=== Structural Advantage ===");
    println!("Flat tokens: {} (no hierarchy)", tokens.len());
    println!(
        "Structural nodes: {} (with roles, captures, spans)",
        tree.node_count()
    );
    println!(
        "Annotated features: {} (embedding-ready metadata)",
        features.len()
    );
}

fn print_tree_limited(node: &TreeNode, indent: usize, source: &str, budget: &mut usize) {
    if *budget <= 0 {
        return;
    }
    *budget -= 1;

    let pad = "  ".repeat(indent);
    let text = node.span.text(source);
    let display_text = truncate_char(&text, 40);

    // Show captures inline.
    let capture_info: Vec<String> = node
        .captures
        .iter()
        .map(|(name, child)| {
            let cap_text = truncate_char(child.span.text(source), 20);
            format!("@{}={}", name, cap_text)
        })
        .collect();

    if capture_info.is_empty() {
        println!("{}{} ({})", pad, node.label, display_text);
    } else {
        println!(
            "{}{} ({}) [{}]",
            pad,
            node.label,
            display_text,
            capture_info.join(", ")
        );
    }

    for child in &node.children {
        print_tree_limited(child, indent + 1, source, budget);
    }
    // Also show captures as part of the tree.
    for (name, capture) in &node.captures {
        if *budget <= 0 {
            break;
        }
        let pad = "  ".repeat(indent + 1);
        println!("{}@{}", pad, name);
        print_tree_limited(capture, indent + 2, source, budget);
    }
}

fn truncate_char(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let cutoff = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    format!("{}...", &s[..cutoff])
}

struct EmbeddingFeature {
    role: String,
    class: String,
    label: String,
}

fn extract_embedding_features(
    tree: &TreeNode,
    annotations: &BTreeMap<pesti_structural::Nonterminal, pesti_structural::RuleAnnotation>,
) -> Vec<EmbeddingFeature> {
    let mut features = Vec::new();
    walk_for_features(tree, annotations, &mut features);
    features
}

fn walk_for_features(
    node: &TreeNode,
    annotations: &BTreeMap<pesti_structural::Nonterminal, pesti_structural::RuleAnnotation>,
    out: &mut Vec<EmbeddingFeature>,
) {
    if let Some(ann) = annotations.get(&Nonterminal(node.label.clone())) {
        let label = extract_label(node, &ann.embed_fields);
        out.push(EmbeddingFeature {
            role: ann.role.clone(),
            class: ann.similarity_class.clone().unwrap_or_default(),
            label,
        });
    }
    for child in &node.children {
        walk_for_features(child, annotations, out);
    }
    // Also traverse named captures — the evaluator stores matched sub-trees there.
    for capture in node.captures.values() {
        walk_for_features(capture, annotations, out);
    }
}

fn extract_label(node: &TreeNode, fields: &[String]) -> String {
    let mut parts = Vec::new();
    for field in fields {
        if let Some(text) = find_capture_text(node, field) {
            if !text.is_empty() {
                parts.push(text);
            }
        }
    }
    if parts.is_empty() {
        "<no captures>".to_string()
    } else {
        parts.join(" ")
    }
}

/// Recursively search a node and its descendants for a named capture,
/// returning the text of the first match found.
fn find_capture_text(node: &TreeNode, name: &str) -> Option<String> {
    // Check this node's captures directly.
    if let Some(captured) = node.captures.get(name) {
        return Some(collect_text(captured));
    }
    // Search children.
    for child in &node.children {
        if let Some(text) = find_capture_text(child, name) {
            return Some(text);
        }
    }
    // Search nested captures.
    for capture in node.captures.values() {
        if let Some(text) = find_capture_text(capture, name) {
            return Some(text);
        }
    }
    None
}

fn collect_text(node: &TreeNode) -> String {
    let mut parts = Vec::new();
    for child in &node.children {
        if child.children.is_empty() {
            parts.push(child.label.clone());
        } else {
            parts.push(collect_text(child));
        }
    }
    parts.join(" ")
}
