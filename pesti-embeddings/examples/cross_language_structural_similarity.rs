//! Cross-language structural similarity using universal structure tokens + embeddings.
//!
//! Demonstrates comparing code structures across different languages using
//! LanguageAdapters from the universal-structure crate combined with pesti-embeddings.

use anyhow::Result;
use pesti_embeddings::Embedder;
use universal_structure::{LanguageAdapter, SpanKind};
use universal_structure::adapters::{RustAdapter, PythonAdapter, GoAdapter};

/// Compute cosine similarity.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Combine embeddings with weighted average.
fn combine_embeddings(
    text_emb: &[f32],
    struct_emb: &[f32],
    text_weight: f32,
    struct_weight: f32,
) -> Vec<f32> {
    let mut combined = Vec::with_capacity(text_emb.len());
    for i in 0..text_emb.len() {
        combined.push(text_emb[i] * text_weight + struct_emb[i] * struct_weight);
    }
    let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for i in 0..combined.len() {
            combined[i] /= norm;
        }
    }
    combined
}

/// Convert SpanTree to structural representation string.
fn tree_to_structural_repr(tree: &universal_structure::SpanTree) -> String {
    tree.spans
        .iter()
        .map(|span| format!("{:?}", span.kind))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/crombo/projects/active/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    println!("Loading embedding model...");
    let embedder = Embedder::load(&model_path)?;

    // Equivalent algorithms in different languages
    let rust_impl = r#"fn fibonacci(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}"#;

    let python_impl = r#"def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"#;

    let go_impl = r#"func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    a, b := 0, 1
    for i := 2; i <= n; i++ {
        a, b = b, a + b
    }
    return b
}"#;

    // Analyze each implementation with LanguageAdapters
    println!("\n=== Cross-Language Structural Similarity ===");

    let rust_adapter = RustAdapter::new();
    let python_adapter = PythonAdapter::new();
    let go_adapter = GoAdapter::new();

    let rust_tree = rust_adapter.tokenize_and_adapt(rust_impl);
    let python_tree = python_adapter.tokenize_and_adapt(python_impl);
    let go_tree = go_adapter.tokenize_and_adapt(go_impl);

    // Print structural representations
    println!("\nStructural representations:");
    println!("Rust:   {}", tree_to_structural_repr(&rust_tree));
    println!("Python: {}", tree_to_structural_repr(&python_tree));
    println!("Go:     {}", tree_to_structural_repr(&go_tree));

    // Embed each implementation (text + structure)
    let impls = vec![
        ("Rust", rust_impl, &rust_tree),
        ("Python", python_impl, &python_tree),
        ("Go", go_impl, &go_tree),
    ];

    let mut embeddings = Vec::new();
    for (name, code, tree) in &impls {
        let text_emb = embedder.embed(code)?;
        let struct_repr = tree_to_structural_repr(tree);
        let struct_emb = embedder.embed(&struct_repr)?;
        let combined = combine_embeddings(&text_emb, &struct_emb, 0.5, 0.5);
        embeddings.push((name.to_string(), combined));
    }

    // Compare pairwise similarities
    println!("\nPairwise similarity (structure-aware):");
    for i in 0..embeddings.len() {
        for j in i + 1..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            println!("  {} vs {}: {:.4}", embeddings[i].0, embeddings[j].0, sim);
        }
    }

    // Compare with text-only baseline
    println!("\nPairwise similarity (text only):");
    let text_embeddings = vec![
        ("Rust", embedder.embed(rust_impl)?),
        ("Python", embedder.embed(python_impl)?),
        ("Go", embedder.embed(go_impl)?),
    ];

    for i in 0..text_embeddings.len() {
        for j in i + 1..text_embeddings.len() {
            let sim = cosine_similarity(&text_embeddings[i].1, &text_embeddings[j].1);
            println!("  {} vs {}: {:.4}", text_embeddings[i].0, text_embeddings[j].0, sim);
        }
    }

    Ok(())
}