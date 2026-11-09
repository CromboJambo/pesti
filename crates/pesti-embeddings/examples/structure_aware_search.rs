//! Structure-aware code search prototype.
//!
//! Demonstrates using pesti-structural-tokenizer + pesti-embeddings to find
//! structurally similar code patterns across a codebase.

use anyhow::Result;
use pesti_embeddings::Embedder;
use pesti_structural_tokenizer::StructuralTokenizer;

/// Create structural representation of code.
fn structural_repr(tokens: &[pesti_structural_tokenizer::StructuralToken]) -> String {
    tokens
        .iter()
        .map(|t| format!("{:?}", t.kind))
        .collect::<Vec<_>>()
        .join(" ")
}

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

fn main() -> Result<()> {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/crombo/projects/active/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    println!("Loading embedding model...");
    let embedder = Embedder::load(&model_path)?;
    let tokenizer = StructuralTokenizer::new();

    // Query: find functions with similar structure to this one
    let query_code = r#"fn calculate(a: i32, b: i32) -> i32 {
    return a + b;
}"#;

    // Codebase snippets to search through
    let snippets = vec![
        ("simple_add", r#"fn add(x: i32, y: i32) -> i32 {
    return x + y;
}"#),
        ("multiply", r#"fn multiply(a: i32, b: i32) -> i32 {
    return a * b;
}"#),
        ("loop_process", r#"fn process(items: Vec<i32>) {
    for item in items {
        println!("{}", item);
    }
}"#),
        ("conditional", r#"fn check(n: i32) -> bool {
    if n > 0 {
        return true;
    }
    return false;
}"#),
        ("nested_loop", r#"fn matrix_multiply(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut result = vec![vec![0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}"#),
    ];

    // Embed query
    println!("\nQuery: find structurally similar functions");
    let query_tokens = tokenizer.tokenize(query_code)?;
    let query_struct = structural_repr(&query_tokens);
    let query_text_emb = embedder.embed(query_code)?;
    let query_struct_emb = embedder.embed(&query_struct)?;
    let query_combined = combine_embeddings(&query_text_emb, &query_struct_emb, 0.5, 0.5);

    println!("Query structure: {}", query_struct);

    // Search through snippets
    println!("\nSearch results:");
    let mut results = Vec::new();
    for (name, code) in &snippets {
        let tokens = tokenizer.tokenize(code)?;
        let struct_repr = structural_repr(&tokens);
        let text_emb = embedder.embed(code)?;
        let struct_emb = embedder.embed(&struct_repr)?;
        let combined = combine_embeddings(&text_emb, &struct_emb, 0.5, 0.5);

        let similarity = cosine_similarity(&query_combined, &combined);
        results.push((name.to_string(), similarity, struct_repr));
    }

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, similarity, structure) in &results {
        println!("  {} - {:.4} ({})", name, similarity, structure);
    }

    // Identify structurally similar matches
    println!("\nStructurally similar matches (>0.95):");
    for (name, similarity, _) in &results {
        if *similarity > 0.95 {
            println!("  ✓ {}", name);
        }
    }

    Ok(())
}