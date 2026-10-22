//! Cross-language structural similarity prototype.
//!
//! Demonstrates comparing code structures across different languages using
//! universal structure tokens + embeddings. This shows that structurally
//! similar algorithms written in different languages can be identified as
//! similar despite syntactic differences.

use anyhow::Result;
use pesti_embeddings::Embedder;
use pesti_structural_tokenizer::StructuralTokenizer;

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

    // Embed each implementation
    println!("\n=== Cross-Language Structural Similarity ===");
    let impls = vec![
        ("Rust", rust_impl),
        ("Python", python_impl),
        ("Go", go_impl),
    ];

    let mut embeddings = Vec::new();
    for (name, code) in &impls {
        // For non-Rust languages, use text-only embedding since structural tokenizer is Rust-specific
        let text_emb = embedder.embed(code)?;
        embeddings.push((name.to_string(), text_emb));
    }

    // Compare pairwise similarities
    println!("\nPairwise similarity (text only):");
    for i in 0..embeddings.len() {
        for j in i + 1..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].1, &embeddings[j].1);
            println!("  {} vs {}: {:.4}", embeddings[i].0, embeddings[j].0, sim);
        }
    }

    // Now try with structural representation for Rust (baseline)
    println!("\n=== Structural Representation Comparison ===");
    let rust_tokens = tokenizer.tokenize(rust_impl)?;
    let rust_struct = format!("{:?}", rust_tokens.iter().map(|t| t.kind.clone()).collect::<Vec<_>>());
    println!("Rust structural tokens: {}", rust_struct);

    // For Python/Go, we'd need language adapters - for now just show Rust baseline
    println!("\nNote: Cross-language structural comparison requires LanguageAdapters");
    println!("for each target language. This prototype demonstrates the concept.");

    Ok(())
}