//! Structure-aware code embeddings prototype.
//!
//! Demonstrates combining pesti-structural-tokenizer with pesti-embeddings
//! to create embeddings that capture both semantic meaning and syntactic structure.
//!
//! Approach: Embed text and structural representation separately, then combine
//! in vector space with learnable weights.

use anyhow::{anyhow, Result};
use pesti_embeddings::Embedder;
use pesti_structural_tokenizer::StructuralTokenizer;
use std::path::Path;

/// Create a structural representation of code as text for embedding.
fn structural_repr(tokens: &[pesti_structural_tokenizer::StructuralToken]) -> String {
    tokens
        .iter()
        .map(|t| format!("{:?}", t.kind))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute similarity between two embeddings (cosine).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Combine two embeddings with weighted average.
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
    // Normalize
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

    println!("Loading embedding model from: {}", model_path);
    let embedder = Embedder::load(&model_path)?;
    let tokenizer = StructuralTokenizer::new();

    // Test cases: structurally similar vs dissimilar code
    let code_a = r#"fn add(a: i32, b: i32) -> i32 {
    return a + b;
}"#;

    let code_b = r#"fn multiply(x: i32, y: i32) -> i32 {
    return x * y;
}"#;

    let code_c = r#"fn process(items: Vec<i32>) {
    for item in items {
        println!("{}", item);
    }
}"#;

    // Embed raw text only (baseline)
    println!("\n=== Baseline: Raw Text Embeddings ===");
    let emb_a_raw = embedder.embed(code_a)?;
    let emb_b_raw = embedder.embed(code_b)?;
    let emb_c_raw = embedder.embed(code_c)?;

    let sim_ab_raw = cosine_similarity(&emb_a_raw, &emb_b_raw);
    let sim_ac_raw = cosine_similarity(&emb_a_raw, &emb_c_raw);

    println!("Similarity(add vs multiply): {:.4}", sim_ab_raw);
    println!("Similarity(add vs loop):     {:.4}", sim_ac_raw);

    // Embed with structural representation (separate channels)
    println!("\n=== Structure-Aware Embeddings ===");
    let tokens_a = tokenizer.tokenize(code_a)?;
    let tokens_b = tokenizer.tokenize(code_b)?;
    let tokens_c = tokenizer.tokenize(code_c)?;

    let struct_a = structural_repr(&tokens_a);
    let struct_b = structural_repr(&tokens_b);
    let struct_c = structural_repr(&tokens_c);

    println!("Structural repr A: {}", struct_a);
    println!("Structural repr B: {}", struct_b);
    println!("Structural repr C: {}", struct_c);

    // Embed text and structure separately, then combine
    let emb_a_text = embedder.embed(code_a)?;
    let emb_a_struct = embedder.embed(&struct_a)?;
    let combined_a = combine_embeddings(&emb_a_text, &emb_a_struct, 0.5, 0.5);

    let emb_b_text = embedder.embed(code_b)?;
    let emb_b_struct = embedder.embed(&struct_b)?;
    let combined_b = combine_embeddings(&emb_b_text, &emb_b_struct, 0.5, 0.5);

    let emb_c_text = embedder.embed(code_c)?;
    let emb_c_struct = embedder.embed(&struct_c)?;
    let combined_c = combine_embeddings(&emb_c_text, &emb_c_struct, 0.5, 0.5);

    let sim_ab_combined = cosine_similarity(&combined_a, &combined_b);
    let sim_ac_combined = cosine_similarity(&combined_a, &combined_c);

    println!("Similarity(add vs multiply): {:.4}", sim_ab_combined);
    println!("Similarity(add vs loop):     {:.4}", sim_ac_combined);

    // Analyze improvement
    println!("\n=== Analysis ===");
    let delta_ab = sim_ab_combined - sim_ab_raw;
    let delta_ac = sim_ac_combined - sim_ac_raw;

    println!(
        "Delta(ab): {:.4} (positive = more similar, as expected for same structure)",
        delta_ab
    );
    println!(
        "Delta(ac): {:.4} (negative = less similar, as expected for different structure)",
        delta_ac
    );

    if delta_ab > 0.0 && delta_ab > delta_ac {
        println!("✓ Structure-aware embeddings successfully capture syntactic similarity!");
        println!("  Structurally similar code (A-B) became {} more similar", delta_ab);
        println!("  Structurally different code (A-C) became only {} more similar", delta_ac);
    } else {
        println!("✗ No clear improvement from structural representation.");
    }

    Ok(())
}