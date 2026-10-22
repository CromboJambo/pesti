//! Structure-aware code embeddings with CLI integration.
//!
//! Demonstrates how structure-aware embeddings could be integrated into the
//! pesti-embeddings CLI with a --structure-aware flag and language adapter support.

use anyhow::Result;
use pesti_embeddings::Embedder;
use universal_structure::{LanguageAdapter, RustAdapter};

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

/// Scale a vector by a scalar.
fn scale_vector(v: &[f32], scalar: f32) -> Vec<f32> {
    v.iter().map(|x| x * scalar).collect()
}

/// Element-wise add two vectors.
fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Create structure-aware embedding with language adapter support.
async fn create_structure_aware_embedding(
    embedder: &Embedder,
    source_code: &str,
    adapter: Box<dyn LanguageAdapter>,
) -> Result<Vec<f32>> {
    // Embed the original source code
    let text_embedding: Vec<f32> = embedder.embed(source_code)?;

    // Tokenize using the language-specific adapter and adapt to universal representation
    let span_tree = adapter.tokenize_and_adapt(source_code);
    let structural_repr = format!("{} spans", span_tree.spans.len());

    // Embed the structural representation
    let structural_embedding: Vec<f32> = embedder.embed(&structural_repr)?;

    // Combine with equal weights (50/50)
    let text_part = scale_vector(&text_embedding, 0.5);
    let struct_part = scale_vector(&structural_embedding, 0.5);
    Ok(add_vectors(&text_part, &struct_part))
}

fn main() -> Result<()> {
    println!("=== Structure-Aware Embeddings CLI Integration Demo ===");
    println!("\nThis example demonstrates how structure-aware embeddings could be integrated into the pesti-embeddings CLI.");
    println!("\nProposed CLI usage:");
    println!("  pesti-embeddings embed --structure-aware --language rust <<EOF");
    println!("fn add(a: i32, b: i32) -> i32 {{ a + b }}");
    println!("EOF");
    println!("\nOr for similarity comparison:");
    println!("  pesti-embeddings similarity --structure-aware --language rust file1.rs file2.rs");

    Ok(())
}