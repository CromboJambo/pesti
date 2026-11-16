//! Structure-aware code embeddings prototype.
//!
//! Demonstrates combining pesti-structural (PEG grammar parser) with
//! pesti-embeddings to create embeddings that capture both semantic meaning
//! and syntactic structure.
//!
//! Three approaches compared:
//! 1. Baseline: raw text embeddings only
//! 2. Legacy structural: pesti-structural-tokenizer kind sequence
//! 3. PEG-based: pesti-structural annotated parse tree features

use anyhow::Result;
use pesti_embeddings::{Embedder, StructuralConfig, StructuralEmbedder};
use pesti_structural_tokenizer::StructuralTokenizer;
use std::path::Path;

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
        combined.push(struct_emb[i].mul_add(struct_weight, text_emb[i] * text_weight));
    }
    // Normalize
    let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut combined {
            *v /= norm;
        }
    }
    combined
}

/// Create a structural representation of code as text for embedding (legacy).
fn structural_repr(tokens: &[pesti_structural_tokenizer::StructuralToken]) -> String {
    tokens
        .iter()
        .map(|t| format!("{:?}", t.kind))
        .collect::<Vec<_>>()
        .join(" ")
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

    // --- Approach 1: Baseline (raw text only) ---
    println!("\n=== Approach 1: Raw Text Embeddings ===");
    let emb_a_raw = embedder.embed(code_a)?;
    let emb_b_raw = embedder.embed(code_b)?;
    let emb_c_raw = embedder.embed(code_c)?;

    let sim_ab_raw = cosine_similarity(&emb_a_raw, &emb_b_raw);
    let sim_ac_raw = cosine_similarity(&emb_a_raw, &emb_c_raw);

    println!("Similarity(add vs multiply): {:.4}", sim_ab_raw);
    println!("Similarity(add vs loop):     {:.4}", sim_ac_raw);

    // --- Approach 2: Legacy structural (token kind sequence) ---
    println!("\n=== Approach 2: Legacy Structural Tokens ===");
    let tokens_a = tokenizer.tokenize(code_a)?;
    let tokens_b = tokenizer.tokenize(code_b)?;
    let tokens_c = tokenizer.tokenize(code_c)?;

    let struct_a = structural_repr(&tokens_a);
    let struct_b = structural_repr(&tokens_b);
    let struct_c = structural_repr(&tokens_c);

    println!("Structural repr A: {}", struct_a);
    println!("Structural repr B: {}", struct_b);
    println!("Structural repr C: {}", struct_c);

    let emb_a_struct = embedder.embed(&struct_a)?;
    let combined_a_legacy = combine_embeddings(&emb_a_raw, &emb_a_struct, 0.5, 0.5);

    let emb_b_struct = embedder.embed(&struct_b)?;
    let combined_b_legacy = combine_embeddings(&emb_b_raw, &emb_b_struct, 0.5, 0.5);

    let emb_c_struct = embedder.embed(&struct_c)?;
    let combined_c_legacy = combine_embeddings(&emb_c_raw, &emb_c_struct, 0.5, 0.5);

    let sim_ab_legacy = cosine_similarity(&combined_a_legacy, &combined_b_legacy);
    let sim_ac_legacy = cosine_similarity(&combined_a_legacy, &combined_c_legacy);

    println!("Similarity(add vs multiply): {:.4}", sim_ab_legacy);
    println!("Similarity(add vs loop):     {:.4}", sim_ac_legacy);

    // --- Approach 3: PEG-based structural features (pesti-structural) ---
    println!("\n=== Approach 3: PEG Grammar Features ===");
    let grammar_path = "crates/pesti-structural/grammars/rust.peg";
    if !Path::new(grammar_path).exists() {
        eprintln!(
            "Warning: grammar file not found at {}, skipping PEG-based approach",
            grammar_path
        );
    } else {
        let mut peg_embedder = StructuralEmbedder::new(StructuralConfig {
            grammar_path: grammar_path.to_string(),
            text_weight: 0.5,
            struct_weight: 0.5,
        });

        // Extract features for each code sample
        let features_a = peg_embedder.extract_features(code_a)?;
        let features_b = peg_embedder.extract_features(code_b)?;
        let features_c = peg_embedder.extract_features(code_c)?;

        println!("Features extracted:");
        println!("  Code A (add):       {} features", features_a.len());
        println!("  Code B (multiply):  {} features", features_b.len());
        println!("  Code C (loop):      {} features", features_c.len());

        // Show feature details for code A
        if !features_a.is_empty() {
            println!("\nSample features from code A:");
            for (i, f) in features_a.iter().take(5).enumerate() {
                println!(
                    "  [{}] role={} class={} label={}",
                    i + 1,
                    f.role,
                    f.class,
                    f.label
                );
            }
        }

        // Format structural representations
        let struct_repr_a = StructuralEmbedder::format_structural_repr(&features_a);
        let struct_repr_b = StructuralEmbedder::format_structural_repr(&features_b);
        let struct_repr_c = StructuralEmbedder::format_structural_repr(&features_c);

        println!("\nStructural repr A:\n{}", struct_repr_a);
        println!("\nStructural repr B:\n{}", struct_repr_b);
        println!("\nStructural repr C:\n{}", struct_repr_c);

        // Embed and combine
        let emb_a_peg = embedder.embed(&struct_repr_a)?;
        let combined_a_peg = combine_embeddings(&emb_a_raw, &emb_a_peg, 0.5, 0.5);

        let emb_b_peg = embedder.embed(&struct_repr_b)?;
        let combined_b_peg = combine_embeddings(&emb_b_raw, &emb_b_peg, 0.5, 0.5);

        let emb_c_peg = embedder.embed(&struct_repr_c)?;
        let combined_c_peg = combine_embeddings(&emb_c_raw, &emb_c_peg, 0.5, 0.5);

        let sim_ab_peg = cosine_similarity(&combined_a_peg, &combined_b_peg);
        let sim_ac_peg = cosine_similarity(&combined_a_peg, &combined_c_peg);

        println!("Similarity(add vs multiply): {:.4}", sim_ab_peg);
        println!("Similarity(add vs loop):     {:.4}", sim_ac_peg);

        // Show similarity groups
        let groups = StructuralEmbedder::similarity_groups(&features_a);
        if !groups.is_empty() {
            println!("\nSimilarity groups (code A):");
            for (class, indices) in &groups {
                println!("  {}: features {:?}", class, indices);
            }
        }

        // --- Final comparison ---
        println!("\n=== Comparison ===");
        println!("{:<30} {:>12} {:>12}", "Approach", "sim(A,B)", "sim(A,C)");
        println!(
            "{:<30} {:>12.4} {:>12.4}",
            "Raw text", sim_ab_raw, sim_ac_raw
        );
        println!(
            "{:<30} {:>12.4} {:>12.4}",
            "Legacy structural tokens", sim_ab_legacy, sim_ac_legacy
        );
        println!(
            "{:<30} {:>12.4} {:>12.4}",
            "PEG grammar features", sim_ab_peg, sim_ac_peg
        );

        // Analyze which approach best separates structurally similar vs different code
        let separation_raw = sim_ab_raw - sim_ac_raw;
        let separation_legacy = sim_ab_legacy - sim_ac_legacy;
        let separation_peg = sim_ab_peg - sim_ac_peg;

        println!("\nSeparation (sim(A,B) - sim(A,C)):");
        println!("  Raw text:       {:.4}", separation_raw);
        println!("  Legacy tokens:  {:.4}", separation_legacy);
        println!("  PEG features:   {:.4}", separation_peg);

        let approaches = [
            ("Raw text", separation_raw),
            ("Legacy tokens", separation_legacy),
            ("PEG features", separation_peg),
        ];
        let best = approaches
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("\n✓ Best structural separation: {} ({:.4})", best.0, best.1);
    }

    Ok(())
}
