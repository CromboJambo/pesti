//! Real GGUF model comparison using structure-aware embeddings.
//!
//! Loads actual GGUF model files, extracts architectural metadata, embeds the
//! descriptions, and compares models based on structural similarity.

use anyhow::{anyhow, Result};
use pesti_embeddings::Embedder;
use pesti_gguf::parse_gguf;
use std::fs;
use std::path::Path;

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Extract model architecture description from GGUF metadata.
fn extract_model_description(model_path: &str) -> Result<String> {
    let header = parse_gguf(Path::new(model_path))?;

    // Build architectural description using header methods
    let mut desc = format!("GGUF Model");
    if let Some(arch) = header.architecture() {
        desc.push_str(&format!(" arch={}", arch));
    }
    if let Some(n_layers) = header.block_count() {
        desc.push_str(&format!(" n_layers={}", n_layers));
    }
    if let Some(n_embd) = header.embedding_length() {
        desc.push_str(&format!(" n_embd={}", n_embd));
    }

    Ok(desc)
}

/// Find GGUF models on the system.
fn find_gguf_models() -> Vec<String> {
    let mut found = Vec::new();

    // Brave Local AI Models
    let brave_dir = "/home/crombo/.config/BraveSoftware/Brave-Browser-Beta/BraveLocalAIModels";
    if let Ok(entries) = fs::read_dir(brave_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Look for model.gguf inside subdirectories (recursive search)
                fn find_models_in_dir(path: &Path, results: &mut Vec<String>) {
                    if let Ok(entries) = fs::read_dir(path) {
                        for entry in entries.flatten() {
                            let entry_path = entry.path();
                            if entry_path.is_dir() {
                                find_models_in_dir(&entry_path, results);
                            } else if entry_path.file_name().unwrap().to_string_lossy().ends_with(".gguf") {
                                results.push(entry_path.to_string_lossy().to_string());
                            }
                        }
                    }
                }
                let mut brave_models = Vec::new();
                find_models_in_dir(&path, &mut brave_models);
                found.extend(brave_models);
            }
        }
    }

    // HuggingFace cache
    let hf_dir = "/home/crombo/.cache/huggingface/hub";
    if let Ok(entries) = fs::read_dir(hf_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Look for GGUF files in snapshots
                if let Ok(sub_entries) = fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.is_dir() && sub_path.file_name().unwrap().to_string_lossy().starts_with("snapshots") {
                            if let Ok(snapshot_entries) = fs::read_dir(&sub_path) {
                                for snap_entry in snapshot_entries.flatten() {
                                    let snap_path = snap_entry.path();
                                    if snap_path.is_dir() {
                                        // Look for .gguf files
                                        if let Ok(files) = fs::read_dir(&snap_path) {
                                            for file in files.flatten() {
                                                let file_name = file.file_name().to_string_lossy().to_string();
                                                if file_name.ends_with(".gguf") {
                                                    found.push(file.path().to_string_lossy().to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Ollama models
    let ollama_dir = "/home/crombo/.local/share/ollama/models";
    if let Ok(entries) = fs::read_dir(ollama_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Look for GGUF files recursively
                fn find_gguf_in_dir(path: &Path, results: &mut Vec<String>) {
                    if let Ok(entries) = fs::read_dir(path) {
                        for entry in entries.flatten() {
                            let entry_path = entry.path();
                            if entry_path.is_dir() {
                                find_gguf_in_dir(&entry_path, results);
                            } else {
                                let file_name = entry_path.file_name().unwrap().to_string_lossy().to_string();
                                if file_name.ends_with(".gguf") {
                                    results.push(entry_path.to_string_lossy().to_string());
                                }
                            }
                        }
                    }
                }
                find_gguf_in_dir(&path, &mut found);
            }
        }
    }

    found
}

fn main() -> Result<()> {
    println!("=== Real GGUF Model Comparison ===");
    println!();

    // Find models on the system
    let model_paths = find_gguf_models();

    if model_paths.is_empty() {
        println!("No GGUF model files found on the system.");
        return Ok(());
    }

    println!("Found {} GGUF model file(s):", model_paths.len());

    // Extract descriptions and embed them
    let mut descriptions = Vec::new();
    for path in &model_paths {
        match extract_model_description(path) {
            Ok(description) => {
                println!("  {} - {}", Path::new(path).file_name().unwrap().to_string_lossy(), description);
                descriptions.push((path.clone(), description));
            }
            Err(e) => {
                println!("  {} - Error: {}", Path::new(path).file_name().unwrap().to_string_lossy(), e);
            }
        }
    }

    if descriptions.len() < 2 {
        println!("\nNeed at least 2 models to compare.");
        return Ok(());
    }

    // Use Qwen2.5-0.5B-Instruct as the embedding model (known working GGUF)
    let embedder_model = "/home/crombo/projects/active/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("Loading embedding model...");
    let embedder = Embedder::load(embedder_model)?;

    // Embed each model description
    println!("Embedding model descriptions...");
    let mut embeddings = Vec::new();
    for (path, desc) in &descriptions {
        match embedder.embed(desc) {
            Ok(embedding) => {
                embeddings.push((path.clone(), desc.clone(), embedding));
            }
            Err(e) => {
                println!("  Failed to embed {}: {}", Path::new(path).file_name().unwrap().to_string_lossy(), e);
            }
        }
    }

    // Compare all pairs
    println!("\n=== Model Similarity Matrix ===");
    println!("Cosine similarity between model architecture descriptions:\n");

    let n = embeddings.len();
    print!("{:<20}", "Model");
    for (_, _, _) in &embeddings {
        print!("{:>10}", "");
    }
    println!();

    for (i, (path_a, desc_a, emb_a)) in embeddings.iter().enumerate() {
        let name_a = Path::new(path_a).file_name().unwrap().to_string_lossy();
        print!("{:<20}", &name_a[..std::cmp::min(name_a.len(), 18)]);

        for (j, (_, _, emb_b)) in embeddings.iter().enumerate() {
            if i == j {
                print!("{:>10}", "1.000");
            } else {
                let sim = cosine_similarity(emb_a, emb_b);
                print!("{:>10.3}", sim);
            }
        }

        println!();
    }

    // Analyze results
    println!("\n=== Analysis ===");
    let mut max_sim = -1.0;
    let mut min_sim = 1.0;
    for (i, (_, desc_a, emb_a)) in embeddings.iter().enumerate() {
        for (j, (path_b, desc_b, emb_b)) in embeddings.iter().enumerate() {
            if i < j {
                let sim = cosine_similarity(emb_a, emb_b);
                if sim > max_sim {
                    max_sim = sim;
                }
                if sim < min_sim {
                    min_sim = sim;
                }
            }
        }
    }

    println!("Highest similarity: {:.3}", max_sim);
    println!("Lowest similarity: {:.3}", min_sim);
    println!("Similarity range: {:.3}", max_sim - min_sim);

    // Identify most similar pair
    let mut best_pair = (0, 0, -1.0);
    for (i, (_, desc_a, emb_a)) in embeddings.iter().enumerate() {
        for (j, (path_b, desc_b, emb_b)) in embeddings.iter().enumerate() {
            if i < j {
                let sim = cosine_similarity(emb_a, emb_b);
                if sim > best_pair.2 {
                    best_pair = (i, j, sim);
                }
            }
        }
    }

    let name_i = Path::new(&embeddings[best_pair.0].0).file_name().unwrap().to_string_lossy();
    let name_j = Path::new(&embeddings[best_pair.1].0).file_name().unwrap().to_string_lossy();
    println!("\nMost similar models: {} and {} (similarity={:.3})", name_i, name_j, best_pair.2);

    Ok(())
}