//! Quick benchmark for pesti-gguf parse time measurements
//! Run with: cargo run --release --example quick_bench
//! 
//! Automatically discovers GGUF models from the conformance-corpus directory.

use pesti_gguf::parse_gguf;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Find all GGUF files in the conformance-corpus directory
fn find_conformance_models() -> Vec<PathBuf> {
    let manifest_dir = std::env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("Failed to compute workspace root");
    
    let corpus_path = workspace_root.join("projects").join("pesti").join("conformance-corpus");
    
    if !corpus_path.exists() {
        eprintln!("⚠️  Conformance corpus not found at {:?}", corpus_path);
        return vec![];
    }
    
    let mut models = vec![];
    for entry in std::fs::read_dir(&corpus_path).expect("Failed to read corpus dir") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            models.push(path);
        }
    }
    
    // Sort by filename for consistent output
    models.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    models
}

fn main() -> Result<(), pesti_gguf::GgufError> {
    println!("=== PESTI-GGUF PARSE TIME BENCHMARK ===\n");
    
    let models = find_conformance_models();
    
    if models.is_empty() {
        eprintln!("No GGUF files found in conformance-corpus/");
        return Ok(());
    }
    
    println!("Found {} model(s) to benchmark\n", models.len());
    
    for path in &models {
        let metadata = std::fs::metadata(path)?;
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        
        // Single run (no warm-up, no iterations for clean output)
        let start = Instant::now();
        match parse_gguf(path) {
            Ok(header) => {
                let duration = start.elapsed();
                
                // Extract model name from filename (e.g., "qwen2.5-0.5b-instruct-q4_k_m" → "0.5B Q4_K_M")
                let model_name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                
                // Simple heuristic to extract size/quantization from filename
                let display_name = if model_name.contains("0.5b") {
                    format!("{} {}", 
                        if model_name.contains("q4_k_m") { "0.5B Q4_K_M" }
                        else if model_name.contains("q5_k") { "0.5B Q5_K" }
                        else if model_name.contains("q6_k") { "0.5B Q6_K" }
                        else if model_name.contains("q8_0") { "0.5B Q8_0" }
                        else if model_name.contains("f16") { "0.5B F16" }
                        else { "0.5B" },
                        model_name
                    )
                } else if model_name.contains("3b") {
                    format!("{} {}", 
                        if model_name.contains("q4_k_m") { "3B Q4_K_M" }
                        else { "3B" },
                        model_name
                    )
                } else {
                    model_name.to_string()
                };
                
                println!("{}:", display_name);
                println!("  File size:      {:.1} MB", file_size_mb);
                println!("  Parse time:     {:?} ({:.2} ms)", 
                    duration, 
                    duration.as_secs_f64() * 1000.0);
                println!("  KV pairs:       {}", header.kv_pairs.len());
                println!("  Tensors:        {}", header.tensors.len());
                println!();
            }
            Err(e) => {
                eprintln!("⚠️  Skipping {:?}: {}", path, e);
            }
        }
    }
    
    Ok(())
}
