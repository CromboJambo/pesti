//! Test utilities for conformance tests

use std::path::Path;

/// Helper to get the path to a conformance corpus file relative to the workspace root.
/// 
/// This avoids hardcoding `/home/crombo/projects/llm-workspace` and makes tests portable.
pub fn conformance_corpus_path(filename: &str) -> std::path::PathBuf {
    // Start from CARGO_MANIFEST_DIR (pesti-gguf crate directory)
    let manifest_dir = std::env!("CARGO_MANIFEST_DIR");
    
    // Navigate up two levels to reach workspace root (/home/crombo/projects/pesti),
    // then into conformance-corpus
    Path::new(manifest_dir)
        .parent()  // pesti-gguf/ -> /home/crombo/projects/pesti/
        .and_then(|p| p.parent())  // /home/crombo/projects/pesti/ -> /home/crombo/projects/
        .map(|p| p.join("pesti").join("conformance-corpus").join(filename))
        .expect("Failed to compute conformance corpus path")
}
