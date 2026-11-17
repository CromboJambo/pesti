//! Test utilities for conformance tests

use std::path::Path;

/// Helper to get the path to a conformance corpus file.
///
/// The corpus is a sibling of the `pesti` project (`../pesti/conformance-corpus`
/// relative to this crate). We probe a list of candidate locations and return
/// the first that actually contains the requested file, so tests run against
/// the real corpus when present and fail with a clear message when absent.
pub fn conformance_corpus_path(filename: &str) -> std::path::PathBuf {
    let manifest_dir = std::env!("CARGO_MANIFEST_DIR");
    let crate_dir = Path::new(manifest_dir);

    // Candidate corpus roots, most-specific first.
    let candidates: Vec<std::path::PathBuf> = vec![
        // Sibling project layout: <projects>/pesti/conformance-corpus
        crate_dir
            .parent()
            .map(|p| p.join("pesti").join("conformance-corpus"))
            .unwrap_or_default(),
        // Direct parent: <parent>/conformance-corpus
        crate_dir
            .parent()
            .map(|p| p.join("conformance-corpus"))
            .unwrap_or_default(),
        // Two levels up
        crate_dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("conformance-corpus"))
            .unwrap_or_default(),
        // Three levels up
        crate_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .map(|p| p.join("conformance-corpus"))
            .unwrap_or_default(),
    ];

    for dir in &candidates {
        let full = dir.join(filename);
        if full.exists() {
            return full;
        }
    }

    // None found: return the primary candidate so the error message shows the
    // expected location (the caller's `.expect`/IO error will surface it).
    candidates
        .into_iter()
        .next()
        .map(|d| d.join(filename))
        .unwrap_or_else(|| crate_dir.join("conformance-corpus").join(filename))
}
