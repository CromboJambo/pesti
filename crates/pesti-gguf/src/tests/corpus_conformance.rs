//! Corpus-wide conformance: for every real GGUF file in the conformance corpus,
//! verify that pesti-gguf's `data_section_start` + `stored_size()` math reconstruct
//! the file's data section *exactly* (zero-byte delta).
//!
//! This is the end-to-end regression guard for the class of bugs where a wrong
//! dtype ID, a wrong quant block size, or an undercounted header field makes the
//! parser read tensor data at the wrong offset. A zero delta across the whole
//! corpus means:
//!   - `data_section_start` is correct (header size + alignment), and
//!   - every tensor's `stored_size()` (dtype + block math) is correct, because
//!     the writer pads each tensor to `general.alignment`, so the data section is
//!     exactly the sum of the padded per-tensor sizes.
//!
//! These are `#[ignore]`d because they read multi-hundred-MB (up to 2 GB) files.
//! Run with: `cargo test -- --ignored corpus`
use std::fs;
use std::path::Path;

use crate::parse_gguf;

fn corpus_root() -> Option<std::path::PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        crate_dir.parent()?.join("pesti").join("conformance-corpus"),
        crate_dir.parent()?.join("conformance-corpus"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

fn align_up(v: u64, a: u64) -> u64 {
    if a == 0 {
        return v;
    }
    (v + a - 1) / a * a
}

/// Verify one real file reconstructs its data section with zero delta.
fn check_file(path: &Path) -> Result<(u64, u64, u64), String> {
    let header = parse_gguf(path).map_err(|e| format!("parse failed: {e}"))?;
    if header.tensors.is_empty() {
        return Err("no tensors (stub/header-only file)".to_string());
    }
    let align = header.data_alignment.unwrap_or(1).max(1);
    let fsize = fs::metadata(path)
        .map_err(|e| format!("metadata: {e}"))?
        .len();
    if fsize <= header.data_section_start {
        return Err(format!(
            "file size {fsize} <= data_section_start {}",
            header.data_section_start
        ));
    }
    let actual_data = fsize - header.data_section_start;

    // Sum of every tensor's size padded to the alignment (writer pads each tensor).
    let mut padded_sum: u64 = 0;
    for t in &header.tensors {
        let sz = t
            .stored_size()
            .map_err(|e| format!("tensor '{}' stored_size: {e}", t.name))?;
        padded_sum = padded_sum
            .checked_add(align_up(sz, align))
            .ok_or_else(|| format!("tensor '{}' padded sum overflow", t.name))?;
    }

    let delta = padded_sum as i128 - actual_data as i128;
    if delta != 0 {
        return Err(format!(
            "data-section delta {delta} bytes (padded_sum={padded_sum}, actual={actual_data})"
        ));
    }
    Ok((fsize, actual_data, padded_sum))
}

/// Every real GGUF file in the corpus reconstructs with a zero-byte data-section delta.
#[test]
#[ignore = "Reads the full conformance corpus (multi-GB)"]
fn test_corpus_data_section_zero_delta() {
    let root = corpus_root().expect("conformance corpus directory not found");
    let mut files: Vec<std::path::PathBuf> = fs::read_dir(&root)
        .expect("read corpus dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "gguf"))
        .collect();
    files.sort();

    let mut checked = 0usize;
    let mut skipped = Vec::new();
    for path in &files {
        // Skip tiny stub files (header-only placeholders), not real models.
        let len = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if len < 1024 {
            skipped.push((path.file_name().unwrap().to_string_lossy().to_string(), len));
            continue;
        }
        match check_file(path) {
            Ok((fsize, actual, padded)) => {
                checked += 1;
                eprintln!(
                    "  ✓ {:40} file={fsize:>11} data={actual:>11} padded={padded:>11} delta=0",
                    path.file_name().unwrap().to_string_lossy()
                );
            }
            Err(e) => {
                panic!(
                    "✗ {} FAILED: {}",
                    path.file_name().unwrap().to_string_lossy(),
                    e
                );
            }
        }
    }
    assert!(checked > 0, "no real corpus files were checked");
    let skipped_list = skipped
        .iter()
        .map(|(n, l)| format!("{n}({l}B)"))
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "\n✓ {checked} real corpus files reconstruct with zero data-section delta \
         ({} stub file(s) skipped: {skipped_list})",
        skipped.len()
    );
}
