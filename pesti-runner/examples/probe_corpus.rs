//! Comprehensive GGUF corpus integrity validator.
//!
//! For every tensor, computes the ACTUAL stored byte size (offset delta to the
//! next tensor, or file_end - offset for the last one) and compares its
//! bytes/element against the canonical ggml density for the tensor's LABELED
//! dtype. A mismatch means the file's dtype label does not match the actual
//! byte layout on disk — i.e. the file was not produced by a correct
//! llama.cpp quantizer (or is synthetic/corrupt).
//!
//! Usage: probe_corpus <file.gguf> [more files...]
use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufDtype;

/// Canonical bytes/element from ggml-common.h static_asserts.
/// QK_K=256, K_SCALE_SIZE=12, QK4_0=QK5_0=Q8_0=32, ggml_half=2 bytes.
fn canonical_density(dtype: GgufDtype) -> Option<f64> {
    Some(match dtype {
        GgufDtype::F32 => 4.0,
        GgufDtype::F16 => 2.0,
        GgufDtype::BF16 => 2.0,
        GgufDtype::F64 => 8.0,
        GgufDtype::I8 => 1.0,
        GgufDtype::I16 => 2.0,
        GgufDtype::I32 => 4.0,
        GgufDtype::I64 => 8.0,
        GgufDtype::Q4_0 => 18.0 / 32.0, // 0.5625 (f16 d + qs[16])
        GgufDtype::Q4_1 => 20.0 / 32.0, // 0.625
        GgufDtype::Q5_0 => 22.0 / 32.0, // 0.6875
        GgufDtype::Q5_1 => 24.0 / 32.0, // 0.75 (2xf16 + u32 hmask + qs[16])
        GgufDtype::Q8_0 => 34.0 / 32.0, // 1.0625
        GgufDtype::Q8_1 => 36.0 / 32.0, // 1.125
        GgufDtype::Q2K | GgufDtype::Q2K_S | GgufDtype::Q2K_M => 84.0 / 256.0, // 0.328125
        GgufDtype::Q3K | GgufDtype::Q3K_S => 110.0 / 256.0, // 0.4296875
        GgufDtype::Q4K | GgufDtype::Q4K_M | GgufDtype::Q4K_S => 144.0 / 256.0, // 0.5625
        GgufDtype::Q5K | GgufDtype::Q5K_M | GgufDtype::Q5K_S => 176.0 / 256.0, // 0.6875
        GgufDtype::Q6K | GgufDtype::Q6K_S => 210.0 / 256.0, // 0.8203125
        GgufDtype::Q8K | GgufDtype::Q8K_M => 292.0 / 256.0, // 1.140625
        _ => return None, // unknown / not in canonical table
    })
}

/// Which canonical dtype the ACTUAL byte density is closest to (for a hint).
fn nearest_known_density(actual_bpe: f64) -> Option<(&'static str, f64)> {
    let known: [(&str, f64); 12] = [
        ("Q4_0", 0.5625),
        ("Q4_1", 0.625),
        ("Q5_0", 0.6875),
        ("Q5_1", 0.75),
        ("Q8_0", 1.0625),
        ("Q8_1", 1.125),
        ("Q2_K", 0.328125),
        ("Q3_K", 0.4296875),
        ("Q4_K", 0.5625),
        ("Q5_K", 0.6875),
        ("Q6_K", 0.8203125),
        ("Q8_K", 1.140625),
    ];
    let mut best: Option<(&str, f64)> = None;
    for (name, d) in known {
        let err = (d - actual_bpe).abs();
        if best.map_or(true, |(_, be)| err < be) {
            best = Some((name, err));
        }
    }
    best
}

fn check_file(path: &str) -> i32 {
    println!("\n==================================================================");
    println!("FILE: {path}");
    println!("==================================================================");
    let header = match parse_gguf(std::path::Path::new(path)) {
        Ok(h) => h,
        Err(e) => {
            println!("  PARSE FAILED: {e}");
            return 1;
        }
    };
    let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  file_len={}  data_start={}  tensors={}  arch={:?}",
        file_len,
        header.data_section_start,
        header.tensors.len(),
        header.architecture()
    );

    let ts = &header.tensors;

    // Tensors must be stored in ascending-offset order for the data section to
    // be a contiguous pack. Verify that, and detect any gaps/overlaps.
    let mut sorted_ok = true;
    for w in ts.windows(2) {
        if w[1].offset <= w[0].offset {
            sorted_ok = false;
            break;
        }
    }

    // Build offset-sorted list of (abs_offset, dtype, name, elems) so that the
    // "actual stored size" of a tensor = next_sorted_abs - this_abs.
    let mut by_off: Vec<(u64, GgufDtype, &str, f64)> = ts
        .iter()
        .map(|t| {
            (
                t.offset + header.data_section_start,
                GgufDtype::from_u32(t.dtype),
                t.name.as_str(),
                t.element_count() as f64,
            )
        })
        .collect();
    by_off.sort_by_key(|x| x.0);

    let data_end = file_len; // last tensor runs to EOF
    let mut mismatch = 0usize;
    let mut checked = 0usize;
    let mut gap_bytes = 0u64;
    let mut dtype_hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    for (i, (abs, dtype, name, elems)) in by_off.iter().enumerate() {
        *dtype_hist.entry(format!("{dtype:?}")).or_insert(0) += 1;
        if *elems == 0.0 {
            continue;
        }
        let actual = if i + 1 < by_off.len() {
            by_off[i + 1].0 - *abs
        } else {
            data_end - *abs
        } as f64;
        let bpe = actual / elems;
        let expected = canonical_density(*dtype);
        checked += 1;

        let status = match expected {
            Some(exp) => {
                let tol = (exp * 0.005).max(0.0005).max(4.0 / elems);
                if (bpe - exp).abs() <= tol {
                    "ok"
                } else {
                    mismatch += 1;
                    "MISMATCH"
                }
            }
            None => "unknown-dtype",
        };

        let show = status != "ok"
            || name.starts_with("token_embd")
            || name.starts_with("output");
        if show {
            let hint = nearest_known_density(bpe)
                .map(|(n, e)| format!("  ~{n} (err {:.4})", e))
                .unwrap_or_default();
            println!(
                "  [{status:>8}] {:45} dtype={dtype:?} n={elems:.0} actual={:.0}B bpe={:.5} expected={:?}{hint}",
                name,
                actual,
                bpe,
                expected.map(|e| format!("{e:.5}")).unwrap_or_else(|| "-".into()),
            );
        }

        // Gap detection: if the tensor's canonical size is smaller than the
        // delta, the difference is unexplained padding.
        if let Some(exp) = expected {
            let canonical = (exp * *elems) as u64;
            let delta = by_off.get(i + 1).map(|n| n.0 - *abs).unwrap_or(data_end - *abs);
            if delta > canonical {
                gap_bytes += delta - canonical;
            }
        }
    }

    let total_data = data_end - header.data_section_start;
    println!(
        "\n  data_section: {} bytes  (sorted_by_offset={})  unexplained_gaps={}B",
        total_data, sorted_ok, gap_bytes
    );

    println!("\n  dtype histogram:");
    for (k, v) in &dtype_hist {
        println!("    {k:>12} : {v} tensors");
    }
    println!(
        "\n  SUMMARY: {checked} tensors checked, {mismatch} dtype/byte-size MISMATCHES"
    );
    if mismatch == 0 {
        println!("  => FILE INTERNALLY CONSISTENT with canonical ggml densities");
        0
    } else {
        println!("  => FILE HAS DTYPE-LABEL / BYTE-LAYOUT MISMATCHES (not a clean llama.cpp export)");
        1
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_corpus <file.gguf> [more...]");
        std::process::exit(2);
    }
    let mut any_fail = false;
    for a in &args {
        if check_file(a) != 0 {
            any_fail = true;
        }
    }
    std::process::exit(if any_fail { 1 } else { 0 });
}
