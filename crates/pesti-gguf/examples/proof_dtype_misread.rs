//! PROOF: pesti-gguf reads a real GGUF file's quant types and sizes correctly.
//!
//! Parses conformance-corpus/qwen2.5-0.5b-instruct-q2_k.gguf (a real
//! mixed-quant model) and verifies that pesti-gguf's dtype resolution and
//! stored_size() math agree with the authoritative llama.cpp layout:
//!
//!   - each tensor's data is placed at an offset aligned to
//!     `general.alignment` (32 here), and
//!   - the data section size == sum of every tensor's size *padded* to that
//!     alignment (ggml/src/gguf.cpp: the writer pads each tensor, so the last
//!     tensor's pad shows up as trailing bytes).
//!
//! Ground truth (verified against the file's own tensor table + llama.cpp
//! ggml type table):
//!   - 121 tensors, raw dtype ID 0  = GGML_TYPE_F32    (4 B per elem)
//!   - 120 tensors, raw dtype ID 20 = GGML_TYPE_IQ4_NL (18 B per 32 elems)
//!   -  24 tensors, raw dtype ID 11 = GGML_TYPE_Q3_K   (110 B per 256 elems)
//!   -  24 tensors, raw dtype ID 6  = GGML_TYPE_Q5_0   (22 B per 32 elems)
//!   -   1 tensor,  raw dtype ID 8  = GGML_TYPE_Q8_0   (33 B per 32 elems)
//!
//! Run: cargo run --example proof_dtype_misread
use std::collections::BTreeMap;
use pesti_gguf::{GgufDtype, parse_gguf};

fn align_up(v: u64, a: u64) -> u64 {
    if a == 0 {
        return v;
    }
    (v + a - 1) / a * a
}

fn main() {
    // Accept an explicit path, else default to the pesti conformance corpus.
    let path = std::env::args().nth(1).map(std::path::PathBuf::from).unwrap_or_else(|| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("pesti/conformance-corpus/qwen2.5-0.5b-instruct-q2_k.gguf")
    });
    println!("parsing: {}", path.display());
    let header = parse_gguf(&path).expect("parse failed");
    println!(
        "version={} kv={} tensors={}\n",
        header.version,
        header.kv_pairs.len(),
        header.tensors.len()
    );

    // Group tensors by the dtype pesti-gguf resolved, sum raw + padded sizes.
    let align = header.data_alignment.unwrap_or(1).max(1);
    let mut by_dtype: BTreeMap<String, (u32, u64, u64, u64)> = BTreeMap::new();
    for t in &header.tensors {
        let dt = GgufDtype::from_u32(t.dtype);
        let sz = t.stored_size().unwrap_or(0);
        let e = t.element_count();
        let entry = by_dtype
            .entry(format!("{} (raw id {})", dt.name(), t.dtype))
            .or_insert((0, 0, 0, 0));
        entry.0 += 1;
        entry.1 += sz;
        entry.2 += e;
        entry.3 += align_up(sz, align);
    }

    println!(
        "{:<22} {:>7} {:>13} {:>13} {:>13}",
        "pesti reads as", "tensors", "raw bytes", "elem count", "padded bytes"
    );
    let mut total_raw: u64 = 0;
    let mut total_padded: u64 = 0;
    for (name, (count, raw, _elems, padded)) in &by_dtype {
        println!("{:<22} {:>7} {:>13} {:>13} {:>13}", name, count, raw, _elems, padded);
        total_raw += raw;
        total_padded += padded;
    }

    let fsize = std::fs::metadata(&path).unwrap().len();
    let data_bytes = fsize - header.data_section_start;
    println!("\nfile size            : {fsize}");
    println!("data_section_start   : {}", header.data_section_start);
    println!("actual data bytes    : {data_bytes}");
    println!("pesti raw size sum   : {total_raw}");
    println!("pesti padded size sum: {total_padded}");

    // The writer pads every tensor to `align`, so the data section is exactly
    // the sum of the padded sizes. This must match the real file byte-for-byte.
    let delta = data_bytes as i128 - total_padded as i128;
    println!("delta (actual - padded): {delta} bytes");
    if delta == 0 {
        println!("\nPASS: pesti-gguf's dtype + stored_size() match the real file exactly.");
    } else {
        println!("\nFAIL: {delta}-byte mismatch - dtype or size math is still wrong.");
        std::process::exit(1);
    }
}
