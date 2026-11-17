# pesti-gguf

[![Crates.io](https://img.shields.io/crates/v/pesti-gguf.svg)](https://crates.io/crates/pesti-gguf)
[![Docs.rs](https://docs.rs/pesti-gguf/badge.svg)](https://docs.rs/pesti-gguf)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)

A parser for [GGUF](https://github.com/ggml-org/llama.cpp/blob/master/docs) model weight files (the file type used by llama.cpp, Ollama, and others). Written in pure Rust with three dependencies: serde, serde_json, and thiserror.

GGUF files from different quantization tools carry inconsistent metadata: a tensor can claim one dtype while its data is laid out differently. This crate exposes the raw header and tensor data, and provides a format-inference module for detecting the actual quantization from stored byte sizes.

## Quick start

```bash
cargo add pesti-gguf
```

```rust
use pesti_gguf::parse_gguf;
use std::path::Path;

fn main() -> Result<(), pesti_gguf::GgufError> {
    let header = parse_gguf(Path::new("model.gguf"))?;

    println!("Version: {}", header.version);
    println!("KV pairs: {}", header.kv_pairs.len());
    println!("Tensors: {}", header.tensors.len());

    Ok(())
}
```

## Features

- v1/v2/v3 GGUF format support with version-aware parsing
- Structured error types (`GgufError`: `InvalidMagic`, `UnsupportedVersion`, etc.)
- Alignment validation from the `general.alignment` KV pair
- Length limits on metadata keys and tensor names (1 MiB max)
- `stored_size()` per tensor, mirroring ggml's `ggml_row_size`
- `extract_tensor_bytes*` helpers to slice a tensor's data out of a file
- Format-inference module for detecting the actual quantization from data size

## Dtype coverage

`GgufDtype` maps raw GGML type IDs to the variants in llama.cpp's `ggml_type` enum. Removed or absent IDs (4, 5, 31, 32, 33, 36, 37, 38) resolve to `Unknown`. `stored_size()` computes `type_size * ne / blck_size` and requires the element count to be a multiple of the block size, matching ggml's behavior.

## Format inference

Some GGUF files have metadata that does not match the stored data. The `format_inference` module estimates the real format from a tensor's byte size:

```rust
use pesti_gguf::{parse_gguf, format_inference::infer_tensor_format, GgufDtype};
use std::path::Path;

let header = parse_gguf(Path::new("model.gguf"))?;

for tensor in &header.tensors {
    let n_elements = tensor.element_count() as usize;
    let raw_data = /* extract from file */;

    let inferred = infer_tensor_format(
        GgufDtype::from_u32(tensor.dtype),
        n_elements,
        &raw_data,
    )?;

    for hint in &inferred {
        println!("Possible format: {} (confidence: {:.2})",
            hint.dtype, hint.confidence);
    }
}
```

- Returns multiple candidates sorted by confidence (0.0-1.0)
- Block sizes verified against llama.cpp's `ggml_type_traits`
- Currently covers Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0
- `validate_tensor_metadata()` flags dtype/size mismatches and suspicious layer formats

## Conformance testing

Tests validate the parser against real GGUF files in a `conformance-corpus` directory (a sibling of the `pesti` project, or a parent directory). The corpus path is resolved by probing a list of candidate locations.

The corpus-wide test checks that, for every real file, `data_section_start` plus the sum of each tensor's alignment-padded `stored_size()` reconstructs the data section with a zero-byte delta. This is the end-to-end guard against wrong dtype IDs, block sizes, or header sizes.

Because the corpus files are large (up to 2 GB), the corpus tests are `#[ignore]`d by default. Run them with:

```bash
cargo test -- --ignored corpus
```

To add a model, download any GGUF file from [HuggingFace](https://huggingface.com/models?library=gguf) and place it in `conformance-corpus/`.

## CLI

A `gguf-inspect` binary is available behind the `cli` feature:

```bash
cargo run -p pesti-gguf --features cli --bin gguf-inspect -- <model.gguf>
```

Flags:

- `--tensors-only` / `-t`: list tensor names only
- `--extract <NAME>` / `-e <NAME>`: extract a specific tensor as raw bytes (hex dump of first 64 bytes)
- `--json`: dump the full parsed header as JSON (for conformance comparison)

```bash
cargo run -p pesti-gguf --features cli --bin gguf-inspect -- -t /path/to/model.gguf
```

## Comparison with llama.cpp

| Feature       | pesti-gguf                              | llama.cpp          |
|---------------|-----------------------------------------|--------------------|
| Language      | Pure Rust                               | C++                |
| Dependencies  | serde, serde_json, thiserror            | CUDA libs, OpenBLAS |
| Memory safety | Compile-time guarantees                 | Runtime checks     |
| FFI required  | No                                      | N/A (native)       |
| WASM ready    | Yes                                     | Requires Emscripten |

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

---

Maintained by [@crombojambo](https://github.com/crombojambo).
