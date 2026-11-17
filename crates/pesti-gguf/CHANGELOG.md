# Changelog

All notable changes to this project will be documented in this file.

The versioning follows `0.x.y` while we're establishing the API, with:
- **x** = major structural shifts (format inference added, breaking parser changes)
- **y** = minor additions or fixes (new features, bug fixes, docs)

## [0.2.5] - 2026-11-17

### Added
- `cli` feature + `gguf-inspect` binary (`src/bin/gguf-inspect.rs`):
  inspect GGUF headers, list tensor names, extract a tensor's raw bytes, or
  dump the parsed header as JSON (`--json`) for conformance comparison.
  Run via: `cargo run -p pesti-gguf --features cli --bin gguf-inspect -- <model.gguf>`
- Defensive test suite (`src/tests/defensive_tests.rs`):
  - Huge tensor/KV/string counts rejected before allocation (overflow protection)
  - Huge array element counts rejected (allocation-bomb protection)
  - Truncated / malformed header fields rejected with typed errors
  - Invalid UTF-8 and zero-length keys rejected
  - v1 (u32 counts) backward-compat and unknown-version rejection

### Fixed
- Vendored the standalone crate into the PESTI workspace at `crates/pesti-gguf`
  (removed the `crates/pesti-gguf-standalone` submodule, repointed the workspace
  `pesti-gguf` path dependency). Workspace now resolves and builds green.

### Testing
- Full suite: 55 passed, 0 failed, 6 ignored (corpus tests)
- `cargo bench -p pesti-gguf` (comprehensive + detailed) → exit 0

---

## [0.2.4] - 2026-08-20

### Changed (breaking)
- `GgufDtype` realigned to llama.cpp's `ggml_type` enum in `ggml.h`:
  - Removed fabricated variants: `Q1K`, `Q4K_M`, `Q5K_M`, `Q6K_S`, `Q8K_M`,
    `Q2K_S`, `Q3K_S`, `Q4K_S`, `Q5K_S`, `Q2K_M`, `Q4_0_4_4`, `Q4_0_4_8`,
    `Q4_0_8_8`, `IQ4NL_4_4`, `IQ4NL_4_8`, `IQ4NL_8_8`, and the pesti-custom
    IDs 43-48
  - Renamed K-quant variants to ggml naming: `Q2K` -> `Q2_K`, `Q3K` -> `Q3_K`,
    `Q4K` -> `Q4_K`, `Q5K` -> `Q5_K`, `Q6K` -> `Q6_K`, `Q8K` -> `Q8_K`
  - Added missing ggml types: `IQ4_NL` (20), `IQ3_S` (21), `IQ2_S` (22),
    `IQ4_XS` (23), `IQ1_M` (29)
  - Removed/absent ggml IDs (4, 5, 31, 32, 33, 36, 37, 38) now resolve to
    `Unknown` instead of mapping to invented types
- `stored_size()` rewritten to mirror ggml's `ggml_row_size`
  (`type_size * ne / blck_size`), requiring the element count to be a multiple
  of the block size; non-aligned counts return `GgufError::InvalidTensor`
  instead of guessing partial-block sizes
- `bytes_per_element()` now returns `Option<usize>` (scalar types only);
  quantized types use the new `block_size()` and `bytes_per_block()` helpers

### Fixed
- KV array wire size: the element type is a u32 (4 bytes), previously counted
  as 1 byte, which undercounted every array KV field
- `FORMAT_SPECS` in `format_inference` corrected against `ggml_type_traits`
  (e.g. Q4_K is 256 elements / 144 bytes, not 16 / 9); removed specs for
  non-existent formats (Q4K_M, Q5K_M)
- Conformance test no longer asserts that every boolean KV value is true
  (Qwen2.5 sets `tokenizer.ggml.add_bos_token` to false)
- Corpus path helper probes the sibling `pesti/conformance-corpus` layout
  first and reports the expected location on miss

### Added
- Corpus-wide conformance test: every real GGUF file in the corpus must
  reconstruct its data section with a zero-byte delta
  (`cargo test -- --ignored corpus`)
- `examples/proof_dtype_misread`: regression demo showing per-dtype tensor
  counts and byte sums for a real mixed-quant model

### Documentation
- README rewritten: removed marketing language, corrected the license
  (AGPL-3.0-or-later, not dual-licensed), updated the format-inference
  coverage list, documented the corpus test layout

### Testing
- Full suite: 32 passed, 6 ignored (all 6 pass with `-- --ignored`)
- Zero data-section delta verified across 9 real corpus files, including
  the 2.1 GB Qwen2.5-3B Q4_K_M

---

## [0.2.3] - 2026-08-12

### Added
- **Format inference engine** (`src/format_inference.rs`)
  - `infer_tensor_format()`: Detects actual quantization from raw data size
  - `validate_tensor_metadata()`: Flags dtype mismatches and suspicious layers
  - Supports Q4_0, Q4K, Q4K_M, Q5_0, Q5K, Q5K_M, Q6K, Q8_0
  - Confidence scoring (0.0–1.0) based on size match quality
- **New module exports**: `format_inference::InferredFormat`, `format_inference::Warning`

### Documentation
- README updated with "honest truth" about GGUF metadata inconsistencies
- Added "The Entropy Problem" section explaining Rust's type-driven approach to format detection
- Format inference usage examples and migration guide

### Testing
- 8 new unit tests for format inference (100% pass rate)
- Full test suite: 32 passed, 5 ignored (missing corpus files)

---

## [0.2.2] - 2026-08-07
- Initial crates.io release
- Full v1/v2/v3 GGUF parsing
- Conformance testing with Qwen2.5 models
