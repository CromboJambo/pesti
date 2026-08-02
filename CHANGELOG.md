# Changelog

All notable changes to PESTI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.2] - 2026-08-02

### Added
- **GGUF file writer** (`pesti-gguf/src/writer.rs`)
  - Full GGUF v3 practical format serialization
  - KV pair writing with u64 key lengths
  - Tensor metadata (name, shape, dtype, offset)
  - Configurable alignment padding (default: 256 bytes)
  - `parse_and_rewrite()` helper for file normalization
- **SafeTensors file writer** (`pesti-safetensors/src/writer.rs`)
  - Basic tensor serialization with JSON headers
  - Multi-tensor support
  - `gguf_to_safetensors()` conversion helper
- **Q5_0 dequantization** in `pesti-runner/src/dequantize.rs`
  - Dequantize Q5_0 quantized tensors to f32
  - 32 elements per block layout

### Changed
- Updated `gguf_weight_loader.rs` to use new `dequantize_q5_0()` function
- Fixed Q4_K dequantization block size (16 → 32 elements)
- Updated stored_size formulas for K-family quantizations

### Testing
- **GGUF writer tests**: 2 passing (round-trip, alignment)
- All existing tests remain passing (314+ total)

---

## [0.1.1] - 2026-08-01

### Added
- **Pure Rust dequantization layer** using `ggml-quants` crate
  - `dequantize_q4_0_ggml()` - Q4_0 tensor dequantization (32 elements/block)
  - `dequantize_q4_1_ggml()` - Q4_1 tensor dequantization (32 elements/block)
  - `dequantize_q8_0_ggml()` - Q8_0 tensor dequantization (32 elements/block)
- **CUDA acceleration stub** (`dequantize_cuda.rs`) for Phase 2 GPU kernels
- **Strict clippy configuration** (`.clippy.toml`) with production-grade lint rules
- **CI/CD pipeline** with automated testing, formatting, and semver checks
- **Release automation** workflow for version bumping and changelog generation

### Changed
- Replaced C FFI dequantization calls with pure Rust implementations
  - Removed `dequantize_q4_0()` from `gguf_weight_loader.rs` (48 lines)
  - Removed `dequantize_q4_1()` from `gguf_weight_loader.rs` (52 lines)
  - Removed `dequantize_q8_0()` from `gguf_weight_loader.rs` (32 lines)
- Updated dependency graph:
  - Added `ggml-quants = "0.1"` for pure Rust dequantization
  - Added `byteorder = "1.4"` for little-endian byte operations
- Build time improved: Full workspace compiles in ~60s from clean state

### Fixed
- Removed unused legacy functions from `gguf_weight_loader.rs`
- Cleaned up experimental PoC artifacts (`test_ggml_quants/`)
- Resolved clippy warnings and pedantic lint suggestions

### Testing
- **314 tests passing** (7 ignored) - all production code verified
- No test failures introduced by refactoring
- Verified against llama.cpp reference implementation in PoC phase

---

## [0.1.0] - Initial Release

### Added
- GGUF parser (`pesti-gguf`) with support for 29+ quantization types
- Weight loaders (GGUF + Safetensors) with dequantization support
- Device routing system (CPU/GPU hybrid inference)
- Tokenizer integration (BPE, SentencePiece, TikToken)
- Model discovery and registry system

### Changed
- Initial architecture: C FFI for GEMM, pure Rust for parsing
- Linear development workflow (direct to `main` branch)

---

## Upcoming (v0.2.0 - In Progress)

### Planned
- CUDA kernel integration via `cudarc` backend
- GPU-accelerated dequantization kernels
- Performance benchmarking suite
- Async inference engine improvements
