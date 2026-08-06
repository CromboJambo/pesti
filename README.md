# PESTI
```
▄▄▄▄▄▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄         ▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄
█████ ▀█████▄  █████ ▀████▄   ▄▓▓▓▓▀▓▓▓▓▄  ▐▓█▓█▀▓███▓▀▓▓█▓▌ █▓█▓█
▓███▓  ▐█▓██▓▌ ▓█▓█▓  ▐▓█▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▌ ▓▓▓▓▓ ▐▓▓▓▌ ▓▓▓▓▓
▓▓▓▓▓   ▓▓▓▓▓▓ ▓▓▓▓▓   ▒▓▒▒░ ▓▒▓▒▒   ▒▒▓▒▓ ▀     ▒▓▒▓▓     ▀ ▓▓▓▒▓
▓▓▓▒▓   ▒▒▒▒▒▒ ▒▓▓▒▓         ▓▒▓▓▒▄              ▒▒▒▒▓       ▓▒▓▓▒
▒▓▒▓▒ ▄▒▒▒▒▒▒▌ ░▓▒▓▒▓▒       ▀▀▀▓▒▓▒░▒▒▀▄        ▒░▒▓▒       ░▒░▓▒
▓░▓▒▒▓▒▒▒░▒░▀  ░░▒░░   ░░░░░        ▀░▒░▒▒       ░░░░░       ░░░░░
░▀█░▀          ▀░▀░░   ▀░▀░▀ ░░▀░░   ▀░░░▄       ░░░▀░       ▀░▀░░
  ▀               ▀     ▀     ▀  ▀   ▒▀ ▀░       ▀ ▀          ▀  ▀
▓ ▓▄▀          ▄▓ ▄▓   ▓ ▄▓▄ ▓▄ ▓▄   ▓▄▄▓▓       █ ▄▓        ▓▄ ▓▄
▒▄▒▒▒          ▒▒▄░▒  ▓▒▒▒▒▓ ▄▒▓▒▄  ▄▄▒▒▒▒       ▓▓▓▒▒       ▄▒▓▒▄
░░░░░          ░░░▒░▄░░░░░░▀ ▀▄░░░░░░░░░▄▀      ▄▒▒▒░░▄      ░░░░░
```
**Portable Execution Substrate for Transformer Inference**

A high-performance Rust inference engine for LLMs.

⚠️ **Warning**: This is a learning project with intentional chaos. 
You'll find more bugs than features, but the GPU kernels actually work.

## Overview

PESTI is a modular inference engine that delivers **~217-222 tok/s** on CPU and optimized GPU paths for models up to 2B parameters. Built with:

- **Native Rust 2024** with `unsafe` code only where necessary (forbidden by default)
- **CUDA tensor core kernels** (WGMMA, CUTLASS GEMM) via `cudarc`
- **Full GGUF v3 support** for all K-family quantizations (Q2_K through Q8_K)
- **Chunked batch processing** to minimize FFI overhead
- **Feature-gated architecture** for CPU-only and GPU-accelerated builds

## Quick Start

```bash
# Clone the repository
git clone https://github.com/CromboJambo/pesti.git
cd pesti

# Build with CUDA acceleration (requires NVIDIA GPU, sm_89+)
cargo build --package pesti-runner --features cuda

# Run inference on a GGUF model
cargo run --package pesti-runner --example infer -- \
  --model models/tinyllama-1.1b-q4_k_m.gguf \
  --prompt "Once upon a time" \
  --tokens 50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PESTI Runner                              │
├─────────────────────────────────────────────────────────────┤
│  InferenceEngine (dispatch layer)                           │
│  ├─ GPU Path: CudaGemmKernel (WGMMA/cublas)                │
│  ├─ GPU Path: CudaAttentionKernel (WGMMA PTX kernel)       │
│  └─ CPU Fallback: CpuGemmKernel + CpuAttentionKernel       │
├─────────────────────────────────────────────────────────────┤
│  GGUF Weight Loader                                         │
│  ├─ Dequantization: Q4_K, Q5_K, Q6_K, Q8_K (pure Rust)    │
│  └─ Reverse inference for dtype mismatch detection          │
├─────────────────────────────────────────────────────────────┤
│  Transformer Modules                                        │
│  ├─ RoPE position-corrected forward pass                  │
│  ├─ RMSNorm with fused scaling                              │
│  ├─ Linear layers (GEMM-optimized)                         │
│  └─ KV cache (O(N) autoregressive decode)                 │
└─────────────────────────────────────────────────────────────┘
```

## Performance

### CPU Benchmark (TinyLlama-1.1B, 4 threads)

| Quantization | Speed (tok/s) |
|--------------|---------------|
| Q3_K_M       | 218.5         |
| Q4_K_M       | 216.8         |
| Q5_K_M       | 221.6         |
| Q8_0         | 221.8         |

**Key insight**: Performance varies by <3% across all quantization levels—compute-bound inference dominates over dequantization cost for small models.

### GPU Benchmark (RTX 5060 Ti, sm_120)

- **WGMMA attention kernel**: Consumer Blackwell optimized
- **CUTLASS GEMM**: Ada Lovelace tensor cores (sm_8.9+)
- Target: ~6-8 tok/s (baseline), with future optimization for higher throughput

## Feature Status

| Component | Status | Notes |
|-----------|--------|-------|
| GGUF Parser | ✅ 100% | All 29+ quantization types supported |
| K-Family Conformance | ✅ 8/8 | Byte-exact match within tolerance |
| WGMMA Attention Kernel | ✅ Implemented | sm_12.0 (RTX 50-series) |
| CUTLASS GEMM | ✅ Implemented | sm_8.9+ (Ada Lovelace) |
| CPU Fallback | ✅ Production | `--no-default-features` supported |
| RoPE Correctness | ✅ Fixed | Position bug resolved in v0.1.5 |
| tcgen05 Path | ⏳ Stub | Datacenter Blackwell (sm_100) needs TMA |

## Recent Changes (v0.1.5)

### Complete K-Family Conformance (8/8 Passing) ✅

- **Q4_K/Q5_K dequantization overflow fixed**: Split `qs` into `qs_low` + `qs_high` u32 values
- **Q6_K logic corrected**: Proper 8-byte block structure handling
- **Optional output layer detection**: Graceful handling of Q2_K/Q3_K models without LM head

### RoPE Position Bug Fix

- Fixed `forward_layers` to use `start_pos` instead of `start_pos + layer_idx`
- Ensures correct rotary position embedding across all layers

### Debug Spam Cleanup

- Removed 24+ `eprintln!` statements from transformer, GGUF loader, and inference engine
- Cleaner production logs with reduced I/O overhead

## Project Structure

```
pesti/
├── pesti-conformance/      # Model conformance testing suite
├── pesti-runner/           # Core inference engine (GGUF/Safetensors)
│   ├── src/
│   │   ├── kernel/         # GPU/CPU attention & GEMM kernels
│   │   ├── transformer/    # Transformer module implementations
│   │   └── gguf_weight_loader.rs
├── pesti-gguf/             # GGUF v3 parser & writer
├── pesti-safetensors/      # Safetensors serialization
├── cuda-oxide/             # CUDA device detection & runtime (cudarc wrapper)
└── examples/               # Benchmark & inference examples
```

## Dependencies

### Runtime

- **Rust nightly 1.99+** (for `std::simd` features)
- **CUDA 12.05+** (for GPU acceleration)
- **NVIDIA GPU**: sm_8.9+ recommended (Ada Lovelace or newer)

### Build

```toml
[workspace.dependencies]
cudarc = "0.19.4"        # CUDA runtime & cuBLAS
gemm = "0.19.0"          # BLAS operations
rayon = "1.10.0"         # Parallelism
safetensors = "0.7.0"    # Model serialization
```

## Development Workflow

### Feature Gating

```bash
# CPU-only build (no CUDA)
cargo build --package pesti-runner --no-default-features

# GPU-accelerated build (requires CUDA toolkit)
cargo build --package pesti-runner --features cuda

# With mistral.rs backend (experimental)
cargo build --package pesti-runner --features cuda,mistralrs
```

### Testing

```bash
# Run all tests
cargo test --workspace

# K-family conformance suite
cd pesti-conformance && cargo test

# Performance benchmark
cargo run --package pesti-runner --example cpu_attention_bench
```

## Contributing

PESTI follows a **working in public** philosophy:

- Linear development on `main` branch (no feature branches)
- Many small commits (25+) showing iterative progress
- Technical transparency via engineering notes and benchmarks
- Open to PRs for kernel optimizations, bug fixes, and documentation

See [docs/](docs/) for architecture deep-dives and session summaries.

## License

**AGPL-3.0-or-later** — free for open-source and commercial use with copyleft provisions.

---

**Version**: 0.1.5 (in progress)  
**Status**: Production-ready CPU path, GPU kernels functional but end-to-end verification pending  
**Repository**: [github.com/nousresearch/pesti](https://github.com/CromboJambo/pesti)
