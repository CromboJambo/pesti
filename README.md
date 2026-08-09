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

A learning scaffold for understanding LLM inference internals. Not a product, not a competitor—just my way of climbing onto the real codebase later.

## Why This Exists

Most LLM inference engines (llama.cpp, candle, burn) are **production-ready** but hard to learn from if you're new to:
- GGUF binary format internals
- GPU tensor core kernels (WGMMA, tcgen05)
- Quantization math (Q4_K, Q5_K, Q8_K dequantization)
- Attention kernel optimization

PESTI gives you:

✅ **Learning-first design** - Every component documented with "why" not just "how"  
✅ **Progressive complexity** - CPU → GEMM proxy → fused kernels  
✅ **Honest roadmap** - No benchmark chasing, just understanding  
✅ **Feature-gated builds** - Compile without CUDA for pure Rust learning  

## What's Inside

### Phase 1: Foundations ✅
- **[pesti-gguf](https://crates.io/crates/pesti-gguf)** - Full K-family quantization support (external crate)
- **CPU inference engine** (`pesti-runner`) - Pure Rust transformer primitives
- **Conformance testing** - Byte-exact dequantization verification

### Phase 2: GPU Integration 🆕 (Working via GEMM Proxy)
- **CUTLASS integration** via `cudarc` - Battle-tested NVIDIA kernels
- **GEMM-based attention** - Q @ K^T → softmax → S @ V
- **GPU softmax kernel** - Optional CUDA acceleration with feature gating
- **End-to-end verification** - Real GGUF model inference


## Quick Start

```bash
# Clone the workspace
git clone https://github.com/crombojambo/pesti.git
cd pesti

# Run CPU-only inference (no CUDA needed)
cargo run --package pesti-runner --example cpu_baseline

# Run GPU-enabled inference (requires CUDA 12.5+)
cargo run --package pesti-runner --example e2e_gpu_inference --features cuda

# Verify K-family quantization conformance
cargo test --package pesti-conformance
```

## Dependencies

The workspace depends on the external **pesti-gguf** crate:

- **[pesti-gguf](https://crates.io/crates/pesti-gguf)** v0.2.1 - GGUF parser (external dependency)
- **cudarc** - CUDA runtime bindings (optional, feature-gated)
- **half**, **num-traits**, **serde** - Core numeric & serialization types

### Adding pesti-gguf to your project

```bash
cargo add pesti-gguf@0.2.1
```

Or directly from crates.io:

```bash
cargo add https://crates.io/crates/pesti-gguf
```
