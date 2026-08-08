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

## Project Structure

```
pesti/
├── Cargo.toml                 # Workspace root
├── README.md                  # This file
├── CHANGELOG.md               # Version history & decisions
├── ROADMAP.md                 # Honest learning journey
│
├── pesti-gguf/                # GGUF parser crate (external dependency) → [crates.io](https://crates.io/crates/pesti-gguf)
├── pesti-runner/              # Main inference engine
│   ├── src/
│   │   ├── kernel/            # GPU primitives (GEMM, attention, softmax)
│   │   ├── transformer/       # CPU inference primitives
│   │   └── inertia.rs         # GPU unavailability handling
│   └── examples/              # CPU & GPU examples
│
├── pesti-conformance/         # Quantization verification tests
├── references/                # Architecture docs & EDRs
│   ├── gpu-softmax.md        # New: GPU softmax implementation guide
│   └── cuda-reintegration-checklist.md
│
└── benchmarks/               # Performance measurements (Python scripts)
```

## Engineering Decisions

See [CHANGELOG.md](./CHANGELOG.md) for full details. Key decisions:

### EDR-001: Consumer GPU Architecture Choice ✅
**Selected**: GEMM-based attention via `cudarc::cublas`  
**Why**: Works on RTX 4070 Ti SUPER (Ada Lovelace) right now, proven path via llama.cpp

### EDR-002: CUTLASS vs Custom PTX ✅
**Selected**: Integrate CUTLASS via cudarc  
**Why**: NVIDIA's reference implementation, saves 8-12 hours of PTX debugging

### EDR-003: K-Family Quantization Fix ✅
**Fixed**: Q4_K/Q5_K/Q8_K dequantization overflow  
**Result**: Conformance improved from 2/8 (25%) → 8/8 (100%)

### EDR-004: GPU Softmax with Feature Gating 🆕 (New)
**Selected**: Optional CUDA-accelerated softmax via `SoftmaxKernel` trait  
**Why**: Keeps codebase buildable without CUDA, enables future fused kernels

## What This Is NOT

❌ A roadmap to beat llama.cpp at benchmarks  
❌ A product launch timeline  
❌ A way to become famous in the Rust/LLM space  

## What This IS

✅ My learning scaffold for understanding LLM inference  
✅ Proof that I can build systems-level software  
✅ A vehicle to eventually navigate llama.cpp with confidence  

## Status & Updates

- **Last updated**: August 2026
- **Branch**: `main` → origin/main (ahead 7, behind 0)
- **Conformance**: 8/8 K-family quantizations passing (100%)
- **GPU support**: Working via GEMM proxy (Q @ K^T → softmax → S @ V)

---

*This roadmap will change as I learn more. If it looks perfect, it's lying.*
