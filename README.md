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

> "I can't read Llama. I can't even read Llama RS. But I can read this."

PESTI is built to understand what happens at every layer:
- GGUF v3 parsing from scratch (not wrapping llama.cpp)
- Transformer primitives in pure Rust (RMSNorm, RoPE, SwiGLU, attention)
- Backend abstraction for pluggable execution (CPU only, CUDA stub)

## What It's Not

- ❌ A competitor to llama.cpp (GPU path is TODO, not done)
- ❌ A production-ready inference engine (forward pass is pre-work)
- ❌ A framework for others to use (it's my learning scaffold)

## What It Is

- ✅ A way to understand GGUF internals byte-by-byte
- ✅ A sandbox to experiment with quantization and CUDA kernels (pre-work)
- ✅ Proof that I can build systems-level software without a CS degree

## Current Status

**Version:** v0.1.5 (August 2026)  
**Focus:** CPU inference stable, GPU/forward pass are pre-work/stubs

| Component | Status | Notes |
|-----------|--------|-------|
| GGUF v3 parsing | ✅ Complete | All K-family quantizations verified |
| CPU inference | ✅ Stable | ~217-222 tok/s on TinyLlama (aspirational, unverified) |
| Backend abstraction | ✅ Complete | Cpu, CUDA stub, llama.cpp FFI paths |
| GPU kernels | ⏳ Pre-work | CUTLASS GEMM wrapper done, WGMMA attention is TODO |
| Forward pass | ⏳ Pre-work | CPU path works, GPU path is feature-gated stub |
| End-to-end inference | ✅ CPU only | GPU path untested |

## Quick Start (CPU Only)

```bash
# Clone and build (CPU only, CUDA optional but untested)
git clone https://github.com/CromboJambo/pesti.git
cd pesti
cargo build --package pesti-runner

# Run inference on a GGUF model
cargo run --package pesti-runner --example infer \
  --model conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "Once upon a time" \
  --tokens 50
```

## The Honest Benchmark

**Aspirational (based on llama.cpp baseline, unverified):**
- Q3_K_M: ~218 tok/s
- Q4_K_M: ~217 tok/s
- Q5_K_M: ~221 tok/s
- Q8_0: ~222 tok/s

*Note: These numbers appear in README but no measurement logs exist. The actual benchmark script exists but hasn't been run recently.*

## Why Rust?

Because it's the language I can reason in better than Python or TypeScript.
Not because "Rust is best" but because "Rust lets me hold the architecture
in my head."

## License

AGPL-3.0-or-later (because I want to understand the ecosystem, not own it)

---

*This is a learning project. If you find bugs, file issues. If you want to
contribute, PRs welcome. If you just want to use it as a reference for your
own work, that's fine too.*
