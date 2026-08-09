# PESTI Development Roadmap (Honest Version)

This roadmap tracks my learning journey through LLM inference internals.
It's organized by milestones, not product features.

## Phase 1: Foundations (✅ Complete)
### GGUF v3 Parsing
- [x] Full support for all K-family quantizations (Q2_K through Q8_0)
- [x] Byte-exact dequantization within tolerance
- [x] Tensor metadata extraction with architecture-specific fallback keys
- [x] **Learning outcome:** Understanding how models are serialized

### CPU Inference Engine
- [x] Transformer primitives (RMSNorm, RoPE, SwiGLU, attention) in pure Rust
- [x] Autoregressive generation loop with Top-P/Top-K sampling
- [x] Backend abstraction layer for pluggable execution
- [x] **Learning outcome:** Understanding how models execute

### Tokenizer Integration
- [x] GGUF tokenizer metadata extraction (vocab, BOS/EOS tokens)
- [x] Rust tokenizer loading from GGUF header
- [x] encode/decode API in CpuModel
- [x] **Learning outcome:** Understanding tokenization pipeline

### Benchmarking Baseline
- [x] CPU-only generation example (`examples/cpu_generation.rs`)
- [x] llama.cpp comparison (110.9 t/s on Qwen2.5-0.5B)
- [x] Parser performance: ~0.127s vs Python-based tools (~2.8s)
- [x] **Learning outcome:** Establishing apples-to-apples benchmark methodology

### Notes
- Tokenizer integration verified with Qwen2.5 model (32k vocab, BOS=151643, EOS=151645)
- CPU-only stub mode allows compilation without CUDA dependencies
- Full forward pass inference pending transformer implementation
- llama.cpp build: ~110.9 tokens/s on same hardware

## Phase 2: GPU Integration (✅ Working via GEMM Proxy)

### CUDA Skeleton
- [x] CUTLASS GEMM wrapper via `cudarc`
- [x] GEMM-based attention kernel (Q @ K^T → softmax → S @ V)
- [x] End-to-end GPU inference verification with real GGUF model
- [x] **GPU softmax kernel** - Optional CUDA-accelerated softmax with feature gating
- **Learning outcome**: Understanding how GPUs accelerate inference

### Forward Pass
- [x] CPU forward pass works (full autoregressive generation)
- [x] GPU forward pass via dispatch layer
- [ ] Byte-exact comparison between CPU and GPU paths (optional refinement)
- **Learning outcome:** Understanding the difference between CPU and GPU execution

### Notes
- Current implementation uses GEMM ops as building blocks rather than fused WGMMA attention PTX
- This is a valid engineering choice: proves GPU inference works before optimizing with dedicated kernels
- Dedicated WGMMA PTX kernel can be added in Phase 3 as a performance optimization

## Phase 3: Upstream Contribution (❌ Not Started)

### llama.cpp PRs
- [ ] Find bugs based on what I learned from PESTI
- [ ] Submit fixes or improvements
- [ ] Establish reputation as "the person who understands GGUF"
- **Learning outcome:** Understanding the ecosystem and community

## What This Is NOT

- ❌ A roadmap to beat llama.cpp at benchmarks
- ❌ A product launch timeline
- ❌ A way to become famous in the Rust/LLM space

## What This IS

- ✅ My learning scaffold for understanding LLM inference
- ✅ Proof that I can build systems-level software
- ✅ A vehicle to eventually navigate llama.cpp with confidence

---

*Last updated: August 2026*  
*This roadmap will change as I learn more. If it looks perfect, it's lying.*
