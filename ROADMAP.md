# PESTI Roadmap

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 29)

Working GPU inference path for Qwen2.5-0.5B-Instruct with production profiling complete. Llama.cpp replacement track delivered (CLI, studio, server).

**Throughput:** pesti-runner: ~307 tok/s vs llama.cpp: 504.04 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti) — Phase 1 target of 100 tok/s exceeded. Remaining gap is optimization target.

**Week 29 focus:** Optimization path decision — GEMM compute at practical optimum via plain cuBLAS; evaluating KV cache quantization, speculative decoding, or custom non-GEMM kernels.

## Roadmap Hierarchy

This document is the top-level project roadmap. Detailed implementation plans live in module-specific roadmaps and specs:

| Document | Scope |
|----------|-------|
| [docs/ROADMAP.md](docs/ROADMAP.md) | Detailed phase-by-phase work, completed weeks, optimization analysis |
| [pesti-runner/ROADMAP.md](pesti-runner/ROADMAP.md) | Inference engine module: GPU dispatch, transformer layers, kernel integration |
| [docs/REFACTOR_SPEC.md](docs/REFACTOR_SPEC.md) | Architecture refactor plan (fused attention, GEMM integration, KV cache quantization) |
| [docs/spec-gpu-optimization-phases.md](docs/spec-gpu-optimization-phases.md) | GPU optimization phase specifications |
| [docs/SPEC_LLM_RUNNER_TRAINING.md](docs/SPEC_LLM_RUNNER_TRAINING.md) | LLM runner, tuning, and safetensors track |

## Completed Work (Summary)

### GPU Inference Substrate
- ✅ Fused attention kernel with numerical conformance vs llama.cpp
- ✅ KV cache autoregressive validation suite
- ✅ Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- ✅ Long sequence support verified to seq=4096
- ✅ F16 GPU inference via cuBLAS hgemm
- ✅ Trait-based linear layer integration
- ✅ Non-matmul GPU kernels: SwiGLU, RMSNorm, RoPE, Softmax
- ✅ GPU weight caching via lazy upload
- ✅ Deep profiling and GEMM optimization analysis
- ✅ cuBLASLt/CUTLASS evaluation (found slower for LLM decode shapes)

### Llama.cpp Replacement Track
- ✅ Phase 1: `pesti` CLI binary with llama.cpp-compatible argument parsing
- ✅ Phase 2: `pesti-studio` model management (hardware detection, HF Hub search/download)
- ✅ Phase 3: Auto-tuning engine with benchmarking infrastructure
- ✅ Phase 4: Enhanced server mode (hot-swapping, connection pooling, SSE streaming)

## Upcoming Work

### Optimization Path Decision (Week 29)
Based on Week 28 findings, GEMM compute is at practical optimum. Remaining throughput gap likely stems from other factors. Candidate optimization paths:
1. KV cache quantization — reduce memory bandwidth pressure
2. Speculative decoding — increase effective tokens/sec
3. Custom CUDA kernels for non-GEMM ops — attention, RMSNorm, RoPE optimization headroom

### LLM Runner, Tuning & Safetensors Track (Planned)
- QLoRA fine-tuning pipeline with hardware-aware configuration
- Dataset loading, training loop, checkpointing, adapter saving
- Target: within 2x of Unsloth Python implementation on same hardware
- See [docs/SPEC_LLM_RUNNER_TRAINING.md](docs/SPEC_LLM_RUNNER_TRAINING.md) for full spec

## Known Issues / Debt

| Issue | Status | Impact |
|-------|--------|--------|
| pesti-safetensors: 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config) | Open | Can't fully validate quantized model loading |
| Examples don't compile after API changes | Recurring | Developer experience, not runtime |

## Failure Modes (Reference)

When heading toward these patterns, expect trouble:

**CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Always sync via `cuda_shim::stream_synchronize` (event-based) or `context_synchronize` for legacy-default-stream work. This will silently produce wrong results, not crash.

**Kernel stack arrays:** Stack array sizes in kernels are bounds, not dynamic allocation. Indexing past `MAX_SEQ = 512` causes silent corruption, not bounds check failure.

**Numerical stability at scale:** Softmax overflow only manifests at long sequences (seq=4096+). Short-sequence tests pass; production fails. Always test with seq=4096+.

**Examples rot fast:** After API changes, examples break before tests do. Treat example compilation as part of the test suite, not documentation.

**GPU memory allocation is cheap:** Don't over-optimize by avoiding `cudaMalloc`. The cost is in synchronization and kernel launches, not allocation.

---
*Updated: October 12, 2026 — Roadmap hierarchy restructured, Week 29 planning*