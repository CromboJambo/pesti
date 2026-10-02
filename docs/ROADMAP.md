# PESTI Roadmap

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 21)

Working GPU inference path for Qwen2.5-0.5B-Instruct:
- Fused attention kernel passes numerical conformance vs llama.cpp
- KV cache autoregressive validation suite
- Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- Long sequence support verified to seq=4096

**Current throughput:** pesti-runner achieves 307.68 tok/s on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti) vs llama.cpp's 504.04 tok/s — Phase 1 target of 100 tok/s exceeded.

## Completed Weeks

- **Week 25:** Optimization and Scale — established comparable tok/s benchmark (307.68 vs llama.cpp's 504.04), completed F16 GPU inference via cuBLAS hgemm, trait-based linear layer integration, non-matmul GPU kernels (SwiGLU/RMSNorm/RoPE/Softmax), and token embedding fix for GGUF weight loading.
- **Week 23:** Long-Sequence Prefill Throughput — measured prefill speed across sequence lengths; identified attention kernel O(n²) scaling as bottleneck for long-context workloads.

## Upcoming Work (Week 26+)

Per REFACTOR_SPEC.md Phase 4 and remaining Week 25 items:
- Profile GEMM vs attention kernel time split at production sequence lengths to identify remaining bottlenecks
- KV cache quantization (Q4_K) to reduce memory bandwidth bottleneck
- Spike: TMA descriptors for async prefetching

## Architecture Refactor

See [docs/REFACTOR_SPEC.md](REFACTOR_SPEC.md) for the complete top-down refactor plan covering:
- Fused attention kernel (Phase 1)
- GEMM integration into inference path (Phase 2)
- Non-matmul GPU kernels: SwiGLU, RMSNorm, RoPE, Softmax (Phase 3)
- KV cache quantization with on-the-fly dequantization (Phase 4)
- Execution graph and kernel fusion (Phase 5)

This replaces the week-by-week itemization. Each phase has explicit deliverables, conformance requirements, and success metrics.

**Known Issues / Debt**

| Issue | Status | Impact |
|-------|--------|--------|
| pesti-safetensors: 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config) | Open | Can't fully validate quantized model loading |
| Examples don't compile after API changes | Recurring | Developer experience, not runtime |

**Failure Modes (Reference)**

When heading toward these patterns, expect trouble:

**CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Always sync via `cuda_shim::stream_synchronize` (event-based) or `context_synchronize` for legacy-default-stream work. This will silently produce wrong results, not crash.

**Kernel stack arrays:** Stack array sizes in kernels are bounds, not dynamic allocation. Indexing past `MAX_SEQ = 512` causes silent corruption, not bounds check failure.

**Numerical stability at scale:** Softmax overflow only manifests at long sequences (seq=4096+). Short-sequence tests pass; production fails. Always test with seq=4096+.

**Examples rot fast:** After API changes, examples break before tests do. Treat example compilation as part of the test suite, not documentation.

**GPU memory allocation is cheap:** Don't over-optimize by avoiding `cudaMalloc`. The cost is in synchronization and kernel launches, not allocation.

---
*Updated: September 17, 2026 — roadmap consolidated into REFACTOR_SPEC.md based on codebase analysis*
