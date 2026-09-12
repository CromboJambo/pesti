# PESTI Roadmap

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 21)

Working GPU inference path for Qwen2.5-0.5B-Instruct:
- Fused attention kernel passes numerical conformance vs llama.cpp
- KV cache autoregressive validation suite
- Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- Long sequence support verified to seq=4096

**Throughput:** 0.52 tok/s on RTX 3070 Ti (Qwen2.5-0.5B-Instruct Q4_K_M, seq=64). Baseline: llama.cpp achieves 77.1 tok/s on TinyLlama Q8 (RTX 4070 Ti SUPER) — different model/hardware, not directly comparable yet.

## Upcoming Work

### Week 22: Remaining Debt
- [ ] Fix 4 failing pesti-safetensors tests (Q4_K/Q5_K/Q6_K dequant + config extraction)
- [ ] Address remaining clippy warnings (unused vars in stub code, missing Safety docs)
- [ ] Spike: batched generation for parallel prompts

### Week 23+: Optimization and Scale
- [ ] Establish comparable tok/s benchmark against llama.cpp on same model/hardware
- [ ] Profile GEMM vs attention kernel time split at production sequence lengths
- [ ] KV cache quantization (Q4_K) to reduce memory bandwidth bottleneck
- [ ] Spike: TMA descriptors for async prefetching

## Known Issues / Debt

| Issue | Status | Impact |
|-------|--------|--------|
| pesti-safetensors: 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config) | Open | Can't fully validate quantized model loading |
| Examples don't compile after API changes | Recurring | Developer experience, not runtime |
| RoPE broadcasting required explicit expand() | Fixed | Caught by conformance testing |

## Failure Modes (Reference)

When heading toward these patterns, expect trouble:

**CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Always sync via `cuda_shim::stream_synchronize` (event-based) or `context_synchronize` for legacy-default-stream work. This will silently produce wrong results, not crash.

**Kernel stack arrays:** Stack array sizes in kernels are bounds, not dynamic allocation. Indexing past `MAX_SEQ = 512` causes silent corruption, not bounds check failure.

**Numerical stability at scale:** Softmax overflow only manifests at long sequences (seq=4096+). Short-sequence tests pass; production fails. Always test with seq=4096+.

**Examples rot fast:** After API changes, examples break before tests do. Treat example compilation as part of the test suite, not documentation.

**GPU memory allocation is cheap:** Don't over-optimize by avoiding `cudaMalloc`. The cost is in synchronization and kernel launches, not allocation.

## Completed Milestones (Summary)

- **Weeks 15-17:** GPU kernel integration — CUDA dispatch system, fused attention kernel, KV cache validation suite
- **Week 18:** Stabilize and measure — fixed examples, established benchmarks, documented VRAM usage
- **Week 19:** Optimization pass — profiled kernels, measured GEMM speedup, identified memory bandwidth bottleneck
- **Week 20:** FP16 KV cache integration — implemented and verified; real tok/s measurement achieved
- **Week 21:** Cleanup — llama-cpp-2 API compatibility, clippy errors resolved, workspace formatted

Detailed logs in git history. Key commits: `b844f4b` (week 20 results), `a4dd72c` (week 21 cleanup).

---
*Updated: September 11, 2026 — based on git history and test results, not planning documents*
