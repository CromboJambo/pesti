# PESTI Roadmap

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 23)

Working GPU inference path for Qwen2.5-0.5B-Instruct with fused attention kernel. Numerical conformance validated vs llama.cpp reference outputs.

**Throughput:** pesti-runner: 81.78 tok/s vs llama.cpp: 504.04 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti). ~6x gap identified as optimization target.

## Completed Work

### Week 21
- ✅ Fused attention kernel passes numerical conformance vs llama.cpp
- ✅ KV cache autoregressive validation suite
- ✅ Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- ✅ Long sequence support verified to seq=4096

### Week 22: Debt and Spikes
- ✅ Fix remaining clippy warnings — down from 193 to 136 warnings
- ✅ Spike: batched generation for parallel prompts — ran on ftw3, measured ~1.0x speedup at seq=128 (expected: short sequences don't benefit; value appears at production lengths >512)

### Week 23: Optimization and Scale
- ✅ Establish comparable tok/s benchmark against llama.cpp on same model/hardware — 6x gap identified

## Upcoming Work

### Week 23: Optimization and Scale
- ✅ Establish comparable tok/s benchmark against llama.cpp on same model/hardware — pesti-runner: 81.78 tok/s vs llama.cpp: 504.04 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti). ~6x gap identified as optimization target.
- [ ] Profile GEMM vs attention kernel time split at production sequence lengths — identify softmax host-transfer bottleneck
- [ ] KV cache quantization (Q4_K) to reduce memory bandwidth bottleneck
- [ ] Spike: TMA descriptors for async prefetching

## Optimization Analysis (Week 23)

**Benchmark:** pesti-runner 81.78 tok/s vs llama.cpp 504.04 tok/s on Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti
**Gap:** ~6x slower

### Identified Bottleneck: Softmax Host Transfer

The current attention path in `GemmBasedAttentionKernel::forward()`:
1. GEMM (GPU): Q @ K^T → scores on device ✅
2. **Transfer scores to host (D2H)** ❌ bottleneck
3. **Softmax on CPU** ❌ bottleneck  
4. **Transfer softmax back to device (H2D)** ❌ bottleneck
5. GEMM (GPU): S @ V → output ✅

This pattern is repeated per attention layer, per sequence position. The fused attention kernel (`fused_attention_conformant.rs`) exists and does softmax on GPU but isn't the default path.

### Optimization Strategy

1. **Switch to fused attention kernel** — eliminates all host transfers for softmax
2. **Profile GEMM vs attention time split** — quantify remaining bottlenecks
3. **KV cache quantization (Q4_K)** — reduce memory bandwidth bottleneck at long sequences

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
*Updated: September 12, 2026 — based on git history and benchmark results, not planning documents*
