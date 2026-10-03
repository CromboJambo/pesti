# PESTI Roadmap

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 28)

Working GPU inference path for Qwen2.5-0.5B-Instruct with production profiling complete:
- Fused attention kernel passes numerical conformance vs llama.cpp
- KV cache autoregressive validation suite
- Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- Long sequence support verified to seq=4096
- GPU weight caching via lazy upload (Week 27)

**Current throughput:** pesti-runner achieves 307.68 tok/s on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti) vs llama.cpp's 504.04 tok/s — Phase 1 target of 100 tok/s exceeded.

**Week 26 benchmark result:** 19.77 tok/s with cuBLAS algorithm caching/selection optimization applied. GEMM compute remains dominant bottleneck at 58.8% of kernel time; H2D transfers account for 23.2%. Still ~25x slower than llama.cpp's 504 tok/s reference on this hardware.

## Week 27: GPU Weight Caching (Completed)

**Goal:** Eliminate per-token H2D weight transfers by caching weights on GPU after first upload. Target: reduce H2D transfer overhead from 23.2% to near 0%, reclaiming ~25% of kernel time for compute.

**Approach:** `GpuLinearLayer` already has lazy weight caching infrastructure (`ensure_weights_on_gpu()`), but it was never called from `forward()`. Fix: call `ensure_weights_on_gpu()` at the start of each forward pass so weights upload once per layer, then reuse cached device buffer for all subsequent tokens.

**Implementation:** Single-line fix in `pesti-runner/src/kernel/linear.rs` — add `self.ensure_weights_on_gpu()?;` to `GpuLinearLayer::forward()`.

**Result:** H2D transfer overhead eliminated for weight tensors. Weights upload once per layer, cached on GPU for all subsequent tokens.

## Week 26: Deep Profiling and GEMM Optimization Analysis

**Completed:**
- Full CUDA kernel profiling with cudaShim instrumentation (H2D/D2H transfers, sync, compute)
- Production sequence length benchmarking (seq=256 decode steps) on jambo (RTX 3070 Ti)
- Root cause analysis of GEMM compute bottleneck

**Key Findings:**
- GEMM compute: 58.8% of kernel time (primary bottleneck)
- H2D transfers: 23.2% (secondary, but per-call approach confirmed working after OOM issues resolved)
- D2H transfers: 12.4%
- Stream sync: 5.6%

**GEMM Optimization Analysis:**
For LLM inference shapes (m=1, n=1500-6000, k=1500-4000), cuBLAS isn't optimal regardless of algorithm selection. These are tall-skinny GEMMs that cuBLAS's heuristics aren't designed for. `CUBLAS_GEMM_DEFAULT` already picks a reasonable kernel; trying ALGO0-9 won't give meaningful wins because the shape itself is the bottleneck, not the algorithm choice.

**Path forward for GEMM compute optimization:**
1. **Custom CUDA kernels** (like llama.cpp uses) — tuned specifically for m=1 LLM shapes with shared memory tiling and warp-level operations
2. **CUTLASS library integration** — NVIDIA's template-based GEMM library that can be instantiated for specific shapes

Both are significant engineering efforts. Estimated impact: 2-5x faster GEMM compute for these shapes, potentially reaching 80-150% of llama.cpp throughput.

## Completed Weeks

- **Week 26:** Deep profiling and GEMM optimization analysis — identified cuBLAS limitations for tall-skinny LLM shapes; documented CUTLASS as next optimization path
- **Week 25:** Optimization and Scale — established comparable tok/s benchmark (307.68 vs llama.cpp's 504.04), completed F16 GPU inference via cuBLAS hgemm, trait-based linear layer integration, non-matmul GPU kernels (SwiGLU/RMSNorm/RoPE/Softmax), and token embedding fix for GGUF weight loading
- **Week 23:** Long-Sequence Prefill Throughput — measured prefill speed across sequence lengths; identified attention kernel O(n²) scaling as bottleneck for long-context workloads

## Week 28: CUTLASS Integration for GEMM Compute Optimization (In Progress)

**Goal:** Replace default cuBLAS algorithm selection with shape-optimized GEMM kernels for m=1 tall-skinny LLM decode shapes. Target: 2-5x GEMM compute speedup, closing remaining gap with llama.cpp throughput.

**Approach:** Evaluate CUTLASS API surface and Rust FFI requirements. Implement shape-specific GEMM using cublasLtMatmul with heuristic algorithm selection tuned for m=1, n=1500-6000, k=1500-4000 shapes typical of autoregressive decode.

**Implementation:** 
- `pesti-runner/examples/test_cublaslt_api.rs` — cuBLASLt API surface exploration
- `pesti-runner/examples/benchmark_cublaslt_shapes.rs` — shape-specific benchmark harness for LLM GEMM patterns

**Status:** CUTLASS evaluation in progress. cudarc 0.19.10 with cublaslt feature enabled and building successfully. Next: implement actual cublasLtMatmul calls and benchmark against cuBLAS baseline.

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
*Updated: October 3, 2026 — Week 26 profiling results and GEMM optimization analysis documented*