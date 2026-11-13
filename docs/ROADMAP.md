# PESTI Detailed Phase Roadmap

*Detailed week-by-week work, optimization analysis, and phase deliverables.*

[← Back to main roadmap](../ROADMAP.md) | [pesti-runner module roadmap](../pesti-runner/ROADMAP.md)

**Goal:** Portable execution substrate for transformer inference — stable Rust, GPU-first via CUDA dispatch, validated against llama.cpp reference outputs.

## Current State (Week 28 Complete)

Working GPU inference path for Qwen2.5-0.5B-Instruct with production profiling complete:
- Fused attention kernel passes numerical conformance vs llama.cpp
- KV cache autoregressive validation suite
- Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- Long sequence support verified to seq=4096
- GPU weight caching via lazy upload (Week 27)
- cuBLASLt/CUTLASS integration complete (Week 28)

**Current throughput:** pesti-runner achieves **~73 tok/s** on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti) after persistent weight caching + GQA optimization (Week 29). Post-refactor baseline was ~16 tok/s; optimizations achieved 4.5x speedup.

**Week 29 finding:** Persistent GPU weight caching (upload once at construction time) combined with GQA optimization (reduced redundant K/V cache transfers) delivered major performance improvement. Architecture simplification completed — removed dead inference engine variants, unused kernel modules, and obsolete examples.

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

## Week 29: Persistent Weight Caching + GQA Optimization (Completed)

**Goal:** Improve throughput after architecture simplification by eliminating redundant GPU memory transfers.

**Approach:** Upload layer weights to GPU once at construction time instead of per forward pass, and optimize GQA attention to reduce redundant K/V cache transfers across query heads.

**Implementation:**
- Modified `GpuLinearLayer::new()` to upload weights immediately at construction
- Simplified weight cache by removing LRU eviction overhead (weights stay on GPU for lifetime)
- Optimized GQA attention kernel to share K/V across query heads efficiently
- Used candle_bridge cached GEMM infrastructure for persistent weight caching

**Result:** ~16 tok/s → **~73 tok/s** (4.5x speedup) on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti).

## Week 28: CUTLASS Integration for GEMM Compute Optimization (Completed)

**Goal:** Replace default cuBLAS algorithm selection with shape-optimized GEMM kernels for m=1 tall-skinny LLM decode shapes. Target: 2-5x GEMM compute speedup, closing remaining gap with llama.cpp throughput.

**Approach:** Implemented cublasLtMatmul path via cudarc safe API bindings with heuristic-based algorithm selection tuned for LLM decode patterns.

**Implementation:**
- `pesti-runner/src/kernel/cuda_bridge.rs` — added `gemm_f16_cublaslt()` function using cuBLASLt's CUTLASS kernel dispatch
- `pesti-runner/tests/test_cublaslt_gemm.rs` — integration test suite (correctness + performance benchmarks)

**Results:**
| Shape | cuBLAS | cuBLASLt | Speedup |
|-------|--------|----------|---------|
| m=1, n=4096, k=4096 | 0.32ms | 0.78ms | **0.41x** (slower!) |
| m=1, n=2048, k=2048 | 0.09ms | 0.56ms | **0.16x** (slower!) |
| m=32, n=512, k=1024 | 0.03ms | 0.55ms | **0.06x** (slower!) |

**Key finding:** cuBLASLt CUTLASS kernels are slower than plain cuBLAS for LLM decode shapes due to `cublasLtMatmulAlgoGetHeuristic` overhead (~0.5ms) dominating small shape calls. Plain cuBLAS's internal kernel selection is already optimal for these patterns. GEMM compute path is at practical optimum via plain cuBLAS + shape-based algorithm caching (Week 26).

**Conclusion:** CUTLASS integration complete; no further GEMM compute optimization expected from this path. Next optimization targets: KV cache quantization, speculative decoding, or custom CUDA kernels for non-GEMM operations.

## Week 30: State Persistence + Optimization Path

**Completed:** PESTI State persistence layer (`pesti-state` crate) — SQLite-backed tracking of models, adapters, and operation audit logs. Content-addressed model registry enables revalidation triggers when base models change. Tracked weight-space sessions with full provenance. CLI query interface (`pesti-state list-models|list-adapters|show-adapter|session-log`). See `specs/PESTI_STATE_SPEC.md`.

**Optimization (ongoing):** Based on Week 29 findings, persistent weight caching + GQA optimization delivered major improvement (4.5x). Remaining throughput gap (~73 vs 504 tok/s) stems from non-GEMM operations and memory bandwidth. Candidate optimization paths:
1. **KV cache quantization** — reduce KV cache memory bandwidth pressure (highest potential impact)
2. **Speculative decoding** — increase effective tokens/sec without reducing per-token latency
3. **Custom CUDA kernels for non-GEMM ops** — attention, RMSNorm, RoPE may have optimization headroom
4. **Larger model benchmarking** — test on 3B+ models where GPU acceleration matters more

Decision pending further profiling to identify actual bottleneck vs llama.cpp reference implementation.

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

## Complexity Architecture (rustc-inspired query system)

**Spec:** `specs/PESTI_COMPLEXITY_ARCHITECTURE_SPEC.md` (committed `b93b6b2`)
**Inspiration:** rustc's query system — memoized, constraint-based decision making instead of scattered procedural branching.

**Phase 1 — KernelSelector module (Complete, `c207562`):**
- New `kernel/selector.rs`: structured kernel selection via `KernelRequest` → `KernelId`
- `KernelRegistry`: static registry built from runtime detection (SM version, feature flags)
- Constraint system: `ArchConstraint`, `DTypeConstraint`, `ShapeConstraint`
- `PerfClass` ranking for deterministic best-kernel choice
- Guaranteed CPU fallback — selection never fails
- 9 unit tests covering selection, memoization, constraints, and fallback

**Phase 2 — Dispatch integration (In progress):**
- ✅ `dispatch_linear`: three-way branching replaced with selector query (`ca18a9e`)
- Extracted shared helpers: `transpose_weights()`, `add_bias()`, `cuda_sm()`
- Eliminated duplicated weight-transpose code (was in 3 branches, now 1 function)
- ⬜ `dispatch_attention`: complex CUDA path selection → selector query
- ⬜ `dispatch_gemm`: simple arch-based selection → selector query

**Impact:** Adding a new kernel backend = one registry entry + one match arm, not modifying every dispatch method. The combinatorial explosion of (arch × dtype × shape) branches is now a data-driven lookup.

## Type-Safe Masking & Tensor Layer Separation (November 2026)

**Specs:** `specs/SOFTMASK_TYPE_SPEC.md`, `specs/TENSOR_LAYER_SEPARATION_SPEC.md`

### Softmask: Type-Safe Attention Masking (Complete)

Replaced ad-hoc boolean masking with a type-safe `Softmask` abstraction that unifies causal, sliding-window, and custom eligibility patterns across all three CPU attention implementations.

- **Phase 1** (`63920d2`): New `softmask.rs` module — `Softmask` trait with `is_eligible()`, `CausalMask`, `SlidingWindowMask`, `CompositeMask` implementations; empty-segment policy (debug panic / release zeros)
- **Phase 2** (`f67cb31`): Migrated `cpu_optimized` attention path to `Softmask`; cross-implementation parity tests locking masking semantics
- **Phase 3**: Migrated remaining CPU paths; 6 masking parity tests; ~13–16% release-mode overhead documented as acceptable trade-off for type safety

**Key principle:** "Don't change the measurements to force the system into alignment. Preserve the measurements and represent the constraints separately." Softmask is now the authority on eligibility — future attention implementations must not recreate causal-mask logic independently.

### Tensor Layer Separation (Phase 2 Complete)

Layered weight storage architecture separating tensor bytes (Layer 0), structure/metadata (Layer 1), and system factors (Layer 2).

- **Phase 1** (`d183c0a`): `weight_store.rs` foundation — `CoreTensor`, `TensorInfo`, `TensorStructure`, `SafetensorsFile`, `WeightStore`; memmap2 integration; zero-copy tensor access
- **Phase 2** (this commit): Layer-2 `SystemFactors` — RoPE config, normalization stats, attention pattern, quantization scheme; `InferenceReader` trait extending `TensorReader`; dual-source config parsing (HuggingFace `config.json` + safetensors `__metadata__`); computed structure fields (`num_layers`, `hidden_size`)
- **Phase 3** (planned): Migrate `LlamaModel::load_safetensors()` to use `WeightStore::load_inference()`; eliminate redundant config parsing in model construction

**Design:** Strictly additive — `LlamaModel::load_safetensors()` unchanged. `WeightStore::load_minimal()` preserves Phase 1 behavior (Layers 0+1 only); `load_inference()` adds Layer-2 system factors for full inference readiness.

**Failure Modes (Reference)**

When heading toward these patterns, expect trouble:

**CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Always sync via `cuda_shim::stream_synchronize` (event-based) or `context_synchronize` for legacy-default-stream work. This will silently produce wrong results, not crash.

**Kernel stack arrays:** Stack array sizes in kernels are bounds, not dynamic allocation. Indexing past `MAX_SEQ = 512` causes silent corruption, not bounds check failure.

**Numerical stability at scale:** Softmax overflow only manifests at long sequences (seq=4096+). Short-sequence tests pass; production fails. Always test with seq=4096+.

**Examples rot fast:** After API changes, examples break before tests do. Treat example compilation as part of the test suite, not documentation.

**GPU memory allocation is cheap:** Don't over-optimize by avoiding `cudaMalloc`. The cost is in synchronization and kernel launches, not allocation.

---
*Updated: November 12, 2026 — Softmask complete (Phases 1-3), Tensor Layer Separation Phase 2 complete*

## Llama.cpp Replacement Track (Parallel)

Separate from the GPU substrate work above, this track delivers a drop-in replacement for llama.cpp with Unsloth-style hardware-aware UX.

**Completed:**
- ✅ Phase 1: `pesti` CLI binary — llama.cpp-compatible argument parsing (-m/--model, -p/--prompt, -n, --temp, --top-k, --top-p, --repeat-penalty, --threads, --ctx-size, --gpu-layers)
- ✅ Phase 2: `pesti-studio` model management — hardware detection (CPU/RAM/GPU), local model listing, HF Hub search and download via reqwest API integration

**Phase 3 Complete (October 10, 2026):** Auto-Tuning Engine
- Benchmarking infrastructure in `pesti-studio/src/benchmark.rs`
- `pesti-studio benchmark <model>` command measures tok/s across n_ctx/n_batch/thread configurations
- Hardware-aware config generation based on VRAM vs model size heuristics
- Tested on ftw3 (RTX 3070 Ti, 8GB VRAM) and jambo (RTX 4070 Ti SUPER, 16GB VRAM)

**Phase 4 Complete (October 10, 2026):** Enhanced Server Mode
- pesti-server v2 with model hot-swapping (`/v1/models/load`, `/v1/models/unload`)
- Connection pooling for concurrent requests (configurable via `PESTI_POOL_SIZE`)
- SSE streaming for chat completions (`stream: true`)
- OpenAI-compatible API surface at `/v1/chat/completions` and `/v1/models`

**Llama.cpp replacement track complete.** pesti now provides:
- Drop-in CLI replacement (`pesti` binary with llama.cpp-compatible args)
- Hardware-aware model management (`pesti-studio detect/run/models/download/search/benchmark`)
- Auto-tuning engine that benchmarks configurations and recommends optimal settings
- Production server with hot-swapping, pooling, and streaming

## LLM Runner, Tuning & Safetensors Track (Planned)

See [docs/SPEC_LLM_RUNNER_TRAINING.md](SPEC_LLM_RUNNER_TRAINING.md) for the complete spec covering:

**Component 1: LLM Runner (Inference)** — Complete
- pesti CLI binary with llama.cpp-compatible interface
- pesti-server v2 with OpenAI-compatible API, hot-swapping, streaming
- Performance: **~73 tok/s** on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti) after persistent weight caching + GQA optimization

**Component 2: Model Tuning & Training** — Planned (Weeks 1-8)
- QLoRA fine-tuning pipeline with hardware-aware configuration
- Dataset loading, training loop, checkpointing, adapter saving
- Target: within 2x of Unsloth Python implementation on same hardware

**Component 3: Safetensors Weight Management** — In Progress
- pesti-safetensors crate with loading/saving/inspection capabilities
- VRAM estimation engine for inference vs training workloads
- Support for all major quantization formats (Q4_K, Q5_K, Q6_K)

---
*Updated: October 10, 2026 — LLM Runner & Training spec added*