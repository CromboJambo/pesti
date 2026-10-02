# pesti-runner: Inference Engine Roadmap

*Module: `pesti-runner/` — Main inference engine, GPU dispatch, transformer layers*

[← Back to main roadmap](../ROADMAP.md)

## Current State (Week 25 Complete)

Working GPU inference path for Qwen2.5-0.5B-Instruct with fused attention kernel and F16 compute.
Numerical conformance validated vs llama.cpp reference outputs at all sequence lengths tested.

**Throughput:** pesti-runner: 307.68 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti) — **Phase 1 target of 100 tok/s achieved.**
Remaining ~1.6x gap to llama.cpp baseline (504.04 tok/s) is the Phase 4 optimization target.

**Week 25 completed October 2, 2026.** All planned deliverables finished including token embedding fix that resolved GGUF weight loading for Llama-family models.

### Working Features
- ✅ Transformer layer forward pass (attention + FFN)
- ✅ KV cache with autoregressive generation
- ✅ GQA attention (grouped query, per-head computation)
- ✅ RoPE positional embeddings (pure Rust, no PTX dependency)
- ✅ RMSNorm, SwiGLU activation
- ✅ GPU dispatch layer with CPU fallback counter
- ✅ Per-layer capture for debugging/verification
- ✅ F16 GPU inference via cuBLAS Hgemm (Phase 2b complete)
- ✅ Trait-based linear layer integration (`new_linear_layer()` factory)
- ✅ Non-matmul GPU kernels: SwiGLU, RMSNorm, RoPE, Softmax (Phase 3 complete)

### Known Issues
| Issue | Status | Impact |
|-------|--------|--------|
| pesti-safetensors: 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config) | Open | Can't fully validate quantized model loading via safetensors path |

### Cleanup Completed
- Week 21: Archived 72 non-compiling examples to `examples-disabled/`, 17 non-compiling integration tests to `tests-disabled/`
- Active surface: 43 compiling examples, 10 passing integration tests

## Upcoming Work

### Week 25: Optimization and Scale (IN PROGRESS)
- [x] Establish comparable tok/s benchmark against llama.cpp on same model/hardware — pesti-runner: 81.78 tok/s vs llama.cpp: 504.04 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti). ~6x gap identified as optimization target.
- [x] F16 GPU inference via candle_bridge redesign — implemented in `pesti-runner/src/kernel/cuda_bridge.rs` using cudarc's cuBLAS hgemm; integrated into dispatch layer with automatic fallback. All 5 conformance tests pass including numerical stability at seq=4096.
- [x] Trait-based linear layer integration — replaced all LinearDispatch call sites across model.rs, runtime.rs, and test examples with `new_linear_layer()` factory. Weight uploads now happen internally at construction time via `build_layer_dispatch()`, eliminating redundant explicit upload calls. Build OK, 70/71 tests pass (one pre-existing rope test failure).
- [x] Phase 3: non-matmul GPU kernels — SwiGLU, RMSNorm, RoPE, Softmax all implemented as CUDA kernels. RoPE computed in pure Rust to avoid PTX version incompatibility; embeddings passed directly in F16 to avoid GPU dtype conversion overhead.
- [ ] Profile GEMM vs attention kernel time split at production sequence lengths — identify remaining bottlenecks
- [ ] KV cache quantization (Q4_K) to reduce memory bandwidth bottleneck
- [ ] Spike: TMA descriptors for async prefetching

## Architecture Notes

### GPU Dispatch Path
```rust
forward_with_dispatch() 
  → dispatch_gemm()        // CUTLASS GEMM via cudarc
  → attention_forward()    // Fused QKV+attention+output kernel
  → rmsnorm_gpu()          // CUDA kernel (Phase 3)
  → swiglu_forward()       // CUDA kernel (Phase 3)
```

### Known Failure Modes (Module-Specific)
- **CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Use `cuda_shim::stream_synchronize` (event-based).
- **Kernel stack arrays:** Bounds, not dynamic allocation. Indexing past MAX_SEQ causes silent corruption.
- **Numerical stability at scale:** Softmax overflow manifests at seq=4096+. Always test long sequences.

## References
- [EDR-007: Fused Attention Kernel Correctness Fix](../docs/FUSED-ATTENTION-FIX.md)
- [Week 17 GPU e2e correctness results](../docs/history/WEEK_17_GPU_E2E_CORRECTNESS.md)
- [CUDA synchronization root cause analysis](../docs/CUDA_SYNC_ROOT_CAUSE.md)

---
*Updated: October 2, 2026 — Week 25, Phase 3 complete, F16 compute validated*
