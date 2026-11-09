# pesti-runner: Inference Engine Roadmap

*Module: `pesti-runner/` — Main inference engine, GPU dispatch, transformer layers*

[← Back to main roadmap](../ROADMAP.md)

## Current State (Week 28 Complete)

Working GPU inference path for Qwen2.5-0.5B-Instruct with fused attention kernel and F16 compute.
Numerical conformance validated vs llama.cpp reference outputs at all sequence lengths tested.

**Throughput:** pesti-runner: ~307 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti) — Phase 1 target of 100 tok/s achieved.
Remaining ~1.6x gap to llama.cpp baseline (504.04 tok/s) is the optimization target.

### Working Features
- ✅ Transformer layer forward pass (attention + FFN)
- ✅ KV cache with autoregressive generation
- ✅ GQA attention (grouped query, per-head computation)
- ✅ RoPE positional embeddings (pure Rust, no PTX dependency)
- ✅ RMSNorm, SwiGLU activation
- ✅ GPU dispatch layer with CPU fallback counter
- ✅ Per-layer capture for debugging/verification
- ✅ F16 GPU inference via cuBLAS Hgemm
- ✅ Trait-based linear layer integration (`new_linear_layer()` factory)
- ✅ Non-matmul GPU kernels: SwiGLU, RMSNorm, RoPE, Softmax
- ✅ GPU weight caching via lazy upload

### Known Issues
| Issue | Status | Impact |
|-------|--------|--------|
| pesti-safetensors: 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config) | Open | Can't fully validate quantized model loading via safetensors path |

## Upcoming Work

### Optimization Path Decision (Week 29)
GEMM compute at practical optimum via plain cuBLAS. Evaluating optimization targets:
- [ ] Profile GEMM vs attention kernel time split at production sequence lengths
- [ ] KV cache quantization (Q4_K) to reduce memory bandwidth bottleneck
- [ ] Spike: TMA descriptors for async prefetching

## Architecture Notes

### GPU Dispatch Path
```rust
forward_with_dispatch() 
  → dispatch_gemm()        // CUTLASS GEMM via cudarc
  → attention_forward()    // Fused QKV+attention+output kernel
  → rmsnorm_gpu()          // CUDA kernel
  → swiglu_forward()       // CUDA kernel
```

### Known Failure Modes (Module-Specific)
- **CUDA synchronization:** `cuStreamSynchronize` is a no-op under `CU_CTX_SCHED_AUTO`. Use `cuda_shim::stream_synchronize` (event-based).
- **Kernel stack arrays:** Bounds, not dynamic allocation. Indexing past MAX_SEQ causes silent corruption.
- **Numerical stability at scale:** Softmax overflow manifests at seq=4096+. Always test long sequences.

## References
- [Main roadmap](../ROADMAP.md) — Project-level goals and completed work
- [Detailed phase work](../docs/ROADMAP.md) — Week-by-week optimization analysis
- [Architecture refactor spec](../docs/REFACTOR_SPEC.md) — Phase deliverables and conformance requirements
- [EDR-007: Fused Attention Kernel Correctness Fix](../docs/FUSED-ATTENTION-FIX.md)
- [Week 17 GPU e2e correctness results](../docs/history/WEEK_17_GPU_E2E_CORRECTNESS.md)
- [CUDA synchronization root cause analysis](../docs/CUDA_SYNC_ROOT_CAUSE.md)

---
*Updated: October 12, 2026 — Module roadmap restructured with links to main roadmap*