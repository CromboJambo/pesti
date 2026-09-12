# pesti-runner: Inference Engine Roadmap

*Module: `pesti-runner/` — Main inference engine, GPU dispatch, transformer layers*

[← Back to main roadmap](../ROADMAP.md)

## Current State (Week 21)

Working GPU inference path for Qwen2.5-0.5B-Instruct with fused attention kernel.
Numerical conformance validated vs llama.cpp reference outputs.

### Working Features
- ✅ Transformer layer forward pass (attention + FFN)
- ✅ KV cache with autoregressive generation
- ✅ GQA attention (grouped query, per-head computation)
- ✅ RoPE positional embeddings
- ✅ RMSNorm, SwiGLU activation
- ✅ GPU dispatch layer with CPU fallback counter
- ✅ Per-layer capture for debugging/verification

### Known Issues
| Issue | Status | Impact |
|-------|--------|--------|
| 4 failing pesti-safetensors tests (Q4_K/Q5_K/Q6_K dequant) | Open | Can't fully validate quantized model loading |

### Cleanup Completed (Week 21)
- Archived 72 non-compiling examples to `examples-disabled/` (old probes, debug dumps, deprecated API benchmarks)
- Archived 17 non-compiling integration tests to `tests-disabled/` (CUDA feature gating issues)
- Active surface: 43 compiling examples, 10 passing integration tests

## Upcoming Work

### Week 22: Debt and Spikes ✅ COMPLETE
- [x] Fix remaining clippy warnings (unused vars in stub code, missing Safety docs) — down from 193 to 136 warnings
- [x] Spike: batched generation for parallel prompts — ran on ftw3 (RTX 3070 Ti), measured ~1.0x speedup at seq=128 with Qwen2.5-0.5B-Instruct. Expected: short sequences don't benefit from batching; value appears at production sequence lengths (>512).

### Week 23: Optimization and Scale
- [x] Establish comparable tok/s benchmark against llama.cpp on same model/hardware — pesti-runner: 81.78 tok/s vs llama.cpp: 504.04 tok/s (Qwen2.5-0.5B-Instruct-Q4_K_M, RTX 3070 Ti). ~6x gap identified as optimization target.
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
- [EDR-007: Fused Attention Kernel Correctness Fix](../docs/FUSED-ATTENTION-FIX.md)
- [Week 17 GPU e2e correctness results](../docs/history/WEEK_17_GPU_E2E_CORRECTNESS.md)
- [CUDA synchronization root cause analysis](../docs/CUDA_SYNC_ROOT_CAUSE.md)

---
*Updated: September 11, 2026*