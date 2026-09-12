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
| Examples don't compile after API changes | Recurring | Developer experience, not runtime |

## Upcoming Work

### Week 22: Debt and Spikes
- [ ] Fix remaining clippy warnings (unused vars in stub code, missing Safety docs)
- [ ] Spike: batched generation for parallel prompts

### Week 23+: Optimization and Scale
- [ ] Establish comparable tok/s benchmark against llama.cpp on same model/hardware
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