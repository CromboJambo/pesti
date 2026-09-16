# PESTI GPU Optimization Phases (Updated)

**Status:** Active — Phase 6 in progress  
**Last updated:** September 16, 2026  
**Reference architecture:** mistral.rs + cudarc safe API patterns

## Current State

| Metric | Value |
|--------|-------|
| Hardware | RTX 4070 Ti SUPER (AD103, Ada Lovelace) |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M |
| CPU baseline | ~83 tok/s (all 5 CPU phases complete) |
| cuBLAS bridge | Functional but not yet conformance-tested |

## Completed Phases

### Phase 1: Tiled GEMM (~76 tok/s) ✓
- CPU tiled matrix multiplication for better cache utilization

### Phase 2: Fused Dequantize+GEMM (~75-77 tok/s) ✓
- Combined dequantization with matrix multiply to reduce memory traffic

### Phase 3: Optimized Attention Kernel (~76 tok/s) ✓
- Logsumexp trick for numerical stability at long sequences
- Single-kernel softmax + fused multiply-add

### Phase 4: AVX2 SIMD Dequantization (77.34 tok/s) ✓
- Nightly Rust std::simd intrinsics for vectorized dequantize
- Commit: `0d915f4`

### Phase 5: F16 KV Cache (+6.7% tok/s) ✓
- Migrated KV cache from F32 to F16 device buffers
- Commit: `1ff852a`

---

## Phase 6: cudarc API Conformance (IN PROGRESS)

**Goal:** Align pesti-runner's CUDA usage with cudarc's recommended patterns before optimizing. Measure first, optimize second.

### Why now?
We built the cuBLAS bridge against cuda-oxide's API and then migrated to cudarc via a shim layer. The shim works but doesn't follow cudarc's safe API patterns — we're mixing safe/result/sys levels, creating redundant abstractions, and not using cudarc's provided utilities. We need to measure the baseline *before* optimizing so we know what our optimizations actually deliver.

### Conformance changes (in progress)
1. **Event reuse pattern** — replace per-call CudaEvent create/destroy with reusable static event ✓
2. **Remove custom IntoResult trait** — cudarc provides `.result()` directly on CUresult ✓
3. **Evaluate CudaSlice<T>** for KV cache buffers (better safety, same performance)
4. **Stream API alignment** — verify we're using cudarc's stream patterns correctly

### What to keep custom
- PTX kernel loading at runtime (domain-specific optimization)
- Fused attention kernel (no cudarc equivalent)
- KV cache management logic (application-level)

### Success Criteria
- [ ] All CUDA operations use cudarc safe API where available
- [ ] No redundant wrapper traits or types
- [ ] Baseline tok/s measured with conformance-compliant code
- [ ] Conformance tests pass vs llama.cpp reference

---

## Phase 7: WGMMA Tensor Core GEMM Kernel (Weeks 25-26)

**Goal:** Replace cuBLAS with custom WGMMA kernel for Ada Lovelace tensor core utilization.

### Why WGMMA?
Ada Lovelace has WGMMA instructions operating on 16x16x32 tiles with F16 inputs and F32 accumulation — exactly what LLM inference needs.

### Implementation
- Kernel: `pesti-runner/src/kernel/ptx/wgmma_gemm.cu`
- NVRTC compilation via cudarc's nvrtc bindings
- Replace cuBLAS calls in dispatch layer where dimensions align (multiples of 16)

### Success Criteria
- [ ] Matches cuBLAS Hgemm output within F16 tolerance (1e-3)
- [ ] Achieves >2x throughput improvement over cuBLAS path
- [ ] Handles non-aligned dimensions via padding or fallback

---

## Phase 8: Fused Attention with WGMMA (Weeks 27-28)

**Goal:** Replace separate QK^T + softmax + PV kernels with single fused kernel using WGMMA.

### Implementation
- Kernel: `pesti-runner/src/kernel/ptx/fused_attention.cu`
- Compute softmax(Q @ K^T / sqrt(d)) @ V in one kernel launch
- Use logsumexp trick for numerical stability at long sequences

### Success Criteria
- [ ] Matches separate kernel computation within tolerance
- [ ] Reduces attention memory bandwidth by >50%
- [ ] Conformance tests pass at seq=4096+

---

## Phase 9: Quantized Kernels (Q4_K_M) for GPU (Weeks 29-30)

**Goal:** Port Q4_K_M dequantization + GEMM fusion to GPU.

### Implementation
- Kernel: `pesti-runner/src/kernel/ptx/q4k_decode.cu`
- Decode Q4_K_M blocks on-the-fly during GEMM in registers
- WGMMA operates on decoded F16 tiles

### Success Criteria
- [ ] Correctly decodes and computes against CPU reference
- [ ] Achieves >3x throughput improvement over F16 path
- [ ] Handles all Q4_K_M edge cases

---

## Phase 10: Speculative Decoding & Batch Inference (Weeks 31-32)

**Goal:** Leverage GPU parallelism for speculative decoding and batched generation.

### Success Criteria
- [ ] >1.5x effective throughput improvement on typical prompts
- [ ] Batch inference scales linearly up to GPU memory limits

---

## Target Performance Trajectory

| Phase | Target tok/s | Improvement vs CPU |
|-------|-------------|-------------------|
| CPU baseline (Phase 5) | ~83 | — |
| Phase 6 (cudarc conformance) | ~80-90 | Baseline measurement |
| Phase 7 (WGMMA GEMM) | >200 | 2.4x |
| Phase 8 (Fused attention) | >500 | 6x |
| Phase 9 (Q4_K_M GPU) | >1500 | 18x |
| Phase 10 (Speculative) | >2000 effective | 24x |

---

## Key Design Decisions

### cudarc vs Custom FFI
- Use cudarc's safe API for standard operations (memory, streams, cuBLAS)
- Keep custom PTX compilation and kernel launching for domain-specific ops
- Measure at each phase — don't assume cudarc patterns are faster, just safer/cleaner

### Error Handling
```rust
// Use cudarc's Result-based error handling
let result = cublas.gemm(&config)?;  // Returns Result<T, DriverError>
```

### Testing Strategy
1. Unit tests: Each kernel has CPU reference for comparison
2. Integration tests: Full model forward pass vs llama.cpp output
3. Performance benchmarks: tok/s measured after each phase
4. Numerical conformance: seq=4096+ tests for overflow/precision issues