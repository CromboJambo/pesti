# PESTI Top-Down Refactor: Complete Inference Stack

**Status:** Proposed  
**Date:** 2026-09-17  
**Motivation:** GPU GEMM kernels are proven (73.4 tok/s on ftw3), but the actual transformer inference path uses separate CPU F32 loops for every non-matmul op. The matmul dispatch layer exists but isn't wired into model execution. This spec defines the refactor to unify these into a complete GPU-accelerated inference stack.

## Current State: Two Parallel Paths

```
GPU GEMM path (proven, not used by inference):
  F32 → f16 convert → dispatch_gemm() → CudaGemmKernel → WGMMA/tcgen05/mma.sync PTX → f16→F32 convert

Transformer inference path (used, but CPU-only):
  model.forward() → layer.forward() → linear.forward() [hand-written rayon matmul]
                   → rms_norm.forward() [CPU loop]
                   → rope.apply() [CPU loop]
                   → attention scores/softmax [CPU nested loops]
                   → swiglu() [CPU loop]
```

The GPU path is validated in tests and examples. The inference path works but never touches GPU hardware despite `--features cuda`.

## Phase 1: Fused Attention Kernel (Week 23-24)

**Goal:** Single CUDA kernel that computes attention scores + softmax + V-weighted sum, replacing the nested CPU loops in `layer.rs:forward_with_cache()`.

**Why first:** Attention is O(n²) and dominates decode latency. The Q/K/V projections are matmuls (Phase 2), but the score computation and softmax have no GPU path at all right now.

**Design:**
- Input: Q head (on device), K cache slice, V cache slice, seq_len, head_dim
- Output: Attention output for this head
- RoPE applied to Q before kernel launch (separate small kernel or CPU)
- Kernel handles: score computation → softmax → V-weighted sum in one pass

**Deliverables:**
1. `attention_kernel.cu` — fused attention for single query against KV cache
2. Rust binding via `cuda_shim::launch_kernel`
3. Conformance test vs existing CPU attention output (use `comprehensive_attention_conformance` as template)
4. Integration into `Attention::forward_with_cache()` path

**Success metric:** Attention kernel passes conformance at <1e-4 error vs CPU reference.

## Phase 2: GEMM Integration Into Inference Path (Week 25-26)

**Goal:** Replace `linear.rs:forward()`'s hand-written rayon matmul with the existing GPU GEMM dispatch when CUDA is available.

**Current blocker:** Linear layer converts F32→F16, calls GEMM, converts back to F32 on every single call. This is correct but wasteful — intermediate results should stay F16 on device as long as possible.

**Design (two sub-phases):**

### 2a: Direct Integration (Week 25)
- Modify `linear.rs` to accept an optional `GemmKernel` trait object
- When CUDA available, pass `CudaGemmKernel` instance
- Keep F32→F16→F32 conversion at layer boundary for now (correctness first)
- Test: end-to-end inference produces identical output to CPU path

### 2b: Device-Resident Tensors (Week 26)
- Introduce `DeviceTensor` wrapper that tracks location (host vs device) and dtype
- RMSNorm, RoPE, SwiGLU receive/return `DeviceTensor<f16>` when on GPU path
- Conversion happens only at model input/output boundaries
- Requires Phase 3 kernels for non-matmul ops

**Success metric:** End-to-end inference uses GPU GEMM for all weight projections (visible in profiler).

## Phase 3: Non-Matmul GPU Kernels (Week 27-29)

**Goal:** GPU implementations of the remaining CPU-only ops.

**Priority order (by performance impact):**

1. **SwiGLU activation kernel** — element-wise, trivially parallelizable
   - Input: gate tensor [batch, intermediate_dim], up tensor [batch, intermediate_dim]
   - Output: silu(gate) * up
   - Can be fused with W2 matmul epilogue (advanced)

2. **RMSNorm kernel** — row-wise normalization
   - Two passes: compute RMS, then normalize and scale by weight
   - Weight tensor stays on device as persistent parameter

3. **RoPE application kernel** — applied to Q and K heads
   - Can be fused into attention kernel (Phase 1) or done separately
   - Precompute rotation angles on host, pass as constant memory

4. **Softmax kernel** — if not fully absorbed into fused attention
   - Row-wise max, exp, sum, divide
   - Standard pattern, well-documented in CUDA samples

**Deliverables per kernel:**
- PTX/CUDA source file
- Rust binding with proper stride/dimension handling
- Unit test vs CPU reference implementation
- Integration into transformer layer forward pass

## Phase 4: KV Cache Quantization (Week 30)

**Goal:** Store KV cache in Q4_K format on device, dequantize within attention kernel.

**Prerequisite:** Phase 1 (fused attention kernel must handle dequantized K/V access).

**Design:**
- KV cache stores quantized blocks instead of F32 values
- Attention kernel dequantizes K rows on-the-fly during score computation
- V dequantization integrated into weighted sum step
- Memory savings: 8x (F32→Q4_K) for cache, dominant at long context

**Success metric:** Long-sequence inference uses <1/8 the KV cache memory with <1% accuracy degradation.

## Phase 5: Execution Graph & Kernel Fusion (Week 31+)

**Goal:** Fuse consecutive operations into single kernel launches to eliminate intermediate memory round-trips.

**Opportunities:**
- RMSNorm + Q/K/V projection matmul (pre-fusion)
- Matmul + SwiGLU activation (epilogue fusion)
- Attention score computation + softmax + V-weighted sum (already Phase 1)

**Approach:** Investigate whether hand-written fused PTX kernels outperform cuBLASLt with epilogue at these tensor shapes. Small M values (batch=1, seq=1 for decode) may favor custom fusion over library calls.

## What This Replaces

The current roadmap items are subsumed by this spec:
- Week 23 KV Cache Quantization → Phase 4
- Week 24 F16 GPU Inference Redesign → Phases 2b + 3
- Week 25 Long-Sequence Benchmarking → happens naturally as each phase completes

The "refactor intermediate op layering" item added to the roadmap is Phase 2b's device-resident tensor design.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PTX version drift between dev and target machines | Kernels fail to launch | Build PTX on target (ftw3) during CI; pin CUDA toolkit version in docs |
| Fused kernels harder to debug than separate ops | Development velocity drops | Maintain CPU reference path for conformance testing at every phase |
| Device memory pressure from persistent tensors | OOM on smaller GPUs | Profile memory usage per phase; implement device buffer pooling if needed |
| RoPE precision loss in F16 | Degraded quality at long context | Keep RoPE angles in F32, apply in F32, convert result to F16 |

## Exit Criteria for Full Refactor

End-to-end inference on ftw3 with RTX 3070 Ti achieves:
- >150 tok/s on TinyLlama Q8 (current: 73.4 tok/s)
- All transformer ops executing on GPU (verified via profiler)
- KV cache in quantized format for sequences >256 tokens
- Conformance within 1e-4 of CPU reference across all test prompts
