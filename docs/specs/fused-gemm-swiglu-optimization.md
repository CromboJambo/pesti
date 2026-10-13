# Pesti GPU Optimization Spec: Fused GEMM+Activation Kernel

## Current State (as of 2026-10-13)

- **Benchmark:** `benchmark_pesti_runner.rs` (authoritative)
- **Model:** Qwen2.5-0.5B-Instruct-Q4_K_M
- **Current performance:** 16.64 tok/s (GPU, CUDA)
- **Target:** ~141 tok/s (llama.cpp CPU Q4_0 baseline)
- **Gap:** ~8.5x slower on GPU than CPU competitor

## Bottleneck Analysis (from benchmark_kernel_timing.rs)

| Kernel | Calls | Time (ms) | % of Decode |
|--------|-------|-----------|-------------|
| GEMM | 975 | 1796.746 | 74.9% |
| Attention | 936 | 598.890 | 25.0% |
| Other | - | 3.801 | 0.1% |

## Optimization Target: Fused GEMM+SwiGLU Kernel

### Why This Is High-ROI

The transformer forward pass has this pattern for FFN layers:

```
x → GEMM(W_up) → SwiGLU activation → GEMM(W_down) → x'
```

Currently, each step is a separate kernel launch:
1. `gemm_f16` → writes intermediate tensor to global memory (~4864 elements × 2 bytes = ~9.7KB per layer)
2. `swiglu_kernel` → reads intermediate, computes activation, writes result to global memory
3. `gemm_f16` → reads activation result, computes final GEMM

**Waste:** The intermediate tensor is written and immediately read back from global memory — a full round-trip that could be eliminated.

### Existing Infrastructure

The codebase already has fused kernel infrastructure:
- `FusedGemmSwigluKernel` found in 13 locations
- `set_fused_kernel`, `upload_fused_weights` functions exist
- Weight caching already implemented (`gemm_f16_with_persistent_weights`)

### Implementation Plan

#### Step 1: Enable Fused Kernel for FFN Layers

Modify `transformer/layer.rs` to use fused GEMM+SwiGLU for the up/gate projections:

```rust
// Current (separate kernels):
let h = self.gate_proj.forward(x, seq_len)?;
let v = self.up_proj.forward(x, seq_len)?;
let h = swiglu_kernel(&h, &v)?;
let out = self.down_proj.forward(&h, seq_len)?;

// Optimized (fused):
let out = self.fused_ffn_forward(x, seq_len)?;  // GEMM+SwiGLU in one kernel
```

#### Step 2: Verify Numerical Correctness

Run conformance tests comparing fused vs. separate kernel outputs:
- Load model with both configurations
- Run same prompt through both
- Compare token-by-token output (should be identical or within floating-point tolerance)

#### Step 3: Benchmark Impact

Run `benchmark_pesti_runner.rs` and compare:
- tok/s before vs. after fusion
- GEMM call count reduction (expect ~50% fewer GEMM calls for FFN layers)
- Memory traffic reduction (no intermediate tensor writes)

### Expected Performance Gain

- **Eliminates ~84 kernel launches per decode step** (16 layers × 2 FFN projections + activation)
- **Reduces global memory writes by ~9.7KB × 16 layers × 2 = ~310KB per step**
- **Estimated speedup:** 15-30% (based on similar optimizations in other LLM runtimes)

### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Fused kernel produces different results | Run conformance tests first; fall back to separate kernels if needed |
| Fused kernel is slower due to complexity | Benchmark both paths; keep separate as fallback |
| Memory layout incompatibility | Verify tensor layouts match fused kernel requirements |

### Success Criteria

- [ ] Fused kernel produces identical output to separate kernels (within FP16 tolerance)
- [ ] tok/s improvement of at least 10% (target: 20%+)
- [ ] GEMM call count reduced by ~50% for FFN layers
- [ ] No regression in attention or other kernel timings

### Files to Modify

- `pesti-runner/src/transformer/layer.rs` — switch to fused FFN forward path
- `pesti-runner/src/kernel/swiglu.rs` — may need layout adjustments
- `pesti-runner/examples/benchmark_pesti_runner.rs` — add fused kernel benchmark variant (optional)

### Reference Implementation

The fused GEMM+SwiGLU kernel pattern is well-established in other LLM runtimes:
- llama.cpp: `llama_moe_scale_experts_c` uses fused operations
- vLLM: implements fused MoE kernels for similar patterns
- PyTorch: `torch.nn.functional.silu(x * y)` can be fused with matmul via custom autograd

### Next Steps After Fusion

If fusion achieves <10% improvement, investigate:
1. Attention kernel optimization (currently 25% of decode time)
2. KV cache layout optimization
3. Batched GEMM for multiple tokens

## Benchmark Commands

```bash
# Authoritative benchmark
cargo run --release --features cuda --example benchmark_pesti_runner

# Per-kernel timing (optimization tool)
cargo run --release --features cuda --example benchmark_kernel_timing

# Conformance test
cargo run --release --features cuda --example conformance_test
```

## Notes for Fresh Session

- Working directory: `/home/crombo/projects/active/pesti`
- Build with `--features cuda` (not `cuda-shim`)
- Model path: `conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf` (symlink)
- GPU: RTX 3070 Ti, CUDA compute capability 8.6 (sm_86)
- Nsight Compute not installed; use benchmark_kernel_timing.rs for profiling
