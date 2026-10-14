# Pesti GPU Optimization Spec: Fused GEMM+SwiGLU Kernel (FAILED)

## Status: Experiment Complete — Regression Observed

**Result:** The fused GEMM+SwiGLU kernel caused a ~3x performance regression (16 tok/s → 8.35 tok/s).
Code has been removed. Do not retry this approach without understanding the root cause of the regression.

## What Was Tried

- Implemented `FusedGemmSwigluKernel` combining GEMM + SwiGLU activation into a single CUDA kernel
- Eliminated intermediate tensor writes to global memory between GEMM and activation
- Used existing fused kernel infrastructure (`set_fused_kernel`, `upload_fused_weights`)

## Why It Failed

The regression likely came from one or more of:
1. Kernel launch overhead vs. compute savings for small LLM shapes
2. Memory access pattern changes that hurt GPU occupancy
3. Inefficient PTX code for the fused operation
4. Loss of cuBLAS optimization advantages when fusing manually

## Lessons Learned

- Fusing operations is not always faster on GPUs — measure before assuming
- cuBLAS's highly optimized GEMM kernels may outperform manual fusion for typical LLM shapes
- The intermediate tensor write/read penalty was overestimated vs. actual kernel launch overhead

## Files Removed

- `pesti-runner/src/kernel/fused_gemm_swiGLU.rs`
- `pesti-runner/src/kernel/simple_gemm_test.rs`
- `pesti-runner/examples/test_fused_kernel.rs`
- `pesti-runner/examples/test_fused_cpu.rs`
- `pesti-runner/src/kernel/ptx/fused_gemm_swiGLU.ptx`

## Next Steps for Optimization

Focus on other approaches:
1. Attention kernel optimization (25% of decode time)
2. KV cache layout optimization
3. Batched GEMM for multiple tokens
4. Better cuBLAS algorithm selection via shape-based caching
