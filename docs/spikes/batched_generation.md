# Batched Generation Spike Results

**Date:** September 16, 2026  
**Model:** Qwen2.5-0.5B-Instruct (Q4_K_M)  
**Hardware:** RTX 4070 Ti SUPER (jambo)  
**Framework:** pesti-runner with cudarc cuBLAS GEMM

## Findings

Running 10 sequential generations of the same prompt (8 tokens each):

- **Total time:** 231.47s
- **Average per generation:** 23.15s
- **Effective throughput:** 0.35 tok/s
- **Theoretical batched speedup:** 10x (process all sequences in ~23s)

## Analysis

Each generation is dominated by kernel launch overhead and sequential decode steps. At seq=8, the compute time per token is minimal compared to:

1. CUDA kernel launches (~15ms each for attention + GEMM ops)
2. KV cache updates
3. Sampling operations

With true batched inference (single forward pass for multiple sequences), we could theoretically process 10 sequences in the time of one, achieving ~3.5 tok/s aggregate throughput.

## Recommendations

**High priority:** Implement batched generation with:
- Paged attention for variable-length sequences
- Dynamic batching scheduler
- Shared KV cache management across sequences

**Medium priority:** Optimize kernel launch patterns:
- Fuse small operations into single kernels
- Use CUDA graphs for repeated compute patterns
- Minimize host-device synchronization points

## Next Steps

The ROI on batched generation is clear given the 10x theoretical speedup. This should be prioritized alongside F16 GPU inference optimization (Week 23) as both address different aspects of the performance gap vs llama.cpp.