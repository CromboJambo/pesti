# PESTI Week 29: Optimization Path Decision

## Context
PESTI is a portable execution substrate for transformer inference, written in Rust with CUDA dispatch via cudarc bindings. Current throughput on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti) is ~307 tok/s vs llama.cpp's 504 tok/s reference.

## Week 28 Results
CUTLASS integration completed but revealed that cuBLASLt CUTLASS kernels are **slower** than plain cuBLAS for LLM decode shapes (m=1, tall-skinny):

| Shape | cuBLAS | cuBLASLt | Speedup |
|-------|--------|----------|---------|
| m=1, n=4096, k=4096 | 0.32ms | 0.78ms | **0.41x** (slower!) |
| m=1, n=2048, k=2048 | 0.09ms | 0.56ms | **0.16x** (slower!) |
| m=32, n=512, k=1024 | 0.03ms | 0.55ms | **0.06x** (slower!) |

Root cause: `cublasLtMatmulAlgoGetHeuristic` overhead (~0.5ms) dominates small shape calls. Plain cuBLAS's internal kernel selection is already optimal for these patterns.

## Week 29 Decision Point
GEMM compute path is at practical optimum via plain cuBLAS + shape-based algorithm caching (Week 26). Remaining throughput gap likely stems from other factors. Three candidate optimization paths:

1. **KV cache quantization** — reduce KV cache memory bandwidth pressure
2. **Speculative decoding** — increase effective tokens/sec without reducing per-token latency  
3. **Custom CUDA kernels for non-GEMM ops** — attention, RMSNorm, RoPE already on GPU but may have optimization headroom

## Your Task
1. Profile pesti-runner vs llama.cpp to identify the actual bottleneck causing the 307 vs 504 tok/s gap
2. Based on profiling data, select ONE of the three optimization paths above
3. Implement the selected optimization
4. Benchmark against both pesti baseline and llama.cpp reference

## Key Files
- `pesti-runner/src/kernel/cuda_bridge.rs` — cuBLAS integration (gemm_f16_cublaslt added Week 28)
- `pesti-runner/src/transformer/model.rs` — transformer forward pass, KV cache handling
- `pesti-runner/examples/benchmark_pesti_runner.rs` — benchmarking infrastructure
- `docs/ROADMAP.md` — roadmap with detailed history and findings

## Build & Test Commands
```bash
# Build pesti-runner with CUDA support
cargo build --release -p pesti-runner --features cuda

# Run specific tests
cargo test --release -p pesti-runner --features cuda --test <test_name>

# List available tests
cargo test --release -p pesti-runner --features cuda --list
```

## Hardware Setup
- Target: RTX 3070 Ti (compute capability 8.6)
- Model: Qwen2.5-0.5B-Instruct-Q4_K_M (~1GB VRAM)
- Reference: llama.cpp achieves 504 tok/s on this hardware/model

## Important Notes
- pesti-runner uses cudarc's safe API bindings (CudaSlice, not raw pointers)
- F16 precision throughout inference path
- KV cache currently stored as full F16 tensors — prime target for quantization optimization
- All optimizations must maintain numerical conformance with llama.cpp reference outputs
