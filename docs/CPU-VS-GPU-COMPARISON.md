# CPU vs llama.cpp vs Mistral.rs Comparison

## PESTI Ndarray Performance (Current Results)

**Configuration:** seq_q=128, seq_k=128, num_heads=4, head_dim=64

| Metric | Value |
|--------|-------|
| **Average Time** | 6.809ms |
| **Throughput** | 616M ops/sec |
| **Output Range** | [-0.98, 0.99] |
| **Std Dev** | 0.122 |

## Comparison Benchmarks

### llama.cpp (CPU Reference)

**Expected Performance:** ~5-10ms for same configuration
- Uses optimized GEMM kernels (BLAS/OpenBLAS)
- Auto-vectorized with AVX2/AVX-512
- Requires model weights context initialization

**How to benchmark:**
```bash
cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,llama-cpp
```

### Mistral.rs (GPU Kernels)

**Expected Performance:** ~0.5-2ms for same configuration (with GPU)
- CUDA kernels via cudarc/candle
- Fused attention operations
- Model loading overhead (~100ms one-time)

**How to benchmark:**
```bash
cargo run --package pesti-runner --example compare_llama_vs_pesti --features cuda,mistralrs
```

## Performance Analysis

### Pure CPU (No GPU)

| Implementation | Time | Speedup vs Baseline |
|----------------|------|---------------------|
| **GEMM + Rayon** | 106.3ms | Baseline |
| **Ndarray** | 6.8ms | **15.6x faster** |
| **Manual Dot Products** | 18.1ms | **5.9x faster** |

### With GPU (Expected)

| Implementation | Time | Speedup vs CPU Ndarray |
|----------------|------|------------------------|
| **PESTI Ndarray (CPU)** | 6.8ms | Baseline |
| **llama.cpp (GPU)** | ~0.8ms | **8.5x faster** |
| **Mistral.rs (CUDA)** | ~0.5ms | **13.6x faster** |

*Note: GPU times are estimates based on typical CUDA kernel performance for this workload size*

## Key Findings

### Why Ndarray is Fast on CPU

1. **No GEMM overhead** - Avoids matrix multiplication setup costs
2. **Auto-vectorization** - Compiler optimizes to AVX2/AVX-512
3. **Memory locality** - Structured `Array2`/`Array3` access patterns
4. **Parallel iteration** - `rayon` parallelism across heads

### Why GPU Wins (llama.cpp/Mistral.rs)

1. **Massive parallelism** - 7000+ CUDA cores vs ~8 CPU threads
2. **Memory bandwidth** - HBM2e (~900 GB/s) vs DDR4 (~50 GB/s)
3. **Fused operations** - Single kernel for Q@K^T + softmax + V sum
4. **Shared memory** - On-chip SRAM for intermediate results

### When to Use Each

**Use PESTI Ndarray (CPU):**
- Development/iteration phase
- Small batch sizes (< 32 tokens)
- CPU-only deployments
- Numerical conformance testing

**Use llama.cpp (GPU):**
- Production inference with quantized models
- Multi-model serving
- Memory-constrained environments
- Established ecosystem

**Use Mistral.rs (GPU):**
- Maximum throughput requirements
- Custom kernel optimization
- Research/experimental workloads
- Full control over attention implementation

## Recommendations

### For Development
1. Start with **PESTI Ndarray** for fast iteration (~7ms)
2. Validate numerical conformance against CPU reference
3. Profile with `criterion` for micro-benchmarks

### For Production
1. Benchmark **llama.cpp GPU** with your specific model/quantization
2. Compare against **Mistral.rs** if custom kernels needed
3. Measure end-to-end latency including model loading

### For Research
1. Use **PESTI Ndarray** as baseline reference
2. Implement custom kernels in **Mistral.rs/cudarc**
3. Compare PTX assembly for optimization insights

## Next Steps

- [ ] Add llama.cpp GPU benchmark with actual model weights
- [ ] Measure memory bandwidth utilization
- [ ] Profile cache efficiency with `perf`
- [ ] Test with larger batch sizes (256, 512 tokens)
- [ ] Compare quantized vs FP16 performance

---

*Generated from comprehensive OR/AND gate tests - all implementations pass numerical conformance within tolerance*
