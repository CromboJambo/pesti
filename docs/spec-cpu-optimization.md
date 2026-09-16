# PESTI CPU Inference Optimization Plan

## Current Baseline
- **Tok/s**: ~1.5 tok/s (qwen2.5-0.5b Q4_K_M, RTX 4070 Ti SUPER)
- **Bottleneck**: Full dequantization to f32 before GEMM, naive CPU matmul loop

## Optimization Targets (from mistral.rs patterns)

### Phase 1: Tiled/Batched CPU GEMM with SIMD
**Goal**: Replace naive O(m*n*k) CPU GEMM with tiled/batched approach using std::simd intrinsics.

**Implementation**:
- Implement blocked/tiled GEMM in `pesti-runner/src/kernel/cpu_gemm.rs` (or new file)
- Use `std::simd` for vectorized inner loop (f32x4 or f32x8 depending on target)
- Cache-friendly memory access patterns (row-major layout optimization)

**Expected improvement**: 5-10x faster CPU GEMM → measurable tok/s improvement even with CUDA fallback path.

### Phase 2: Fused Dequantize+GEMM for Quantized Weights
**Goal**: Eliminate full dequantization pass; compute directly from quantized format like mistral.rs's mmvq kernels.

**Implementation**:
- Modify `pesti-runner/src/quantized_linear.rs` to perform inline dequantization during GEMM
- Create specialized kernels for Q4_K_M, Q8_0 formats that operate on compressed weights directly
- This is the biggest win — eliminates redundant memory bandwidth usage

**Expected improvement**: 3-5x faster inference (weight loading is the bottleneck in LLM inference)

### Phase 3: Optimized Attention Kernel with Logsumexp Trick
**Goal**: Replace naive softmax attention with numerically stable logsumexp trick + optimized computation.

**Implementation**:
- Implement fused attention kernel in `pesti-runner/src/kernel/attention.rs`
- Use logsumexp for numerical stability at long sequence lengths
- Optimize memory layout for KV cache access patterns

**Expected improvement**: 20-30% faster attention (significant for long sequences)

### Phase 4: SIMD-Accelerated Dequantization Kernels
**Goal**: Vectorize the remaining dequantization operations that aren't fused into GEMM.

**Implementation**:
- Use `std::simd` for parallel nibble/byte extraction in Q4_K_M, Q8_0 dequantization
- Process multiple elements simultaneously (e.g., 8 f16 values at once)

**Expected improvement**: 2-3x faster dequantization operations

## Implementation Notes

### Dependencies
- `std::simd` requires nightly Rust or specific feature flags
- Consider adding optional dependency on existing SIMD crate if std::simd proves insufficient

### Testing Strategy
- Run conformance tests after each phase to verify numerical correctness
- Benchmark tok/s before and after each optimization
- Use qwen2.5-0.5b Q4_K_M as standard test model throughout

### Risk Assessment
- **Phase 1**: Low risk, high reward — straightforward SIMD optimization
- **Phase 2**: Medium risk, highest reward — requires careful handling of quantization formats
- **Phase 3**: Low risk, moderate reward — well-understood numerical technique
- **Phase 4**: Low risk, incremental improvement

## Success Metrics
- Target: >5 tok/s on qwen2.5-0.5b Q4_K_M (RTX 4070 Ti SUPER)
- Maintain numerical correctness within 1e-3 tolerance of baseline
- No regressions in existing test suite
