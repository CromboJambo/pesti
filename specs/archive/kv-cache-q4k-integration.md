# KV Cache Q4_K Quantization Integration Spec

## Goal
Reduce KV cache memory bandwidth pressure during autoregressive decode by ~72% through Q4_K quantization, targeting measurable throughput improvement on pesti-runner's GPU inference path.

## Current State (Baseline)
- **Benchmark**: 73 tok/s on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti)
- **Attention kernel time**: ~24% of decode time (from Week 26 profiling)
- **KV cache format**: FP16 (2 bytes per element)
- **Existing infrastructure**: `q4k_kvcache.rs` module exists but not integrated into inference path

## Target Models
Qwen2.5 family with GQA (grouped-query attention):
- num_kv_heads: 8
- head_dim: 64
- embed_dim: 896
- intermediate_dim: 1792

## Expected Improvements
From `benchmark_kvcache_q4k.rs` analysis:
- **Memory savings**: 71.9% (32 MB → 9 MB for Qwen2.5-0.5B)
- **Bandwidth per token at seq=512**: 8.2 MB → 2.3 MB
- **Estimated attention kernel speedup**: 3.56x if memory-bound
- **Overall throughput target**: 90-100 tok/s (optimistic), minimum measurable improvement

## Q4_K Format (ggml canonical block_q4_K)
Each block stores 256 values using 144 bytes:
- 2 bytes: scale factor (f16)
- 2 bytes: min offset (f16)  
- 12 bytes: per-group scales/mins (packed 6-bit pairs, 8 groups of 32 values each)
- 128 bytes: quantized data (nibbles, 2 values per byte)

## Implementation Phases

### Phase 1: Q4_K Quantization Function
**File**: `pesti-runner/src/kernel/q4k_quantize.rs` (new)

Implement `quantize_q4k_block(input: &[f32], output: &mut [u8])` — inverse of existing `dequantize_q4k_block()` in pesti-safetensors.

**Algorithm**:
1. Compute min/max of input values
2. Calculate block scale and offset
3. For each group of 32 values, compute group-specific scale/min
4. Quantize each value to 4-bit nibble
5. Pack into ggml canonical format

**Validation**: Round-trip test — quantize then dequantize, verify error < epsilon (target: mean absolute error < 0.01 for typical attention score ranges).

### Phase 2: Inference Path Integration
**Files**: 
- `pesti-runner/src/kernel/dispatch.rs` — modify KV cache write path
- `pesti-runner/src/transformer/model.rs` — add Q4_K cache configuration

**Changes**:
1. Add `use_q4k_kvcache` flag to `LlamaModel` config
2. When enabled, quantize K/V rows before writing to cache:
   ```hemlock
   // Before (FP16):
   key_cache.write_k_at(pos, &k_row_f16);
   
   // After (Q4_K):
   let k_q4k = quantize_q4k_block(&k_row_f32);
   q4k_cache.write_kv_at(pos, &k_q4k, &v_q4k);
   ```
3. Use `fused_decode_attention_q4k` kernel for attention when Q4_K cache is active

### Phase 3: Benchmarking and Validation
**Benchmark**: Compare tok/s with and without Q4_K KV cache on identical model/prompt

**Commands**:
```hemlock
# Baseline (FP16 KV cache)
cargo run --release --example benchmark_pesti_runner

# With Q4_K KV cache
cargo run --release --example benchmark_pesti_runner -- --q4k-kvcache
```

**Success criteria**:
- Measurable throughput improvement (even small)
- No regression in output quality (same tokens generated)
- Memory usage reduced as predicted by benchmarks

## Risk Factors
1. **Quantization error propagation**: Q4_K may introduce subtle numerical differences that cascade through layers, potentially changing generation results
2. **Kernel integration complexity**: Wiring the Q4_K fused kernel into the dispatch path requires careful handling of memory layouts
3. **Diminishing returns**: If attention is not actually memory-bound on this hardware/architecture, improvement may be minimal

## Rollback Plan
If Q4_K KV cache causes regression or quality issues:
- Keep FP16 KV cache as default
- Make Q4_K opt-in via config flag or CLI argument
- Document trade-offs clearly in benchmark results

## Timeline Estimate
- Phase 1 (quantization function): 2-3 hours
- Phase 2 (inference integration): 4-6 hours  
- Phase 3 (benchmarking/validation): 1-2 hours
- Total: ~8-11 hours of focused work

## References
- `pesti-runner/src/kernel/q4k_kvcache.rs` — existing Q4_K cache container
- `pesti-runner/src/kernel/fused_decode_attention_q4k.rs` — Q4_K attention kernel
- `pesti-runner/examples/benchmark_kvcache_q4k.rs` — memory savings benchmark
- `pesti-safetensors/src/gguf_converter.rs::dequantize_q4_k()` — reference dequantization