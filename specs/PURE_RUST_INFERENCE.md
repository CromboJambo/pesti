# Pure-Rust Inference Optimization Plan

## Goal
Make PESTI's native Rust transformer path competitive with llama.cpp for inference speed, enabling pure-Rust inference without C dependencies.

## Current State
- Native transformer (`transformer/`) has full inference: `generate()`, KV cache, attention
- All weights loaded as f32 even from quantized GGUF (dequantize-on-load)
- `QuantizedLinear` exists but is NOT used in transformer path
- `tile_dequant.rs` has scalar dequantize for Q4_K/Q8_0/etc (no SIMD)
- Attention computed with per-head scalar loops
- No batched prefill
- CUDA support via candle-kernels (generic, not tuned)

## Target Performance
Match or exceed llama.cpp on CPU for Q4_K models at 1.5B-7B scale:
- Decode: ≥ 40 tok/s (Qwen2.5-Coder-1.5B Q4_K, current f32 baseline ~41 tok/s via LlamaRunner)
- Prefill: ≥ 100 tok/s

## Optimization Phases

### Phase 1: Quantized GEMM (CPU, SIMD) — HIGHEST IMPACT
**Impact:** 60-70% of decode time is matmul. Q4_K weights are 4x smaller than f32 → 4x less memory bandwidth.

**Work:**
1. Wire `QuantizedLinear` into transformer layers (replace f32 Linear)
2. Implement fused dequant+GEMM with AVX2 intrinsics:
   - Dequantize Q4_K block to f32 in registers
   - Multiply-accumulate with input vector
   - No intermediate f32 weight storage
3. Use `gemm` crate (already in workspace) for f32 GEMM fallback
4. Support Q4_K, Q8_0 initially (most common quantizations)

**Files:**
- New: `src/kernel/q4k_gemm.rs` — SIMD dequant+GEMM kernel
- Modify: `transformer/linear.rs` — add quantized forward path
- Modify: `transformer/model.rs` — load weights as QuantizedLinear

### Phase 2: Batched Prefill
**Impact:** 30-50% faster prompt processing (GPU/CPU saturation)

**Work:**
1. Add `forward_batch(&self, tokens: &[u32]) -> Result<Vec<f32>>` to model
2. Process prompt in chunks of 64-512 tokens
3. Use batched GEMM for QKV projections
4. Update KV cache to accept batch appends

**Files:**
- Modify: `transformer/model.rs` — add batched forward
- Modify: `transformer/kv_cache.rs` — batch append
- Modify: `transformer/layer.rs` — batched attention

### Phase 3: Fused Attention
**Impact:** 20-40% faster attention (eliminate intermediate matrix materialization)

**Work:**
1. Fuse QKV projection + RoPE + attention into single pass per head
2. Avoid materializing full S×S score matrix for short sequences
3. Use online softmax (flash attention style) for long sequences
4. SIMD dot products for Q·K scoring

**Files:**
- Modify: `transformer/layer.rs` — fused attention kernel
- New: `src/kernel/fused_attention.rs` (if needed)

### Phase 4: Quantized KV Cache
**Impact:** 50% less memory bandwidth for attention reads at long context

**Work:**
1. Store K/V as Q8_0 instead of f32
2. On-the-fly dequantize during attention scoring
3. Optional: per-layer quantization (recent layers f32, older Q8_0)

**Files:**
- Modify: `transformer/kv_cache.rs` — quantized storage
- Modify: `transformer/layer.rs` — dequantize in attention loop

### Phase 5: Custom CUDA Kernels
**Impact:** 2-5x faster on GPU for supported models

**Work:**
1. Q4_K dequant+GEMM CUDA kernel (fused)
2. Fused attention CUDA kernel (QKV → softmax → output)
3. RoPE CUDA kernel
4. cuBLASLt for f32 GEMM fallback

**Files:**
- New: `src/cuda/q4k_gemm.cu`
- New: `src/cuda/fused_attention.cu`
- Modify: build system for CUDA compilation

## Execution Order
1. Phase 1 (Quantized GEMM) — foundation, biggest win
2. Phase 2 (Batched Prefill) — builds on Phase 1
3. Phase 3 (Fused Attention) — builds on Phase 1+2
4. Phase 4 (Quantized KV) — incremental improvement
5. Phase 5 (CUDA) — separate track, can parallelize

## Success Criteria
- [ ] Decode ≥ 40 tok/s Qwen2.5-Coder-1.5B Q4_K on CPU (native path)
- [ ] No f32 weight storage for quantized models
- [ ] Memory usage ≤ llama.cpp for same model/quantization
- [ ] Numerical parity with LlamaRunner (perplexity diff < 0.1)
