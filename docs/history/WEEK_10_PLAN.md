# Week 10 Plan: Performance Optimization Sprint 🚀

**Date**: August 15, 2026  
**Goal**: Verify RoPE fix + implement shared memory tiling for real GPU speedup  
**Timeline**: 12-14 days (2 weeks of focused grinding)

---

## 🎯 Week 10 Objectives

### Primary Goal
**Achieve <1e-4 relative error vs llama.cpp AND establish measurable performance baseline on real model.**

### Secondary Goals
1. Implement single-kernel fused attention (RoPE + scores + softmax + V)
2. Add shared memory tiling for 3-5x speedup on long sequences
3. Benchmark on Qwen2.5-0.5B and establish GPU vs CPU baseline

---

## 📊 Current State Assessment

### ✅ What's Working (End of Week 9)
1. **Model discovery** from `CRABJAR_MODEL_PATHS` environment variable
2. **GGUF loading** via llama.cpp FFI (Q4_K_M quantization verified)
3. **CUDA infrastructure**: PTX loading, kernel launch, stream management
4. **RoPE implementation**: Half-swap rotation aligned with llama.cpp ✅
5. **Softmax**: Numerical stability with max subtraction trick
6. **Causal mask**: Applied before softmax (standard pattern)
7. **Conformance test framework**: Byte-exact comparison vs llama.cpp

### ⚠️ What's Pending Verification
1. **Numerical conformance**: RoPE fix needs re-test after fresh PTX compilation
2. **End-to-end throughput**: Still using stub/sequential implementation (~83-100 tok/s)
3. **Single-kernel fusion**: Currently two-kernel approach (scores → softmax)

### 🎯 Target Metrics
| Metric | Current | Week 10 Target | Notes |
|--------|---------|----------------|-------|
| Max relative error vs llama.cpp | 46,964× (before fix) | <1e-4 | Pending verification |
| Softmax sum | 1.0 ✅ | 1.0 ±1e-6 | Already working |
| Throughput on Qwen2.5-0.5B | ~83-100 tok/s | ~110-120 tok/s | +25-40% with tiling |
| Kernel launches per token | 2 (scores → softmax) | 1 (fused) | Single-kernel fusion |

---

## 🗓️ Week 10 Schedule

### Day 1-2: Verify RoPE Fix ✅ Priority!

**Goal**: Confirm the half-swap rotation fix actually resolves the 46,964× error

#### Tasks
```bash
# 1. Force fresh PTX compilation (clear nvcc cache)
rm pesti-runner/src/kernel/ptx/attention_rope_softmax.ptx
nvcc -arch=sm_89 --ptx attention_rope_softmax.cu -o attention_rope_softmax.ptx

# 2. Re-run conformance test
cargo test --package pesti-runner fused_attention_llama_conformance \
    --features cuda,flash-attention -- --nocapture

# 3. Check results
Expected: Max relative error <1e-4 (machine epsilon level)
If still failing: Debug RoPE intermediate values (copy Q/K back from GPU)
```

#### Success Criteria
- [ ] PTX compiled successfully with no errors
- [ ] Test shows max relative error <1e-4
- [ ] Softmax sums = 1.0 ±1e-6 (already working ✅)
- [ ] Attention scores match llama.cpp within tolerance

#### If Fails: Debug Strategy
1. **Copy RoPE-applied Q/K back from GPU** and compare with CPU reference
2. **Print intermediate values**: cos/sin frequencies, rotated dimensions
3. **Verify dimension pairing**: Check that `d` pairs with `d + head_dim/2` correctly
4. **Compare with HuggingFace**: Use their reference implementation as ground truth

---

### Day 3-5: Single-Kernel Fusion ⚡

**Goal**: Merge two-kernel approach (scores → softmax) into single fused kernel

#### Current Architecture (Two Kernels) ❌
```
Kernel 1: Q @ K^T → scores_buffer[]
Kernel 2: Read scores_buffer[] → softmax → multiply by V → output_buffer[]
```
**Issues**:
- Two kernel launches = more overhead
- Global memory write/read of scores buffer (bandwidth bottleneck)
- Synchronization complexity between kernels

#### Target Architecture (Single Kernel) ✅
```cuda
__global__ void fused_attention_kernel(
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    half* __restrict__ out_ptr,
    float scale,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    if (q_pos >= seq_len_q || head >= num_heads) return;
    
    // Step 1: Apply RoPE to Q and K (half-swap rotation)
    float q_val = apply_rope_half_swap(q_ptr, k_ptr, q_pos, head, tid, ...);
    float k_val = ...;
    
    // Step 2: Compute attention scores: Q @ K^T with causal mask
    float max_score = -FLT_MAX;
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        if (k_pos > q_pos) continue;  // Causal mask
        
        float raw_score = q_val * k_val * scale;
        if (raw_score > max_score) max_score = raw_score;
    }
    
    // Step 3: Softmax with max subtraction trick
    float exp_sum = 0.0f;
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        if (k_pos > q_pos) continue;
        
        float raw_score = ...;
        float exp_val = expf(raw_score - max_score);
        exp_sum += exp_val;
    }
    
    // Step 4: Weighted sum of V
    float out_val = 0.0f;
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        if (k_pos > q_pos) continue;
        
        float raw_score = ...;
        float softmax_weight = expf(raw_score - max_score) / exp_sum;
        
        int v_idx = k_pos * num_heads * head_dim + head * head_dim + tid;
        float v_val = __half2float(v_ptr[v_idx]);
        out_val += softmax_weight * v_val;
    }
    
    // Step 5: Store output (FP32 → FP16)
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + tid;
    out_ptr[out_idx] = __float2half(out_val);
}
```

#### Implementation Tasks
- [ ] Create `pesti-runner/src/kernel/ptx/fused_attention_kernel.cu`
- [ ] Copy RoPE, scores, softmax, V-multiply logic into single kernel
- [ ] Test correctness with small inputs (seq_len=2, dim=16)
- [ ] Compare output vs two-kernel version (should be identical)
- [ ] Update Rust loader to use new PTX file

#### Expected Benefits
- **-1 kernel launch** per attention layer
- **Better memory locality** (no intermediate global write/read)
- **Simpler synchronization** (one kernel, one launch)

---

### Day 6-7: Shared Memory Tiling 🚀

**Goal**: Achieve 3-5x speedup on long sequences (512+ tokens)

#### Current Approach: Sequential ❌
```cuda
// Each thread loads Q/K/V from global memory for every k_pos iteration
for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
    float k_val = __half2float(k_ptr[k_idx]);  // Global memory read!
    float v_val = __half2float(v_ptr[v_idx]);   // Global memory read!
}
```
**Problem**: O(n²) global memory accesses → bandwidth bottleneck

#### Target Approach: Shared Memory Tiling ✅
```cuda
__shared__ half q_tile[TILE_SIZE];
__shared__ half k_tile[TILE_SIZE];
__shared__ half v_tile[TILE_SIZE];

const int TILE_SIZE = 128;

for (int tile_start = 0; tile_start < seq_len_kv; tile_start += TILE_SIZE) {
    // Step 1: Load tile into shared memory (once per block)
    int k_idx = tile_start * num_heads * head_dim + head * head_dim + tid;
    if (tid < head_dim) {
        k_tile[tid] = k_ptr[k_idx];
        v_tile[tid] = v_ptr[k_idx];
    }
    __syncthreads();  // Ensure all threads loaded
    
    // Step 2: Compute Q @ K^T from shared memory (no global access!)
    float score = 0.0f;
    for (int t = 0; t < TILE_SIZE && (tile_start + t) < seq_len_kv; t++) {
        if ((tile_start + t) > q_pos) continue;  // Causal mask
        
        float k_val = __half2float(k_tile[t]);
        score += q_val * k_val * scale;
    }
    
    // Step 3: Accumulate softmax + V-multiply from tile
    for (int t = 0; t < TILE_SIZE && (tile_start + t) < seq_len_kv; t++) {
        if ((tile_start + t) > q_pos) continue;
        
        float v_val = __half2float(v_tile[t]);
        out_val += softmax_weight * v_val;
    }
    
    __syncthreads();  // Prepare for next tile
}
```

#### Implementation Tasks
- [ ] Add shared memory declarations to fused kernel
- [ ] Implement tile loading loop (outer loop over seq_len_kv)
- [ ] Add `__syncthreads()` synchronization points
- [ ] Handle edge cases (seq_len not divisible by TILE_SIZE)
- [ ] Test correctness with small sequences first

#### Performance Expectations
| Sequence Length | Current (sequential) | With Tiling | Speedup |
|-----------------|----------------------|-------------|---------|
| 32 tokens | ~0.16s kernel launch | ~0.14s | +15% |
| 128 tokens | ~0.6s | ~0.25s | +2.4x |
| 512 tokens | ~2.8s | ~0.7s | +4x |
| 2048 tokens | ~12s | ~2.5s | +4.8x |

**Key insight**: Tiling shines on long sequences where memory bandwidth dominates!

---

### Day 8-10: WGMMA Tensor Core Instructions 🎯

**Goal**: Leverage RTX 4070 Ti SUPER tensor cores for Q @ K^T GEMM

#### Current Approach: FP32 Sequential ❌
```cuda
// Each thread does sequential FP32 dot product
float score = 0.0f;
for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
    float k_val = __half2float(k_ptr[k_idx]);
    score += q_val * k_val;  // Scalar multiply-add
}
```

#### Target Approach: WGMMA Tensor Cores ✅
```ptx
// WGMMA tile: 16x8 matrix multiply-accumulate
// Input: FP16 tiles, Output: FP32 accumulator
wgmma.sync.aligned.m16n8k16.f32.f16.f16.f32
    {%w0,%w1,%w2}, %w3, [%rdA], [%rDB], %fC;
```

**Pattern**:
```cuda
__shared__ half q_tile[16][TILE_SIZE];  // FP16 tiles in shared memory
__shared__ half k_tile[TILE_SIZE][16];

// Load Q tile into shared memory (FP16)
for (int i = 0; i < 16; i++) {
    q_tile[i][tid] = q_ptr[...];
}
__syncthreads();

// Load K tile into shared memory (FP16)
for (int j = 0; j < TILE_SIZE / 16; j++) {
    k_tile[tid * 16 + j] = k_ptr[...];
}
__syncthreads();

// Launch WGMMA: 16x8 matrix multiply-accumulate
wgmma.sync.aligned.m16n8k16.f32.f16.f16.f32
    {%w0,%w1,%w2}, %w3, [%rdA], [%rDB], %fC;

// Accumulate result into score
score += fp32_accum_to_fp32(%w0);
```

#### Implementation Tasks
- [ ] Study WGMMA instruction syntax (cudacc docs)
- [ ] Modify fused kernel to use FP16 tiles for Q @ K^T
- [ ] Ensure tensor core alignment requirements (16x8 tiles)
- [ ] Handle dimension mismatches (head_dim may not be multiple of 16)
- [ ] Benchmark vs sequential implementation

#### Performance Expectations
| Model Size | Sequential FP32 | WGMMA FP16 | Speedup |
|------------|-----------------|------------|---------|
| Qwen2.5-0.5B | ~95 tok/s | ~105 tok/s | +10% |
| Qwen2.5-3B | ~18 tok/s | ~45 tok/s | +2.5x |
| Llama 3.1 8B | ~10 tok/s | ~60 tok/s | +6x |

**Key insight**: WGMMA shines on large models where GEMM dominates!

---

### Day 11-12: End-to-End Benchmark 📊

**Goal**: Establish measurable GPU vs CPU baseline on real model

#### Test Setup
```bash
# Run with CPU backend (llama.cpp)
export CRABJAR_MODEL_PATHS="$HOME/pesti-models"
cargo run --package pesti-runner --example test_load_and_generate \
    --features cuda \
    --release

# Run with GPU Flash Attention (once implemented)
cargo run --package pesti-runner --example test_load_and_generate \
    --features cuda,flash-attention \
    --release
```

#### Metrics to Collect
- **Throughput**: tokens/sec (greedy sampling, temperature=0.0)
- **Latency**: time per token (ms/token)
- **Memory usage**: VRAM consumption (peak and steady-state)
- **Kernel launch overhead**: Time spent in CUDA kernel launches vs computation

#### Expected Results (Qwen2.5-0.5B)
| Backend | Throughput | Latency | Notes |
|---------|------------|---------|-------|
| CPU llama.cpp | ~95 tok/s | ~10.5 ms/token | Baseline |
| GPU GEMM-based | ~87-95 tok/s | ~10.5-11.5 ms/token | Small model: minimal benefit |
| GPU Flash Attention (Week 10 target) | ~110-120 tok/s | ~8.3-9.1 ms/token | +15-25% speedup |

**Key insight**: Small models (0.5B) don't benefit much from Flash Attention yet! Real speedup (+40-50%) expected on 3B+ models with long sequences (512+ tokens).

---

### Day 13-14: Documentation & Cleanup 📝

#### Tasks
- [ ] Update `WEEK_9_SUMMARY.md` with actual verification results
- [ ] Create `WEEK_10_SUMMARY.md` with performance metrics
- [ ] Commit all changes with clear commit messages
- [ ] Update `ROADMAP.md` with new progress status
- [ ] Add notes on lessons learned and pitfalls encountered

#### Deliverables
- ✅ Verified numerical conformance (<1e-4 relative error)
- ✅ Single-kernel fused attention implementation
- ✅ Shared memory tiling for long sequences
- ✅ Performance baseline on real model
- ✅ Clean git history with logical commit order

---

## 🎯 Success Criteria

### Minimum Viable Success (Week 10 Complete)
- [ ] RoPE fix verified: Max relative error <1e-4 vs llama.cpp
- [ ] Single-kernel fusion implemented and tested
- [ ] Shared memory tiling working on sequences ≥128 tokens
- [ ] Throughput >100 tok/s on Qwen2.5-0.5B (vs ~83-100 baseline)

### Stretch Goals (Nice to Have)
- [ ] WGMMA tensor core instructions implemented
- [ ] Performance benchmark on 3B+ model showing +4x speedup
- [ ] Streaming output for autoregressive generation
- [ ] Documentation: "GPU Kernel Development Best Practices" guide

---

## 🚦 Risk Mitigation

### If RoPE Fix Doesn't Work (Day 1-2)
**Fallback**: Debug step-by-step with intermediate value extraction
1. Copy Q/K back from GPU after RoPE, compare with CPU reference
2. Print cos/sin frequencies to verify correct formula
3. Use HuggingFace transformers as ground truth (Python reference)

### If Single-Kernel Fusion is Too Complex (Day 3-5)
**Fallback**: Start with simpler fusion (RoPE + scores only), add softmax/V later
1. Implement RoPE + scores in one kernel (easiest win)
2. Verify correctness before adding softmax
3. Add V-multiply in third iteration

### If Shared Memory Tiling Doesn't Speed Up (Day 6-7)
**Fallback**: Analyze memory bandwidth vs compute bound
1. Use `nsys` profiling to identify bottleneck
2. If bandwidth-bound: Increase TILE_SIZE or reduce global reads
3. If compute-bound: WGMMA may be better investment

### If WGMMA Is Too Hard (Day 8-10)
**Fallback**: Skip tensor cores for now, focus on shared memory tiling
1. Sequential + shared memory already gives 3-5x speedup
2. WGMMA can be added in Week 11+ once baseline is solid

---

## 📚 References

- `WEEK_9_SUMMARY.md` — RoPE alignment complete
- `WEEK_8_NUMERICAL_CONFORMANCE.md` — Bug discovery and analysis
- `WEEK_7_SOFTMAX_INTEGRATION.md` — Softmax numerical stability
- `HuggingFace transformers` — `apply_rotary_pos_emb` reference implementation
- `llama.cpp ggml.c` — RoPE and attention reference
- `NVIDIA WGMMA docs` — Tensor core instruction syntax

---

## 🎉 Final Verdict: Ready to Start Week 10! ✅

**Current state**: Infrastructure solid, RoPE formula fixed, verification pending  
**Next milestone**: Numerical conformance <1e-4 + measurable performance baseline  
**Timeline**: 12-14 days of focused grinding (Day 1 = fresh PTX compilation)

**Strategic position**: We're at the "infrastructure solid → optimization sprint" transition point. Once RoPE fix is verified, we can dive into shared memory tiling and WGMMA for real GPU speedup! 🚀

---

**Author**: PESTI Engineering Team  
**Date**: August 15, 2026  
**Status**: Week 9 complete, Week 10 plan ready. Let's grind! 💪
