# Week 12/12: Numerical Conformance & Performance Optimization

## Overview
Final week of PESTI project - transform infrastructure into production-ready inference with numerical accuracy and performance matching llama.cpp baselines.

## Current State (End of Week 11)

### ✅ Completed Infrastructure
- **GGUF loading**: 291 tensors loaded from Qwen2.5-0.5B Q4_K_M model
- **CUDA runtime**: RTX 4070 Ti SUPER context initialized
- **KV cache**: 2 MiB per cache (8 KV heads × 64 dim × 2048 seq)
- **Batch prefill**: Working with seq_len > 1 (5,285 tok/s at seq_len=16)
- **Full pipeline**: End-to-end inference example operational

### ⚠️ Current Limitations
- **CPU fallback**: Attention computation running on CPU, not GPU kernels
- **Missing RoPE**: Rotary position embeddings not yet implemented
- **No KV updates**: Generation loop doesn't actually update KV cache
- **Performance gap**: ~35% of llama.cpp baseline (needs 100%+)

## Week 12 Goals

### Primary Objectives
1. **Numerical Conformance**: Match llama.cpp outputs within floating-point tolerance
2. **GPU Integration**: Move all kernels to CUDA for real performance gains
3. **Performance Target**: Achieve ~72 tok/s (llama.cpp baseline for Qwen2.5-0.5B f16)

### Success Metrics
- ✅ Conformance test passes with < 1e-4 max absolute error vs llama.cpp
- ✅ All attention kernels running on GPU (no CPU fallback)
- ✅ End-to-end throughput ≥ 50 tok/s on RTX 4070 Ti SUPER
- ✅ KV cache updates working correctly during autoregressive generation

---

## Phase 1: Numerical Conformance Testing

### Goal
Validate that PESTI produces identical outputs to llama.cpp for the same input.

### Approach
```rust
// Example conformance test structure
fn numerical_conformance_test() {
    // 1. Load model with both backends
    let pesti_model = load_gguf_weights("model.gguf")?;
    let llama_model = llama_cpp::load("model.gguf")?;
    
    // 2. Run same prompt through both
    let prompt = "The quick brown fox jumps over the lazy dog";
    let pesti_output = pesti_model.generate(prompt, max_tokens=50)?;
    let llama_output = llama_model.generate(prompt, max_tokens=50)?;
    
    // 3. Compare token-by-token and log probabilities
    compare_outputs(&pesti_output, &llama_output);
}
```

### Tasks
- [ ] **1.1**: Install `llama.cpp` CLI for reference outputs
  ```bash
  cd /tmp && git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make -j$(nproc)
  ```

- [ ] **1.2**: Create reference output generator
  ```bash
  ./llama.cpp/llama-cli -m model.gguf -p "The quick brown fox" -n 50 --temp 0.0
  ```

- [ ] **1.3**: Implement token-level comparison in conformance test
  - Exact match: ✅ Same tokens produced
  - Logprob tolerance: ±0.1 per token
  - Max absolute error: < 1e-4 on logits

- [ ] **1.4**: Add regression tests for known prompts
  - Short prompt (16 tokens): Basic sanity check
  - Medium prompt (64 tokens): Attention pattern validation
  - Long prompt (256 tokens): KV cache correctness

### Deliverables
- `pesti-runner/examples/conformance_reference.rs` - Reference comparison tool
- `pesti-runner/tests/numerical_conformance.rs` - Automated conformance suite
- `docs/numerical-conformance.md` - Error analysis and tolerance documentation

---

## Phase 2: CUDA Attention Kernel Integration

### Goal
Replace CPU attention fallback with optimized CUDA kernel.

### Current Implementation (CPU Fallback)
```rust
// pesti-runner/examples/full_inference.rs
for q_pos in 0..seq_len {
    for h in 0..num_heads {
        let q_base = (q_pos * num_heads + h) * head_dim;
        for k_pos in 0..cache_len.min(10) {
            // CPU computation
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_base + d] * k[k_base + d];
            }
            scores[...] = dot / (head_dim as f32).sqrt();
        }
    }
}
```

### Target Implementation (CUDA Kernel)
```rust
// pesti-runner/src/kernel/attention.cu
__global__ void attention_kernel(
    const float* __restrict__ q,  // [seq_len, num_heads, head_dim]
    const float* __restrict__ k,  // [cache_len, num_kv_heads, head_dim]
    const float* __restrict__ v,  // [cache_len, num_kv_heads, head_dim]
    float* __restrict__ scores,   // [seq_len, num_heads, cache_len]
    int seq_len, int cache_len,
    int num_heads, int num_kv_heads, int head_dim
) {
    int q_pos = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    
    // Load q[d] and k[d] into shared memory
    // Compute dot product in parallel
    // Store to scores[q_pos, h, k_pos]
}
```

### Tasks
- [ ] **2.1**: Design CUDA kernel architecture
  - Option A: One token at a time (simple, good for learning)
  - Option B: Full batch processing (optimal performance)
  - Decision: Start with Option A for Week 12, refactor to B in future

- [ ] **2.2**: Implement basic scaled dot-product attention kernel
  ```rust
  // pesti-runner/src/kernel/attention.rs
  pub fn attention_kernel(
      q: &CudaSlice<f32>,
      k: &CudaSlice<f32>,
      v: &CudaSlice<f32>,
      seq_len: usize,
      num_heads: usize,
      head_dim: usize,
  ) -> Result<CudaSlice<f32>> {
      // Launch CUDA kernel
      // Copy results back to host
  }
  ```

- [ ] **2.3**: Add softmax and output projection kernels
  - Softmax: Stabilize numerically (subtract max)
  - Output projection: Matrix multiply with W_o

- [ ] **2.4**: Fuse kernels into single pass (one-stage fusion)
  - QKV projections → Attention → Softmax → Output in one kernel
  - Reduces global memory writes by 70%

### Deliverables
- `pesti-runner/src/kernel/attention.cu` - CUDA attention implementation
- `pesti-runner/src/kernel/attention.rs` - Rust FFI bindings
- Performance benchmark showing GPU vs CPU speedup (target: 10x+)

---

## Phase 3: RoPE Embedding Implementation

### Goal
Add rotary position embeddings for positional awareness.

### Theory
RoPE rotates query/key vectors by angle θ based on position:
```
q_rot[d] = q[d] * cos(θ) - q[d+1] * sin(θ)
k_rot[d] = k[d] * cos(θ) - k[d+1] * sin(θ)
```

Where θ = pos^(2d/d_model) for d ∈ [0, head_dim/2)

### Tasks
- [ ] **3.1**: Implement RoPE frequency computation
  ```rust
  pub fn compute_rope_freqs(head_dim: usize, seq_len: usize) -> Vec<f32> {
      (0..head_dim/2)
          .map(|d| 1.0f32.powf(-(2.0 * d as f32) / head_dim as f32))
          .collect()
  }
  ```

- [ ] **3.2**: Create RoPE application kernel
  ```rust
  __global__ void rope_kernel(
      float* q, float* k,  // in-place rotation
      const float* freqs,  // precomputed frequencies
      int seq_len, int head_dim
  ) {
      int pos = blockIdx.x;
      int d = threadIdx.x;
      
      float cos_val = cos(freqs[d] * pos);
      float sin_val = sin(freqs[d] * pos);
      
      // Apply rotation
      q[pos * head_dim + d] *= cos_val;
      q[pos * head_dim + d + 1] *= sin_val;
  }
  ```

- [ ] **3.3**: Integrate RoPE into prefill loop
  - Compute frequencies once per session (cached)
  - Apply to Q and K before attention computation
  - Skip for V (only Q/K need rotation)

### Deliverables
- `pesti-runner/src/kernel/rope.rs` - RoPE implementation
- Updated benchmark showing positional awareness improvement

---

## Phase 4: KV Cache Updates During Generation

### Goal
Implement proper autoregressive KV cache management.

### Current State (Prefill Only)
```rust
// Only prefill, no generation updates
for q_pos in 0..seq_len {
    // Compute attention scores against cached K/V
}
// Missing: Update cache with new K/V for next token
```

### Target State (Full Autoregressive Loop)
```rust
let mut kv_cache = Kvcache::new(...);

// Prefill phase
let logits = prefill(prompt, &mut kv_cache)?;

// Generation loop
for _ in 0..gen_len {
    let next_token = sample(logits);
    
    // Update KV cache with new position
    kv_cache.write_kv_at(global_pos, &k_row, &v_row)?;
    
    // Compute logits for next token
    logits = compute_logits(next_token, &mut kv_cache)?;
}
```

### Tasks
- [ ] **4.1**: Implement KV cache write method
  ```rust
  impl Kvcache {
      pub fn write_kv_at(
          &mut self,
          pos: usize,
          k_row: &[f32],
          v_row: &[f32],
      ) -> Result<()> {
          // Copy to GPU memory at position pos
          cuda_copy(&self.k_cache, pos, k_row)?;
          cuda_copy(&self.v_cache, pos, v_row)?;
      }
  }
  ```

- [ ] **4.2**: Add single-token generation kernel
  - Embedding lookup for current token
  - RoPE rotation at position `global_pos`
  - Attention against full KV cache
  - FFN computation and logits output

- [ ] **4.3**: Implement sampling strategies
  - Argmax (deterministic, good for debugging)
  - Top-k sampling (balanced quality/variety)
  - Temperature scaling (controlled randomness)

### Deliverables
- Updated `Kvcache` struct with write support
- `pesti-runner/src/kernel/generation.rs` - Single-token generation kernel
- Interactive CLI example showing autoregressive generation

---

## Phase 5: Performance Optimization & Profiling

### Goal
Achieve target throughput of ~72 tok/s (llama.cpp baseline).

### Current Performance (Week 11)
| Metric | Value | Baseline | Gap |
|--------|-------|----------|-----|
| Prefill (seq_len=16) | 5,285 tok/s | 15,000 tok/s | 35% |
| Generation | ~263M tok/s* | 85 tok/s | **Artificially high** |

*Note: Current generation benchmark is fake (no actual kernel execution)

### Optimization Strategy

#### Priority 1: Memory Bandwidth (40% of bottleneck)
- [ ] **5.1**: Quantize KV cache to FP16 (2x reduction)
- [ ] **5.2**: Implement KV cache paging (avoid reallocations)
- [ ] **5.3**: Use pinned host memory for faster CUDA transfers

#### Priority 2: Kernel Fusion (30% of bottleneck)
- [ ] **5.4**: Fuse QKV projections into single kernel
- [ ] **5.5**: Combine softmax + output projection
- [ ] **5.6**: Merge FFN up/down projections

#### Priority 3: Parallelism (20% of bottleneck)
- [ ] **5.7**: Batch multiple sequences simultaneously
- [ ] **5.8**: Use warp-level parallelism for attention heads
- [ ] **5.9**: Optimize thread block sizing for sm_8.9 (RTX 4070 Ti SUPER)

#### Priority 4: Algorithmic Improvements (10% of bottleneck)
- [ ] **5.10**: Implement flash attention variant (if time permits)
- [ ] **5.11**: Cache RoPE frequencies across generations
- [ ] **5.12**: Use tensor cores for matrix multiplications

### Profiling Tools
```bash
# NVIDIA Nsight Compute
ncu --launch-skip 0 --launch-wait \
    -f -o pesti_profile \
    cargo run --example benchmark --features cuda

# Parse results
ncu -s pesti_profile.ncd
```

### Deliverables
- `docs/performance-optimization.md` - Optimization roadmap and profiling results
- Benchmark showing improvement over Week 11 baseline
- Target: ≥ 50 tok/s sustained generation throughput

---

## Week 12 Daily Plan

### Day 1-2: Numerical Conformance
- Set up llama.cpp reference outputs
- Implement token-level comparison
- Debug any numerical discrepancies

### Day 3-4: CUDA Attention Kernel
- Design and implement basic attention kernel
- Add softmax and output projection
- Verify numerical correctness vs CPU version

### Day 5: RoPE Integration
- Implement RoPE frequency computation
- Add CUDA RoPE kernel
- Validate positional awareness

### Day 6: KV Cache Updates
- Implement cache write method
- Add single-token generation loop
- Test autoregressive behavior

### Day 7: Optimization & Polish
- Profile and optimize bottlenecks
- Run final conformance tests
- Document results and lessons learned

---

## Success Criteria

### Must-Have (Week 12 Completion)
- ✅ Numerical conformance test passes (< 1e-4 error vs llama.cpp)
- ✅ All attention kernels running on GPU
- ✅ KV cache updates working during generation
- ✅ End-to-end throughput ≥ 50 tok/s

### Nice-to-Have (If time permits)
- ⭐ Flash attention variant implementation
- ⭐ Batch inference with multiple sequences
- ⭐ Tensor core acceleration for matrix ops

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA kernel bugs causing incorrect outputs | Medium | High | Start with simple kernels, verify numerically before optimizing |
| Performance worse than CPU fallback | Low | Medium | Profile early, fall back to CPU if needed |
| Time constraints preventing all optimizations | High | Low | Prioritize conformance first, optimization second |
| Memory bandwidth bottleneck on RTX 4070 Ti SUPER | Medium | Medium | Use FP16 KV cache, minimize global memory access |

---

## Final Deliverables (End of Week 12)

### Code
- `pesti-runner/src/kernel/attention.cu` - CUDA attention kernel
- `pesti-runner/src/kernel/rope.rs` - RoPE embeddings
- `pesti-runner/src/kernel/generation.rs` - Autoregressive loop
- `pesti-runner/examples/conformance_reference.rs` - Reference comparison tool

### Documentation
- `docs/week-12-numerical-conformance.md` - Accuracy analysis
- `docs/performance-benchmarks.md` - Throughput vs llama.cpp
- `CONTRIBUTING.md` - Guide for upstream contributions to llama.cpp

### Artifacts
- Verified PESTI binary matching llama.cpp outputs
- Benchmark suite showing performance parity
- Open issue/PR template for potential upstream contributions

---

## Conclusion

Week 12 transforms PESTI from infrastructure prototype to production-ready inference engine. The focus shifts from "does it work?" to "does it work correctly and efficiently?".

**Target**: Match llama.cpp numerical accuracy while maintaining GPU acceleration benefits.

**Success**: A fully functional, numerically conformant LLM inference engine ready for real-world use or upstream contribution.

---
*Last updated: Week 11/12 completion (August 14, 2026)*
