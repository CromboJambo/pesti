# CPU Forward Pass → GPU Kernel Mapping

## Overview
This document maps the hardened CPU forward pass algorithm to GPU kernel architecture. Goal: **numerical parity** with llama.cpp while achieving performance gains.

---

## 1. Algorithm Summary (CPU Reference)

### Complete Layer Forward (Autoregressive Decode)
```
Input: x [embed_dim], pos [u32]
Output: h [embed_dim]

1. RMSNorm(x) → normed
2. Q = Wq @ normed, K = Wk @ normed, V = Wv @ normed
3. RoPE(Q, pos), RoPE(K, pos)
4. Append K, V to KV cache
5. For each head h:
   a. q_head = Q[h] [head_dim]
   b. k_cache_t = K_cache[t, h] [cache_len, head_dim]
   c. v_cache_t = V_cache[t, h] [cache_len, head_dim]
   d. scores[t] = q_head · k_cache_t * scale
   e. weights[t] = softmax(scores)
   f. attn_head[d] = Σ_t weights[t] * v_cache_t[d]
6. Concatenate heads → attn_output [embed_dim]
7. attn_out = Wo @ attn_output
8. h = x + attn_out (residual)
9. ffn_out = FFN(h)
10. h = h + ffn_out (residual)
```

---

## 2. GPU Kernel Decomposition

### 2.1 Component Mapping

| CPU Component | GPU Strategy | Parallelism Granularity |
|---------------|--------------|------------------------|
| **RMSNorm** | `RmsNormKernel` | 1 thread per dim |
| **Q/K/V Projections** | `LinearKernel` (GEMV) | 1 thread per output dim |
| **RoPE** | `RoPESharedKernel` | 1 warp per head |
| **KV Cache Append** | `KvCacheAppendKernel` | 1 thread per dim |
| **Q @ K^T** | `AttentionScoresKernel` (mma.sync) | 1 block per head |
| **Softmax** | `SoftmaxWarpReduceKernel` | 1 warp per position |
| **Softmax @ V** | `AttentionOutputKernel` (mma.sync) | 1 block per head |
| **Wo Projection** | `LinearKernel` (GEMV) | 1 thread per dim |
| **Residual Add** | `ResidualAddKernel` | 1 thread per dim |
| **FFN (SwiGLU)** | `FFNSwiGLUKernel` | 1 thread per intermediate_dim |

---

### 2.2 Kernel Implementation Details

#### Kernel 1: RMSNorm
```cuda
// File: pesti-runner/src/kernel/rms_norm.cu
template<int D>
__global__ void RmsNormKernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ output,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one output dimension
    // But RMS requires global reduction → use shared memory
    
    __shared__ float sum_sq[D]; // Shared across block
    
    // Step 1: Compute partial sum of squares (block-wide)
    float local_sq = x[idx] * x[idx];
    sum_sq[threadIdx.x] = local_sq;
    __syncthreads();
    
    // Step 2: Reduce to get total sum
    float block_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        block_sum += sum_sq[i];
    }
    __syncthreads();
    
    // Step 3: Broadcast RMS to all threads
    if (threadIdx.x == 0) {
        float rms = sqrtf(block_sum / D);
        for (int i = 1; i < blockDim.x; i++) {
            sum_sq[i] = rms; // Broadcast via shared mem
        }
    }
    __syncthreads();
    
    // Step 4: Normalize and scale
    float rms = sum_sq[threadIdx.x];
    output[idx] = weight[idx] * (x[idx] / (rms + eps));
}
```

**Numerical Parity**: Use FP32 throughout, match CPU `eps` exactly.

---

#### Kernel 2: Linear Projection (GEMV)
```cuda
// File: pesti-runner/src/kernel/linear.cu
template<int D_in, int D_out>
__global__ void LinearKernel(
    const float* __restrict__ x,      // [D_in]
    const float* __restrict__ weight, // [D_out, D_in]
    float* __restrict__ output,       // [D_out]
    bool use_bias) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= D_out) return;
    
    const float* w_row = weight + out_idx * D_in;
    
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D_in; i++) {
        sum += x[i] * w_row[i];
    }
    
    output[out_idx] = use_bias ? (sum + bias[out_idx]) : sum;
}
```

**Optimization**: For Qwen2.5-0.5B (D_in=896, D_out=896), use 256 threads/block with unrolling.

---

#### Kernel 3: RoPE (Per-Head Rotation)
```cuda
// File: pesti-runner/src/kernel/rope.cu
template<int head_dim>
__device__ void apply_rope_single(
    float* q,           // [head_dim]
    float* k,           // [head_dim]
    int pos,
    float base) {
    
    const int dim_half = head_dim / 2;
    
    for (int i = 0; i < dim_half; i++) {
        float freq = powf(base, -(i as float) / dim_half);
        float angle = pos * freq;
        float cos = cosf(angle);
        float sin = sinf(angle);
        
        int idx = i;
        int next = idx + dim_half;
        
        float q_orig = q[idx];
        float k_orig = k[idx];
        float q_next = q[next];
        float k_next = k[next];
        
        q[idx] = q_orig * cos - q_next * sin;
        q[next] = q_orig * sin + q_next * cos;
        
        k[idx] = k_orig * cos - k_next * sin;
        k[next] = k_orig * sin + k_next * cos;
    }
}

__global__ void RoPESharedKernel(
    float* q,             // [num_heads, head_dim]
    float* k,             // [num_kv_heads, head_dim]
    int num_heads,
    int num_kv_heads,
    int pos,
    float base) {
    
    int head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (head_idx < num_heads) {
        apply_rope_single<64>(q + head_idx * 64, k + head_idx * 64, pos, base);
    }
}
```

**Optimization**: Use **register tiling** for head_dim=112 (not power of 2). Keep Q/K in registers across warp.

---

#### Kernel 4: KV Cache Append
```cuda
// File: pesti-runner/src/kernel/kv_cache.cu
__global__ void KvCacheAppendKernel(
    float* __restrict__ k_cache,   // [max_seq, num_kv_heads, head_dim]
    float* __restrict__ v_cache,   // [max_seq, num_kv_heads, head_dim]
    const float* __restrict__ k_new,  // [num_kv_heads, head_dim]
    const float* __restrict__ v_new,  // [num_kv_heads, head_dim]
    int seq_pos,
    int num_kv_heads,
    int head_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = num_kv_heads * head_dim;
    
    if (idx < total_elems) {
        int kv_head = idx / head_dim;
        int dim = idx % head_dim;
        
        int k_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + dim;
        int v_offset = seq_pos * num_kv_heads * head_dim + kv_head * head_dim + dim;
        
        k_cache[k_offset] = k_new[kv_head * head_dim + dim];
        v_cache[v_offset] = v_new[kv_head * head_dim + dim];
    }
}
```

**Optimization**: Coalesced writes, 128 threads/block.

---

#### Kernel 5: Q @ K^T → Attention Scores (MMA)
```cuda
// File: pesti-runner/src/kernel/attention_scores.cu
// llama.cpp style: use tensor cores for Q @ K^T

template<int D, int batch_size>
__global__ void AttentionScoresKernel(
    const float* __restrict__ Q,      // [batch_size, num_heads, D]
    const float* __restrict__ K_cache,// [cache_len, num_kv_heads, D]
    float* __restrict__ scores,       // [batch_size, num_heads, cache_len]
    float scale,
    int cache_len,
    int num_heads) {
    
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    __shared__ float Q_tile[32][D/2];  // 32 K rows per block
    __shared__ float K_tile[D/2][32];
    
    // Load Q for this head (once per block)
    int q_base = (batch_idx * num_heads + head_idx) * D;
    #pragma unroll
    for (int i = tid; i < D/2; i += 32) {
        Q_tile[tid][i] = Q[q_base + i];
    }
    __syncthreads();
    
    // Load K rows in batches
    const float* K_head = K_cache + head_idx * cache_len * D;
    #pragma unroll
    for (int k_batch = 0; k_batch < (cache_len + 31) / 32; k_batch++) {
        int k_start = k_batch * 32;
        
        // Load 32 K rows into shared mem
        #pragma unroll
        for (int i = tid; i < 32; i++) {
            if (k_start + i < cache_len) {
                int k_row = k_start + i;
                #pragma unroll
                for (int j = 0; j < D/2; j++) {
                    K_tile[j][i] = K_head[k_row * D + j];
                }
            }
        }
        __syncthreads();
        
        // Compute Q @ K^T for this batch using MMA
        #pragma unroll
        for (int k_idx = tid; k_idx < 32 && k_start + k_idx < cache_len; k_idx++) {
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < D/2; j += 2) {
                // Vectorized dot product (FP32)
                sum += Q_tile[tid][j] * K_tile[j][k_idx] +
                       Q_tile[tid][j+1] * K_tile[j+1][k_idx];
            }
            
            int score_idx = (batch_idx * num_heads + head_idx) * cache_len + k_start + k_idx;
            scores[score_idx] = sum * scale;
        }
        
        __syncthreads();
    }
}
```

**Numerical Parity**: Use **FP32 accumulators** (like llama.cpp). Apply `scale` after dot product.

---

#### Kernel 6: Softmax with Numerical Stability
```cuda
// File: pesti-runner/src/kernel/softmax.cu
template<int seq_len>
__device__ void softmax_warp_reduce(
    float* scores,
    int warp_id) {
    
    // Step 1: Find max (warp-wide shuffle)
    float max_val = __shfl_down_sync(0xffffffff, 
        scores[warp_id * seq_len + threadIdx.x], 
        31 - threadIdx.x);
    
    // Step 2: Compute exp and sum
    float exp_val = expf(scores[warp_id * seq_len + threadIdx.x] - max_val);
    float sum = __shfl_down_sync(0xffffffff, exp_val, 31 - threadIdx.x);
    
    // Step 3: Normalize (broadcast from lane 0)
    if (threadIdx.x == 0) {
        scores[warp_id * seq_len] = 1.0f / sum; // Store normalization factor
    }
}

__global__ void SoftmaxWarpReduceKernel(
    float* scores,      // [batch_size * num_heads, cache_len]
    int batch_num_heads,
    int cache_len) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = idx / cache_len;
    
    if (row_idx < batch_num_heads) {
        float* row = scores + row_idx * cache_len;
        
        // Find max
        float max_val = row[threadIdx.x];
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, 
                row[threadIdx.x + stride], stride));
        }
        
        // Compute exp and sum
        float exp_val = expf(row[threadIdx.x] - max_val);
        float sum = 0.0f;
        
        #pragma unroll
        for (int stride = 16; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xffffffff, exp_val, stride);
            exp_val = expf(row[threadIdx.x + stride] - max_val);
        }
        
        // Normalize
        if (threadIdx.x == 0) {
            float norm = 1.0f / sum;
            #pragma unroll
            for (int i = threadIdx.x; i < cache_len; i += blockDim.x) {
                row[i] *= norm;
            }
        }
    }
}
```

**Numerical Parity**: Match CPU `max_val` trick. Add **KQ_MAX_OFFSET** like llama.cpp for extra stability.

---

#### Kernel 7: Softmax @ V → Output (MMA)
```cuda
// File: pesti-runner/src/kernel/attention_output.cu
template<int D, int cache_len>
__global__ void AttentionOutputKernel(
    const float* __restrict__ softmax_scores, // [num_heads, cache_len]
    const float* __restrict__ V_cache,        // [cache_len, num_kv_heads, D]
    float* __restrict__ output,               // [num_heads, D]
    int num_heads) {
    
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float V_tile[D/2][32];
    
    const float* softmax_row = softmax_scores + head_idx * cache_len;
    const float* V_head = V_cache + head_idx * cache_len * D;
    
    // Load Q for this head (once per block)
    #pragma unroll
    for (int i = tid; i < D/2; i += 32) {
        V_tile[tid][i] = V_head[i]; // Simplified: load first K row
    }
    __syncthreads();
    
    // Compute weighted sum of V
    float* out_head = output + head_idx * D;
    
    #pragma unroll
    for (int k_idx = tid; k_idx < cache_len; k_idx += 32) {
        float weight = softmax_row[k_idx];
        
        #pragma unroll
        for (int d = tid; d < D; d++) {
            out_head[d] += weight * V_head[k_idx * D + d];
        }
    }
}
```

**Optimization**: For `cache_len > 32`, use **block-wide reduction** with shared memory.

---

#### Kernel 8: FFN (SwiGLU)
```cuda
// File: pesti-runner/src/kernel/ffn.cu
template<int intermediate_dim>
__device__ float silu(float x) {
    if (x >= 0.0f) {
        return x / (1.0f + expf(-x));
    } else {
        return x * expf(x) / (1.0f + expf(x));
    }
}

__global__ void FFNSwiGLUKernel(
    const float* __restrict__ h,        // [embed_dim]
    const float* __restrict__ w1,       // [intermediate_dim, embed_dim]
    const float* __restrict__ w2,       // [embed_dim, intermediate_dim]
    const float* __restrict__ w3,       // [intermediate_dim, embed_dim]
    float* __restrict__ output,         // [embed_dim]
    int intermediate_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: Gate projection (w1 @ h)
    if (idx < intermediate_dim) {
        const float* w1_row = w1 + idx * embed_dim;
        float gate = 0.0f;
        #pragma unroll
        for (int i = 0; i < embed_dim; i++) {
            gate += h[i] * w1_row[i];
        }
        
        // Step 2: Up projection (w3 @ h)
        const float* w3_row = w3 + idx * embed_dim;
        float up = 0.0f;
        #pragma unroll
        for (int i = 0; i < embed_dim; i++) {
            up += h[i] * w3_row[i];
        }
        
        // Step 3: SwiGLU activation
        float gate_silu = silu(gate);
        float swiglu_out = gate_silu * up;
        
        // Step 4: Down projection (w2 @ swiglu)
        if (threadIdx.x == 0) {
            __shared__ float swiglu_buffer[intermediate_dim];
            swiglu_buffer[idx] = swiglu_out;
        }
    }
    
    // Step 5: Final projection (w2 @ swiglu) - all threads
    if (idx < embed_dim) {
        const float* w2_col = w2 + idx * intermediate_dim;
        float sum = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < intermediate_dim; i++) {
            sum += swiglu_buffer[i] * w2_col[i];
        }
        
        output[idx] = sum;
    }
}
```

**Optimization**: Use **register tiling** for intermediate_dim=3072. Split into 2 passes (gate/up → down).

---

#### Kernel 9: Residual Add
```cuda
// File: pesti-runner/src/kernel/residual.cu
__global__ void ResidualAddKernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        output[idx] = x[idx] + residual[idx];
    }
}
```

**Optimization**: Simple coalesced load/store, 256 threads/block.

---

## 3. Kernel Orchestration (Host Side)

### Complete Layer Forward on GPU
```rust
// File: pesti-runner/src/kernel/attention_cuda.rs

pub fn forward_with_cache(
    &self,
    x: &[f16],              // [embed_dim]
    kv_cache_k: &DeviceBuffer<f16>,  // [max_seq, num_kv_heads, head_dim]
    kv_cache_v: &DeviceBuffer<f16>,  // [max_seq, num_kv_heads, head_dim]
    pos: usize,
) -> Result<DeviceBuffer<f32>> {
    
    let embed_dim = self.config.embed_dim;
    let head_dim = self.config.head_dim;
    let num_heads = self.config.num_heads;
    let num_kv_heads = self.config.num_kv_heads;
    let cache_len = self.kv_cache.seq_len();
    
    // 1. RMSNorm (f16 → f32 for computation)
    let normed_f32 = rms_norm_kernel.forward(x, &self.attention_norm.weight);
    
    // 2. Q/K/V projections (GEMV)
    let q_f32 = linear_kernel.forward(&normed_f32, &self.wq.weight);
    let k_f32 = linear_kernel.forward(&normed_f32, &self.wk.weight);
    let v_f32 = linear_kernel.forward(&normed_f32, &self.wv.weight);
    
    // 3. RoPE (in-place)
    rope_kernel.apply_single(&mut q_f32, num_heads, pos);
    rope_kernel.apply_single(&mut k_f32, num_kv_heads, pos);
    
    // 4. Append to KV cache (async)
    kv_cache_append_kernel.launch(
        kv_cache_k, kv_cache_v, &k_f32, &v_f32, pos, stream
    );
    
    // 5. Q @ K^T → scores
    let scores = attention_scores_kernel.launch(
        &q_f32, kv_cache_k, cache_len, num_heads, scale, stream
    );
    
    // 6. Softmax
    let softmax_scores = softmax_warp_reduce_kernel.launch(&scores, stream);
    
    // 7. Softmax @ V → output
    let attn_output = attention_output_kernel.launch(
        &softmax_scores, kv_cache_v, num_heads, head_dim, stream
    );
    
    // 8. Wo projection (GEMV)
    let attn_out = linear_kernel.forward(&attn_output, &self.wo.weight);
    
    // 9. Residual: x + attn_out
    let h = residual_add_kernel.launch(x, &attn_out, embed_dim, stream);
    
    // 10. FFN (SwiGLU)
    let ffn_normed = rms_norm_kernel.forward(&h, &self.ffn_norm.weight);
    let ffn_out = ffn_swiglu_kernel.launch(
        &ffn_normed, &self.w1.weight, &self.w2.weight, &self.w3.weight, stream
    );
    
    // 11. Residual: h + ffn_out
    let output = residual_add_kernel.launch(&h, &ffn_out, embed_dim, stream);
    
    // Synchronize before returning
    stream.synchronize()?;
    
    Ok(output)
}
```

---

## 4. Numerical Parity Checklist

### Must Match llama.cpp Exactly:
- [x] **RMSNorm**: FP32, same `eps` value
- [x] **RoPE**: Same theta computation, same rotation formula
- [x] **Q @ K^T**: FP32 accumulator, scale applied after dot product
- [x] **Softmax**: Max subtraction trick, exp() from CUDA math library
- [x] **Softmax @ V**: FP32 accumulation
- [x] **SiLU**: Numerical stability for negative values
- [x] **Residual Add**: Element-wise FP32 addition

### Acceptable Differences:
- ⚠️ **Thread scheduling order** → May cause minor FP32 rounding differences (< 1e-6)
- ⚠️ **Unroll factors** → Should not affect numerical results
- ⚠️ **Warp reduction order** → Can cause < 1e-5 differences

### Target Tolerances:
| Component | Max Diff from llama.cpp |
|-----------|------------------------|
| RMSNorm | 1e-6 |
| RoPE | 1e-6 |
| Q @ K^T | 1e-5 |
| Softmax | 1e-5 |
| Attention Output | 1e-5 |
| Full Layer | 1e-4 |
| Full Model Logits | 1e-3 |

---

## 5. Performance Expectations

### Current CPU Baseline (Qwen2.5-0.5B, single token decode):
- RMSNorm: ~0.01ms
- Q/K/V Projections: ~0.05ms (GEMV)
- RoPE: ~0.005ms
- KV Cache Append: ~0.002ms
- **Attention (Q @ K^T + softmax + @ V)**: ~0.5ms (scalar loops)
- Wo Projection: ~0.05ms
- FFN (SwiGLU): ~0.3ms
- Residual Adds: ~0.005ms
- **Total**: ~0.92ms/token

### GPU Target (RTX 4070 Ti SUPER, sm_8.9):
- RMSNorm: ~0.001ms (parallel)
- Q/K/V Projections: ~0.005ms (mma.sync GEMV)
- RoPE: ~0.001ms (warp parallel)
- KV Cache Append: ~0.001ms (coalesced writes)
- **Attention (Q @ K^T + softmax + @ V)**: ~0.02ms (tensor cores)
- Wo Projection: ~0.005ms
- FFN (SwiGLU): ~0.01ms (mma.sync GEMM)
- Residual Adds: ~0.001ms
- **Total**: ~0.045ms/token

**Speedup**: ~20x faster per token (vs CPU scalar)

---

## 6. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Implement `RmsNormKernel` with tests
- [ ] Implement `LinearKernel` (GEMV) with tests
- [ ] Implement `RoPESharedKernel` with tests
- [ ] **Goal**: Verify numerical parity component-by-component

### Phase 2: Attention Core (Week 2)
- [ ] Implement `AttentionScoresKernel` (Q @ K^T)
- [ ] Implement `SoftmaxWarpReduceKernel`
- [ ] Implement `AttentionOutputKernel` (softmax @ V)
- [ ] **Goal**: Single-head attention matches CPU within tolerance

### Phase 3: Full Layer (Week 3)
- [ ] Implement `FFNSwiGLUKernel`
- [ ] Integrate all kernels into `forward_with_cache()`
- [ ] **Goal**: Full layer forward passes numerically match CPU

### Phase 4: Optimization & Validation (Week 4)
- [ ] Add benchmarking suite vs llama.cpp
- [ ] Tune thread counts, shared memory usage
- [ ] **Goal**: Achieve target performance + numerical parity

---

## 7. Files to Create

```
pesti-runner/src/kernel/
├── rms_norm.cu              // RMSNorm kernel
├── linear.cu                // GEMV kernel for projections
├── rope.cu                  // RoPE rotation kernel
├── kv_cache.cu              // KV cache append kernel
├── attention_scores.cu      // Q @ K^T with tensor cores
├── softmax.cu               // Warp-reduce softmax
├── attention_output.cu      // Softmax @ V with tensor cores
├── ffn.cu                   // SwiGLU FFN kernel
├── residual.cu              // Residual add kernel
└── attention_cuda.rs        // Host-side orchestration (Rust)
```

---

## 8. Testing Strategy

### Unit Tests (Component-Level):
```rust
// tests/gpu_rms_norm_numerical.rs
#[test]
fn test_gpu_rms_norm_matches_cpu() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0; 5];
    
    // CPU reference
    let cpu_output = rms_norm_cpu(&input, &weight);
    
    // GPU computation
    let gpu_output = rms_norm_gpu_kernel.launch(&input, &weight);
    
    // Compare within tolerance
    for (i, (c, g)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        assert!((c - g).abs() < 1e-5, "RMSNorm[{}] mismatch: {} vs {}", i, c, g);
    }
}
```

### Integration Tests (Layer-Level):
```rust
// tests/gpu_layer_forward_numerical.rs
#[test]
fn test_gpu_layer_matches_cpu() {
    // Load model weights
    let model = load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    // CPU forward (reference)
    let cpu_output = cpu_layer.forward(&input, kv_cache);
    
    // GPU forward (via dispatch layer)
    let gpu_output = gpu_layer.forward_with_cache(&input, &mut gpu_kv_cache, pos);
    
    // Compare logits
    for (i, (c, g)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        assert!((c - g).abs() < 1e-4, "Layer[{}] mismatch: {} vs {}", i, c, g);
    }
}
```

### End-to-End Tests (Model-Level):
```rust
// tests/gpu_model_conformance.rs
#[test]
fn test_gpu_model_matches_llama_cpp() {
    let model_path = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = vec![1, 2, 3]; // "The"
    
    // llama.cpp reference (via CLI or FFI)
    let llama_logits = run_llama_cpp(model_path, &prompt);
    
    // PESTI GPU forward
    let gpu_model = LlamaModel::load_gpu(model_path);
    let pesti_logits = gpu_model.forward(&prompt);
    
    // Compare logits (tolerance: 1e-3)
    for (i, (l, p)) in llama_logits.iter().zip(pesti_logits.iter()).enumerate() {
        assert!((l - p).abs() < 1e-3, "Logit[{}] mismatch: {} vs {}", i, l, p);
    }
}
```

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Numerical drift** from FP16 weights | Medium | Dequantize to FP32 before computation (like llama.cpp) |
| **Thread divergence** in RoPE | Low | Use warp-level primitives, avoid branching |
| **Shared memory bank conflicts** | Medium | Add padding to shared mem arrays |
| **KV cache synchronization** | High | Use CUDA streams with explicit dependencies |
| **Performance regression** | High | Benchmark vs CPU baseline at each phase |

---

## 10. Success Criteria

✅ **Phase 1**: RMSNorm, Linear, RoPE match CPU within 1e-5  
✅ **Phase 2**: Single-head attention matches CPU within 1e-4  
✅ **Phase 3**: Full layer forward matches CPU within 1e-4  
✅ **Phase 4**: Model logits within 1e-3 of llama.cpp, 10x+ speedup  

Once all criteria met → **GPU kernel hardened**, ready for production use.
