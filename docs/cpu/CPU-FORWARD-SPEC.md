# CPU Forward Pass - Numerical Conformance Specification

## Overview
This document defines the exact algorithm for PESTI's CPU forward pass, with reference to llama.cpp implementation. Goal: **lock in CPU algo first**, then map to GPU when hardened.

---

## 1. Model Architecture Summary

### Qwen2.5-0.5B Configuration (Test Model)
```rust
embed_dim: 896
num_heads: 8
num_kv_heads: 8
head_dim: 112 (896 / 8)
intermediate_dim: 3072
vocab_size: 32000
max_seq_len: 2048
rope_base: 10000.0
rms_norm_eps: 1e-5
```

---

## 2. Forward Pass Pipeline

### 2.1 Token Embedding
**Input**: `token_id: u32`  
**Output**: `[embed_dim]` vector of f32

```rust
// Linear lookup from embedding table
let embedding = token_embeddings[token_id]; // [embed_dim]
```

**Reference**: llama.cpp uses direct weight lookup, no projection.

---

### 2.2 RMSNorm (Pre-Attention)
**Input**: `[embed_dim]`  
**Output**: `[embed_dim]` normalized vector

```rust
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut output = vec![0.0; x.len()];
    
    // Compute RMS
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / x.len() as f32).sqrt();
    
    // Normalize and scale
    for (i, &x_val) in x.iter().enumerate() {
        output[i] = weight[i] * (x_val / (rms + eps));
    }
    
    output
}
```

**Reference**: Identical to llama.cpp `ggml_rms_norm` (see `ggml-cpu.cpp`).

---

### 2.3 RoPE (Rotary Positional Embeddings)
**Input**: `[num_heads * head_dim]` query/key vectors  
**Output**: Rotated vectors in-place

```rust
fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, base: f32, head_dim: usize) {
    let dim_half = head_dim / 2;
    
    // Compute frequencies: theta_i = base^(-2i/dim)
    let theta: Vec<f32> = (0..dim_half)
        .map(|i| base.powf(-(i as f32) / dim_half as f32))
        .collect();
    
    for head in 0..num_heads {
        for (i, &freq) in theta.iter().enumerate() {
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            
            let idx = head * head_dim + i;
            let next = idx + dim_half;
            
            // Apply rotation
            let q_orig = q[idx];
            let q_next = q[next];
            let k_orig = k[idx];
            let k_next = k[next];
            
            q[idx] = q_orig * cos - q_next * sin;
            q[next] = q_orig * sin + q_next * cos;
            
            k[idx] = k_orig * cos - k_next * sin;
            k[next] = k_orig * sin + k_next * cos;
        }
    }
}
```

**Reference**: llama.cpp `ggml_rope_norm` in `fattn-common.cuh`. Key differences:
- llama.cpp uses **f16** internally for GPU kernels (this CPU impl uses f32)
- llama.cpp applies RoPE at **append time** to KV cache, not at query time

---

### 2.4 Scaled Dot-Product Attention (Core Algorithm)

**Input**: 
- `q_rotated`: `[num_heads * head_dim]` (single token's query after RoPE)
- `k_cache`: `[cache_len, num_kv_heads, head_dim]` (KV cache)
- `v_cache`: `[cache_len, num_kv_heads, head_dim]` (KV cache)
- `scale`: `1.0 / sqrt(head_dim)`

**Output**: `[embed_dim]` attention output

#### Step 2.4.1: Q @ K^T → Attention Scores
```rust
fn compute_attention_scores(q_head: &[f32], k_cache: &[Vec<f32>]) -> Vec<f32> {
    let mut scores = vec![0.0; k_cache.len()];
    
    for (t, k_pos) in k_cache.iter().enumerate() {
        // Dot product for this head
        let dot: f32 = q_head.iter().zip(k_pos.iter()).map(|(a, b)| a * b).sum();
        scores[t] = dot * scale;
    }
    
    scores
}
```

**Reference**: llama.cpp `vec_dot_KQ` in `fattn-common.cuh`. Same algorithm, but:
- llama.cpp uses **tensor core MMA** (mma.sync) for Q @ K^T
- llama.cpp processes **blocks of 256 KV rows** per softmax rescaling (see `nbatch_fa`)

#### Step 2.4.2: Softmax with Numerical Stability
```rust
fn softmax(scores: &[f32]) -> Vec<f32> {
    // Find max for numerical stability (llama.cpp FATTN_KQ_MAX_OFFSET)
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(scores - max)
    let exps: Vec<f32> = scores.iter()
        .map(|&s| (s - max_val).exp())
        .collect();
    
    // Normalize
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / scores.len() as f32; scores.len()]
    }
}
```

**Reference**: llama.cpp `softmax` in `fattn.cu`. Key differences:
- llama.cpp uses **KQ_MAX_OFFSET = 3.0 * log(2)** to shift range (see `fattn-common.cuh`)
- llama.cpp flushes exp values < -20.0 to zero (`SOFTMAX_FTZ_THRESHOLD`)
- llama.cpp accumulates in **FP32** even with f16 inputs

#### Step 2.4.3: Softmax @ V → Output per Head
```rust
fn weighted_v_sum(softmax_weights: &[f32], v_cache: &[Vec<f32>], head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; head_dim];
    
    for (t, v_pos) in v_cache.iter().enumerate() {
        let weight = softmax_weights[t];
        for d in 0..head_dim {
            output[d] += weight * v_pos[d];
        }
    }
    
    output
}
```

**Reference**: llama.cpp `VKQ` accumulator in `fattn-mma-f16.cuh`. Same algorithm.

#### Step 2.4.4: Output Projection
```rust
fn wo_projection(attn_output: &[f32], wo_weight: &Linear) -> Vec<f32> {
    // Linear layer: output = attn_output @ W_o^T
    wo_weight.forward(attn_output, 1)
}
```

**Reference**: llama.cpp `ggml_matmul` for output projection.

---

### 2.5 Feed-Forward Network (SwiGLU)

**Input**: `[intermediate_dim * 2]` (gate + up projections)  
**Output**: `[intermediate_dim]` FFN output

```rust
fn swiglu(gate: &[f32], up: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0; size];
    
    for i in 0..size {
        // SiLU activation: x / (1 + exp(-x)) with numerical stability
        let sigmoid = if gate[i] >= 0.0 {
            1.0 / (1.0 + (-gate[i]).exp())
        } else {
            gate[i] / (1.0 + gate[i].exp())
        };
        
        output[i] = sigmoid * gate[i] * up[i];
    }
    
    output
}

fn ffn_forward(h: &[f32], w1: &Linear, w2: &Linear, w3: &Linear) -> Vec<f32> {
    let gate = w1.forward(h, 1); // [intermediate_dim]
    let up = w3.forward(h, 1);   // [intermediate_dim]
    
    let swiglu_out = swiglu(&gate, &up, intermediate_dim);
    w2.forward(&swiglu_out, 1)  // [embed_dim]
}
```

**Reference**: llama.cpp `ggml_swiglu` in `ggml-cpu.cpp`. Same algorithm.

---

### 2.6 Residual Connections

```rust
fn residual_add(x: &[f32], residual: &[f32]) -> Vec<f32> {
    x.iter().zip(residual.iter()).map(|(a, b)| a + b).collect()
}
```

**Reference**: llama.cpp uses fused add in most kernels. Same algorithm.

---

## 3. Complete Layer Forward (With KV Cache)

```rust
fn transformer_layer_forward(
    x: &[f32],              // [embed_dim]
    kv_cache: &mut LayerKvCache,
    pos: usize,
) -> Vec<f32> {
    let embed_dim = x.len();
    
    // 1. Pre-attention RMSNorm
    let normed = rms_norm(x, &attention_norm.weight, eps);
    
    // 2. Q/K/V projections
    let q_proj = wq.forward(&normed, 1);      // [num_heads * head_dim]
    let k_proj = wk.forward(&normed, 1);      // [num_kv_heads * head_dim]
    let v_proj = wv.forward(&normed, 1);      // [num_kv_heads * head_dim]
    
    // 3. Apply RoPE (per-head, per-position)
    let mut q_rope = q_proj.clone();
    let mut k_rotated = k_proj.clone();
    apply_rope_single(&mut q_rope, num_heads, pos);
    apply_rope_single(&mut k_rotated, num_kv_heads, pos);
    
    // 4. Append K (RoPE-rotated) and V to cache
    kv_cache.append(&k_rotated, &v_proj);
    
    // 5. Compute attention against full cache
    let scale = 1.0 / (head_dim as f32).sqrt();
    let cache_len = kv_cache.seq_len();
    
    let mut attn_output = vec![0.0; embed_dim];
    
    for head in 0..num_heads {
        let kv_group = head / (num_heads / num_kv_heads);
        
        // Q for this head
        let q_head = &q_rope[head * head_dim..(head + 1) * head_dim];
        
        // K, V for this head from cache
        let k_head = kv_cache.k_head(kv_group); // [cache_len, head_dim]
        let v_head = kv_cache.v_head(kv_group); // [cache_len, head_dim]
        
        // Q @ K^T → scores
        let mut scores = vec![0.0; cache_len];
        for t in 0..cache_len {
            let k_pos = &k_head[t * head_dim..(t + 1) * head_dim];
            let dot: f32 = q_head.iter().zip(k_pos.iter()).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        
        // Softmax
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / cache_len as f32; cache_len]
        };
        
        // Softmax @ V → output head
        let out_head = &mut attn_output[head * head_dim..(head + 1) * head_dim];
        for t in 0..cache_len {
            let v_pos = &v_head[t * head_dim..(t + 1) * head_dim];
            for d in 0..head_dim {
                out_head[d] += weights[t] * v_pos[d];
            }
        }
    }
    
    // Output projection
    let attn_out = wo.forward(&attn_output, 1);
    
    // Residual: x + attn_out
    let mut h = residual_add(x, &attn_out);
    
    // FFN sub-layer
    let normed_ffn = rms_norm(&h, &ffn_norm.weight, eps);
    let ffn_out = ffn_forward(&normed_ffn, &w1, &w2, &w3);
    
    // Residual: h + ffn_out
    residual_add(&h, &ffn_out)
}
```

---

## 4. Key Differences from llama.cpp GPU Implementation

| Aspect | PESTI CPU | llama.cpp GPU |
|--------|-----------|---------------|
| **Data type** | f32 throughout | f16 weights, f32 accumulators |
| **Q @ K^T** | Scalar dot product | Tensor core MMA (mma.sync) |
| **Softmax** | CPU host, FP32 | GPU device, with KQ_MAX_OFFSET shift |
| **RoPE** | Applied at query time | Applied at cache append time |
| **KV cache** | Host memory (f32) | Device memory (f16 for quantized K/V) |
| **Numerical stability** | Standard softmax max trick | FATTN_KQ_MAX_OFFSET + FTZ threshold |

---

## 5. Conformance Test Plan

### Phase 1: Unit Tests (Component-Level)

#### Test 1.1: RMSNorm
```rust
#[test]
fn test_rms_norm_numerical() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let eps = 1e-5;
    
    let output = rms_norm(&input, &weight, eps);
    
    // Reference: compute manually
    let expected: Vec<f32> = input.iter()
        .map(|&x| x / ((input.iter().map(|&v| v*v).sum::<f32>() / 5.0).sqrt() + eps))
        .collect();
    
    for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((out - exp).abs() < 1e-6, "RMSNorm mismatch at {}: {} vs {}", i, out, exp);
    }
}
```

#### Test 1.2: RoPE
```rust
#[test]
fn test_rope_numerical() {
    let head_dim = 8;
    let base = 10000.0;
    let pos = 5;
    
    let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut k = q.clone();
    
    apply_rope(&mut q, &mut k, pos, base, head_dim);
    
    // Verify rotation matrix application (manual computation)
    // ...
}
```

#### Test 1.3: Softmax
```rust
#[test]
fn test_softmax_numerical() {
    let scores = vec![2.0, 1.0, 0.1];
    let output = softmax(&scores);
    
    // Verify sum to 1.0
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    
    // Verify against known values
    let expected = vec![0.701, 0.241, 0.058]; // approximate
    for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!((out - exp).abs() < 1e-3);
    }
}
```

#### Test 1.4: SwiGLU
```rust
#[test]
fn test_swiglu_numerical() {
    let gate = vec![1.0, 2.0, 3.0];
    let up = vec![0.5, 0.6, 0.7];
    
    let output = swiglu(&gate, &up, 3);
    
    // Manual verification
    let expected_0 = (1.0 / (1.0 + (-1.0).exp())) * 1.0 * 0.5;
    assert!((output[0] - expected_0).abs() < 1e-6);
}
```

### Phase 2: Integration Tests (Layer-Level)

#### Test 2.1: Single Transformer Layer (No Cache)
```rust
#[test]
fn test_single_layer_forward() {
    let config = TransformerConfig {
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 112,
        embed_dim: 896,
        intermediate_dim: 3072,
        // ...
    };
    
    let input = vec![0.01; config.embed_dim];
    
    let cpu_output = cpu_layer.forward(&input, 1, 1, 0);
    
    // Compare against llama.cpp reference output (if available)
    // For now, verify dimensional consistency
    assert_eq!(cpu_output.len(), config.embed_dim);
}
```

#### Test 2.2: Single Layer With KV Cache
```rust
#[test]
fn test_layer_forward_with_cache() {
    let mut kv_cache = LayerKvCache::new(num_heads, num_kv_heads, head_dim, max_seq_len);
    
    let input = vec![0.01; embed_dim];
    
    for pos in 0..10 {
        let output = layer.forward_with_cache(&input, &mut kv_cache, pos);
        
        // Verify output dimension
        assert_eq!(output.len(), embed_dim);
        
        // Verify cache grows correctly
        assert_eq!(kv_cache.seq_len(), pos + 1);
    }
}
```

### Phase 3: Full Model Conformance

#### Test 3.1: End-to-End Forward (Single Token)
```rust
#[test]
fn test_full_model_single_token() {
    let model_path = "/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let token_ids = vec![1, 2, 3]; // "The"
    
    // Run PESTI CPU forward
    let pesti_logits = model.forward(&token_ids, 0).unwrap();
    
    // Run llama.cpp reference (via CLI or FFI)
    // llama.cpp --model $model_path -n 3 -p "The"
    
    // Compare logits (tolerance: 1e-4 for f32)
    for (i, (p, l)) in pesti_logits.iter().zip(llama_logits.iter()).enumerate() {
        let diff = (p - l).abs();
        assert!(diff < 1e-4, "Logit mismatch at {}: PESTI={}, llama.cpp={}", i, p, l);
    }
}
```

#### Test 3.2: Autoregressive Generation Conformance
```rust
#[test]
fn test_autoregressive_generation() {
    let prompt = "The quick brown fox";
    let max_tokens = 50;
    
    // PESTI generation
    let pesti_tokens = generate_autoregressive(&model, prompt, max_tokens);
    
    // llama.cpp generation (via CLI)
    let llama_tokens = run_llama_cpp_server(prompt, max_tokens);
    
    // Compare token sequences
    assert_eq!(pesti_tokens, llama_tokens);
}
```

---

## 6. Numerical Tolerance Guidelines

| Component | Tolerance | Rationale |
|-----------|-----------|-----------|
| RMSNorm | 1e-6 | Pure FP32 math, no accumulation errors |
| RoPE | 1e-6 | Sin/cos from standard library |
| Softmax | 1e-5 | exp() approximation + division |
| Q @ K^T | 1e-5 | Accumulation over head_dim=112 |
| Softmax @ V | 1e-5 | Weighted sum accumulation |
| Full layer | 1e-4 | Cumulative errors across components |
| Full model logits | 1e-3 | Many layers, quantization effects |

---

## 7. Next Steps: GPU Mapping

Once CPU is hardened (all tests pass):

### Step 7.1: Identify Parallelizable Operations
- **Q @ K^T**: Each head → independent CUDA block
- **Softmax**: Each position/head pair → warp reduction
- **Softmax @ V**: Each output dim → parallel accumulation

### Step 7.2: Choose Kernel Strategy
**Option A (Current)**: GEMM-based attention via existing `CudaGemmKernel`
- Pro: Reuses working GEMM infrastructure
- Con: H2D transfers for softmax, 2 GEMM ops

**Option B**: Dedicated Flash Attention PTX kernel
- Pro: Single kernel, no H2D transfers
- Con: Complex implementation, needs PTX generation

### Step 7.3: Numerical Parity Strategy
- Use **f32 accumulators** on GPU (like llama.cpp)
- Apply **KQ_MAX_OFFSET** shift for softmax stability
- Match **softmax FTZ threshold** (-20.0)

---

## 8. Files to Create/Modify

### New Files:
1. `tests/cpu_attention_numerical.rs` - Unit tests for attention components
2. `tests/rope_numerical.rs` - RoPE conformance tests
3. `tests/swiglu_numerical.rs` - FFN component tests
4. `docs/CPU-FORWARD-SPEC.md` - This document (refined)

### Modified Files:
1. `pesti-runner/src/transformer/layer.rs` - Add KV cache struct if missing
2. `pesti-runner/src/transformer/model.rs` - Verify forward_with_cache implementation
3. `Cargo.toml` - Add `llama-cpp-2 = "0.1.146"` as dev-dependency for reference

---

## 9. Success Criteria

✅ **Phase 1**: All unit tests pass with tolerance ≤ 1e-5  
✅ **Phase 2**: Layer tests pass with tolerance ≤ 1e-4  
✅ **Phase 3**: Full model logits within 1e-3 of llama.cpp reference  
✅ **Documentation**: Complete spec with numerical formulas and references

Once all criteria met → **CPU is hardened**, proceed to GPU mapping.
