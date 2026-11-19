# Spec: F16 GPU Inference via candle_bridge Redesign

**Status:** Draft — Week 23 deliverable  
**Author:** crombo (with Hermes Agent)  
**Last updated:** September 13, 2026

## Problem Statement

pesti-runner's pure Rust inference path currently uses `candle_core::Tensor` through the `candle_bridge` layer for GPU operations. This introduces two critical issues:

1. **Memory bloat (2x overhead):** All data is converted to F32 tensors internally, regardless of input type. For Qwen2.5-0.5B-Instruct (~600MB weights in F16), this requires ~1.2GB for weight tensors alone on GPU, plus activation tensors during forward pass → OOM on consumer GPUs.

2. **Performance ceiling:** F32 GEMM operations process 4 bytes per element instead of 2, halving effective memory bandwidth utilization compared to native F16 inference (cublasHgemm).

## Current Architecture

```
pesti-runner/src/transformer/model.rs
    ↓ calls
pesti-runner/src/kernel/dispatch.rs
    ↓ uses
pesti-runner/src/kernel/candle_bridge.rs
    ↓ wraps
candle_core::Tensor API (always F32 internally)
    ↓ uses
CUDA cuBLAS Sgemm operations
```

### Key bottleneck: `candle_bridge::gemm`

```rust
pub fn gemm(
    a: &[f16],  // input is F16
    b: &[f16],  // weights are F16
    ...
) -> Result<Vec<f32>, candle_core::Error> {
    let device = bridge_device();

    // Converts F16 → F32 tensor (doubles memory!)
    let a_t = Tensor::from_vec(a.to_vec(), (m, k), &device)?;
    let b_t = Tensor::from_vec(b.to_vec(), (k, n), &device)?;

    gemm_with_tensors(&a_t, &b_t, c, m, k, n, alpha, beta)
}
```

## Proposed Architecture

Replace `candle_bridge` with a direct CUDA kernel launcher that passes F16 pointers to cuBLAS Hgemm operations:

```
pesti-runner/src/transformer/model.rs
    ↓ calls
pesti-runner/src/kernel/dispatch.rs
    ↓ uses (REDESIGNED)
pesti-runner/src/kernel/cuda_bridge.rs
    ↓ wraps
cuBLAS cublasHgemm / cublasGemmEx(CUBLAS_COMPUTE_16F)
    ↓ uses
CUDA driver API directly (no candle_core dependency)
```

### Key design principles

1. **True F16 tensors on GPU:** Pass `f16*` pointers directly to cuBLAS, no conversion to F32.
2. **Minimal allocation:** Allocate once per layer during model load, reuse across forward passes.
3. **Zero-copy where possible:** Avoid intermediate host→device copies for cached weight tensors.

## Implementation Plan

### Phase 1: CUDA Bridge Layer (Week 23)

**File:** `pesti-runner/src/kernel/cuda_bridge.rs` (new)

```rust
use cudarc::driver::*;
use half::f16;

pub struct CudaBridge {
    stream: CuStream,
    cublas_handle: cuBLASHandle,
}

impl CudaBridge {
    pub fn new() -> Result<Self, String> { ... }

    /// F16 GEMM: Y = alpha * A @ B + beta * Y
    /// Uses cublasGemmEx with CUBLAS_COMPUTE_16F for true half-precision compute
    pub fn gemm_f16(
        &self,
        a: &DeviceBuffer<f16>,  // [m, k]
        b: &DeviceBuffer<f16>,  // [k, n] (transposed weight)
        y: Option<&mut DeviceBuffer<f32>>,  // output [m, n]
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<DeviceBuffer<f32>, String> { ... }

    /// F16→F32 conversion kernel for attention scores
    pub fn convert_f16_to_f32(&self, input: &DeviceBuffer<f16>, size: usize) 
        -> Result<DeviceBuffer<f32>, String> { ... }
}
```

### Phase 2: Integration with Dispatch Layer (Week 23-24)

**File:** `pesti-runner/src/kernel/dispatch.rs` (modify)

Replace all `candle_bridge::gemm` calls with `cuda_bridge.gemm_f16`:

```rust
// Before:
let result = candle_bridge::gemm(&x_f16, &w_t, None, m, k, n, 1.0, 0.0)?;

// After:
let result = self.cuda_bridge.gemm_f16(
    &x_buf, &w_buf, None, 
    batch_size, k, out_features,
    1.0, 0.0
)?;
```

### Phase 3: Weight Tensor Caching (Week 24)

**File:** `pesti-runner/src/transformer/model.rs` (modify)

Upload weights to GPU once during model load as F16 tensors:

```rust
fn build_layer_dispatch(layer: &TransformerLayer, bridge: &CudaBridge) -> LayerDispatch {
    // Upload weight once, cache on GPU
    let wq_buf = bridge.upload_f16(&layer.attention.wq.weight);
    
    AttentionDispatch {
        wq_buf,  // Cached F16 device buffer
        ...
    }
}
```

## Expected Performance Improvements

| Metric | Current (F32 via candle) | Target (F16 direct CUDA) | Improvement |
|--------|--------------------------|--------------------------|-------------|
| GPU memory usage | ~2.4GB for Qwen2.5-0.5B | ~1.2GB | 2x reduction |
| Memory bandwidth | ~30 GB/s effective | ~60 GB/s effective | 2x improvement |
| Throughput (tok/s) | N/A (OOM) | >82.9 tok/s (match FFI baseline) | First functional GPU inference |

## Risks & Mitigations

1. **cuBLAS API complexity:** Direct cuBLAS calls require careful handle/stream management.  
   *Mitigation:* Wrap in RAII structs with clear ownership semantics.

2. **Numerical precision:** F16 compute may differ from F32 reference.  
   *Mitigation:* Run conformance tests against llama.cpp outputs at seq=4096+ to verify stability.

3. **Platform portability:** Direct CUDA calls reduce CPU-only fallback options.  
   *Mitigation:* Keep existing `#[cfg(feature = "cuda")]` gating; maintain CPU path as separate branch.

## Success Criteria

1. ✅ Qwen2.5-0.5B-Instruct runs end-to-end on RTX 4070 Ti SUPER without OOM
2. ✅ Achieves ≥82.9 tok/s (matches current FFI baseline)
3. ✅ All existing conformance tests pass (numerical correctness preserved)
4. ✅ GPU memory usage <1.5GB for Qwen2.5-0.5B model

## Dependencies

- `cudarc` crate (already used in pesti-runner for CUDA kernel compilation)
- cuBLAS library (standard on any CUDA installation)
- No new system dependencies required

---
*This spec supersedes the previous approach of trying to make candle_core work with F16 data. The direct CUDA bridge provides true half-precision inference without abstraction overhead.*