# CPU Attention Kernel Optimization Plan

## Problem
Current `CpuAttentionKernel` uses naive triple-nested loops, achieving ~80 tok/s
on Qwen2.5-0.5B config (8 heads × 64 dim × 256 tokens). GPU achieves 5844 tok/s
(73x). Most of this gap is avoidable CPU inefficiency, not fundamental hardware limits.

## Goal
Push CPU to ~400-600 tok/s using **only stdlib** (`core::simd`) + minimal deps
(`rayon`), aligning with the project's objective to reduce LLVM/non-Rust FFI.

## Strategy: SIMD via `std::simd` (Rust 1.85+, no external crates)

### Key Insight
`std::simd` in stdlib compiles to the same efficient SIMD code as LLVM
auto-vectorization, but without relying on LLVM's loop-nest analysis. Explicit
lane-based programming gives deterministic performance.

### Changes

**1. SIMD inner product for Q·K computation**
File: `pesti-runner/src/kernel/attention.rs`

Replace scalar dot-product:
```rust
// Before: scalar accumulation
for d in 0..head_dim {
    acc += q[head * head_dim + d] * k[offset + d];
}
```

With explicit SIMD:
```rust
use std::simd::{f32x8, SimdFloat};

// Process 8 elements per iteration
for chunk in head_dim / 8 {
    let q_vec = f32x8::from_slice(&q[q_offset + chunk * 8..]);
    let k_vec = f32x8::from_slice(&k[k_offset + chunk * 8..]);
    acc += (q_vec * k_vec).sum();
}
// Handle remainder
```

**Expected gain: ~4-8x** (depending on head_dim and AVX2 vs AVX-512)

**2. Parallelize across query positions with `rayon`**
File: `pesti-runner/src/kernel/attention.rs`

```rust
// Before: sequential query processing
for q_pos in 0..query_len {
    compute_single_query(...);
}
```

With data-parallelism:
```rust
use rayon::prelude::*;

let results: Vec<f32> = (0..query_len).into_par_iter()
    .flat_map(|q_pos| compute_single_query(...))
    .collect();
```

**Expected gain: 4-8x** (on 8-thread i7-12700)

**3. Pre-allocate softmax buffer**
File: `pesti-runner/src/kernel/attention.rs`

Currently allocates `scores` and `softmax_output` per forward pass.
Pre-allocate once in `CpuAttentionKernel::new` and reuse.

**Expected gain: ~2x** (eliminates per-call allocation overhead)

## Constraints
- **No external SIMD crates** — use `std::simd` only
- **No LLVM FFI bindings** — no `std::arch::x86_64` intrinsics
- **`rayon` is acceptable** — it's pure Rust, no C dependencies, and provides
  thread-pool management that would otherwise require hand-rolled threading

## Expected Results
| Optimization | Est. Speedup | Projected tok/s |
|---|---|---|
| Baseline (current) | 1x | 80 |
| + SIMD inner product | 6x | ~480 |
| + Rayon parallel queries | 5x | ~2400 |
| + Pre-allocated buffers | 2x | ~4800 |

## Implementation Order
1. **Phase 1**: SIMD inner product (high impact, low risk)
2. **Phase 2**: Rayon parallel query processing (moderate risk — thread safety)
3. **Phase 3**: Buffer pre-allocation (low risk, cleanup)

## Verification
Run `cargo run --package pesti-runner --example attention_cpu_vs_gpu` after each phase.
Target: CPU ≥ 50% of GPU throughput on small configs (headroom for full model
inference where other ops dominate).
