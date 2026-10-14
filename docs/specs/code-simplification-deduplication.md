# Pesti Code Simplification & Deduplication Spec

## Goal
Reduce code complexity, remove redundant/experimental code, and eliminate dead paths while maintaining the **16 tok/s baseline** on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti).

## Current Baseline
- **Performance:** 16.10 tok/s (verified after fused kernel regression removal)
- **Benchmark command:** `cargo run --release --features cuda --example benchmark_pesti_runner`
- **Model path:** `conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf`

## Phase 1: Identify Redundant Code

### Test/Example Files to Audit
Many test files were created during optimization experiments and may be obsolete:

```bash
# List all examples and tests
ls pesti-runner/examples/
ls pesti-runner/src/kernel/*test*.rs
```

**Candidates for removal or consolidation:**
- `benchmark_kernel_timing.rs` — deprecated per earlier decision (use benchmark_pesti_runner)
- Any test files that only test removed/experimental features
- Duplicate test coverage across multiple files

### Kernel Module Audit
The kernel module has grown with many experimental implementations:

```bash
ls pesti-runner/src/kernel/
```

**Look for:**
- Multiple implementations of the same operation (e.g., different attention kernels)
- Experimental variants that were never integrated into production
- Stub files that duplicate real implementations

## Phase 2: Deduplication Strategy

### Rule: One Production Path, Zero Dead Code
For each operation type, determine:
1. Which implementation is used in production inference?
2. Which are experimental/legacy?

**Keep:** Production path + one well-tested fallback (CPU)
**Remove:** Everything else unless explicitly needed for benchmarking

### Specific Areas to Investigate

#### Attention Kernels
Multiple attention implementations exist:
- `attention.rs` — base implementation
- `fused_attention_conformant.rs` — fused variant
- `batched_parallel_attention.rs` — batched variant
- `flash_attention_v2.rs` — flash attention
- `fused_decode_attention.rs` — decode-specific fusion
- `one_stage_attention.rs` — full fusion

**Question:** Which one is actually used in production? Check `dispatch.rs` and `inference_engine_conformant.rs`.

#### GEMM Kernels
- `gemm.rs` — base implementation
- `gemm_cutlass.rs` — CUTLASS variant
- `wgmma_gemm.rs` — WGMMA tensor core variant

**Question:** Is CUTLASS actually faster? If not, remove it.

#### KV Cache Implementations
- `kvcache.rs` — standard
- `optimized_kvcache.rs` — FP16/paged optimizations
- `q4k_kvcache.rs` — quantized variant

**Question:** Which is used in production? Are the others tested and working?

## Phase 3: Simplification Actions

### Remove Obsolete Files
Delete files that are confirmed unused. Track with git so they can be recovered if needed.

### Consolidate Test Coverage
If multiple test files cover the same functionality, merge them into one comprehensive test suite.

### Eliminate Redundant Features
If two features do the same thing (e.g., two different weight caching approaches), keep only the better one.

## Phase 4: Verification After Each Change

After each simplification step:
```bash
# Verify baseline performance is maintained
cargo run --release --features cuda --example benchmark_pesti_runner
# Expected: ~16 tok/s (±10% acceptable)

# Run conformance tests if available
cargo run --release --features cuda --example conformance_test
```

## Decision Framework

For each piece of code, ask:
1. Is this used in production inference? → **Keep**
2. Is this the only implementation of its function? → **Keep**
3. Is this actively being developed/improved? → **Keep**
4. Is this older/slower than another implementation? → **Remove**
5. Is this untested or experimental? → **Remove or mark clearly**

## Success Criteria
- [ ] No more than one production implementation per operation type
- [ ] All test files have a clear purpose and aren't duplicating coverage
- [ ] Benchmark remains at ~16 tok/s (no regression)
- [ ] Codebase is smaller and easier to navigate
- [ ] No dead code paths remain

## Files Likely to Change
- `pesti-runner/examples/` — consolidate/remove obsolete tests
- `pesti-runner/src/kernel/` — remove experimental variants
- Possibly `pesti-runner/src/kernel/dispatch.rs` if removing features changes dispatch logic
