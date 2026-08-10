# GPU Kernel Optimization Plan

## Objective
Replace 2-GEMM CPU-softmax attention path with fused kernel to eliminate H2D overhead and improve performance.

## Current Path (Option A - GEMM-based)

```
Q @ K^T via mma.sync → scores_f16 → D2H → CPU softmax(f32) → f16 conversion → H2D → S @ V via mma.sync → D2H
```

**Transfers**: 2 H2D + 2 D2H (including intermediate softmax round-trip)  
**Precision loss**: f32→f16 during softmax transfer  
**Latency**: ~0.05ms per attention step (PCIe overhead dominates for small matrices)

## Target Path (Option B - Fused Kernel)

```
Q @ K^T + RoPE + softmax + S @ V via fused mma.sync kernel → D2H
```

**Transfers**: 1 H2D (query+KV) + 1 D2H (output)  
**Precision loss**: Only final output conversion  
**Latency**: ~0.01ms per attention step (eliminates PCIe round-trips)

## PTX Assets Available

| File | Arch | Target GPU | Status |
|------|------|------------|--------|
| `gemm_mma_sync.ptx` | mma.sync | Consumer Blackwell RTX 5060Ti/5090 | ✅ Wired |
| `attention_rope_softmax.ptx` | mma.sync | Consumer Blackwell RTX 4070Ti SUPER | ⚠️ Not wired |
| `attention_tcgen05.ptx` | tcgen05 | Datacenter B200 | ⚠️ Not wired |
| `attention_wgmma.ptx` | wgmma | Hopper A100/A800 | ⚠️ Not wired |

## Implementation Steps

### Step 1: Create FusedAttentionKernel (Week 1)
- Add `FusedAttentionKernel` struct to `pesti-runner/src/kernel/attention.rs`
- Load PTX from `include_str!("ptx/attention_rope_softmax.ptx")`
- Implement `forward()` using same trait interface as `GemmBasedAttentionKernel`

### Step 2: Wire into InferenceEngine (Week 1)
- Update `InferenceEngine::new()` to select fused kernel when available
- Add architecture check for sm_89+ (RTX 40-series, consumer Blackwell)

### Step 3: Benchmark & Validate (Week 2)
- Compare fused vs GEMM-based on Qwen2.5-0.5B
- Verify numerical conformance against CPU baseline
- Measure tok/s speedup in decode mode

## Expected Improvements

| Metric | Current | Target | Change |
|--------|---------|--------|--------|
| H2D/D2H transfers per step | 4 | 2 | **-50%** |
| Intermediate softmax precision loss | f32→f16 | None | **+24 bits** |
| Latency (per attention) | ~50μs | ~10μs | **5x faster** |
| Throughput (decode mode) | ~80 tok/s | ~120 tok/s | **+50%** |

## Dependencies

- `attention_rope_softmax.ptx` must exist and compile on sm_89+
- Consumer GPU available for testing (RTX 40-series or RTX 50-series)
- Conformance corpus for numerical validation

## Risk Assessment

**Low risk**: The PTX already exists, we just need to wire it in.  
**Medium risk**: Kernel may have different performance characteristics than GEMM-based path.  
**Mitigation**: Keep `GemmBasedAttentionKernel` as fallback option.
