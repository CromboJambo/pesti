# Week 15 Spec: CUDA Integration & Tokenizer Validation

**Date**: August 18, 2026  
**Status**: In Progress — Critical bugs blocking GPU path  
**Goal**: Integrate CUDA GEMM kernels from Week 13 and validate real GGUF tokenizer

---

## 🎯 Executive Summary

Week 14 achieved **~100 tok/s** on Qwen2.5-0.5B with CPU path. Week 15 aims to integrate CUDA GEMM for **5-8× speedup** (target: 500-800 tok/s).

**Current blockers**:
1. **FFN dimension mismatch** — Qwen2.5 FFN weights loaded with swapped in/out features
2. **Tokenizer returns 0 tokens** — GGUF tokenizer parsing issue
3. **CUDA path not wired** — `forward_layers()` still uses CPU-only implementation

---

## 📊 Week 14 Baseline (Reference)

| Metric | Value |
|--------|-------|
| Model | Qwen2.5-0.5B-Instruct (Q4_K_M) |
| Prompt | "The quick brown fox jumps over the lazy dog." |
| Generated tokens | 64 |
| Generation time | ~0.64s |
| **Throughput** | **~100 tok/s** ⭐ |
| Weight loading | 30.55s (one-time) |
| Hardware | RTX 4070 Ti SUPER (sm_8.9) + RTX 5060 Ti |

---

## 🔍 Week 15 Findings (Updated: August 18, 2026)

### CUDA Integration Status: ⚠️ PARTIALLY WORKING

**Attempted**: `week15_cuda_integration.rs` with `--features cuda`  
**Result**: Model loads successfully, but crashes in w2 (ffn_down) forward pass

```
DEBUG Llama: ffn_gate - in=896, out=4864  ✅ CORRECT
DEBUG Llama: ffn_down - in=4864, out=896  ✅ CORRECT  
DEBUG Llama: ffn_up - in=896, out=4864   ✅ CORRECT
Linear::forward: in=4864, out=896, weight.len=1361920  ❌ WRONG SIZE
FFN: intermediate_dim=4864, gate.len=4864, up.len=4864  ✅ CORRECT
thread 'main' panicked at pesti-runner/src/transformer/linear.rs:149:45:
range end index 1366784 out of range for slice of length 1361920
```

**Root cause**: **GGUF metadata vs actual data size mismatch**
- GGUF claims ffn_down shape: `[4864, 896]` = 4,358,144 elements
- Actual quantized data in file: ~1,361,920 elements (31% of claimed)
- Dequantization produces 1,361,920 f32 floats instead of expected 4,358,144

**Technical details**:
- GGUF v3 format stores tensor metadata separately from raw quantized data
- `tensor_shape()` reads **claimed shape from metadata** (lines 61-74 of gguf_weight_loader.rs)
- `dequantize_tensor()` tries to **infer element count from data size** (lines 127-176)
- For Q4_K_M: 54 bytes → 32 f32 floats after dequant
- But w2's claimed shape says 4,358,144 elements, while actual data only supports ~1.36M

**Why this happens**: The GGUF file (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) has **inconsistent metadata**:
- Some tensors (ffn_gate, ffn_up) have correct shape claims
- Others (ffn_down) have inflated shape claims that don't match actual data size
- This could be due to: quantization tool bug, GGUF version mismatch, or model conversion issue

### Real Tokenizer Status: ⚠️ BLOCKED BY FFN BUG

**Attempted**: `week15_real_tokenizer.rs` with `--features cuda`  
**Result**: Same w2 crash before tokenizer can be tested

```
Built model in 43.94s
Loaded real GGUF tokenizer with 32000 tokens
Encoded 0 tokens (simplified)  ← Simplified tokenization, not real GGUF tokenizer
Error: ModelLoad("empty context")
```

**Note**: The Week 15 examples use **simplified whitespace tokenization**, not the real GGUF tokenizer from `load_tokenizer_from_gguf()`. The real tokenizer path is gated behind CUDA feature and never reaches execution due to w2 crash.

---

## 🛠️ Critical Fixes Needed (Updated)

### ✅ COMPLETED: Fix FFN Dimension Swap (`model.rs`)

**File**: `pesti-runner/src/transformer/model.rs`  
**Lines**: ~735, 743, 751, 977, 985, 993

**Fixed**: Changed `(w_out, w_in)` to `(w_in, w_out)` for all FFN weight loading:
```rust
// Before (WRONG):
let (w1_out, w1_in) = weights.tensor_shape(&w1_name);
let w1 = Linear::from_f32_weight_with_dims(w1_data, None, w1_in, w1_out);

// After (CORRECT):
let (w1_in, w1_out) = weights.tensor_shape(&w1_name);  // GGUF: [in_features, out_features]
let w1 = Linear::from_f32_weight_with_dims(w1_data, None, w1_in, w1_out);
```

**Verification**: Debug output now shows correct dimensions for all FFN layers.

### ⚠️ PENDING: Fix GGUF Metadata Inference (`gguf_weight_loader.rs`)

**File**: `pesti-runner/src/gguf_weight_loader.rs`  
**Lines**: ~284-320 (Q4_K_M dequantization logic)

**Problem**: `tensor_shape()` returns claimed shape from metadata, but actual data size differs for some tensors.

**Current logic** (lines 127-176):
```rust
// For K-family types, infer element count from data size
let num_blocks = raw_data.len() / block_size;
let inferred_count = num_blocks * elements_per_block;
```

**Issue**: The inference picks the format closest to **claimed element count**, but if the claimed count is wrong, it still uses the wrong shape for Linear initialization.

**Fix needed**: 
1. Always use **inferred element count** from actual data size (not claimed shape)
2. Derive `in_features` and `out_features` from inferred count + known architecture constraints
3. For FFN: if input dim is known (from previous layer), compute intermediate dim = `inferred_count / in_features`

**Implementation plan**:
```rust
// In load_layer_from_gguf(), after loading w2 data:
let inferred_elements = raw_w2_data.len() / 4; // f32 after dequant
let w2_in = config.intermediate_dim; // Known from architecture
let w2_out = inferred_elements / w2_in; // Derive from actual data

let w2 = Linear::from_f32_weight_with_dims(w2_data, None, w2_in, w2_out);
```

### ⏳ FUTURE: Wire CUDA Path (`model.rs`)

**File**: `pesti-runner/src/transformer/model.rs`  
**Lines**: ~1050-1110

Once w2 bug is fixed, enable GPU acceleration:
```rust
pub fn forward_layers(&self, hidden: &[f32], start_pos: usize) -> Result<Vec<f32>> {
    if let Some(ref ctx) = self.dispatch {
        if ctx.gpu_available() {
            return self.forward_with_dispatch(hidden, start_pos);
        }
    }
    self.forward_layers_cpu(hidden, start_pos)
}
```

---

## 📈 Week 15 Success Criteria (Updated)

| Criterion | Target | Status | Notes |
|-----------|--------|--------|-------|
| FFN weights load with correct dimensions | ✅ | ✅ FIXED | All three FFN layers now show correct in/out dims |
| Real GGUF tokenizer encodes prompt | 🎯 | ⏳ BLOCKED | Waiting for w2 fix to unblock model loading |
| CUDA GEMM kernels run in forward pass | 🎯 | ⏳ BLOCKED | Model crashes before reaching CUDA path |
| Measured throughput ≥ 300 tok/s | 🎯 | ⏳ PENDING | Will measure after w2 fix |
| Numerical conformance vs llama.cpp (rtol=1e-3) | 🎯 | ⏳ PENDING | Future validation |

---

## 🚦 Next Steps (Updated Priorities)

### Priority 1: Fix w2 dimension inference (Day 1-2)
**File**: `pesti-runner/src/gguf_weight_loader.rs`  
**Action**: Modify dequantization to always use inferred element count, then derive Linear dimensions from architecture constraints.

**Steps**:
1. Add debug logging in `dequantize_tensor()` to print both claimed and inferred element counts
2. Modify `load_layer_from_gguf()` to pass `intermediate_dim` to FFN weight loading
3. Compute `w2_out = inferred_elements / intermediate_dim` instead of using claimed shape
4. Test with `week15_cuda_integration`

### Priority 2: Validate tokenizer integration (Day 3)
**File**: `pesti-runner/examples/week15_real_tokenizer.rs`  
**Action**: Replace simplified tokenization with real GGUF tokenizer after model loads successfully.

### Priority 3: Enable CUDA path (Day 4-5)
**File**: `pesti-runner/src/transformer/model.rs`  
**Action**: Wire `forward_with_dispatch()` into main inference loop when GPU available.

### Priority 4: Measure & profile (Day 6-7)
**Action**: 
- Run full benchmark with CUDA enabled
- Compare vs Week 14 CPU baseline (~100 tok/s)
- Target: 500-800 tok/s (5-8× speedup)

---

## 📝 Notes for Next Session

### Starting commands (after w2 fix):
```bash
cd /home/crombo/projects/pesti

# Rebuild with w2 fix
cargo build --package pesti-runner --features cuda

# Test model loading
cargo run --package pesti-runner --features cuda --example week15_cuda_integration 2>&1 | grep -E "DEBUG Llama|Linear::forward|panicked"

# If successful, measure throughput
cargo run --package pesti-runner --features cuda --example week14_e2e_decode 2>&1 | tail -20
```

### Key files to review:
- `pesti-runner/src/gguf_weight_loader.rs` (lines 284-320) - Q4_K_M dequantization logic
- `pesti-runner/src/transformer/model.rs` (lines 735-755, 977-997) - FFN weight loading
- `pesti-runner/src/transformer/linear.rs` (line 149) - w2 forward crash location

### Known issues:
- ⚠️ **GGUF metadata inconsistency**: Some tensors have inflated shape claims vs actual data
- ⚠️ **Qwen2.5-0.5B architecture**: FFN uses SwiGLU (gate×up before w2), needs careful dimension tracking
- ⚠️ **RTX 4070 Ti SUPER sm_8.9**: Requires CUTLASS/cublas kernels for optimal performance

---

*Last updated: August 18, 2026 — Week 15 in Progress*  
*Next milestone: Fix w2 dimension inference and validate model loading*
