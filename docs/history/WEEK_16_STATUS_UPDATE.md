# Week 16 Sprint Status Update - August 21, 2026

## ✅ COMPLETED: Real GGUF Tokenizer Integration

**Finding**: The Qwen2.5-0.5B tokenizer **does work correctly** when properly loaded from the GGUF file.

### Verification Evidence

1. **Direct parser test** (`test_tokenizer_read.rs`):
   ```
   Found tokenizer.ggml.tokens
     value_type: Array
     Array length: 151936
     First element type: String
     First token string: '!'
   Extracted 151936 strings
   ```

2. **Encoding test** (`test_week16_prompt.rs`):
   ```
   Prompt: 'The quick brown fox jumps over the lazy dog.'
   Token count: 10
   Tokens: [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]
   Decoded: 'The quick brown fox jumps over the lazy dog.'
   ```

3. **Full inference test** (`full_inference_test.rs`):
   ```
   ✅ Real GGUF tokenizer loaded with 151936 tokens
   Encoded 10 tokens in 0.400ms
   [DEBUG] forward_with_dispatch called with GPU available: true
   Built model in 48.70s
   ```

### Root Cause Analysis

The tokenizer code was **already correct** - the issue was that previous examples weren't actually loading or using it due to:
- Compilation errors in example files (API mismatches)
- The w2 dimension bug blocking execution until Week 16 commit `daa9ac1`

### What Changed

The real GGUF tokenizer has **always worked** - the `pesti-gguf` crate v0.2.4 correctly parses `tokenizer.ggml.tokens` as an array of `GgufKvValue::String` items, and the `PestiTokenizer::string_array()` helper correctly extracts them.

---

## ⚠️ NEW BLOCKER: CUDA Memory Allocation

**Status**: GPU path selected but fails on memory allocation

### Error Details

```
[DEBUG] forward_with_dispatch called with GPU available: true
Error: Dispatch(Memory("alloc B: CUDA error: cuMemAlloc_v2: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")"))
```

### Analysis

1. **GPU detected**: ✅ RTX 4070 Ti SUPER + RTX 5060 Ti (32GB VRAM)
2. **CUDA enabled**: ✅ Feature flag active
3. **GPU path selected**: ✅ `dispatch_gemm()` auto-selects GPU
4. **Memory allocation**: ❌ Fails during KV cache allocation

### Hypothesis

The `CudaMemoryBackend` is being initialized with incorrect size parameters. Looking at the dispatch code:

```rust
// pesti-runner/src/kernel/dispatch.rs
fn build_memory_from_engine(engine: &InferenceEngine) -> MemoryManager {
    #[cfg(feature = "cuda")]
    {
        if let (Some(stream), Some(info)) = (engine.cuda_stream(), engine.cuda_device_info()) {
            return MemoryManager::Cuda(crate::kernel::memory::CudaMemoryBackend::with_device_info(
                stream.clone(),
                info.clone(),
            ));
        }
    }
    MemoryManager::Cpu(...)  // fallback
}
```

The `with_device_info()` constructor should query the real device VRAM, but might be using stale/stub values.

---

## 📊 Current Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Weight loading** | ~48s | ✅ Working |
| **Tokenizer vocab** | 151,936 tokens | ✅ Real GGUF loaded |
| **Prompt encoding** | 10 tokens in 0.4ms | ✅ Correct |
| **GPU path selected** | `forward_with_dispatch()` | ✅ Working |
| **CUDA memory alloc** | ❌ OOM error | ⚠️ **BLOCKER** |
| **Throughput** | N/A (OOM) | ⏳ Pending fix |

---

## 🔜 Next Steps (Week 17)

### Priority 1: Fix CUDA Memory Allocation (Day 1-2)
**Goal**: Resolve OOM error on `cuMemAlloc_v2`  
**Files**: 
- `pesti-runner/src/kernel/memory.rs` - `CudaMemoryBackend::with_device_info()`
- `pesti-runner/src/kernel/dispatch.rs` - `build_memory_from_engine()`

**Actions**:
1. Verify `cuda_device_info()` returns actual VRAM, not stub values
2. Check `CudaMemoryBackend` initialization parameters
3. Ensure stream cloning doesn't cause type mismatches (`(*stream).clone()` vs `stream.clone()`)

### Priority 2: Establish GPU Baseline (Day 3-4)
**Goal**: Measure tokens/sec with working CUDA path  
**Expected**: 
- Conservative estimate: 5-8× speedup over CPU (~3-5 tok/s)
- With proper kernel impl: 10-20× speedup possible

### Priority 3: Verify Numerical Conformance (Day 5)
**Goal**: Byte-exact comparison vs llama.cpp  
**Files**: `pesti-conformance` suite

---

## 📝 Technical Notes

### Files Modified This Session
1. `pesti-runner/examples/debug_tokenizer.rs` - Fixed API to match current interface
2. `pesti-runner/examples/test_week16_prompt.rs` - New verification example
3. `pesti-runner/examples/full_inference_test.rs` - End-to-end inference test
4. `docs/history/WEEK_16_STATUS_REPORT.md` - Updated status documentation

### Key Discovery
**The tokenizer was never broken** - it always worked correctly via `pesti-gguf`. The blockers were:
- Week 13-15: w2 dimension crash prevented any execution
- Week 16 commit `daa9ac1`: Fixed w2, wired GPU path
- Current session: Verified tokenizer works, found CUDA memory issue

---

*Last updated: August 21, 2026 — Tokenizer verified, CUDA memory blocker identified*  
*Next milestone: Fix CUDA allocation to enable GPU inference*
