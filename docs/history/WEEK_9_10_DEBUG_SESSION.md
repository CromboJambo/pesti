# Week 9-10 Debug Session: Adversarial Test Analysis

**Date**: August 13, 2026  
**Status**: Bug isolated, root cause identified

---

## The Problem

The adversarial test fails with huge negative outputs:
```
GPU output: -1238.156250, -309.532806, ... (raw scores?)
CPU expect: -5.0, -4.6, -4.2, ... (attention output)
Max relative error: 1.026294e4 (should be < 1e-4)
```

---

## Key Discovery

**The GPU is computing raw attention scores, NOT softmax-normalized attention output!**

Evidence:
1. For q_pos=0, head=0, causal mask allows only k_pos=0
2. Raw dot product with adversarial inputs (range [-10, 10]) can be ~5000
3. Scaled by 1/sqrt(16) = 0.25 gives ~1250, matching the -1238 output!
4. If softmax were working, probs would sum to 1.0 and V-multiply would produce values in range [-10, 10]

---

## Suspected Root Causes

### Cause #1: Softmax Not Applied
- Kernel computes scores but doesn't apply softmax normalization
- Output is raw dot products instead of attention weights × V

### Cause #2: Shared Memory Race Condition
- `s_exp_sum` reduction might not complete before normalization
- Other threads read uninitialized/zero values
- Result: division by zero or incorrect norm_factor

### Cause #3: Incorrect Causal Mask Logic
- Maybe masking all positions instead of just future ones
- Or masking wrong indices entirely

---

## Debugging Attempts

1. ✅ Added syncthreads after reduction (no change)
2. ✅ Changed max initialization from -FLT_MAX to -1e10f (no change)  
3. ✅ Added explicit masked position checks (no change)
4. ✅ Verified parameter order in kernel launch (fixed earlier)
5. ❌ PTX caching issues prevented reliable recompilation testing

---

## Comparison with Passing Test

**Passing test** (`single_kernel_numerical_conformance_with_rope`):
- Uses `fused_attention_exact_pattern.ptx` (different kernel!)
- Compares scores → softmax → probs vs reference
- Configuration: seq_q=2, seq_k=32, heads=4, dim=16

**Failing test** (`adversarial_attention_conformance`):
- Uses `fused_attention_simple_kernel.ptx` (our new kernel)
- Compares full attention output (softmax + V-multiply) vs reference
- Configuration: seq_q=3, seq_k=8, heads=2, dim=16

**Key insight**: The passing test uses a **different kernel** that was already working! My simple_kernel is untested and buggy.

---

## Current Status

### What Works ✅
- Kernel launches without crashing (no CUDA_ERROR_ILLEGAL_ADDRESS)
- Basic dot product computation works (produces reasonable scores)
- Shared memory allocation works
- Parameter passing fixed (q_ptr, k_ptr, v_ptr, out_ptr order)

### What's Broken ❌
- Softmax normalization not producing correct probs
- V-multiply producing raw scores instead of attention output
- Max relative error: 10,262× target (should be < 1e-4)

---

## Next Steps (Recommended)

### Option A: Debug and Fix Simple Kernel
1. Add explicit debug prints to kernel (printf in CUDA)
2. Verify softmax probs sum to 1.0 for each (q_pos, head)
3. Check that only unmasked positions have non-zero probs
4. Verify V-multiply indexing matches CPU reference

### Option B: Use Exact Pattern Kernel
- The `fused_attention_exact_pattern.ptx` already passes conformance tests
- Use it for production while developing simple_kernel separately
- Simpler: just copy exact_pattern logic into simple_kernel source

### Option C: Simplify Test Expectations
- Change adversarial test to compare scores (not full output)
- Match the passing test's approach
- Easier to verify, but less comprehensive

---

## Recommendation

**Go with Option B**: Copy the working exact_pattern kernel logic into simple_kernel.

Why:
1. We already have a working implementation
2. The bug is in softmax/V-multiply logic which is complex
3. Learning value: understand exact_pattern first, then optimize
4. Time-efficient: 80% of work done, just need to port the logic

---

## Action Items

1. **Immediate**: Read `fused_attention_exact_pattern.ptx` source (if available) or reverse-engineer from PTX
2. **Short-term**: Port exact_pattern softmax + V-multiply logic to simple_kernel
3. **Medium-term**: Add debug prints to verify each step
4. **Long-term**: Optimize with shared memory tiling and WGMMA

---

**Author**: PESTI Engineering Team  
**Date**: August 13, 2026  
**Status**: Root cause identified (softmax not applied), ready for fix 🎯
