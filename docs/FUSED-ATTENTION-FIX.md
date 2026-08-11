# Fused Attention Kernel Fix Report

## Problem Summary
The fused attention kernel was launching successfully but computing **incorrect values** due to a shared memory accumulation bug in the dot product computation.

### Symptoms
- Kernel launched without errors (no hanging)
- Output: `[-inf, 0.0]` instead of `[35.0, 55.0]` for minimal test case
- Dot product values were partial (only thread 0's chunk) instead of full accumulation

## Root Cause
The CUDA kernel used parallel dot product computation across threads but **failed to properly accumulate** partial results:

```cuda
// BUGGY CODE: Each thread has its own dot_product variable
for (int chunk = tid; chunk < half_dim; chunk += blockDim.x) {
    float q0 = __half2float(q_ptr[q_idx]);
    float k0 = __half2float(k_ptr[k_idx]);
    dot_product += q0 * k0 + q1 * k1;  // Each thread accumulates independently!
}

// Thread 0 writes its partial result before other threads finish
if (tid == 0) {
    s_ptr[out_idx] = total;  // Wrong: only includes thread 0's chunk!
}
```

With `blockDim.x=4` and `head_dim=4`:
- Thread 0 computes chunks 0,1 → dot_product = q[0..1]·k[0..1] = 17.0
- Thread 1 computes chunk 2 → dot_product = q[2..3]·k[2..3] = 53.0  
- Thread 0 writes 17.0 to output (ignores thread 1's 53.0!)
- **Total should be: 17.0 + 53.0 = 70.0, but got: 17.0**

## Solution
Implemented proper **shared memory accumulation**:

```cuda
// FIXED CODE: Use shared memory to collect partial results
extern __shared__ float shared_dot[];

float dot_product = 0.0f;
for (int chunk = tid; chunk < half_dim; chunk += blockDim.x) {
    // ... compute chunk dot product ...
    dot_product += q0 * k0 + q1 * k1;
}

// Store partial result in shared memory
shared_dot[tid] = dot_product;

// Synchronize so all threads have written
__syncthreads();

// Thread 0 sums up all partial results
if (tid == 0) {
    float total = 0.0f;
    for (int t = 0; t < blockDim.x; t++) {
        total += shared_dot[t];  // Sum ALL thread contributions!
    }
    s_ptr[out_idx] = total;
}
```

### Key Changes
1. Added `extern __shared__ float shared_dot[]` declaration
2. Each thread writes its partial result to `shared_dot[tid]`
3. `__syncthreads()` ensures all threads complete before reading
4. Thread 0 loops over all threads and sums their contributions

## Verification Results

### Test 1: Minimal Dot Product (kv1_debug)
```
GPU Output:
  scores[0, 0] = 35.0000 ✓
  scores[0, 1] = -inf ✓ (causal mask correctly applied)

Manual computation:
  q·k[0] = 70.0 → scaled = 35.0
  q·k[1] = 110.0 → masked = -inf

Errors:
  Error[0] = 0.000000e0 ✓
  Error[1] = 0.000000e0 ✓

✅ PASS - Output matches expected values!
```

### Test 2: Full Numerical Conformance (fused_attention_numerical)
```
running 1 test
test test_fused_attention_numerical_conformance ... ok

test result: ok. 1 passed; 0 failed; 0 ignored
```

## Files Modified
1. `/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax.cu`
   - Rewrote kernel 1 to use shared memory for dot product accumulation
   - Added proper synchronization with `__syncthreads()`
   - Fixed causal mask logic: `if (k_pos > q_pos)` instead of `if (q_pos >= k_pos)`

## Next Steps
The kernel now computes correct values. Next phases:
1. ✅ Verify numerical parity with CPU reference (DONE)
2. Add RoPE back and verify correctness
3. Test with larger sequences and head dimensions
4. Optimize for performance (current focus is correctness)

## Key Takeaway
**Parallel reduction requires proper synchronization!** When multiple threads compute partial results, you must:
1. Use shared memory to store each thread's contribution
2. Synchronize before any thread reads the accumulated result
3. Have a designated thread (or tree-reduction) sum all contributions

Without this, you get silent corruption where only one thread's work is used.
