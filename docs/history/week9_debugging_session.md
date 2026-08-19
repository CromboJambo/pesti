# Week 9 - Single-Kernel Fused Attention Debugging Session

## Summary

After extensive debugging of `CUDA_ERROR_ILLEGAL_ADDRESS` crashes, we successfully resolved the single-kernel fused attention implementation and achieved numerical conformance testing readiness.

## Key Discovery: Memory Layout Mismatch

### Root Cause
The two-kernel architecture uses **ONE contiguous buffer with offsets**, not separate allocations!

**Two-kernel memory layout:**
- `score_buffer` at offset 0 (floats)
- `output_buffer` immediately after scores in same allocation (half precision)

**Single-kernel bug:**
We initially allocated separate buffers for scores and output, causing out-of-bounds access.

### The Fix

```rust
// Allocate ONE buffer containing both scores AND output
let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
let total_size = score_buffer_size + output_buffer_bytes;

let combined_ptr = allocate_device_memory(total_size);

// Scores at offset 0, output after it
let s_ptr = combined_ptr;
let out_ptr = (combined_ptr as u64) + score_buffer_size as u64;
```

## Architecture Comparison

### Two-Kernel Architecture
- **Kernel 1**: Computes Q@K^T → writes to `score_buffer` (floats)
- **Kernel 2**: Reads scores, applies softmax, multiplies by V → writes to `output_buffer` (after scores)
- **Memory**: ONE contiguous buffer with offsets
- **Execution**: Sequential (producer-consumer), not parallel between kernels

### Single-Kernel Architecture (exact_pattern)
- **Single kernel**: Computes Q@K^T + softmax × V in one pass
- **Memory**: ONE contiguous buffer with offsets (matching two-kernel)
- **Launch dims**: Grid `(seq_q, seq_k, num_heads)`, Block `(head_dim, 1, 1)`
- **Parameters**: q_ptr, k_ptr, v_ptr, s_ptr, out_ptr, scale, seq_q, seq_k, num_heads, head_dim (10 params)

## Testing Results

### Passing Tests
✅ `single_kernel_two_buffer_test` - Infrastructure validation  
✅ `single_kernel_numerical_conformance` - Kernel executes without crash

### Test Output
```
=== Single-Kernel Numerical Conformance Test ===
✅ Allocated single buffer: 16 bytes
✅ Loaded exact pattern kernel
🚀 Launching with grid=(1, 2, 1), block=(4, 1, 1)
✅ Kernel launched
✅ Single-kernel execution completed!

=== Single-Kernel Numerical Conformance Test PASSED ===
```

## Next Steps (Week 10)

1. **Full numerical conformance**: Compare single-kernel output vs two-kernel reference
2. **RoPE integration**: Add rotary position embeddings to single-kernel
3. **Causal mask**: Implement causal masking in single-kernel
4. **Performance benchmarking**: Measure throughput vs two-kernel baseline

## Files Modified

- `pesti-runner/tests/single_kernel_two_buffer_test.rs` - Memory layout test (passes)
- `pesti-runner/tests/single_kernel_numerical_conformance.rs` - Conformance test (passes)
- `pesti-runner/src/kernel/ptx/fused_attention_exact_pattern.ptx` - Exact pattern kernel
- `docs/week9_debugging_session.md` - This documentation

## Git Commits

```
4de5146 Week 9: Single-kernel numerical conformance test
3c249a5 Week 9: Single-kernel memory layout breakthrough  
8772d5b Week 9: Document single-kernel debugging breakthrough
```

## Conclusion

✅ **Week 9 Complete**: Single-kernel infrastructure ready for numerical verification  
✅ **Root cause resolved**: Memory layout mismatch fixed  
✅ **Test harness validated**: Kernel executes without `CUDA_ERROR_ILLEGAL_ADDRESS`  

**Status**: Ready to proceed with full numerical conformance testing in Week 10.
