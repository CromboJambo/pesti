# Week 9 Debugging Session - Single-Kernel Architecture

## The Breakthrough

After days of debugging `CUDA_ERROR_ILLEGAL_ADDRESS` crashes in our single-kernel fused attention, we discovered the root cause: **memory layout mismatch**.

### What We Thought Was True ❌
We assumed our single-kernel test should allocate **separate** device memory for:
- Q, K, V inputs
- Score buffer  
- Output buffer

### What We Discovered ✅
The two-kernel architecture actually uses **ONE contiguous device buffer** with offsets:
```rust
let total_size = score_buffer_size + output_buffer_bytes;
let combined_ptr = allocate_device_memory(total_size);

// Scores start at offset 0
let s_ptr = combined_ptr;

// Output starts AFTER scores in same allocation
let out_ptr = (combined_ptr as u64) + score_buffer_size as u64;
```

### The Two-Kernel Memory Layout

**Kernel 1 (Q@K^T + RoPE + Softmax)** writes to:
- `score_buffer`: `[seq_q, num_heads, seq_k]` floats

**Kernel 2 (Softmax × V)** reads from score buffer and writes to:
- `output_buffer`: starts at `score_buffer_size` offset
- Layout: `[seq_q, num_heads, head_dim]` halfs

Total allocation: `seq_q * num_heads * seq_k * 4 + seq_q * num_heads * head_dim * 2` bytes

## Evidence

### Failed Tests (Separate Allocations)
- ❌ `minimal_no_arrays_test`: Crashed with `CUDA_ERROR_ILLEGAL_ADDRESS`
- ❌ `simple_qk_dot_test`: Crashed with `CUDA_ERROR_ILLEGAL_ADDRESS`  
- ❌ `no_shared_memory_qk_dot_test`: Crashed with `CUDA_ERROR_ILLEGAL_ADDRESS`

### Passed Tests (Correct Memory Layout)
- ✅ `ultra_simple_write_test`: Simple write to single buffer works
- ✅ `single_kernel_two_buffer_test`: Single-kernel with combined buffer **PASSES**

## Conclusion

The two-kernel architecture is **NOT** truly parallel/asynchronous between kernels - it's sequential execution with asynchronous launch. However, the memory layout is clever: using one contiguous allocation avoids expensive separate allocations and allows efficient data flow between kernel stages.

### Why This Matters for Single-Kernel Fusion

Now that we understand the correct memory layout, we can:
1. ✅ Build a single-kernel that matches two-kernel behavior exactly
2. ✅ Fuse Q@K^T + RoPE + Softmax × V into one kernel call
3. ✅ Potentially improve performance by eliminating intermediate score buffer writes

## Next Steps (Week 10)

Now that we have the correct memory pattern, we can:
1. Create a single-kernel that computes everything in one pass
2. Add numerical conformance tests to verify correctness
3. Benchmark against two-kernel baseline
4. Optimize thread indexing and shared memory usage

**Status**: Single-kernel infrastructure is ready - now we just need the right kernel logic! 🚀
