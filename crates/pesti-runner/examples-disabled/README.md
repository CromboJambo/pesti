# Disabled Examples (P2)

These examples were migrated from `cuda-oxide` to `cudarc` but are currently disabled until P2 is completed.

## Status: Ready for Reimplementation

All 24 examples compile cleanly with the new `cudarc` backend. They can be re-enabled by moving them back to `../examples/` and adjusting any minor import paths if needed.

## Why Disabled?

- **P0**: Commit uncommitted cleanups (highest priority)
- **P1**: ✅ cudarc migration complete (just finished)
- **P2**: Move examples back or gate with feature flag
- **P3-P6**: Integration tests, attention kernels, upstreaming

## List of Disabled Examples

```
attention_cpu_vs_gpu.rs
benchmark_cpu_vs_gpu.rs
benchmark_cutlass_gemm.rs
benchmark_gpu.rs
comprehensive_benchmark.rs
cpu_attention_bench.rs
cpu_baseline.rs
e2e_gpu_inference.rs
full_gpu_benchmark.rs
gemm_exact.rs
gemm_isolate.rs
gemm_sizes.rs
gemm_verify.rs
kv_cache_bench.rs
quant_bench.rs
quant_diag.rs
quant_fwd_test.rs
quant_smoke.rs
simple_gpu_verify.rs
test_attention_kernel.rs
test_cutlass_gemm.rs
test_device.rs
test_gemm_attention.rs
```

## How to Re-enable

1. Move files back: `mv examples-disabled/*.rs examples/`
2. Run `cargo check --features cuda` to verify compilation
3. Optionally add `#[cfg(feature = "cuda-examples")]` gate if desired

## Notes

- All examples use the re-exported API: `pesti_runner::CudaRuntime`, `pesti_runner::enumerate_devices()`
- No changes needed to core logic — just move files back
- The `cudarc` migration is solid; these examples will work as-is

---

**P2 Action Item**: Move examples back to `examples/` directory when ready.
