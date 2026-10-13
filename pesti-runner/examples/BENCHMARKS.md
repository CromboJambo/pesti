# Pesti Runner Benchmarks

## Authoritative Benchmarks

### `benchmark_pesti_runner.rs` (Primary)
- **Purpose:** End-to-end token generation speed with kernel timing breakdown
- **What it measures:** Real tok/s including all overhead (model loading, encoding, prefill, decode)
- **Why authoritative:** Measures actual user-facing performance; includes integrated profiling
- **Usage:** `cargo run --release --features cuda --example benchmark_pesti_runner`

### `benchmark_kernel_timing.rs` (Optimization Tool)
- **Purpose:** Per-kernel timing breakdown for identifying bottlenecks
- **What it measures:** Individual kernel execution times (GEMM, attention, etc.)
- **Why useful:** Essential for optimization work — shows where time is spent
- **Usage:** `cargo run --release --features cuda --example benchmark_kernel_timing`

## Deprecated Benchmarks

All other benchmarks have been moved to `examples-disabled/`. They were either:
- Redundant with the authoritative benchmarks
- Specific to past optimization sprints (phase-specific)
- CPU-only tests not relevant for GPU optimization
- Using outdated model loading paths

To restore any deprecated benchmark, move it back from `examples-disabled/` to `examples/`.

## Benchmark Results (Qwen2.5-0.5B-Instruct-Q4_K_M)

| Date | tok/s | Notes |
|------|-------|-------|
| 2026-10-13 | 16.64 | cuBLASLt algo caching, GEMM weight caching |

## Target

- llama.cpp CPU baseline: ~141 tok/s (Q4_0)
- Current gap: ~8.5x slower on GPU than CPU competitor
- Primary bottleneck: GEMM (74.9% of decode time), attention (25.0%)
