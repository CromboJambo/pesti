# Week 30: Throughput Regression Diagnosis and Optimization Path

## Goal
Diagnose and fix the ~10x throughput regression detected since Week 29 (73 tok/s → ~6-7 tok/s), then establish the correct optimization path for continued performance improvement.

## Current State (Baseline)
- **Documented Week 29 throughput**: 73 tok/s on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti)
- **Current measured throughput**: ~6-7 tok/s (benchmark_detailed test: 6.04, benchmark_pesti_runner example: 6.73)
- **Regression window**: Between commit c37a3b3 (Week 29 complete, Oct 15) and current HEAD (Oct 21)
- **Q4_K KV cache status**: `benchmark_tok_q4k` example fails with CUDA OOM during warmup

## Suspected Cause
Commit a1a232d ("feat: add model router and kernel dispatch improvements", Oct 18) significantly rewrote kernel dispatch (339 lines changed in `pesti-runner/src/kernel/dispatch.rs`). This is the most likely source of the regression, potentially breaking the persistent weight caching path established in Week 29.

## Implementation Phases

### Phase 1: Regression Diagnosis (Days 1-2)
**Objective**: Identify exact commit and code change causing throughput drop.

1. **Git bisect** between c37a3b3 and HEAD using `benchmark_detailed` test as oracle
2. **Profile kernel dispatch path** with CUDA instrumentation to determine:
   - Are weights actually being cached on GPU or re-uploaded per token?
   - Is there redundant H2D transfer introduced by dispatch rewrite?
   - Is the correct GEMM path being selected (cuBLAS vs cuBLASLt)?
3. **Fix Q4_K KV cache OOM** issue in `benchmark_tok_q4k` example

**Deliverable**: Root cause analysis document identifying exact regression source and code change responsible.

### Phase 2: Regression Fix (Day 3)
**Objective**: Restore Week 29 throughput level (~70+ tok/s).

Based on Phase 1 findings, implement fix:
- If weight caching broken: restore `upload_weights_to_gpu()` call path in dispatch
- If GEMM selection regressed: restore shape-based algorithm caching from Week 26
- If architecture fundamentally changed: adapt optimization strategy to new codebase

**Validation**: Run `benchmark_detailed` and `benchmark_pesti_runner` — must achieve ≥70 tok/s.

### Phase 3: Optimization Path Decision (Day 4)
**Objective**: Establish correct week 31+ optimization target based on accurate baseline.

Run deep profiling on fixed codebase to identify actual bottleneck vs llama.cpp reference (504 tok/s):
- GEMM compute time percentage
- H2D/D2H transfer overhead
- Kernel launch overhead
- Memory bandwidth utilization

**Decision matrix**:
| Finding | Optimization Path |
|---------|-------------------|
| GEMM > 60% of time | Custom CUDA kernels for m=1 shapes (llama.cpp approach) |
| H2D/D2H > 30% | Persistent weight caching + KV cache quantization |
| Kernel launch overhead high | Batch multiple operations, reduce sync points |
| Memory bandwidth bound | KV cache Q4_K quantization (existing spec) |

### Phase 4: Documentation Sync (Day 5)
**Objective**: Update ROADMAP.md and relevant docs to reflect actual current state.

1. Update throughput numbers with verified benchmarks
2. Document regression findings and fix
3. Capture all work done since Week 29 (new crates, architecture changes)
4. Set realistic week 31+ targets based on Phase 3 decision

## Risk Factors
1. **Regression may be in multiple places**: Kernel dispatch rewrite may have introduced several issues, not just one
2. **Baseline may be inaccurate**: Week 29's 73 tok/s claim may have been measured differently than current benchmarks
3. **Architecture drift**: Significant new crate development since Week 29 may have changed the optimization landscape
4. **Q4_K KV cache OOM** may indicate deeper memory management issues beyond just benchmark configuration

## Rollback Plan
If regression fix introduces other issues:
- Revert to c37a3b3 (Week 29 complete) as stable baseline
- Cherry-pick only necessary new crate development
- Establish proper CI benchmarking to prevent future silent regressions

## Timeline Estimate
- Phase 1 (diagnosis): 8-12 hours
- Phase 2 (fix): 4-8 hours
- Phase 3 (optimization decision): 4-6 hours
- Phase 4 (documentation): 2-4 hours
- Total: ~18-30 hours

## Success Criteria
- Root cause of regression identified and documented
- Throughput restored to ≥70 tok/s (Week 29 level)
- Q4_K KV cache benchmark runs without OOM
- Clear optimization path established for week 31+
- ROADMAP.md updated with accurate current state

## References
- `docs/ROADMAP.md` — main roadmap with Week 29 completion notes
- `specs/kv-cache-q4k-integration.md` — existing KV cache quantization spec
- `pesti-runner/tests/benchmark_detailed.rs` — primary benchmark test
- `pesti-runner/examples/benchmark_pesti_runner.rs` — secondary benchmark example
- Commit c37a3b3 — Week 29 complete (persistent GPU weight caching)
- Commit a1a232d — suspected regression source (kernel dispatch rewrite)