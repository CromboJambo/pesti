# PESTI Attention Kernel Architecture Strategy 🏗️

**Date**: August 15, 2026  
**Status**: ✅ **Dual-architecture approach** | 🚀 **Two-kernel = production baseline**, ⚡ **Single-kernel = optimized alternative**

---

## 📊 Current Architecture State

### Architecture #1: Two-Kernel (Production Baseline)
**File**: `pesti-runner/src/kernel/fused_attention_conformant.rs`  
**PTX**: `attention_rope_softmax.ptx`  
**Status**: ✅ **Active, tested, documented**

**Design**:
- Kernel 1: Compute scores (Q @ K^T + RoPE + causal mask)
- Kernel 2: Apply softmax + multiply by V
- Inter-kernel communication via score buffer

**Pros**:
- ✅ Well-tested and documented
- ✅ q=0 perfect match verified
- ✅ Clear separation of concerns
- ✅ Easy to debug each kernel independently

**Cons**:
- ⚠️ Inter-kernel communication bugs (q=1 uniform distribution issue)
- ⚠️ Extra memory bandwidth (score buffer write/read)
- ⚠️ Harder to optimize together

---

### Architecture #2: Single-Kernel (Optimized Alternative)
**Files**: 
- `pesti-runner/src/kernel/ptx/fused_attention_simple_kernel.cu`
- `pesti-runner/src/kernel/ptx/fused_attention_single_kernel.cu`  
**PTX**: `fused_attention_simple_kernel.ptx` (83KB), `fused_attention_single_kernel.ptx` (20KB)  
**Status**: ⏳ **Created, verified to load, pending numerical testing**

**Design**:
- Single kernel: RoPE → scores → softmax → V-multiply in one launch
- No inter-kernel communication
- Sequential processing (correctness first)

**Pros**:
- ✅ Eliminates inter-kernel bugs
- ✅ Better memory locality
- ✅ Simpler debugging (one kernel instead of two)
- ✅ Foundation for shared memory tiling optimization

**Cons**:
- ⏳ Not yet numerically verified
- ⚠️ Requires integration into test harness
- ⚠️ More complex code (all logic in one place)

---

## 🎯 Strategy: Keep Both, Compare Both

### Why Dual Architecture?

1. **Risk Mitigation**
   - Two-kernel = proven production baseline
   - Single-kernel = alternative to validate/replace later
   - Can roll back if single-kernel has issues

2. **Numerical Comparison**
   - Run both on same inputs
   - Compare outputs directly
   - Identify which architecture is more accurate

3. **Performance Benchmarking**
   - Measure latency vs throughput for both
   - Identify bottlenecks in each design
   - Optimize based on real data

4. **Gradual Migration**
   - Start with two-kernel in production
   - Validate single-kernel numerically
   - Migrate when ready (or keep both as options)

---

## 📋 Implementation Roadmap

### Phase 1: Verification (Week 10-11)
- [ ] Integrate single-kernel into conformance test
- [ ] Compare numerical outputs vs two-kernel
- [ ] Verify q=1 selective attention (not uniform)
- [ ] Target: <1e-4 relative error for both architectures

### Phase 2: Performance Benchmarking (Week 11-12)
- [ ] Measure latency/tokens/sec for both
- [ ] Profile memory bandwidth usage
- [ ] Identify optimization opportunities
- [ ] Document performance characteristics

### Phase 3: Optimization (Week 12-14)
- [ ] Add shared memory tiling to single-kernel
- [ ] Implement WGMMA tensor core instructions
- [ ] Optimize two-kernel if needed
- [ ] Re-benchmark after optimizations

### Phase 4: Migration Decision (Week 14+)
- [ ] Compare accuracy + performance data
- [ ] Choose primary architecture for production
- [ ] Document migration path (if applicable)
- [ ] Update documentation and examples

---

## 🔍 Key Questions to Answer

### Accuracy
- Does single-kernel produce more accurate results than two-kernel?
- Are there edge cases where one fails and the other succeeds?
- What's the numerical error profile for each?

### Performance
- Which architecture is faster on real models (Qwen2.5, Llama 3.1)?
- How does performance scale with sequence length (64 → 512 → 2048)?
- Which has better memory bandwidth utilization?

### Maintainability
- Which is easier to extend with new features (RoPE scaling, YaRN, etc.)?
- Which has clearer code structure for future contributors?
- Which integrates better with the rest of PESTI?

---

## 📊 Decision Criteria

### Keep Two-Kernel If:
- ✅ Numerical accuracy matches or exceeds single-kernel
- ✅ Performance is competitive (within 10%)
- ✅ Easier to maintain and extend
- ✅ Community preference (if contributing upstream)

### Migrate to Single-Kernel If:
- ✅ Significant accuracy improvement (>10x error reduction)
- ✅ Performance advantage (>20% speedup)
- ✅ Clearer architecture for future optimizations
- ✅ Simpler codebase overall

### Keep Both If:
- ✅ Different use cases benefit from each
- ✅ Two-kernel better for small models, single-kernel for large
- ✅ Performance/accuracy trade-offs favor different architectures
- ✅ Feature flexibility (e.g., two-kernel easier to extend)

---

## 🚦 Current Status Summary

| Metric | Two-Kernel | Single-Kernel |
|--------|------------|---------------|
| **Status** | Production baseline | Alternative implementation |
| **Numerical accuracy** | q=0 perfect, q=1 partial | TBD (not tested yet) |
| **Integration** | ✅ Active in codebase | ⏳ PTX created, needs test integration |
| **Documentation** | ✅ Complete | ⏳ Partial |
| **Testing** | ✅ Verified | ⏳ PTX loads, numerical pending |
| **Performance** | Baseline established | TBD |
| **Risk level** | Low | Medium (unverified) |

---

## 💡 Recommendation

**Adopt a "compare and decide" strategy:**

1. **Don't drop either architecture yet** - Both have value
2. **Integrate single-kernel into tests** - Verify numerical accuracy
3. **Benchmark both on real models** - Measure actual performance
4. **Make data-driven decision** - Choose based on accuracy + performance

This approach minimizes risk while maximizing learning opportunities. We can always:
- Use two-kernel for production while validating single-kernel
- Switch to single-kernel once proven superior
- Keep both as configurable options (e.g., `--attention-arch two-kernel|single-kernel`)

---

**Author**: PESTI Engineering Team  
**Date**: August 15, 2026  
**Status**: Dual-architecture strategy approved. Two-kernel = production baseline, single-kernel = optimized alternative. Ready for numerical comparison! 🎯
