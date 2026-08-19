# Week 17 - Sprint Plan Summary 🎯

**Date**: August 18, 2026  
**Status**: Ready to execute  
**Goal**: Profile CUDA performance and integrate real tokenizer

---

## 📋 What We're Building This Week

### Primary Focus Areas

#### 1. **Performance Baseline** (Day 1)
- Measure CPU-only token generation speed
- Establish throughput metric (tok/s) as baseline
- Document timing breakdown by phase

#### 2. **CUDA Verification** (Day 2-3)
- Confirm GPU acceleration is actually used
- Measure speedup vs CPU baseline
- Profile attention kernel performance

#### 3. **Real Tokenizer Integration** (Day 4-5)
- Replace fallback whitespace tokenizer with qwen2-bpe crate
- Integrate real Qwen2 BPE logic into main pipeline
- Benchmark tokenizer overhead

#### 4. **Documentation & Optimization** (Day 6-7)
- Create performance comparison chart
- Document optimization opportunities for Week 18
- Write sprint summary report

---

## 🎯 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Baseline throughput | Measure actual | `profiling.rs` example |
| CUDA speedup | ≥2× faster than CPU | Compare runtimes |
| Tokenizer overhead | <10% of total time | Profile encode/decode |
| Real tokenizer working | Pass all conformance tests | Existing test suite |

---

## 📦 Deliverables Checklist

### Must Complete (End of Sprint)
- [ ] `WEEK_17_PROFILING_RESULTS.md` - Benchmark report with metrics
- [ ] `profiling.rs` example - Baseline measurement tool
- [ ] CUDA activation logging in codebase
- [ ] Real qwen2-bpe tokenizer integrated
- [ ] Performance comparison chart (Markdown table)
- [ ] Optimization TODO list for Week 18

### Nice to Have
- [ ] nvprof/nsys profiling reports
- [ ] Automated benchmark script
- [ ] Video demo of CUDA acceleration

---

## 🚀 Quick Start Script

Run this to get baseline profiling immediately:

```bash
bash /tmp/hermes-week17-start.sh
```

This will:
1. Create `profiling.rs` example
2. Run baseline test with 100 tokens
3. Save results to `/tmp/week17_baseline.txt`
4. Show throughput metric (tok/s)

**Time to first result**: ~5 minutes

---

## 📚 Reference Documents Created

| File | Purpose | Location |
|------|---------|----------|
| `WEEK_17_IMPLEMENTATION_PLAN.md` | Detailed 7-day plan | `/home/crombo/projects/pesti/` |
| `WEEK_17_ACTION_CHECKLIST.md` | Immediate action items | `/home/crombo/projects/pesti/` |
| `/tmp/hermes-week17-start.sh` | Quick start script | `/tmp/` |

---

## 🔗 Dependencies on Week 16

Week 17 builds directly on completed work:
- ✅ Real Qwen2 tokenizer data (50k vocab, 151k merges)
- ✅ CUDA dispatch system wired in `generate()` method
- ✅ All conformance tests passing
- ✅ Model loads through all 32 transformer layers

**No blocking issues** - ready to execute immediately.

---

## 💡 Key Questions to Answer This Week

1. **Performance baseline**: What's our actual CPU-only throughput?
2. **CUDA activation**: Does GPU acceleration actually work on RTX 4070 Ti SUPER?
3. **Speedup factor**: How much faster is CUDA vs CPU? (Expected: 2-8×)
4. **Tokenizer overhead**: How much time does BPE encoding/decoding take?
5. **Optimization opportunities**: Where should we focus next sprint?

---

## 🎬 Ready to Start?

**First action**: Run the quick start script to get baseline metrics:

```bash
bash /tmp/hermes-week17-start.sh
```

Then review results in `/tmp/week17_baseline.txt` and decide on next steps.

---

*Plan created: August 18, 2026*  
*Status: Ready for execution*  
*Next update: After baseline profiling complete*
