# Week 17 Implementation Plan

**Date**: August 18, 2026  
**Sprint Goal**: Profile CUDA performance and integrate real tokenizer into main pipeline

---

## 🎯 Sprint Objectives

### Primary Goals (Must Complete)
1. **Benchmark baseline performance** - Measure CPU-only throughput with real data
2. **Profile CUDA path** - Verify GPU acceleration is actually being used
3. **Integrate qwen2-bpe crate** - Replace fallback tokenizer in main pipeline
4. **Document optimizations** - Capture profiling results and improvement opportunities

### Secondary Goals (Nice to Have)
5. Optimize memory layout for attention kernels
6. Add structured logging for dispatch decisions
7. Create performance comparison chart vs llama.cpp

---

## 📊 Current State Baseline

From Week 16 completion:
- **Model**: Qwen2.5-0.5B-Instruct (Q4_K_M quantized)
- **Vocab size**: 50,257 tokens (real data)
- **Layers**: 32 transformer blocks
- **CUDA path**: Wired but not yet tested with real GPU
- **Fallback**: CPU-only mode working (all tests pass)

**Expected performance targets:**
- CPU baseline: ~0.5-1 tok/s (conservative estimate)
- CUDA target: ~3-5 tok/s (6-8× speedup)
- Tokenizer overhead: Should be <10% of total time with real BPE

---

## 🔧 Implementation Phases

### Phase 1: Baseline Profiling (Day 1-2)

**Goal**: Establish CPU-only performance baseline before CUDA activation

#### Tasks:
1. **Create profiler example** - Measure token generation time breakdown
   ```rust
   // pesti-runner/examples/profiling.rs
   fn main() {
       let model = LlamaModel::load_gguf("path/to/model.gguf").unwrap();
       
       let start = Instant::now();
       let tokens = model.generate("Test prompt", 100).unwrap();
       let total_time = start.elapsed();
       
       println!("Total time: {:.2}s", total_time.as_secs_f64());
       println!("Throughput: {:.2} tok/s", tokens.len() as f64 / total_time.as_secs_f64());
   }
   ```

2. **Add timing instrumentation** - Measure each phase separately
   - Embedding lookup time
   - Forward pass time (per layer)
   - KV cache update time
   - Token sampling time

3. **Run baseline tests** - Generate 100 tokens from same prompt, repeat 5x

4. **Document results** - Save timing breakdown to `WEEK_17_PROFILING_RESULTS.md`

#### Deliverables:
- ✅ `profiling.rs` example
- ✅ Baseline throughput metric (tok/s)
- ✅ Per-phase timing breakdown

---

### Phase 2: CUDA Activation & Verification (Day 3-4)

**Goal**: Confirm GPU acceleration is actually being used and measure speedup

#### Tasks:
1. **Add dispatch logging** - Log when GPU path is taken vs CPU fallback
   ```rust
   if self.dispatch.is_some() {
       debug!("Using GPU dispatch for forward pass");
       // ... GPU path
   } else {
       warn!("Falling back to CPU (no CUDA)");
       // ... CPU path
   }
   ```

2. **Create CUDA test harness** - Verify GPU is detected and active
   ```rust
   fn test_cuda_active() -> Result<bool, RunnerError> {
       let ctx = model.dispatch.as_ref().unwrap();
       
       // Check device info
       let device_info = ctx.device_info()?;
       Ok(device_info.max_memory > 0)
   }
   ```

3. **Run CUDA benchmarks** - Same prompt as Phase 1, measure speedup
   - Generate 100 tokens with CUDA enabled
   - Compare vs CPU baseline
   - Measure GPU memory usage (VRAM allocation)

4. **Profile attention kernels** - Use `nvprof` or `nsys` if available
   ```bash
   nvprof --print-gpu-trace cargo run --example profiling --features cuda
   ```

5. **Document findings** - Speedup factor, VRAM usage, any errors

#### Deliverables:
- ✅ CUDA activation logging
- ✅ GPU detection verification
- ✅ Speedup metric (CUDA vs CPU)
- ✅ VRAM usage report

---

### Phase 3: Real Tokenizer Integration (Day 5-6)

**Goal**: Replace fallback tokenizer with real qwen2-bpe crate

#### Tasks:
1. **Audit current tokenizer path** - Identify all places using `fallback_tokenizer`
   ```bash
   grep -r "fallback_tokenizer\|whitespace" pesti-runner/src/
   ```

2. **Create integration wrapper** - Bridge between `qwen2-bpe` and existing API
   ```rust
   // pesti-runner/src/tokenizer/qwen2_bpe.rs
   pub struct Qwen2BpeTokenizer {
       inner: qwen2_bpe::Qwen2BPE,
   }
   
   impl PestiTokenizer for Qwen2BpeTokenizer {
       fn encode(&self, text: &str) -> Result<Vec<u32>, RunnerError> {
           self.inner.encode(text).map_err(|e| e.into())
       }
       
       fn decode(&self, tokens: &[u32]) -> Result<String, RunnerError> {
           self.inner.decode(tokens).map_err(|e| e.into())
       }
   }
   ```

3. **Update model loading** - Load real tokenizer from GGUF file
   ```rust
   // In LlamaModel::load_gguf()
   if let Some(tokenizer_path) = header.get_kv_str("tokenizer.model") {
       let bpe = Qwen2BPE::from_gguf(Path::new(tokenizer_path))?;
       model.tokenizer = Some(PestiTokenizer::Qwen2Bpe(Qwen2BpeTokenizer { inner: bpe }));
   }
   ```

4. **Add tokenizer feature flag** - Allow CPU-only builds without BPE
   ```toml
   # pesti-runner/Cargo.toml
   [features]
   default = []
   rust-tokenizer = ["qwen2-bpe"]
   cuda = ["cudarc"]
   ```

5. **Run integration tests** - Verify encoding/decoding correctness
   ```bash
   cargo test -p qwen2-bpe --features rust-tokenizer
   ```

6. **Benchmark tokenizer performance** - Compare vs fallback
   ```rust
   // Measure tokens/sec for encode() and decode()
   let start = Instant::now();
   for _ in 0..100 {
       tokenizer.encode("Test prompt").unwrap();
   }
   println!("Encode throughput: {:.2} tok/s", ...);
   ```

#### Deliverables:
- ✅ Real BPE tokenizer integrated
- ✅ Feature flag support (CPU-only builds)
- ✅ Tokenizer performance metrics
- ✅ Encoding/decoding correctness verified

---

### Phase 4: Optimization & Documentation (Day 7)

**Goal**: Capture learnings, document optimizations, prepare for next sprint

#### Tasks:
1. **Profile attention kernels** - Identify hotspots
   - Check if CUDA kernels are actually being used
   - Measure GEMM vs naive implementation speedup
   - Profile memory bandwidth utilization

2. **Document optimization opportunities** - Create TODO list
   ```markdown
   ## Optimization Opportunities (Week 18+)
   
   ### High Priority
   - [ ] Fuse attention QKV projections into single kernel
   - [ ] Implement KV cache quantization (Q4_K)
   - [ ] Add batched generation for parallel prompts
   
   ### Medium Priority
   - [ ] Optimize rope embedding lookup
   - [ ] Reduce memory allocations in forward pass
   - [ ] Profile and optimize softmax implementation
   ```

3. **Create performance comparison chart** - Visual benchmark vs llama.cpp
   ```markdown
   | Model          | Backend  | Throughput (tok/s) | Latency (ms/tok) |
   |----------------|----------|-------------------|------------------|
   | Qwen2.5-0.5B   | CPU      | X.XX              | XXX.X            |
   | Qwen2.5-0.5B   | CUDA     | Y.YY              | YYY.Y            |
   | llama.cpp v1.6 | CUDA     | Z.ZZ              | ZZZ.Z            |
   ```

4. **Write Week 17 summary** - Document what worked, what didn't
   - Successes: What exceeded expectations?
   - Surprises: What was unexpected?
   - Blockers: What prevented progress?
   - Next steps: What should Week 18 focus on?

5. **Clean up temporary files** - Remove profiling artifacts, temp logs

#### Deliverables:
- ✅ Performance comparison chart
- ✅ Optimization TODO list for Week 18
- ✅ Week 17 summary document
- ✅ Clean codebase (no temp files)

---

## 📈 Success Metrics

### Primary KPIs (Must Achieve)
- [ ] **Baseline established**: CPU-only throughput measured (tok/s)
- [ ] **CUDA verified**: GPU acceleration detected and working
- [ ] **Speedup achieved**: CUDA ≥ 2× faster than CPU (conservative target)
- [ ] **Real tokenizer**: qwen2-bpe integrated into main pipeline

### Secondary KPIs (Nice to Have)
- [ ] **Tokenizer overhead**: <10% of total generation time
- [ ] **Memory efficiency**: VRAM usage documented and reasonable
- [ ] **Optimization roadmap**: Clear list of 3-5 high-impact improvements

---

## 🛠️ Tools & Dependencies

### Required Tools
- `cargo` - Rust build system
- `nvprof` / `nsys` (optional) - NVIDIA profiling tools
- `grep`, `awk`, `sed` - Text processing for logs
- `gnuplot` or similar (optional) - Performance charts

### Dependencies to Monitor
- `cudarc` - CUDA runtime bindings (already integrated)
- `qwen2-bpe` - Pure Rust tokenizer (target integration)
- `tracing` / `tracing-subscriber` - Structured logging

---

## 🔄 Risk Mitigation

### Potential Blockers & Solutions

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CUDA kernels not loading on RTX 4070 Ti | Low | High | Fallback to CPU path (already implemented) |
| Real tokenizer slower than fallback | Medium | Medium | Profile and optimize BPE merge logic |
| Memory fragmentation in dispatch system | Low | Medium | Add explicit `drop()` calls, use arena allocator |
| GGUF metadata inconsistencies | High | Low | Already handled by dimension inference (Week 16) |

---

## 📝 Notes & Assumptions

### Assumptions
- RTX 4070 Ti SUPER (sm_8.9) has CUDA drivers properly installed
- Model file (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) is accessible and valid
- GPU memory sufficient for model weights + KV cache (~1GB estimated)

### Unknowns
- Actual speedup factor (depends on kernel efficiency, memory bandwidth)
- Real tokenizer throughput vs fallback
- Attention kernel performance characteristics

---

## 🎯 Week 17 Deliverables Checklist

**End of Sprint Must-Haves:**
- [ ] `WEEK_17_PROFILING_RESULTS.md` - Complete benchmark report
- [ ] `profiling.rs` example - Baseline measurement tool
- [ ] CUDA activation logging in main codebase
- [ ] Real qwen2-bpe tokenizer integrated into `LlamaModel`
- [ ] Feature flag `rust-tokenizer` for optional BPE support
- [ ] Performance comparison chart (Markdown table)
- [ ] Optimization TODO list for Week 18

**Nice-to-Haves:**
- [ ] nvprof/nsys profiling reports (if tools available)
- [ ] Automated benchmark script
- [ ] Video demo of CUDA acceleration in action

---

*Plan created: August 18, 2026*  
*Next review: End of Week 17 sprint (August 25, 2026)*
