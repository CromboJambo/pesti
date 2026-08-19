//! Week 3: Model loading bug fix & real-world benchmark report
//!
//! **Status**: ✅ **IN PROGRESS** - Fixed linear.rs batch inference, identified Qwen2 GGUF format issue, validated with Q4_0 model
//!
//! ---
//!
//! ## Executive Summary
//!
//! Week 3 objectives:
//! 1. ✅ Fix `linear.rs` slice bounds errors (dynamic batch size inference)
//! 2. ✅ Identify Qwen2 GGUF weight shape misinterpretation
//! 3. ⏳ Validate end-to-end inference on working model (Qwen2.5-0.5B-Q4_0)
//! 4. ⏳ Measure real-world token throughput with CUDA features
//! 5. ⏳ Document actual performance gap vs projections
//!
//! **Current Status**: 
//! - Linear batch inference: **FIXED** ✅
//! - Qwen2 GGUF loading: **PARTIALLY FIXED** (Q4_K_M has shape mismatch, Q4_0 works)
//! - End-to-end inference: **WORKING** on Q4_0 model (6.9 tok/s)
//! - Performance measurement: **IN PROGRESS** (need to integrate flash attention kernel)
//!
//! ---
//!
//! ## Week 3 Plan (Honest Edition)
//!
//! ### Phase 1: Bug Fixing (COMPLETED)
//! ✅ **Dynamic batch size inference in `linear.rs`**
//! - Changed from `batch_size` parameter to inferred from input tensor length
//! - Formula: `batch_size = x.len() / out_features`
//! - Resolves slice bounds panic at layer 4+ of Qwen2 models
//!
//! ✅ **Qwen2 architecture detection**
//! - Confirmed Qwen2.5-0.5B uses intermediate_dim=4864 (not 4×embed_dim)
//! - Identified GGUF weight shape mismatch in Q4_K_M quantization
//! - Discovered Q4_0 variant loads correctly
//!
//! ### Phase 2: Validation & Benchmarking (IN PROGRESS)
//! ⏳ **End-to-end inference validation**
//! - Test with multiple models: TinyLlama, Qwen2.5-0.5B-Q4_0
//! - Verify forward pass produces valid outputs
//! - Check KV cache correctness
//!
//! ⏳ **Performance measurement**
//! - Measure tok/s on working model (Qwen2.5-0.5B-Q4_0)
//! - Compare against projections: 35 → 60-70 tok/s target
//! - Document actual gap with evidence
//!
//! ### Phase 3: Documentation (IN PROGRESS)
//! ⏳ **Week 3 report**
//! - Document bug fixes with code diffs
//! - Report real-world metrics (tok/s, latency)
//! - Update projections based on actual measurements
//! - Identify next steps for Week 4
//!
//! ---
//!
//! ## Key Findings
//!
//! ### 1. Linear Batch Inference Fix
//!
//! **Problem**: `Linear::forward` assumed batch_size=1 but input tensor length didn't match expected dimensions.
//!
//! **Root cause**: Hardcoded batch parameter vs actual tensor shape mismatch.
//!
//! **Solution**: Infer batch size dynamically from input tensor:
//! ```rust
//! let batch_size = x.len() / self.out_features;
//! ```
//!
//! **Impact**: Fixes slice bounds panic at layer 4+ of Qwen2 models.
//!
//! ### 2. Qwen2 GGUF Weight Shape Mismatch
//!
//! **Problem**: `Qwen2.5-0.5B-Q4_K_M.gguf` has inconsistent tensor sizes across layers.
//!
//! **Evidence**:
//! - Layer 0: `ffn_down.weight` stored bytes = 3,575,040 (claims shape [4864, 896])
//! - Layer 1: Same size (inconsistent!)
//! - Layer 2: Different size (2,451,456 bytes)
//!
//! **Root cause**: Likely a GGUF quantization bug or dynamic intermediate dimension variant.
//!
//! **Workaround**: Use `Qwen2.5-0.5B-Q4_0.gguf` which loads correctly.
//!
//! ### 3. Performance Baseline
//!
//! **Current throughput**: ~6.9 tok/s (CPU-based forward pass)
//!
//! **GPU path available**: `forward_with_dispatch()` uses CUDA kernels but benchmark uses CPU path.
//!
//! **Next step**: Integrate flash attention kernel into model for GPU acceleration.
//!
//! ---
//!
//! ## Next Steps (Week 3b)
//!
//! 1. ✅ Verify Qwen2.5-0.5B-Q4_0 end-to-end inference
//! 2. ⏳ Modify benchmark to use `forward_with_dispatch()` for GPU path
//! 3. ⏳ Measure tok/s with flash attention enabled
//! 4. ⏳ Compare against projections (target: <10% gap vs mistral.rs)
//! 5. ⏳ Document findings in Week 3 report
//!
//! ---
//!
//! ## Projections Update
//!
//! **Original projection** (Week 2): Flash attention + RoPE caching → 60-70 tok/s (<10% gap)
//!
//! **Updated projection**: Need real measurements from GPU path before updating.
//!
//! **Current baseline**: 6.9 tok/s (CPU-only, no flash attention)
//!
//! **Projected with flash attention**: 
//! - Conservative: 25-35 tok/s (50-70% improvement over CPU)
//! - Optimistic: 50-60 tok/s (7-8x improvement, closer to target)
//!
//! **Confidence**: Medium (flash attention kernel built and tested in isolation)
//!
//! ---
//!
//! ## Risks & Blockers
//!
//! 🔴 **Qwen2 GGUF format inconsistency** - May require custom loader or model conversion
//! 🟡 **Flash attention integration** - Kernel exists but not yet integrated into model forward pass
//! 🟢 **Batch inference fix** - Verified working on Q4_0 model
//!
//! ---
//!
//! ## Metrics to Report (Week 3b)
//!
//! - [ ] End-to-end token throughput (tok/s) with CUDA features
//! - [ ] Flash attention speedup vs GEMM-based attention
//! - [ ] KV cache overhead at different sequence lengths
//! - [ ] Memory bandwidth utilization (if available)
//! - [ ] Comparison against mistral.rs baseline (if model compatible)
//!
//! ---
//!
//! ## References
//!
//! - Week 2 report: `docs/WEEK-2-PERFORMANCE-VERIFICATION.md`
//! - Flash attention kernel: `pesti-runner/src/kernel/flash_attention.rs`
//! - Qwen2 architecture: Hugging Face model card for `Qwen/Qwen2.5-0.5B-Instruct`
//! - GGUF format spec: https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md
//!
//! ---
//!
//! **Author**: crombo (PESTI project)
//! **Date**: 2026-08-12
//! **Status**: In progress (awaiting GPU path benchmark)
