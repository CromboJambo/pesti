# Conformance Testing Plan - Target 90% Before GPU Kernels

## Current State (v0.1.4)

**Corpus available:**
- `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MB, Q4_K_M quantized)
- `qwen2.5-0.5b-instruct-f16.gguf` (small stub file - needs real model)
- `qwen2.5-0.5b-instruct-q8_0.gguf` (644 MB, Q8_0 quantized)

**Current conformance test status:**
- Parser tests: 53/53 passing (GGUF v3 parsing ✅)
- CPU dispatch tests: 1/1 passing (test_dispatch_vs_cpu_output ✅)
- Real model tests: 0/3 passing (all ignored or failing due to path issues)

## Gap Analysis

### What's Working ✅
1. **GGUF v3 parsing** - All parser conformance tests pass
2. **CPU inference path** - Pure Rust transformer works correctly
3. **Dispatch layer** - CPU fallback works with tolerance < 1e-2
4. **Dequantization** - ggml-quants integration verified

### What's Missing ❌
1. **Real model conformance tests** - Tests reference wrong paths
2. **Byte-exact RoPE + attention verification** - Only tolerance-based comparison exists
3. **K-family quantization verification** - Q2_K through Q8_K tests marked `#[ignore]`
4. **Differential testing vs llama.cpp** - No reference implementation comparison
5. **Output head correctness** - LM head not being tested properly

## Implementation Plan

### Phase 1: Fix Test Infrastructure (Week 1)
- [ ] Fix path references in existing tests to point to `/home/crombo/projects/pesti/conformance-corpus/`
- [ ] Ensure f16 model file is downloaded or created from source
- [ ] Update `conformance-gate.sh` to use correct paths

### Phase 2: Implement Byte-Exact Conformance (Week 2)
**Goal:** Compare pesti outputs vs llama.cpp reference byte-for-byte (or within tight tolerance)

**Key components:**
1. **Reference runner** - Download/build llama.cpp binary for differential testing
2. **Deterministic inference** - Lock sampling to argmax (temp=0.0) for reproducibility
3. **Layer-by-layer comparison** - Compare intermediate tensors, not just final logits
4. **RoPE verification** - Extract and compare RoPE embeddings layer-by-layer

**Files to create:**
- `pesti-runner/tests/conformance/rope_verification.rs` - RoPE correctness test
- `pesti-runner/tests/conformance/attention_comparison.rs` - Attention output comparison
- `pesti-runner/tests/conformance/output_head.rs` - LM head correctness test

### Phase 3: K-Family Quantization Tests (Week 3)
**Goal:** Verify all quant types Q2_K through Q8_K produce correct outputs

**Approach:**
1. Download or create small test models for each quant type
2. Run same prompt through pesti + llama.cpp
3. Compare output token distributions (KL divergence < threshold)
4. Remove `#[ignore]` markers once verified

**Files to update:**
- `pesti-runner/src/gguf_weight_loader.rs` - Ensure all K-family loaders tested
- Add integration tests for each quant type

### Phase 4: Conformance Gate Integration (Week 4)
**Goal:** Make conformance testing part of CI/CD pipeline

**Components:**
1. **Floor file** - Establish minimum passing threshold (start at 80%, target 90%)
2. **GitHub Actions workflow** - Run conformance tests on push/PR
3. **Artifact generation** - Save divergence reports for failed models
4. **Delta minimization** - Automatically reduce failure diffs to minimal patches

## Success Metrics

### 90% Conformance Definition:
- ✅ **Parser conformance:** 100% of GGUF v3 spec tests pass
- ✅ **CPU inference:** Byte-exact match with llama.cpp on F16 model (within 1e-4 tolerance)
- ✅ **RoPE correctness:** Attention inputs match within 1e-5 relative error
- ✅ **Attention outputs:** Full attention layer matches within 1e-3 relative error
- ✅ **K-family quantization:** At least 80% of Q2_K-Q8_K tests pass
- ✅ **Output head:** Final logits within 1e-3 of reference

### Current Status vs Target:
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Parser conformance | 53/53 (100%) | 100% | ✅ Complete |
| CPU inference match | N/A | 95%+ | ❌ Need tests |
| RoPE verification | N/A | 100% | ❌ Need tests |
| Attention correctness | N/A | 95%+ | ❌ Need tests |
| K-family quantization | ~0% (ignored) | 80% | ❌ Need tests |
| Output head | N/A | 95%+ | ❌ Need tests |

**Overall: ~15-20% → Target 90%**

## Immediate Next Steps

### Step 1: Fix Paths & Enable Existing Tests (Priority: High)
```bash
# Update test path references
sed -i 's|/home/crombo/projects/conformance-corpus|/home/crombo/projects/pesti/conformance-corpus|g' \
  pesti-runner/tests/dispatch_integration.rs

# Run real model tests
cargo test --package pesti-runner test_dispatch_conformance_real_model -- --include-ignored
```

### Step 2: Create Byte-Exact RoPE Test (Priority: High)
```rust
// pesti-runner/tests/conformance/rope_verification.rs
#[test]
fn test_rope_embedding_correctness() {
    let model_path = Path::new("/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-f16.gguf");
    let mut model = CpuModel::load_gguf(model_path).unwrap();
    
    // Embed token 0
    let embed = model.llama_model.embed(0, 0).unwrap();
    
    // Compare with llama.cpp reference (pre-computed expected values)
    let expected_rope = /* ... */;
    
    assert!(max_abs_diff(&embed, &expected_rope) < 1e-5);
}
```

### Step 3: Download F16 Model for Testing
```bash
# Get reference F16 model from HuggingFace
hf-cli download Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-0.5B-Instruct-f16.gguf \
  --out /home/crombo/projects/pesti/conformance-corpus/
```

### Step 4: Build llama.cpp Reference Binary
```bash
git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp && make -j$(nproc)
# This gives us llama-cli for reference output comparison
```

## Timeline

| Week | Focus | Target Conformance |
|------|-------|-------------------|
| 1 | Fix paths, enable tests | 30% |
| 2 | RoPE + attention tests | 50% |
| 3 | K-family quantization | 70% |
| 4 | Output head + CI gate | 90%+ |

## Risk Mitigation

### If F16 model too large:
- Use smaller model (e.g., Phi-3-mini, TinyLlama)
- Downsample to first 100 tokens for testing

### If llama.cpp reference unavailable:
- Establish internal baseline (CPU path = ground truth)
- Compare dispatch vs CPU with tight tolerance

### If K-family tests fail:
- Isolate which quant types fail
- Focus on Q4_K_M and Q8_0 first (most common in corpus)
- Document known issues vs bugs

---

**Decision point:** Should we push GPU kernels now or wait for 90% conformance?
**Recommendation:** Wait. GPU speedup is meaningless if outputs are wrong.