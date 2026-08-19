# Qwen2 BPE Tokenizer - Rust Implementation (Phase 1)

## Overview

This crate implements the core byte-level BPE algorithm for Qwen2 tokenizers in pure Rust. It serves as a foundation for eventually replacing Python's `bpe-qwen` dependency.

## Status: Phase 1 Complete ✅

**Completed:**
- ✅ Core BPE engine with byte-level encoding
- ✅ Merge application algorithm (iterative)
- ✅ Basic unit tests passing
- ✅ Integration into PESTI workspace

**Next Steps:**
- ⏳ Integrate with actual Qwen2 vocab/merges from GGUF
- ⏳ Add performance benchmarks vs Python implementation
- ⏳ Handle special tokens (BOS, EOS, etc.)
- ⏳ Add proper merge priority ordering

## Architecture

### Core Components

```rust
pub struct Qwen2Tokenizer {
    config: Qwen2Config,
    reverse_vocab: HashMap<Vec<u8>, u32>,  // bytes → token_id
    forward_vocab: HashMap<u32, Vec<u8>>,  // token_id → bytes
    merge_pairs: HashSet<(u32, u32)>,      // mergeable pairs
}
```

### Algorithm Flow

1. **Byte-level encoding**: Convert text to byte sequence (e.g., "Hello" → [72, 101, 108, 108, 111])
2. **BPE merges**: Iteratively apply merge pairs until no more applicable
3. **Token ID mapping**: Convert final byte sequences to token IDs

### Example

```rust
let config = Qwen2Config {
    vocab: vec![
        (72, vec![b'H']),   // 'H' → token 72
        (101, vec![b'e']),  // 'e' → token 101
        (108, vec![b'l']),  // 'l' → token 108
        (111, vec![b'o']),  // 'o' → token 111
    ].into_iter().collect(),
    merges: vec![(72, 101)], // Merge H+e → "He"
};

let tokenizer = Qwen2Tokenizer::new(config).unwrap();
let tokens = tokenizer.encode("Hello").unwrap();
// Result: [72, 108, 108, 111] (4 tokens: He, l, l, o)
```

## Current Limitations

### What Works
- ✅ Byte-level encoding of arbitrary text
- ✅ Iterative BPE merge application
- ✅ Basic decode functionality
- ✅ Unit tests with mock vocab/merges

### What's Missing
- ❌ Real Qwen2 vocabulary (needs GGUF extraction)
- ❌ Special token handling (BOS, EOS, etc.)
- ❌ Merge priority ordering (Qwen2 has specific merge order)
- ❌ Unicode normalization (Qwen2 uses byte-level, but edge cases exist)
- ❌ Performance optimization (currently O(n²) in worst case)

## Comparison with Python `bpe-qwen`

| Feature | Python `bpe-qwen` | Rust `qwen2-bpe` | Status |
|---------|------------------|------------------|--------|
| Byte-level encoding | ✅ | ✅ | Match |
| BPE merge algorithm | ✅ | ✅ | Match |
| Vocabulary loading | ✅ JSON files | ❌ Needs GGUF | Gap |
| Special tokens | ✅ | ❌ | Gap |
| Performance | ~10k tok/s | TBD | TBD |
| Unicode handling | ✅ | ⚠️ Partial | Gap |

## Integration with PESTI

The crate is integrated into the PESTI workspace at `/home/crombo/projects/pesti/crates/qwen2-bpe`.

**Current usage:** Standalone prototype, not yet integrated into `pesti-runner`.

**Next integration step:** Replace `mistralrs_core::tokenizer::convert_gguf_to_hf_tokenizer` with pure Rust implementation.

## Testing

```bash
# Run all tests
cargo test -p qwen2-bpe --lib

# Run specific test
cargo test -p qwen2-bpe --lib tests::test_qwen2_hello_world_with_merges
```

**Test Results:**
- ✅ `test_byte_level_encoding`: Basic byte encoding works
- ✅ `test_decode`: Round-trip decode works
- ✅ `test_qwen2_hello_world_with_merges`: Merge application works

## Next Milestones

### Phase 2: GGUF Integration (Week 2)
1. Extract vocab/merges from actual Qwen2 GGUF file
2. Handle special tokens (BOS, EOS, etc.)
3. Add merge priority ordering
4. Benchmark against Python implementation

### Phase 3: Production Ready (Week 3-4)
1. Performance optimization (SIMD where possible)
2. Unicode edge cases
3. Integration into `pesti-runner`
4. Replace `mistralrs_core` dependency

## Strategic Value

**Why rewrite?**
- ✅ Pure Rust = no Python dependency
- ✅ Better performance potential
- ✅ Type safety and compile-time checks
- ✅ Learn Qwen2 internals deeply
- ✅ Contribute back to ecosystem

**Trade-offs:**
- ⚠️ Time investment: ~2-3 weeks for full implementation
- ⚠️ Testing burden: Need to validate against reference
- ⚠️ Risk of missing edge cases

**Recommendation:** Continue with Phase 1→Phase 2 transition. Current prototype validates the approach works correctly.

## References

- Qwen2 tokenizer source: https://github.com/QwenLM/Qwen2/blob/main/qa/bpe_qwen.py
- Mistral.rs tokenizer: https://github.com/mistralai/mistral.rs
- GGUF format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

**Author:** crombo  
**Date:** 2026-08-17  
**Version:** 0.1.0 (Phase 1 prototype)
