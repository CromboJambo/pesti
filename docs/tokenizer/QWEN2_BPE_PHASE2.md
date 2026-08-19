# Qwen2 BPE - Phase 2 Implementation

## Status
✅ **In Progress** - Vocabulary loaded successfully, merges pending

## Phase 1 Summary (Completed)
- Created `qwen2-bpe` crate with core BPE logic
- Implemented byte-level encoding and merge application
- Fixed infinite loop in merge logic (while-loop vs for-loop)
- All unit tests passing (3/3)

## Phase 2 Goals

### ✅ Completed: Vocabulary Loading
1. **Extract vocabulary from GGUF** using `mistralrs_core::convert_gguf_to_hf_tokenizer`
2. **Load JSON vocabularies** via `Qwen2Tokenizer::load_from_json()`
3. **Verified**: Successfully loads 50,257 tokens from Qwen2.5-0.5B model

```bash
$ cargo run -p qwen2-bpe --example test_qwen2_bpe
Loading Qwen2 tokenizer from: /tmp/qwen2_vocab_dump.json
✅ Tokenizer loaded!
Vocabulary size: 50257 tokens
```

### 🚧 In Progress: Special Tokens & Merges

#### Special Tokens (BOS, EOS, etc.)
- **Issue**: Qwen2 uses special tokens like `<|begin_of_text|>`, `<|end_of_text|>`
- **Approach**: 
  - Extract from tokenizer's special token map
  - Add to vocab with their correct IDs
  - Handle during encoding/decoding

#### Merge Pairs
- **Issue**: Real Qwen2 has ~100k+ merge pairs that must be applied in priority order
- **Current Status**: Merges field exists but not loaded from GGUF
- **Next Steps**:
  1. Extract merges from tokenizers library output
  2. Store as ordered list (priority matters!)
  3. Apply merges in correct sequence during encoding

### 📋 Pending Tasks

#### Task 4: Merge Priority Ordering
- [ ] Extract merge pairs with priority from tokenizers
- [ ] Store as `Vec<(u32, u32, usize)>` (token1, token2, priority)
- [ ] Update `apply_bpe_merges()` to respect priority order
- [ ] Add tests with real Qwen2 merge pairs

#### Task 5: Benchmark vs Python bpe-qwen
- [ ] Install Python `bpe-qwen` library
- [ ] Create comparison script (encode same text, compare token IDs)
- [ ] Measure encoding speed difference
- [ ] Document accuracy results

## Current Architecture

```rust
pub struct Qwen2Tokenizer {
    config: Qwen2Config,
    reverse_vocab: HashMap<Vec<u8>, u32>,
    forward_vocab: HashMap<u32, Vec<u8>>,
    merge_pairs: HashSet<(u32, u32)>, // TODO: Add priority field
}

pub struct Qwen2Config {
    pub vocab: HashMap<u32, Vec<u8>>,
    pub merges: Vec<(u32, u32)>, // TODO: Change to (u32, u32, usize)
}
```

## Files Modified in Phase 2

### `/home/crombo/projects/pesti/crates/qwen2-bpe/src/lib.rs`
- Added `load_from_json()` static method
- Added JSON error type to `Qwen2Error`
- Added serde_json dependency

### `/home/crombo/projects/pesti/crates/qwen2-bpe/examples/test_qwen2_bpe.rs`
- Loads real Qwen2 vocabulary from GGUF dump
- Tests encoding/decoding with actual model vocab

## Next Decision Point

**Question**: Should we:
1. **Extract merges now** (requires digging into tokenizers library output)
2. **Use Python reference first** (validate accuracy, then implement merges in Rust)
3. **Hybrid approach** (Rust for speed, Python for correctness check)

**Recommendation**: Option 2 - Extract a small sample of merge pairs from Python's `bpe-qwen`, add to Rust tests, then validate the full pipeline. This gives us confidence before investing time in complex merge priority logic.

## Evidence

### Vocabulary Extraction
```python
# From dump_vocab.rs example
tokenizer.json contains:
- vocab: { "token_str": token_id, ... } (50,257 entries)
- merges: [] (empty - need to extract separately)
```

### Encoding Results (without merges)
```
Input: "Hello, world!"
Tokens: [39, 68, 75, 75, 78, 11, 32, 86, 78, 81, 75, 67, 0] (13 tokens)
Decoded: 'Hello,Aworld!' (missing spaces due to no merges)

Expected with full merges: ~6-7 tokens (He, llo, ,,  , worl, d, !)
```

The gap shows we need merge pairs to get proper tokenization!

## Timeline

| Phase | Status | ETA |
|-------|--------|-----|
| Phase 1: Core BPE engine | ✅ Complete | Done |
| Phase 2a: Vocabulary loading | ✅ Complete | Done |
| Phase 2b: Merge extraction | 🚧 In Progress | Today |
| Phase 2c: Benchmark | ⏳ Pending | Tomorrow |
| Integration with pesti | ⏳ Pending | Next week |

---

**Author**: crombo  
**Date**: Monday, August 17, 2026  
**Session**: #190-227 (qwen2-bpe development)
