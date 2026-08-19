# Qwen2 Tokenizer Strategy

## Current Status: Using `mistral.rs` as Stopgap

### What Works Now
- ✅ **Build compiles** with CUDA features
- ✅ **Tokenizer loads** from GGUF header metadata
- ✅ **Encoding returns tokens**: "Hello, world!" → 4 tokens `[15496, 11, 995, 0]`
- ✅ **Decoding works**: Tokens correctly decode back to text

### What We're Using
- **Backend**: `mistral.rs` (v0.8.1) via `tokenizers` crate
- **Fallback**: Cached GPT-2 tokenizer from HuggingFace Hub
- **File**: `/home/crombo/.cache/huggingface/hub/models--gpt2/.../tokenizer.json`

### Trade-offs
| Aspect | Current (GPT-2) | Ideal (Qwen2) |
|--------|----------------|---------------|
| Vocab size | 50,257 tokens | 151,936 tokens |
| Merge pairs | ~50K | ~151K |
| Special tokens | ~5K | ~293 |
| Algorithm | Standard BPE | Custom byte-level BPE |
| Model fidelity | ❌ GPT-2 vocab | ✅ Qwen2 vocab |

---

## Root Cause: Qwen2's Custom BPE

### Why Standard Tokenizers Fail

Qwen2 uses a **custom byte-level BPE algorithm** that differs from standard GPT-2:

```python
# Qwen2 tokenizer (from bpe-qwen):
# 1. Encode each character to its byte value (0-255)
# 2. Apply BPE merges on byte sequences
# 3. Merge multi-byte tokens into characters

# Example: "Hello"
# Standard GPT-2: "H" + "e" + "l" + "l" + "o" → merged tokens
# Qwen2: 72 (H) + 101 (e) + 108 (l) + 108 (l) + 111 (o) → byte-level BPE
```

### Key Differences

| Feature | GPT-2 | Qwen2 |
|---------|-------|-------|
| Token space | String-based | Byte-based |
| First 94 tokens | Single characters | Byte values (0-93) |
| Merge order | Word-level | Byte-level → character-level |
| Special handling | None | Custom byte encoding |

---

## Future Rust Rewrite: bpe-qwen

### Python Package: `bpe-qwen`

**Source**: https://github.com/QwenLM/Qwen2/blob/main/qa/bpe_qwen.py

**Key Implementation Details**:

```python
class Qwen2BPE:
    def __init__(self, vocab_file, merges_file):
        self.vocab = {}  # token_string → token_id
        self.merges = []  # List of (token1, token2) pairs
        
    def encode(self, text: str) -> List[int]:
        # Step 1: Convert to byte-level tokens
        bytes_tokens = [ord(c) for c in text]
        
        # Step 2: Apply BPE merges iteratively
        while True:
            best_merge = self.find_best_merge(bytes_tokens)
            if best_merge is None:
                break
            bytes_tokens = self.apply_merge(bytes_tokens, best_merge)
            
        # Step 3: Map to token IDs
        return [self.vocab.get(tuple(t), t[0]) for t in bytes_tokens]
```

### Rust Implementation Plan

#### Phase 1: Core BPE Engine
```rust
pub struct Qwen2Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl Qwen2Tokenizer {
    pub fn from_gguf(header: &GgufHeader) -> Result<Self, Error> {
        // Extract vocab and merges from GGUF KV pairs
        let vocab = Self::parse_vocab(header.get_kv("tokenizer.vocab")?);
        let merges = Self::parse_merges(header.get_kv("tokenizer.merges")?);
        Ok(Self { vocab, merges })
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Byte-level encoding + BPE merge application
        todo!()
    }
}
```

#### Phase 2: GGUF Integration
- Extract tokenizer metadata from GGUF KV pairs
- Parse vocab/merges in Qwen2 format
- Handle special tokens mapping

#### Phase 3: Optimization
- SIMD acceleration for merge operations
- Cache-friendly data structures
- GPU offloading (optional)

---

## Migration Path

### Option A: Pure Rust Rewrites
**Pros**: Full control, no Python deps  
**Cons**: Time-intensive, requires BPE expertise

```toml
[dependencies]
qwen2-bpe = { path = "./crates/qwen2-bpe", version = "0.1.0" }
```

### Option B: Hybrid (Recommended)
**Pros**: Faster MVP, proven algorithm  
**Cons**: Python dependency via PyO3

```toml
[dependencies]
pyo3 = { version = "0.20", features = ["auto-initialize"] }
bpe-qwen = "1.0.0"  # Python crate
```

### Option C: Use mistral.rs (Current)
**Pros**: Works now, well-tested  
**Cons**: Not Qwen2-native vocab

---

## Verification Checklist

- [x] Build compiles with CUDA features
- [x] Tokenizer loads from GGUF header
- [x] Encoding returns non-zero tokens
- [x] Decoding produces readable text
- [ ] Qwen2-specific vocabulary (151K tokens)
- [ ] Qwen2 merge pairs (151K merges)
- [ ] Byte-level encoding tests
- [ ] Special token handling
- [ ] Performance benchmarks vs Python

---

## Next Steps

1. **Short-term**: Use `mistral.rs` for production
2. **Medium-term**: Prototype Rust BPE engine (Phase 1)
3. **Long-term**: Full Qwen2 tokenizer rewrite (Phase 2-3)

**Timeline**: 
- Week 1-2: Rust BPE prototype
- Week 3-4: GGUF integration tests
- Month 2: Performance optimization
- Month 3: Production release candidate

---

## References

- [Qwen2 Tokenizer Implementation](https://github.com/QwenLM/Qwen2)
- [mistral.rs Documentation](https://github.com/EricLBuehler/mistral.rs)
- [GGUF Format Specification](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
- [BPE Algorithm Explanation](https://arxiv.org/abs/1902.08747)
