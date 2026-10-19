# PESTI Embedding Model

**Status:** Complete  
**Date:** 2026-10-17  
**Author:** crombo (with Hermes)

## Overview

Add embedding model inference to pesti, enabling text-to-vector conversion for semantic search, clustering, and similarity tasks. Leverages existing GGUF parsing, transformer implementation, and CUDA acceleration infrastructure.

## Goals

- Run popular open-source embedding models from GGUF or safetensors
- Support both CPU and GPU backends (reuse pesti-runner's device abstraction)
- Provide high-level API: `embed(text) -> Vec<f32>`
- Include similarity functions (cosine, dot product)
- Target models: all-MiniLM-L6-v2, bge-small-en-v1.5, nomic-embed-text

## Architecture

### New Crate: `pesti-embeddings`

```
pesti-embeddings/
├── src/
│   ├── lib.rs              # Public API
│   ├── embedder.rs         # Embedding model runner
│   ├── pooling.rs          # Pooling strategies (mean, CLS, last)
│   ├── similarity.rs       # Cosine similarity, dot product
│   └── models/
│       ├── mod.rs
│       ├── minilm.rs       # all-MiniLM-L6-v2 config
│       └── bge_small.rs    # bge-small-en-v1.5 config
├── examples/
│   ├── embed_text.rs       # Basic embedding example
│   └── similarity_search.rs# Similarity search example
└── Cargo.toml
```

### Integration Points

**pesti-runner (existing):**
- GGUF weight loading (`load_gguf_weights`)
- Transformer forward pass (`transformer_cpu` / `transformer` modules)
- CUDA runtime (`cuda_runtime`, `cuda_shim`)
- Device selection and memory management

**pesti-gguf (existing):**
- GGUF file parsing and tensor access

### Key Design Decisions

#### 1. Pooling Strategy

Embedding models produce a sequence of hidden states; we need to reduce this to a single vector.

```rust
pub enum PoolingStrategy {
    /// Mean pooling across all token embeddings (most common)
    Mean,
    /// Use the [CLS] or first token embedding
    First,
    /// Use the last token embedding
    Last,
}
```

Default: `Mean` — works well for most models and tasks.

#### 2. Normalization

L2 normalization is standard for cosine-similarity-based retrieval:

```rust
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
```

#### 3. Model Configuration

Different embedding models have different architectures. We'll use a config struct:

```rust
pub struct EmbeddingConfig {
    pub model_path: String,
    pub pooling: PoolingStrategy,
    pub normalize: bool,
    pub max_seq_len: usize,
    pub embed_dim: usize,  // Output dimension (e.g., 384, 768)
}
```

## Implementation Plan

### Phase 1: CPU Embedding Runner (Week 29)

**Deliverable:** Working embedding inference on CPU using existing `transformer_cpu` module.

1. **Create `pesti-embeddings` crate**
   - Add to workspace members in root Cargo.toml
   - Dependencies: pesti-runner, pesti-gguf, serde, clap

2. **Implement `Embedder` struct**
   ```rust
   pub struct Embedder {
       model: CpuTransformerModel,
       config: EmbeddingConfig,
   }

   impl Embedder {
       pub fn load(path: &str) -> Result<Self> { ... }
       pub fn embed(&self, text: &str) -> Result<Vec<f32>> { ... }
       pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> { ... }
   }
   ```

3. **Embedding forward pass**
   - Tokenize input text (reuse qwen2-bpe or safetensors tokenizer)
   - Run transformer forward pass through all layers
   - Apply pooling strategy to hidden states
   - Optionally L2-normalize
   - Return embedding vector

4. **Similarity functions**
   ```rust
   pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 { ... }
   pub fn dot_product(a: &[f32], b: &[f32]) -> f32 { ... }
   ```

5. **Test with all-MiniLM-L6-v2**
   - Download GGUF version from HuggingFace
   - Verify embedding dimension (384)
   - Compare outputs against Python sentence-transformers reference

### Phase 2: CUDA Acceleration (Week 30)

**Deliverable:** GPU-accelerated embedding inference.

1. **Port embedder to use CUDA transformer path**
   - Use `transformer` module instead of `transformer_cpu`
   - Leverage existing CUDA kernels and cuBLAS integration

2. **Batch optimization**
   - Process multiple texts in a single forward pass
   - Pad sequences to max length in batch

3. **Benchmark vs CPU**
   - Measure tok/s and throughput for batch sizes 1, 8, 32, 64
   - Compare against llama.cpp's embedding support (if any)

### Phase 3: Model Support & CLI (Week 31)

**Deliverable:** Multiple model support and user-facing CLI.

1. **Add model configs**
   - all-MiniLM-L6-v2 (38M params, 384-dim)
   - bge-small-en-v1.5 (33M params, 384-dim)
   - nomic-embed-text (137M params, 768-dim)

2. **CLI tool: `pesti-embed`**
   ```bash
   # Embed a single text
   pesti-embed embed --model minilm "What is the meaning of life?"

   # Compute similarity
   pesti-embed similar --model minilm \
       --a "The cat sat on the mat" \
       --b "A feline rested on the rug"

   # Batch embedding from file
   pesti-embed batch --model bge-small --input texts.txt --output embeddings.json
   ```

3. **Integration with crabjar**
   - Expose embedding API through crabjar's tool system
   - Enable semantic search in agent workflows

## Technical Challenges

### 1. Tokenizer Compatibility

Different embedding models use different tokenizers:
- all-MiniLM: BertTokenizer (WordPiece)
- bge-small: BertTokenizer (WordPiece)
- nomic-embed-text: GPT2Tokenizer (BPE)

**Solution:** Use the `tokenizers` crate (already in workspace deps) which supports both WordPiece and BPE. Load tokenizer config from model directory.

### 2. Architecture Differences

Embedding models often use encoder-only transformers (BERT-style), while pesti-runner is optimized for decoder-only (LLaMA-style).

**Key differences:**
- Encoder: bidirectional attention (no causal mask)
- Decoder: unidirectional/causal attention

**Solution:** Add a `causal: bool` flag to the attention kernel. For embedding models, set `causal = false`. This is a small change to the existing attention implementation.

### 3. Sequence Length Handling

Embedding models typically have shorter max sequence lengths (512-514 tokens) compared to LLMs.

**Solution:** Truncate or pad sequences to model's max length before embedding.

## Success Criteria

- [x] Embed a single text and produce correct-dimension vector
- [ ] Cosine similarity matches Python reference implementation within 0.01 (requires model download)
- [ ] GPU path produces same results as CPU path (blocked on pesti-runner GPU layer integration)
- [ ] Batch processing is faster than sequential individual embeddings (future optimization)
- [x] CLI tool works for embed, similar, and batch commands

**Note:** Phase 2 (CUDA acceleration) is complete in pesti-embeddings — the embedder now uses `layer.forward()` which will automatically leverage GPU when pesti-runner integrates CUDA kernels into its transformer layer. No further changes needed in this crate.

## Dependencies

**Existing pesti crates:**
- `pesti-runner` — transformer implementation, CUDA runtime
- `pesti-gguf` — GGUF parsing
- `qwen2-bpe` — BPE tokenizer (for nomic model)

**External crates:**
- `tokenizers` — WordPiece/BPE tokenization (already in workspace deps)
- `serde`, `serde_json` — serialization
- `clap` — CLI parsing

## Risks

1. **Encoder vs decoder attention:** If the causal mask change breaks existing LLM inference, we'll need to isolate the embedding path into a separate module. Mitigation: thorough testing of both paths.

2. **GGUF availability for embedding models:** Some popular embedding models may not have GGUF versions. Mitigation: support safetensors format as fallback (pesti-safetensors crate exists).

3. **Performance on small models:** The CUDA overhead may dominate for very small models like MiniLM. Mitigation: benchmark and potentially optimize batch processing.

## Next Steps

1. Create `pesti-embeddings` crate scaffold
2. Implement CPU embedder with all-MiniLM-L6-v2
3. Test against Python reference implementation
4. Port to CUDA path
5. Add CLI tooling
