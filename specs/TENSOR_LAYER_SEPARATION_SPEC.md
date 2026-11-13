# PESTI Tensor Layer Separation: Tall/Skinny KV with Metadata Layers

**Status**: 🚧 Phase 1 COMPLETE — CoreTensor + TensorStructure implemented (weight_store.rs)  
**Scope**: Weight storage, loading, and model representation in pesti-runner  
**Motivation**: Separate minimal inference path from auxiliary metadata; position safetensors as a trainable-adjacent format  
**Related**: `SOFTMASK_TYPE_SPEC.md` (attention masking), `pesti-safetensors/src/schema.rs` (current DB schema)

---

## 0. Goal and Scope

### The idea (in the user's words)

> "Keep it a KV, tall and skinny, simple... put metadata and other system factors... in other layers. Different embedder that reads a different shape that will just see that as tall and skinny no matter what. Like trying to arbitrarily skip layers, but not if you have the tensors. Model weights in safe tensors closer to the thing, like closer to something trainable."

### Goal

Decouple PESTI's weight representation into independent, loadable layers so that:

1. **The KV core is always minimal** — tall and skinny weight matrices, nothing else
2. **Metadata is separate** — config, normalization stats, rope params, quantization scales live in their own layer
3. **Readers see a consistent view** — any embedder/reader sees the same tall/skinny tensor shape regardless of which metadata layers are present
4. **Layers are independently loadable** — inference needs Layer 0+1; fine-tuning needs all layers; you can skip what you don't need
5. **Safetensors becomes trainable-adjacent** — the format carries enough structure that a training framework can consume it directly without conversion

### Not in scope (for now)

- Changing the safetensors file format itself (we use it as-is)
- Modifying GGUF loading (separate concern, same principles apply later)
- Training pipeline integration (this spec enables it but doesn't build it)
- GPU memory layout changes (orthogonal to logical layering)

---

## 1. Current State: Monolithic Weight Bundling

### The problem today

`LlamaModel` (transformer/model.rs) is a monolith:

```rust
pub struct LlamaModel {
    pub config: LlamaConfig,           // arch, dims, rope params, norm eps
    pub token_embeddings: Option<...>, // weight data
    pub output: Option<...>,           // weight data
    pub final_norm: Option<...>,       // weight data
    pub layers: Vec<TransformerLayer>, // per-layer attention + FFN weights
    pub vocab_size: usize,             // config
    pub tokenizer: Option<...>,        // operational
    pub tokenizer_config: Option<...>, // operational
    pub kv_caches: Option<...>,        // runtime state
    // ... dispatch handles, capture buffers, etc.
}
```

Everything is bundled: weight data, architectural config, normalization parameters, tokenizer state, runtime caches, dispatch handles. Adding a "system factor" (custom position encoding, attention bias, LoRA adapter) means modifying this core struct.

### What pesti-safetensors already does right

The SQLite schema (`pesti-safetensors/src/schema.rs`) already separates concerns:

- `model_weights`: model-level metadata (name, repo_id, file_path, tensor_count, dtype, device, checksum, metadata JSON)
- `tensor_metadata`: per-tensor info (tensor_name, shape, dtype, size_bytes, checksum)

But the **runtime model** doesn't reflect this separation. Loading goes: safetensors file → `HashMap<String, Vec<u8>>` → monolithic `LlamaModel`. The metadata layer is discarded after config extraction.

### Consequences

1. **No partial loading**: Can't load just embeddings without all transformer layers
2. **No extensibility**: Adding LoRA adapters or custom position encodings requires modifying core structs
3. **Not trainable-adjacent**: A training framework can't consume the runtime representation — it needs to re-parse the original safetensors file
4. **Config and data entangled**: Changing `rope_base` or `rms_norm_eps` means modifying the same struct that holds weight pointers

---

## 2. Proposed Design: Layered Tensor Architecture

### The layers

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Provenance (optional)                          │
│   - Training logs, dataset info, loss curves            │
│   - Checksums, device placement history                 │
│   - Fine-tuning lineage (base model → adapter → merge)  │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Extension (optional, pluggable)                │
│   - LoRA adapters, QLoRA scales                         │
│   - Custom position encodings                           │
│   - Attention bias matrices                             │
│   - Per-layer config overrides                          │
├─────────────────────────────────────────────────────────┤
│ Layer 2: System (required for inference)                │
│   - Normalization stats (rms_norm weights/biases)       │
│   - RoPE parameters (base, scaling type/factor)         │
│   - Quantization scales and zero points                 │
│   - Attention patterns (causal vs bidirectional)        │
├─────────────────────────────────────────────────────────┤
│ Layer 1: Structure (required for inference)             │
│   - Per-tensor shape, dtype, element count              │
│   - Architecture config (num_layers, heads, dims)       │
│   - Tensor naming conventions and layout                │
├─────────────────────────────────────────────────────────┤
│ Layer 0: Core KV (required for inference)               │
│   - Weight matrices: embeddings, projections, FFN       │
│   - Tall and skinny. Just the numbers.                  │
│   - Self-describing shape via Layer 1                   │
└─────────────────────────────────────────────────────────┘
```

### The principle: tall and skinny

Layer 0 is **always** the same shape regardless of what's in Layers 1-4. A reader that only wants the KV data sees:

```rust
struct CoreTensor {
    name: String,        // "blk.0.attn_q.weight"
    data: &[u8],         // raw bytes, dtype from Layer 1
    shape: Vec<usize>,   // from Layer 1
}
```

That's it. No config, no normalization stats, no tokenizer. The tensor is self-describing enough to be placed in memory and used by a GEMM kernel. Everything else is metadata that **describes how to use** the tensor, not what the tensor **is**.

### The reader abstraction

Different consumers see different slices:

```rust
// Minimal inference engine: Layer 0 + 1
trait TensorReader {
    fn core_tensor(&self, name: &str) -> Option<&CoreTensor>;
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>>;
    fn tensor_dtype(&self, name: &str) -> Option<Dtype>;
}

// Full inference: Layer 0 + 1 + 2
trait InferenceReader: TensorReader {
    fn norm_stats(&self, layer: usize) -> Option<NormStats>;
    fn rope_config(&self) -> RopeConfig;
    fn quant_params(&self, name: &str) -> Option<QuantParams>;
}

// Training/fine-tuning: all layers
trait TrainingReader: InferenceReader {
    fn extensions(&self) -> &[ExtensionFactor];
    fn provenance(&self) -> &Provenance;
}
```

Each trait is a **view** over the same underlying storage. Adding Layer 3 (LoRA adapters) doesn't change what `TensorReader` or `InferenceReader` sees — they still get tall/skinny KV. The training reader gets the extra view.

### Connection to "skipping layers"

The user's intuition: "like trying to arbitrarily skip layers, but not if you have the tensors."

This is exactly right. In a traditional transformer, you can't skip layer 5 without breaking the residual stream — each layer's output feeds the next. But at the **weight storage** level, you *can* skip loading certain weight groups:

- Load only embeddings + output head → single-token classification
- Load layers 0-15 of a 32-layer model → truncated inference (quality loss, but functional)
- Load all layers except FFN → attention-only analysis
- Load base weights + LoRA adapter → fine-tuned behavior without merging

The tensors themselves don't care about the computational graph. They're just tall/skinny matrices. The **graph** is in the metadata layer (Layer 1: which tensor feeds which operation). This separation means you can have the full tensor set and choose your computation path at runtime.

### Connection to "closer to trainable"

Current safetensors files from HuggingFace already have this structure implicitly:
- Weight tensors (Layer 0)
- Shape/dtype in the header (Layer 1)
- Config.json alongside (Layer 2)
- Training metadata in model card (Layer 4)

PESTI's job is to make this explicit in the **runtime representation**, not just the file format. When PESTI loads a safetensors file, it should preserve the layering instead of flattening everything into `LlamaModel`. This means:

1. A fine-tuning pipeline can read PESTI's weight store directly
2. Unsloth (or any trainer) can produce weights that PESTI loads without conversion
3. The "trainable" representation and the "inference" representation are the same data, different views

---

## 3. Data Structures

### Core types

```rust
/// Layer 0: A single weight tensor. Tall and skinny.
#[derive(Debug)]
pub struct CoreTensor {
    pub name: String,
    pub data: Arc<[u8]>,       // raw bytes, zero-copy from mmap
    pub shape: Vec<usize>,     // from Layer 1
    pub dtype: Dtype,          // from Layer 1
}

/// Layer 1: Tensor structure metadata.
#[derive(Debug, Clone)]
pub struct TensorStructure {
    pub tensors: HashMap<String, TensorInfo>,
    pub arch: ModelArch,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
    pub intermediate_dim: usize,
    pub max_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub element_count: usize,
    pub byte_offset: u64,      // offset in the safetensors file
    pub size_bytes: u64,
}

/// Layer 2: System factors required for correct inference.
#[derive(Debug, Clone)]
pub struct SystemFactors {
    pub rope: RopeConfig,
    pub norm_eps: f32,
    pub attention_pattern: AttentionPattern,
    pub quantization: Option<QuantScheme>,
    pub per_layer: Vec<LayerSystemFactors>,
}

#[derive(Debug, Clone)]
pub struct LayerSystemFactors {
    pub rms_norm_weight: Option<Arc<[f32]>>,  // if stored separately from core
    pub attention_bias: Option<Arc<[f32]>>,   // optional bias term
}

/// Layer 3: Pluggable extension factors.
#[derive(Debug)]
pub enum ExtensionFactor {
    Lora(LoraAdapter),
    CustomPositionEncoding(Arc<dyn PositionEncoding>),
    AttentionBias(Arc<dyn AttentionBias>),
    // ... extensible
}

#[derive(Debug)]
pub struct LoraAdapter {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,  // tensor names this adapter applies to
    pub lora_a: HashMap<String, CoreTensor>,
    pub lora_b: HashMap<String, CoreTensor>,
}

/// Layer 4: Provenance and lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub base_model: String,
    pub repo_id: Option<String>,
    pub commit_hash: Option<String>,
    pub fine_tuning_lineage: Vec<LineageEntry>,
    pub checksums: HashMap<String, String>,
    pub loaded_at: DateTime<Utc>,
}
```

### The unified store

```rust
/// The PESTI weight store. All layers, one interface.
pub struct WeightStore {
    /// Layer 0: Core tensors (mmap'd from safetensors file)
    core: Arc<SafetensorsFile>,
    
    /// Layer 1: Structure metadata
    structure: TensorStructure,
    
    /// Layer 2: System factors
    system: SystemFactors,
    
    /// Layer 3: Extensions (loaded on demand)
    extensions: Vec<ExtensionFactor>,
    
    /// Layer 4: Provenance
    provenance: Option<Provenance>,
}

impl WeightStore {
    /// Minimal load: Layer 0 + 1. Enough for raw tensor access.
    pub fn load_minimal(path: &Path) -> Result<Self>;
    
    /// Full inference load: Layer 0 + 1 + 2.
    pub fn load_inference(path: &Path, config_path: Option<&Path>) -> Result<Self>;
    
    /// Training load: all layers.
    pub fn load_training(path: &Path, extensions_dir: Option<&Path>) -> Result<Self>;
    
    /// Add an extension factor at runtime (e.g., load a LoRA adapter).
    pub fn add_extension(&mut self, ext: ExtensionFactor);
}

// Implement the reader traits
impl TensorReader for WeightStore { ... }
impl InferenceReader for WeightStore { ... }
impl TrainingReader for WeightStore { ... }
```

### Migration from LlamaModel

`LlamaModel` becomes a **view** over `WeightStore`, not the storage itself:

```rust
pub struct LlamaModel {
    /// The underlying weight store (all layers)
    store: Arc<WeightStore>,
    
    /// Pre-computed kernel handles (GPU-specific, not part of the store)
    dispatch: Option<DispatchContext>,
    
    /// Runtime state (KV caches, sequence position)
    kv_caches: Vec<(Kvcache, Kvcache)>,
    seq_len: usize,
}

impl LlamaModel {
    /// Build a model from a weight store. The store outlives the model.
    pub fn from_store(store: Arc<WeightStore>) -> Result<Self>;
    
    /// Swap in a different set of extensions (e.g., change LoRA adapter)
    /// without reloading weights.
    pub fn set_extensions(&mut self, extensions: Vec<ExtensionFactor>);
}
```

This means:
- Multiple `LlamaModel` instances can share the same `WeightStore` (same weights, different KV caches for parallel inference)
- Swapping LoRA adapters is cheap (change Layer 3, don't reload Layer 0)
- The weight store is immutable and shareable; the model is mutable runtime state

---

## 4. Safetensors as Trainable-Adjacent Format

### Why safetensors fits

Safetensors already has the right properties:
1. **Header with tensor metadata** (shape, dtype, offset) → Layer 1 for free
2. **Flat binary layout** → zero-copy mmap for Layer 0
3. **No execution on load** → safe to load untrusted weights
4. **Standard format** → HuggingFace ecosystem compatibility

What's missing is the explicit layering in how PESTI **consumes** it. Currently:

```
safetensors file → parse header → HashMap<String, Vec<u8>> → LlamaModel (monolith)
```

Proposed:

```
safetensors file → mmap (Layer 0)
                 → parse header → TensorStructure (Layer 1)
config.json      → SystemFactors (Layer 2)
extensions/      → ExtensionFactor[] (Layer 3, on demand)
model card/metadata → Provenance (Layer 4, optional)
```

### What "trainable-adjacent" means concretely

A training framework (Unsloth, Axolotl, etc.) produces:
- `model.safetensors` — weight tensors
- `config.json` — architecture config
- `adapter_config.json` + `adapter_model.safetensors` — LoRA adapters (if applicable)

PESTI should be able to load **exactly these files** without conversion, preserving the layering:
- `model.safetensors` → Layer 0 + 1
- `config.json` → Layer 2
- `adapter_model.safetensors` → Layer 3 (Lora extension)

And the reverse: PESTI's fine-tuning output (if we ever build that) should produce files in this exact format, consumable by any HuggingFace-compatible trainer.

### The "direction" the user sees

> "if you have the model weights in safe tensors closer to the thing, like closer to something trainable, then like I don't know. I can see the direction."

The direction is: **PESTI becomes a first-class participant in the HuggingFace weight ecosystem, not just a consumer of GGUF files.** The tensor layer separation is what makes this possible — it's the bridge between "inference engine that loads weights" and "weight substrate that training and inference both speak."

---

## 5. Implementation Phases

### Phase 1: CoreTensor + TensorStructure (Layer 0 + 1) ✅ COMPLETE
**Output**: `WeightStore::load_minimal()` works; existing code paths unchanged  
**Scope**:
- Define `CoreTensor`, `TensorStructure`, `TensorInfo` types ✅
- Implement `SafetensorsFile` mmap wrapper with zero-copy tensor access ✅
- Implement `TensorReader` trait ✅
- Add tests: load a real safetensors file, verify shapes/dtypes match header ✅ (8 unit tests)
- **No behavior change** — existing `LlamaModel::load_safetensors()` continues to work ✅

**Implementation notes**:
- Module: `crates/pesti-runner/src/weight_store.rs`
- Added dependency: `memmap2.workspace = true` in pesti-runner Cargo.toml
- `CoreTensor.data` is `Arc<[u8]>` — zero-copy slice into mmap'd file (shared via Arc across tensor views)
- `SafetensorsFile` uses `mmap2::MmapOptions::new().map(&file)` with `MAP_PRIVATE|MAP_NORESERVE` semantics
- Dtype enum includes F32, F16, BF16, I64, I32, I16, I8, U8, BOOL (matches safetensors spec)
- Tests cover: CoreTensor f32 view, wrong-dtype rejection, TensorStructure lookup/prefix, WeightStore minimal load, tensor data extraction, zero-copy verification, missing tensor handling, multi-tensor file parsing

### Phase 2: SystemFactors (Layer 2)
**Output**: `WeightStore::load_inference()` works; config extraction centralized  
**Scope**:
- Define `SystemFactors`, `RopeConfig`, `AttentionPattern` types
- Parse config.json (or extract from safetensors metadata HashMap) into `SystemFactors`
- Implement `InferenceReader` trait
- Migrate `LlamaConfig::from_gguf_header()` logic to work from `TensorStructure` + `SystemFactors`
- Tests: verify rope/norm/attention params match current behavior

### Phase 3: WeightStore as LlamaModel backend
**Output**: `LlamaModel::from_store()` works; shared weights across models  
**Scope**:
- Refactor `LlamaModel` to hold `Arc<WeightStore>` instead of raw weight data
- Keep existing public API (forward, generate, etc.) unchanged
- Implement weight sharing: two `LlamaModel` instances from same store
- Tests: parallel inference with shared weights produces identical results

### Phase 4: Extension factors (Layer 3)
**Output**: LoRA adapter loading and hot-swap  
**Scope**:
- Define `ExtensionFactor` enum, `LoraAdapter` struct
- Implement LoRA weight application in attention/FFN forward pass
- `WeightStore::add_extension()` + `LlamaModel::set_extensions()`
- Tests: load base model + LoRA adapter, verify output differs from base; hot-swap between adapters

### Phase 5: Provenance (Layer 4)
**Output**: Full lineage tracking  
**Scope**:
- Define `Provenance` struct
- Parse model card / metadata for lineage info
- Integration with `pesti-safetensors` SQLite schema (already has checksums, loaded_at)
- Tests: verify provenance round-trips through store

---

## 6. Testing Strategy

### Unit tests (per phase)
- **Phase 1**: Tensor shape/dtype correctness against known safetensors files; zero-copy verification (mmap address matches tensor data pointer)
- **Phase 2**: Config extraction parity — `SystemFactors` from config.json matches current `LlamaConfig::from_gguf_header()` output for same model
- **Phase 3**: Shared-weight inference — two models from same store produce byte-identical logits for same input
- **Phase 4**: LoRA correctness — base + adapter output matches reference implementation (Unsloth or PEFT) within tolerance
- **Phase 5**: Provenance integrity — checksums match, lineage chain is valid

### Integration tests
- Load a real model (Qwen2.5-0.5B or similar small model) through each phase's loading path
- Verify generated text is identical to current monolithic loading
- Measure load time: minimal vs full (expect minimal to be significantly faster)

### Performance benchmarks
- **Load time**: `load_minimal()` vs `load_inference()` vs current monolithic load
- **Memory**: mmap'd tensors should not double memory usage vs current approach
- **LoRA swap time**: hot-swap adapter without reloading base weights (target: <10ms)

---

## 7. Open Questions

1. **GGUF compatibility**: This spec is safetensors-first. Should GGUF loading also go through `WeightStore`, or stay as a separate path? (Recommendation: same `WeightStore` interface, different backend loader.)

2. **Quantization representation**: Currently quantized tensors are stored as raw bytes in Layer 0 with dtype indicating the scheme. Should quantization scales/zero-points be in Layer 0 (co-located) or Layer 2 (system factors)? (Leaning Layer 2 — they're "how to interpret" not "what is".)

3. **Multi-file models**: Large models are split across multiple safetensors files (`model-00001-of-00010.safetensors`). Should `WeightStore` handle sharding transparently? (Yes — Layer 0 becomes a list of mmap'd files, tensor lookup resolves which file contains it.)

4. **GPU weight placement**: Currently weights are copied to GPU during model init. With `WeightStore`, should GPU placement be a Layer 3 extension ("device placement factor") or orthogonal? (Leaning orthogonal — device placement is a runtime concern, not a weight property.)

5. **Interaction with pesti-safetensors crate**: The existing SQLite schema already tracks model_weights and tensor_metadata. Should `WeightStore` integrate with it (query DB for available models) or stay file-based? (Recommendation: integrate — DB is the catalog, WeightStore is the runtime view.)

---

## Appendix A: The Clock Analogy Applied

From the branched chats: clocks on the same slope reveal patterns; ±I duality gives pivot points.

Applied here:
- **Layer 0 (KV)** is the clock face — the raw positions, uninterpreted
- **Layer 1 (Structure)** is the clock mechanism — how the hands (tensors) relate to each other
- **Layer 2 (System)** is the timezone — how to interpret the positions correctly
- **Layer 3 (Extension)** is the alarm — optional behavior layered on top
- **Layer 4 (Provenance)** is the serial number — where this clock came from

A watchmaker (training framework) needs all layers. A person telling time (inference engine) needs Layers 0-2. A thief stealing just the hands (raw tensor extraction) gets Layer 0 — tall and skinny, useless without the mechanism, but *there* if you need to build a different clock.

The "tall and skinny" insight: the hands don't care what timezone they're in or what alarm is set. They're always the same length, pointing at the same angles. The interpretation is separate from the geometry. That's the separation this spec codifies.
