# Weight-Space Programming for PESTI

## §0 Goal + Scope

**Goal**: Add runtime weight-space programming capabilities to PESTI — the ability to read, analyze, and surgically edit model weights at inference time based on behavioral specifications, without requiring a training loop.

**NOT in scope (yet)**:
- ❌ Full LLM-based weight reading via prompt embedding (paper's approach doesn't scale beyond ~5K params)
- ❌ Automatic flaw detection from raw weights alone
- ❌ Certification that a network is correct (paper showed this fails)
- ❌ Replacing PEFT/LoRA training — complementary, not replacement

**Scope**: One small reversible session. Build the API surface and integration points for weight-space operations; defer LLM-driven diagnosis to later phases.

## §1 Context & Why Now

**Principle**: Direct weight manipulation enables surgical model adaptation faster than training loops, with precise parameter-level control.

**Why now**:
- Paper "Weight-space programming for auditing clinical decision networks" (arXiv:2611.05341) demonstrates LLMs can read raw weights and make targeted repairs (median 3 parameter changes vs ~430 for fine-tuning)
- PESTI already has PEFT/LoRA infrastructure with `merge_into`/`unmerge_from` capabilities (`pesti-runner/src/peft/lora.rs`)
- GGUF weight loading/saving is established (`gguf_weight_loader.rs`, `safetensors_weight_loader.rs`)
- Runtime model state management exists via `runtime.rs`

**Measured evidence from paper**:
- Detected 16/18 flawed networks (95% CI 67-97%)
- Repaired every detected flaw with median 3 parameter changes
- Fine-tuning changed nearly all parameters for same result
- Failed at reliably clearing healthy networks or finding learned triggers

**Tooling to reuse**:
- `pesti-runner/src/peft/lora.rs` — LoRAAdapter with merge/unmerge
- `pesti-runner/src/gguf_weight_loader.rs` — GGUF weight I/O
- `pesti-runner/src/runtime.rs` — Runtime model state management
- `pesti-safetensors/src/writer.rs` — Weight serialization

## §2 Architecture

### New Crate: `pesti-weightspace`

```toml
[package]
name = "pesti-weightspace"
version.workspace = true
edition.workspace = true
description = "Weight-space programming API for PESTI"

[dependencies]
pesti-runner = { path = "../pesti-runner" }
pesti-safetensors = { path = "../pesti-safetensors" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### Core API Surface

```rust
// pesti-weightspace/src/lib.rs

/// A weight-space operation: read, write, or compute on model weights
pub enum WeightOp {
    /// Read a specific weight value by layer and index
    Read { layer: String, index: Vec<usize> },
    /// Write a new weight value
    Write { layer: String, index: Vec<usize>, value: f32 },
    /// Apply a batch of weight edits (atomic)
    Patch { edits: Vec<WeightEdit> },
    /// Compute a statistic over weights (for diagnosis)
    Stat { layer: String, op: WeightStatOp },
}

/// A single weight edit for atomic patching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEdit {
    pub param: String,      // "W1", "b1", "W2", etc.
    pub index: Vec<usize>,  // Multi-dimensional index
    pub value: f32,         // New weight value
}

/// Statistic operations for weight analysis
pub enum WeightStatOp {
    /// Column norm (for detecting silenced rules)
    ColumnNorm { column_indices: Vec<usize> },
    /// Switch-on point analysis (for threshold shifts)
    SwitchOnPoint { input_indices: Vec<usize> },
}

/// Weight-space session manager
pub struct WeightSpaceSession {
    model_path: PathBuf,
    loaded_weights: HashMap<String, Tensor>,
    original_snapshot: Option<HashMap<String, Tensor>>,
    pending_edits: Vec<WeightEdit>,
}

impl WeightSpaceSession {
    /// Open a weight-space session on a model file
    pub fn open(model_path: &Path) -> Result<Self>;
    
    /// Read current weights for a layer
    pub fn read_layer(&self, layer: &str) -> Result<&Tensor>;
    
    /// Apply weight edits atomically (all or nothing)
    pub fn apply_edits(&mut self, edits: Vec<WeightEdit>) -> Result<()>;
    
    /// Snapshot current weights for rollback
    pub fn snapshot(&mut self) -> Result<()>;
    
    /// Rollback to last snapshot
    pub fn rollback(&mut self) -> Result<()>;
    
    /// Save edited weights back to GGUF file
    pub fn save_edited_model(&self, output_path: &Path) -> Result<()>;
    
    /// Compute weight statistics for diagnosis
    pub fn compute_stat(&self, op: WeightStatOp) -> Result<f32>;
}
```

### Integration with PEFT Infrastructure

The weight-space API complements existing LoRA/PEFT by providing direct parameter access:

```rust
// Example workflow: targeted weight edit vs LoRA training

// Traditional LoRA approach (training loop required)
let adapter = create_lora_adapter(4096, 4096, &config)?;
train(adapter, dataset);
let merged = adapter.merge_into(&base_weights, None)?;

// Weight-space approach (direct surgical edit)
let session = WeightSpaceSession::open("model.gguf")?;
session.apply_edits(vec![
    WeightEdit { param: "W1".to_string(), index: vec![3, 7], value: 0.5 },
    WeightEdit { param: "b2".to_string(), index: vec![12], value: -0.3 },
])?;
session.save_edited_model("model_patched.gguf")?;
```

### Module Registration

Add to `pesti-runner/src/lib.rs`:
```rust
#[cfg(feature = "weightspace")]
pub mod weightspace;
```

Add to root `Cargo.toml` workspace members:
```toml
members = [
    # ... existing ...
    "pesti-weightspace",
]
```

## §3 Acceptance Criteria

### Build Gates
- [ ] `cargo build -p pesti-weightspace --no-default-features` (CPU-only)
- [ ] `cargo build -p pesti-weightspace` (with CUDA features)
- [ ] No new clippy warnings in weightspace crate

### Unit Tests (per file)
- [ ] `tests/session.rs` — WeightSpaceSession open/read/snapshot/rollback
- [ ] `tests/edits.rs` — WeightEdit application and atomicity
- [ ] `tests/stats.rs` — ColumnNorm and SwitchOnPoint computations
- [ ] `tests/integration.rs` — Full workflow: load, edit, save, verify

### Demo Example
- [ ] `examples/targeted_patch.rs` — Demonstrate surgical weight edit
  - Load small model
  - Apply targeted edit (e.g., change threshold)
  - Verify behavior changed as expected
  - Rollback and verify original behavior restored

## §4 Done-When / Handoff

When complete:
1. Update `ROADMAP.md` — add Weight-Space Programming section with measured results
2. Add EDR to `CHANGELOG.md` documenting the design decision
3. Update `docs/PROJECT_STATUS.md` with new crate info
4. Update `docs/README.md` index with weightspace docs entry

Stop at next gate: LLM-driven weight diagnosis (Phase 2).

## §5 Constraints & Cautions

- **16-bit overflow risk**: Paper showed large weight values (>10^5) fail in fp16 arithmetic. Add validation that edited weights stay within safe range for target precision.
- **Atomicity is critical**: All edits in a patch must succeed or all fail — no partial application.
- **Snapshot before edit**: Always create snapshot before applying edits to enable rollback.
- **Index bounds checking**: Multi-dimensional index access must validate against tensor shape.
- **GGUF format awareness**: Different GGUF versions may have different weight layouts; handle version negotiation.

## §6 Suggested TDD Order

1. Define `WeightEdit` and `WeightOp` types with serialization tests
2. Implement `WeightSpaceSession::open()` with GGUF loading test
3. Implement `read_layer()` with shape validation tests
4. Implement `apply_edits()` with atomicity and rollback tests
5. Implement weight statistics (ColumnNorm, SwitchOnPoint)
6. Implement `save_edited_model()` with round-trip verification
7. Write integration demo showing full workflow

## §7 Open Questions

1. **Should edits be layer-aware or raw tensor?** 
   - Default: Raw tensor access with shape validation; add layer abstraction later if needed.

2. **How to handle quantized weights?**
   - Default: Dequantize → edit → requantize on save. Document precision loss.

3. **LLM integration point?**
   - Default: Provide JSON I/O for weight edits so external LLM can propose patches; PESTI executes them. Don't embed LLM in crate.

4. **Versioning of edited models?**
   - Default: Simple increment (model.gguf → model_v1.gguf); leave sophisticated versioning to user.
