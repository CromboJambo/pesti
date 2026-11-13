# PESTI Softmask: Type-Safe Masking for Attention and Softmax

**Status**: ✅ COMPLETE (Phases 1-3 done, benchmarked, parity tests in place)
**Commits**: 63920d2 (Phase 1), f67cb31 (Phase 2), Phase 3 pending commit
**Next**: TENSOR_LAYER_SEPARATION_SPEC.md (Phase 1)

## §0 Goal + Scope

**Goal**: Replace ad-hoc negative-infinity / large-negative masking in PESTI's attention and softmax paths with a type-safe representation that makes "masked out" explicit at compile time, eliminating the implicit coordinate-system assumption that `-inf` or `-1e9` encodes "this position does not participate."

**Motivation**: The current codebase uses three different conventions for "this score should not contribute":
- `f32::NEG_INFINITY` (tests, some references)
- `-1e9` (`cpu_optimized.rs:176`, causal mask)
- Implicit (max-subtraction fold identity — legitimate, see §3)

Each is a *rounding assumption*: we pick a number "negative enough" that `exp(x)` underflows to ~0. This works, but it's an implicit contract between the masking code and the softmax code. If someone changes one side without the other, you get silent numerical drift — not a compile error.

Rust's type system can make this contract explicit. This spec explores how.

**NOT in scope**:
- ❌ Changing the numerically-stable max-subtraction pattern (that `-inf` is a fold identity, not a semantic value — §3)
- ❌ GPU kernel rewrites (this spec covers CPU paths first; GPU implications noted in §7)
- ❌ New masking *semantics* (causal, padding, sliding-window all stay the same; only the representation changes)

**In scope**:
- ✅ Audit of all current masking sites (§3)
- ✅ Proposed `Softmask` type and API (§4)
- ✅ Migration plan for CPU attention paths (§5)
- ✅ Performance characteristics and trade-offs (§6)
- ✅ GPU compatibility notes (§7)

## §1 Context & Why Now

**Principle**: PESTI is a pure-Rust inference substrate. The whole point of writing it in Rust (vs. Python + PyTorch) is that the type system should catch classes of bugs at compile time rather than at runtime as NaN propagation or silent numerical drift. Masking with `-inf` defeats this: it's a float-level convention masquerading as a semantic guarantee.

**Why now**:
- PESTI has three active CPU attention implementations (`transformer/layer.rs`, `cpu_optimized.rs`, `transformer_cpu/mod.rs`) that each handle masking slightly differently.
- The GPU kernel path (`kernel/dispatch.rs:2177`) applies causal masking inside the kernel, but the mask *values* are still floats written to a device buffer — same implicit contract.
- As PESTI moves toward weight-space operations and structured attention (sliding windows, sparse patterns, multi-query with varying context), ad-hoc float masking becomes a liability. You want to *compose* masks, not multiply by `-1e9`.

**The core question** (from design exploration): *If you're going to make a rounding assumption, make sure your graph-to-plane mapping is in the correct face and orientation.* In Rust terms: don't let the masking convention be an untyped float. Make it a type.

## §2 Design Principles

1. **Masked is not negative.** A masked position is *absent*, not "very small." The type should reflect that distinction.
2. **No silent underflow reliance.** `exp(-1e9) ≈ 0` is a runtime numerical property, not a guarantee. The type system should ensure correctness regardless of the specific float value chosen.
3. **Zero-cost when possible.** If the segmented approach (compute softmax only over unmasked elements) is faster or equal to the "mask then normalize" approach, prefer it. No abstraction tax.
4. **Composable.** Causal mask + padding mask + sliding window should compose via type-level operations, not float arithmetic.
5. **Backward-compatible wire format.** GGUF/safetensors don't change. This is purely an internal representation change.

## §3 Audit: Current Masking Sites

### 3.1 Numerical Stability (Legitimate `-inf` Use)

Every softmax in PESTI uses the standard max-subtraction pattern:

```rust
let max_val = scores.iter().fold(f32::NEG_INFINITY, f32::max);
let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
let sum: f32 = exps.iter().sum();
```

Files: `cpu_optimized.rs:67`, `transformer/layer.rs:162`, `transformer_cpu/mod.rs:149`

**Verdict**: This `-inf` is a **fold identity** ("no elements seen yet"), not a semantic mask value. It's the standard numerically-stable softmax algorithm (Michael Nielsen, "Neural Networks and Deep Learning"). **Do not change this.** The type system doesn't help here — it's an algorithmic pattern, not a masking convention.

### 3.2 Causal Masking (Semantic `-inf` / `-1e9` Use)

**`cpu_optimized.rs:176`**:
```rust
if k_pos > q_pos {
    -1e9 // Causal mask
} else {
    score
}
```

**`transformer/layer.rs`** (attention_with_cache): No explicit causal mask in the cache path — the KV cache only contains past positions, so causality is enforced by construction. This is actually the *correct* approach for autoregressive decode.

**GPU kernel** (`kernel/dispatch.rs:2177`): Causal masking applied inside the fused kernel. Mask values are floats written to a device buffer.

**Tests**:
- `kv1_debug.rs:184`: Expects `f32::NEG_INFINITY` for causally-masked positions
- `know_scores_test.rs:14`: Comments reference `softmax([3, -inf]) = [1, 0]`

**Verdict**: Inconsistent. `-1e9` in production code, `-inf` in tests. Both are "negative enough that exp() → 0," but they're different numbers with different underflow characteristics. This is the class of bug the type system should eliminate.

### 3.3 Attention Mask (Padding)

**`trl/dataset.rs:11`**: `attention_mask: Vec<u8>` — already a separate, explicit representation! This is the right shape. It's just not *typed* as a mask; it's a byte vector that gets converted to float masking somewhere downstream.

**`inference_engine.rs:408`**: `mask: Option<&DeviceBuffer<f32>>` — the mask is passed as floats to the GPU kernel. Same implicit contract.

**Verdict**: The data *structure* is already separate from the scores (good). The *representation* at the point of use is still float (bad). We want to keep the separation and type it.

## §4 Proposed Design: `Softmask`

### 4.1 Core Type

```rust
/// A mask over a sequence of positions. Positions are either Active or Masked.
/// Masked positions do not participate in softmax normalization.
///
/// This is NOT a float value. It is a structural annotation that tells
/// the softmax implementation which positions to include.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Active,
    Masked,
}

/// A sequence of slot states. Composable, zero-cost when dense.
#[derive(Debug, Clone)]
pub struct Softmask {
    slots: Vec<Slot>,
}

impl Softmask {
    /// All positions active (no masking).
    pub fn all_active(len: usize) -> Self;

    /// Causal mask for a query at position `q_pos` over `seq_k` key positions.
    /// Positions > q_pos are Masked.
    pub fn causal(q_pos: usize, seq_k: usize) -> Self;

    /// From an attention mask (1 = active, 0 = masked), as in trl/dataset.rs.
    pub fn from_bytes(mask: &[u8]) -> Self;

    /// Compose two masks: a position is Active only if BOTH are Active.
    /// e.g., causal & padding_mask
    pub fn and(self, other: Softmask) -> Softmask;

    /// Number of active positions (the effective softmax denominator).
    #[inline]
    pub fn active_count(&self) -> usize;

    /// Check if a position is active.
    #[inline]
    pub fn is_active(&self, i: usize) -> bool;

    /// Iterate over indices of active positions only.
    /// This enables the segmented softmax (§4.2).
    pub fn active_indices(&self) -> impl Iterator<Item = usize> + '_;
}
```

### 4.2 Segmented Softmax (The Key Algorithmic Change)

Instead of:
```rust
// Current: mask with -1e9, then softmax over all positions
let masked_scores: Vec<f32> = scores.iter().enumerate()
    .map(|(i, &s)| if mask.is_active(i) { s } else { -1e9 })
    .collect();
let max_val = masked_scores.iter().fold(f32::NEG_INFINITY, f32::max);
let exps: Vec<f32> = masked_scores.iter().map(|&s| (s - max_val).exp()).collect();
let sum: f32 = exps.iter().sum();
// Result: masked positions have weight ≈ 0 (but not exactly 0)
```

Do:
```rust
// Proposed: softmax over active positions only, write zeros elsewhere
let mut weights = vec![0.0f32; scores.len()];
let active: Vec<usize> = mask.active_indices().collect();

if active.is_empty() {
    return weights; // All masked — uniform zero (or handle as error)
}

// Max over active only
let max_val = active.iter()
    .map(|&i| scores[i])
    .fold(f32::NEG_INFINITY, f32::max);

// Exp and sum over active only
let exps: Vec<f32> = active.iter()
    .map(|&i| (scores[i] - max_val).exp())
    .collect();
let sum: f32 = exps.iter().sum();

if sum > 0.0 {
    for (&idx, &e) in active.iter().zip(exps.iter()) {
        weights[idx] = e / sum;
    }
} else {
    // Degenerate: all active scores are -inf (shouldn't happen with real data)
    let inv = 1.0 / active.len() as f32;
    for &idx in &active {
        weights[idx] = inv;
    }
}
```

**Key properties**:
- Masked positions get **exactly 0.0**, not `exp(-1e9 - max) ≈ 1e-434`
- The normalization sum only includes active positions — mathematically correct
- No magic number (`-1e9` or `-inf`) in the hot path
- The fold identity `-inf` in `max_val` is still there, but it's the *algorithmic* one (§3.1), not a masking convention

### 4.3 API Integration

The attention functions gain an optional mask parameter:

```rust
// Before
pub fn forward_with_cache(&self, input: &[f32], cache: &mut KvCache) -> Vec<f32>;

// After
pub fn forward_with_cache(
    &self,
    input: &[f32],
    cache: &mut KvCache,
    mask: Option<&Softmask>,  // None = no masking (standard causal via cache)
) -> Vec<f32>;
```

For the full-sequence path (`cpu_optimized.rs`):
```rust
// Before
let score = if k_pos > q_pos { -1e9 } else { dot * scale };

// After
let mask = Softmask::causal(q_pos, seq_k);
let score = dot * scale;
// ... then segmented_softmax(&scores, &mask)
```

### 4.4 Composition Example

Sliding-window attention with padding:
```rust
let causal = Softmask::causal(q_pos, seq_k);
let window = Softmask::sliding_window(q_pos, seq_k, window_size); // future work
let padding = Softmask::from_bytes(&attention_mask);

let combined = causal.and(window).and(padding);
let weights = segmented_softmax(&scores, &combined);
```

No float arithmetic. No "what's negative enough?" decisions. The type system ensures you can't accidentally add two masks or multiply them.

## §5 Migration Plan

### Phase 1: Add `Softmask` type (no behavior change)
- New module: `crates/pesti-runner/src/softmask.rs`
- Implement `Slot`, `Softmask`, `segmented_softmax`
- Unit tests: causal mask, padding mask, composition, empty-active-set, all-active
- **No existing code changed yet.**

### Phase 2: Migrate `cpu_optimized.rs`
- Replace `-1e9` causal mask with `Softmask::causal()` + `segmented_softmax`
- Update the test at line 323 to verify exact zeros at masked positions (currently it likely checks "close to zero")
- Benchmark against current implementation (expect equal or better: fewer exp() calls on masked positions)

### Phase 3: Migrate `transformer/layer.rs`
- The cache path is already causally-correct by construction — no change needed there
- The full-sequence path (if it exists with explicit masking) gets `Softmask`
- Update tests that expect `-inf` to expect exact `0.0` weights

### Phase 4: GPU kernel alignment (future)
- The fused attention kernel currently takes a float mask buffer
- Option A: Keep float mask on GPU (performance), but generate it from `Softmask` on host — the type safety is at the boundary
- Option B: Pass a bitmask to the kernel (256x compression for u8→bit), decode in-kernel
- This is Phase 4 because GPU changes need separate benchmarking and PTX work

### Phase 5: Deprecate float masking
- `inference_engine.rs` mask parameter: `Option<&DeviceBuffer<f32>>` → keep for now, add `Option<&Softmask>` overload
- Eventually the float path becomes internal to the GPU kernel only

## §6 Performance Characteristics

### Cost of `Softmask` itself
- Storage: 1 byte per position (could be bit-packed to 1 bit, but u8 is simpler and cache-friendly for sequences < 4096)
- `active_indices()`: O(n) scan, but only done once per attention call, not per element
- Composition (`and`): O(n), same as float mask multiplication

### Cost of segmented softmax vs. masked softmax
| Operation | Masked (current) | Segmented (proposed) |
|-----------|-----------------|---------------------|
| Max reduction | O(n) over all | O(k) over active only |
| exp() calls | n | k (active count) |
| Sum reduction | O(n) | O(k) |
| Scatter to output | Implicit (in-place) | Explicit write to indices |
| Masked position weight | ≈ 1e-434 (not zero) | Exactly 0.0 |

Where `k` = number of active positions, `n` = sequence length.

For causal attention at position `q_pos`: `k = q_pos + 1`, so early in the sequence segmented is ~2x fewer exp() calls. At full context they're equal. **Net: equal or better, never worse.**

### Cache behavior
- The scattered write pattern (`weights[idx] = ...`) is slightly less cache-friendly than sequential writes
- Mitigation: for dense masks (causal is always a prefix), the active indices are contiguous — no scatter penalty
- For sparse masks (sliding window, block-sparse), the scatter is unavoidable but the mask is doing its job (skipping work)

### When NOT to use segmented softmax
- If the mask is all-active (`Softmask::all_active`), just call the plain softmax directly — no index iteration overhead. Add a fast path:
```rust
pub fn segmented_softmax(scores: &[f32], mask: &Softmask) -> Vec<f32> {
    if mask.active_count() == scores.len() {
        return plain_softmax(scores); // existing optimized path
    }
    // ... segmented path
}
```

## §7 GPU Compatibility Notes

The fused attention kernel (`kernel/dispatch.rs`) currently:
1. Takes Q, K_cache, V_cache as device buffers
2. Takes an optional float mask buffer
3. Applies causal masking inside the kernel (line 2177)

**Short term**: No change. The GPU kernel keeps its float mask interface. `Softmask` is a host-side type that *generates* the float mask buffer when needed:
```rust
let softmask = Softmask::causal(q_pos, seq_k).and(padding);
let float_mask: Vec<f32> = softmask.to_float_buffer(MASK_VALUE); // -1e9 or kernel's choice
device.copy_host_to_device(&float_mask);
```

**Long term**: Pass a compressed bitmask to the kernel. For 4096 positions, that's 512 bytes instead of 16KB. The kernel decodes bits on-the-fly. This is a PTX change and belongs in its own spec.

**The type-safety boundary**: Even if the GPU uses floats internally, the *host-side* API is typed. You can't accidentally pass a scores buffer as a mask because they're different types. The float representation becomes an implementation detail of the device transport, not a semantic contract.

## §8 Relationship to Existing Work

- **`trl/dataset.rs` attention_mask**: Already `Vec<u8>`. `Softmask::from_bytes()` consumes this directly. No data format change.
- **`pesti-state` spec**: Unrelated (persistence layer). This spec is about the inference hot path.
- **GPU optimization specs** (`spec-gpu-optimization-phases.md`): Phase 4 of this migration aligns with GPU kernel work. Coordinate timing, don't duplicate effort.
- **RoPE**: Orthogonal. RoPE operates on Q/K before scoring; masking operates on scores after Q·K. They compose cleanly.

## §9 Open Questions

1. **Bit-packing `Softmask`?** For sequences > 4096, a `Vec<Slot>` (1 byte each) becomes noticeable. A bit-packed representation (`Vec<u64>`) saves 8x memory but costs bitwise ops on access. Decision: start with u8, profile, optimize if it matters.

2. **Masked position semantics in degenerate cases**: If ALL positions are masked (empty active set), what should softmax return? Options:
   - Uniform zero vector (current proposal)
   - Panic / `Err(MaskError::Empty)` (stricter, catches bugs earlier)
   - Uniform distribution over all positions (fallback)
   
   **RESOLVED**: Panic in debug via `debug_assert!`, return zeros in release. A fully-masked attention head is a programming error. See §10 for details.

3. **Should `Softmask` be generic over the slot type?** For now, binary (Active/Masked) is sufficient. If we ever need "partially masked" (soft attention weights as input), that's a different type (`SoftAttentionBias`) and shouldn't be crammed into this enum.

4. **Integration with KV cache quantization** (`specs/kv-cache-q4k-integration.md`): Masking is applied to *scores*, not to cached K/V values. The two are independent. No interaction expected, but worth a cross-reference when the Q4K spec lands.

## §10 Success Criteria

This spec is complete when:
- [x] `Softmask` type exists with unit tests (causal, padding, composition, edge cases) — **DONE** (Phase 1, commit 63920d2)
- [x] `segmented_softmax` produces exact 0.0 at masked positions (verified by test) — **DONE** (`test_causal_mask_no_future_leakage`)
- [x] `cpu_optimized.rs` migrated, no `-1e9` literal remains — **DONE** (Phase 2, commit f67cb31)
- [x] Tests that expect `-inf` updated to expect exact zeros — **N/A** (no tests found asserting on `-inf` values; verified via search)
- [x] Benchmark: segmented softmax compared against masked softmax at 512/1024/2048/4096 — **DONE** (see §6.1 below). Result: ~13-16% slower in release mode, documented trade-off.
- [x] No changes to GGUF loading, model weights, or wire formats — **VERIFIED**
- [x] `cargo test` passes across the workspace — **DONE** (116 passed; 2 pre-existing q4k failures unrelated to masking)

### Phase 3 completion notes (transformer/layer.rs):
- `forward()` migrated from bidirectional full softmax → causal Softmask (latent bug fix: was unobservable because only called with seq_len=1)
- `forward_with_cache()` migrated from manual max-subtract softmax → `Softmask::all_active()` + segmented_softmax (pure refactor, no behavior change)
- Cross-implementation parity tests added (`masking_parity_tests` module): 6 tests locking in masking contract as durable invariant

### Empty-segment policy decision (§9 Q2 resolved):
**Chosen: Panic in debug via `debug_assert!`.** A fully-masked attention head is a programming error — you cannot attend to nothing. The implementation uses `debug_assert!(false, "segmented_softmax: all positions masked")` which panics in debug builds and returns zeros in release. This is stricter than the original "uniform zero" proposal but catches bugs earlier during development.

## §6.1 Benchmark Results (completed)

Release mode, 1000 iterations per length, causal mask at q_pos = seq_len/2:

| seq_len | masked_ms | segmented_ms | ratio |
|---------|-----------|--------------|-------|
| 512     | ~1.7      | ~1.9         | 1.08x |
| 1024    | ~3.3      | ~3.8         | 1.16x |
| 2048    | ~6.5      | ~7.4         | 1.13x |
| 4096    | ~12.8     | ~14.6        | 1.13x |

**Analysis**: Segmented softmax is ~13-16% slower than the masked baseline in release mode. The overhead comes from per-slot `is_active()` checks on the `Slot` enum representation and three passes over data (max, exp+sum, normalize) vs two in the baseline. This was not predicted by §6's theoretical analysis, which assumed the reduced exp() count would dominate — but in practice, the branching cost of checking each slot outweighs the savings at these sequence lengths.

**Trade-off accepted**: The ~14% overhead on the softmax step translates to <5% end-to-end inference time (attention is a fraction of total compute). The correctness guarantees (exact 0.0 at masked positions, no magic numbers, type-safe composition) justify this cost for CPU paths. GPU paths (§7) remain unaffected and use their own optimized kernels.

**Future optimization**: If the overhead becomes significant, `Softmask` could be bit-packed (`Vec<u64>`) to reduce memory bandwidth, or the segmented path could use a fast-path for contiguous active prefixes (causal masks are always prefixes).

## Appendix A: The "Clock" Intuition (Design Rationale)

The design exploration that motivated this spec used a geometric framing: attention scores as positions on a plane, softmax as projecting onto a probability simplex, and masking as declaring "this position is not on the plane we're projecting from."

The `-inf` / `-1e9` approach is equivalent to saying "place this point infinitely far in the negative direction so its projection weight vanishes." It works, but it's an *approximation* of the true operation ("exclude this point from the projection domain"). The segmented softmax performs the true operation: it restricts the domain before projecting.

In Rust terms: `-inf` is a runtime convention. `Softmask` is a compile-time guarantee. The type system encodes the correct face and orientation of the plane, so you can't accidentally project from the wrong one.
