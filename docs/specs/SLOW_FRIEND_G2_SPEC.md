# Slow-Friend Substrate — Implementation Spec (G2: Expert Scoping)

**Status**: 📋 Ready to implement
**Date**: 2026-09-08
**Prerequisite**: G1 drift signal probe must be PASS (completed Sep 8, 2026)
**Decision record**: EDR-011 (`CHANGELOG.md`)
**Concept / references**: [`docs/concepts/SLOW_FRIEND_SUBSTRATE.md`](../concepts/SLOW_FRIEND_SUBSTRATE.md)
**Roadmap nudges**: `ROADMAP.md` → "Phase 5: Slow-Friend Substrate" (gates G2)

> This spec is self-contained. A fresh agent with only this file + the repo should be able to
> implement G2 without reading the prior conversation. Read top-to-bottom before writing code.

---

## 0. Goal of THIS session (and what it is NOT)

**Do**: Build an expert-relevance prior from the slow friend's stable state, use it to scope
activation at high context, and prove with a probe that scoping keeps activations closer to a
low-context reference than free activation does. Measure via Jaccard similarity of activation
patterns across sequence lengths.

**NOT do in this session**:
- ❌ No MoE router implementation (PESTI doesn't have one yet — use synthetic expert regions).
- ❌ No compaction/re-anchor trigger wiring into the generation loop (needs G2 green first).
- ❌ No separate small CPU model. The slow friend is still a stateful EMA over PESTI's own hidden states.
- ❌ No GPU kernel work. This is pure CPU (`no-default-features` must still build).

**Success = G2 green**: the probe shows scoped Jaccard similarity > free Jaccard similarity at high
context lengths, demonstrating that stable-memory scoping reduces activation drift.

---

## 1. Context you need (no prior session required)

### 1.1 The principle in one line
Use the slow friend's stable state to maintain a coarse expert-relevance map `e_t` that scopes which
compute units are active, keeping activations closer to low-context reference behavior as context grows.

### 1.2 Why this follows G1
G1 proved drift is real and grows with context length (cosine divergence: 0.047 → 0.056 → 0.063 at
seq_len 256/512/1024). Now we test whether the slow friend can *reduce* that drift by scoping which
parts of the model are active based on stable memory, not just detecting it.

### 1.3 Tooling you will reuse (already in the repo)
- `kernel/slow_friend/state.rs` — EMA low-pass state (`SlowFriendState`, `update()`, `summary()`).
- `kernel/slow_friend/divergence.rs` — Divergence metrics (`divergence()`, `Cosine`).
- `pesti-runner/examples/slow_friend_drift.rs` — G1 probe; base for the G2 scoping probe.
- `pesti-runner/src/transformer/model.rs` — `LlamaModel`, `forward_layers_with_cache()`, `embed()`.

---

## 2. Architecture: what to build

### 2.1 New module file: `pesti-runner/src/kernel/slow_friend/scoping.rs`

Expert scoping prior from slow-friend state. CPU-only, feature-independent.

```rust
//! Expert scoping prior from slow-friend state.
//!
//! Projects the stable EMA summary to a bounded relevance map over synthetic
//! "expert" slots (hidden-state regions). Used to scope activation and measure
//! whether stable memory can keep long-context behavior closer to low-context reference.

/// Bounded expert-relevance prior computed from slow-friend state.
pub struct ExpertPrior {
    /// Number of synthetic expert slots.
    num_experts: usize,
    /// Dimensionality of the input hidden state.
    dim: usize,
    /// Projection matrix: [num_experts][dim] — learned or fixed random.
    projection: Vec<Vec<f32>>,
}

impl ExpertPrior {
    /// Create a new expert prior with random projection (seeded for reproducibility).
    pub fn new(dim: usize, num_experts: usize) -> Self;

    /// Compute relevance scores for each expert slot from the stable state.
    /// Returns [num_experts] scores in [0, 1] (softmax-normalized).
    pub fn compute(&self, state: &[f32]) -> Vec<f32>;

    pub fn num_experts(&self) -> usize;
}

/// Apply soft scoping: modulate hidden state regions by expert relevance.
/// Regions with low prior relevance are attenuated (scaled toward zero).
pub fn apply_scoping(hidden: &[f32], prior: &[f32]) -> Vec<f32>;

/// Compute activation pattern: which expert regions are "active" (above threshold).
pub fn activation_pattern(hidden: &[f32], num_experts: usize) -> Vec<bool>;

/// Jaccard similarity between two activation patterns.
pub fn jaccard_similarity(a: &[bool], b: &[bool]) -> f32;
```

Register in `pesti-runner/src/kernel/slow_friend/mod.rs`:
```rust
pub mod scoping;
pub use scoping::{ExpertPrior, apply_scoping, activation_pattern, jaccard_similarity};
```

### 2.2 The probe example: `pesti-runner/examples/slow_friend_scoping.rs`

A CPU-only example (base it on `slow_friend_drift.rs`) that, for a real GGUF model:
1. Loads the model + tokenizer (self-contained, from GGUF).
2. For each seq_len in `[256, 512, 1024]` (env-overridable via `PESTI_SLOW_SEQS`):
   - Builds a deterministic long prompt (repeat seed sentence).
   - Runs forward pass; at each decode step:
     - Take the final-layer hidden state as `h_t`.
     - Update slow friend: `slow.update(h_t)`.
     - Compute expert relevance prior from stable summary.
     - Measure activation pattern with and without scoping.
   - After the run, compute Jaccard similarity between reference (short context) and long-context patterns.
3. Prints a table comparing free vs scoped Jaccard similarity at each length.

**G2 pass criterion**: Scoped Jaccard > Free Jaccard at all long contexts (512 and 1024), demonstrating
that scoping by stable memory keeps activations closer to the low-context reference than free activation does.

---

## 3. Acceptance criteria (all must hold)

- [ ] `cargo build` and `cargo build --no-default-features` both succeed (scoping is CPU-only).
- [ ] `cargo test -p pesti-runner` — new unit tests pass; existing lib tests still pass.
- [ ] **Unit tests** in `slow_friend/scoping.rs`:
  - Expert prior scores sum to 1 (softmax normalization).
  - Activation pattern has correct length.
  - Jaccard similarity is 1.0 for identical patterns, 0.0 for disjoint patterns.
  - Apply scoping attenuates low-relevance regions.
- [ ] **Probe** (`slow_friend_scoping.rs`) runs on a real GGUF model CPU-only and prints the comparison table.
- [ ] Per-step cost of `prior.compute()` + `apply_scoping()` reported in µs (should be negligible vs decode step).

---

## 4. Done-when / handoff checklist

When you finish, update the tracking so the next agent can pick up G3:
1. Append results to `ROADMAP.md` → Phase 5: check off **G2** with measured Jaccard similarities per seq_len.
2. Add a short subsection under **EDR-011** in `CHANGELOG.md` recording the G2 outcome (measured improvement, commit hash).
3. If G2 **PASS**: leave G3 (cost measurement) as the next nudge.
4. If G2 **FAIL or inconclusive**: do NOT proceed to G3. Record the finding and what would change the verdict.

---

## 5. Constraints & cautions (from EDR-011)

- **Framing — peers, not ranked.** The fast friend and slow friend are complementary poles of a polarity.
- **Keep it bounded.** `alpha < 1.0` always. Expert prior scores in [0,1].
- **CPU-only, feature-independent.** Must build under `--no-default-features`. No `cuda`, no `cudarc`.
- **No new heavy deps.** Use std + existing workspace deps (`thiserror` for errors). Do not add a linear-algebra crate.
- **Honest reporting.** If the signal isn't there, say so. A false PASS defeats the whole point of the gate.
- **Don't touch the GPU e2e path** (Week 17). This is additive: a new module + one example + tests.
- **Determinism.** The probe must be reproducible — fixed seed sentence, no sampling randomness.

---

## 6. Suggested order of work (TDD-friendly)

1. Scaffold `slow_friend/scoping.rs` + register in `kernel/slow_friend/mod.rs`. Write the unit tests first (prior computation, activation patterns, Jaccard similarity).
2. Confirm `cargo build --no-default-features` still clean.
3. Build `slow_friend_scoping.rs` on top of `slow_friend_drift.rs`; wire expert prior + scoping; run at seq_len=512 first to validate plumbing, then 1024/2048.
4. Read the table. If scoped > free Jaccard → PASS(G2). Write up numbers in ROADMAP + CHANGELOG.
5. Stop. G3 is a separate session (cost measurement).

---

## 7. Open questions for the implementer (decide + record, don't block)

- **Expert region partitioning**: For dense models without real MoE routing, partition the hidden state into N equal-sized regions as synthetic "experts." Record the choice in CHANGELOG.
- **Threshold for activation**: Use a fixed threshold (e.g., mean magnitude > 0.1) or percentile-based. Fixed is simpler and more reproducible for G2.
- **Update cadence**: Per-step prior computation is fine for G2. If cost is non-negligible, compute every N steps and note it.

---

*This spec is the contract for one coding session: build expert scoping + prove drift reduction (G2).
Everything past G3 is deliberately out of scope so the session stays small, testable, and reversible.*
