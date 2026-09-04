# Slow-Friend Substrate — Implementation Spec (G1: Drift Signal)

**Status**: 📋 Ready to implement (fresh-window coding session)
**Date**: 2026-09-02
**Decision record**: **EDR-011** (`CHANGELOG.md`)
**Concept / references**: [`docs/concepts/SLOW_FRIEND_SUBSTRATE.md`](../concepts/SLOW_FRIEND_SUBSTRATE.md)
**Roadmap nudges**: `ROADMAP.md` → "Phase 5: Slow-Friend Substrate" (gates G1–G4)

> This spec is **self-contained**. A fresh agent with only this file + the repo should be able to
> implement G1 without reading the prior conversation. Read top-to-bottom before writing code.

---

## 0. Goal of THIS session (and what it is NOT)

**Do**: Build a minimal, testable **slow friend** — a cheap, always-on, bounded low-pass summary of
the model's hidden states that runs on CPU and produces a per-step **divergence score** against the
precise path. Then prove with a probe (G1) that this divergence signal **grows smoothly with context
length** on a real model.

**NOT do in this session**:
- ❌ No MoE router, no scoped top-k (that's G2 — needs G1 green first).
- ❌ No compaction/re-anchor trigger wiring into the generation loop (needs G1+G2).
- ❌ No separate small CPU model. The slow friend here is a **stateful EMA over PESTI's own hidden
  states** — zero extra forward passes. (The two-model split is G4, decided later.)
- ❌ No GPU kernel work. This is pure CPU (`no-default-features` must still build).

**Success = G1 green**: the probe shows divergence increasing monotonically (or at least non-decreasing
with a clear trend) as `seq_len` grows 512 → 1024 → 2048, on a real GGUF model, with the slow friend's
per-step cost measured and reported.

---

## 1. Context you need (no prior session required)

### 1.1 The principle in one line
Keep a cheap **stable reference** that does *not* accumulate drift, and measure how far the precise path
drifts from it as context grows. Bounded positive gates on anything that accumulates state; hard sparsity
only for compute. (Full rationale: concept doc §2, EDR-011.)

### 1.2 Why PESTI is already primed for this
PESTI's Week 17 work **measured** the exact drift signal we want to detect: GPU-vs-oracle `max|d|` grows
**5.6e-3 (layer 0) → 5.1e-2 (layer 23)**, prehead 1.4e-1, logits 5.0e-2; norm ratio 0.999–1.001, corr ≥
0.999993, argmax identical — "f16 tensor-core accumulation rounding … not a sync race or layout bug."
Week 15: "23.9/20.7 max logit diff (f16 drift, smooth per-layer growth, no structural jumps)."

So the signal is real and already reproducible. G1's job is to express it as a **continuous, length-indexed
divergence score** from a stable reference — not to re-discover that GPU drifts.

### 1.3 Tooling you will reuse (already in the repo)
- `LlamaModel::capture_per_layer` — pushes each layer's output when set (`None` = normal, no overhead).
- `pesti-runner/examples/dump_all_layers_gpu.rs` — full per-layer hidden dump through the real dispatch path.
- `DispatchContext::gpu_fallback_count()` — assert a run was fully GPU (zero fallbacks) or CPU-only as needed.
- `pesti-runner/examples/cpu_e2e_generate.rs` — CPU-only greedy generation (bypasses GPU OOM). **Use this as the base for the probe** so G1 runs without VRAM pressure.
- numpy-oracle diff path in `conformance-corpus/`.

---

## 2. Architecture: what to build

### 2.1 New module: `pesti-runner/src/kernel/slow_friend/`

A small, CPU-only, **feature-independent** module (no `cuda` gate — it must compile under
`--no-default-features`). Follow the existing kernel module conventions (see `kernel/mod.rs`): a
`mod.rs` root with re-exports, plain Rust, `thiserror` for errors if any.

Files:
```
pesti-runner/src/kernel/slow_friend/
  mod.rs        - module root, pub use of public types
  state.rs      - SlowFriendConfig + SlowFriendState (the EMA low-pass)
  divergence.rs - DivergenceMetric enum + DivergenceScore + comparison functions
```

Register in `pesti-runner/src/kernel/mod.rs` (CPU-only, so NO `#[cfg(feature = "cuda")]`):
```rust
pub mod slow_friend;
```

### 2.2 The state: a bounded low-pass (EMA) over hidden states

This is the GDN *analog* for PESTI's current dense transformer — we don't have a delta-network, so the
stable reference is an exponential moving average of the per-layer (or final-layer) hidden vectors. The
critical property: **bounded, fixed-size, O(1) per step, decays old context** — exactly the "preserve
intent without accumulating unbounded debt" invariant.

```rust
/// Bounded low-pass over a sequence of hidden-state vectors.
/// Fixed size (d), O(1) per update, cannot accumulate unbounded error.
pub struct SlowFriendState {
    /// The running summary vector. Length == `dim`.
    summary: Vec<f32>,
    /// Normalized decay in [0,1). Higher = longer memory / slower friend.
    alpha: f32,
    /// Number of steps applied (for reporting + optional normalization).
    step: u64,
}

pub struct SlowFriendConfig {
    pub dim: usize,      // summary vector length (e.g. final-layer hidden dim)
    pub alpha: f32,      // decay; typical 0.9..0.999. Must be < 1.0 (bounded forget).
}

impl SlowFriendState {
    pub fn new(cfg: &SlowFriendConfig) -> Self;

    /// Fold one hidden-state vector into the running summary. O(dim).
    /// h is [dim]; panics (debug_assert) if lengths mismatch.
    pub fn update(&mut self, h: &[f32]);

    /// Current summary (borrowed). Callers must not retain across `update`.
    pub fn summary(&self) -> &[f32];

    /// L2 norm of the summary — a cheap scalar "checksum" of the stable state.
    pub fn norm(&self) -> f32;

    /// Steps applied so far.
    pub fn step(&self) -> u64;

    /// Reset to empty (used at re-anchor, later gate).
    pub fn reset(&mut self);
}
```

**Update rule** (the bounded gate, made concrete):
```
summary = alpha * summary + (1 - alpha) * h     // per element
step   += 1
```
- `alpha < 1.0` is the **bounded forget**; `(1 - alpha)` is the **bounded write**. Both in [0,1].
- This is deliberately a plain EMA (not delta-rule subtraction). G1 only needs a *stable* reference; the
  delta term is a refinement for later if the EMA proves too lossy. Keep it simple and testable now.

### 2.3 The divergence metric

```rust
/// How far the precise path's current representation has drifted from the stable summary.
pub enum DivergenceMetric {
    /// cosine distance = 1 - cos(a, b). In [0,2]. Scale-invariant — robust to norm drift.
    Cosine,
    /// normalized L2 = ||a-b|| / (||a|| + ||b|| + eps). In [0,1]. Magnitude-sensitive.
    RelL2,
}

#[derive(Debug, Clone, Copy)]
pub struct DivergenceScore {
    pub metric: DivergenceMetric,
    pub value: f32,      // the divergence magnitude (larger = more drift)
    /// optional: the slow-friend norm at this step (for reporting).
    pub ref_norm: Option<f32>,
}

/// Compare a precise-path vector against the stable summary. O(dim).
pub fn divergence(metric: DivergenceMetric, stable: &[f32], precise: &[f32]) -> DivergenceScore;
```

- `Cosine` is the **primary** metric (scale-invariant; PESTI's norm ratio is 0.999–1.001 so cosine won't
  be fooled by a global scale shift). Keep `RelL2` as a secondary for reporting.
- Guard against zero vectors (return `value: 0.0` if either norm ≈ 0) — no NaNs.

### 2.4 The probe example: `pesti-runner/examples/slow_friend_drift.rs`

A CPU-only example (base it on `cpu_e2e_generate.rs`) that, for a real GGUF model:
1. Loads the model + tokenizer (self-contained, from GGUF — Week 15).
2. For each `seq_len` in `[512, 1024, 2048]` (env-overridable via `PESTI_SLOW_SEQS`, comma-separated):
   - Builds a prompt of that length (repeat/expand a seed sentence; deterministic).
   - Runs forward pass; at each decode step:
     - Take the **final-layer hidden state** (via `capture_per_layer` or the pre-head vector) as `h_t`.
     - `slow.update(h_t)` → maintain the stable summary.
     - Compute `precise_t` = a representation of the *current* token from the precise path. For G1 use the
       **pre-head hidden state** (or final-layer output) at the same step as `precise_t`.
     - Record `d_t = divergence(Cosine, slow.summary(), precise_t)` every step (or every N steps).
   - After the run, report per-seq_len: mean `d_t`, max `d_t`, and `d_t` at the last 10% of steps.
3. Prints a small table:

```
seq_len | mean_d | max_d | tail_mean_d | slow_update_us/step
   512  |  ...   |  ...  |     ...     |        ...
  1024  |  ...   |  ...  |     ...     |        ...
  2048  |  ...   |  ...  |     ...     |        ...
```

**G1 pass criterion (print this verdict in the example output):** `tail_mean_d` is **non-decreasing** across
the seq_len ladder (2048 ≥ 1024 ≥ 512 within a small tolerance), OR shows a clear monotonic-upward trend.
Print `PASS(G1)` / `FAIL(G1)` so the result is machine-checkable.

> Design note: because the slow friend *also* consumes the same hidden states, at short context it will be
> close to the precise path (low divergence); as context grows and the precise path accumulates f16 drift,
> divergence should rise. If instead divergence is flat across lengths, that itself is a finding — report it
> honestly (G1 no-go) rather than tuning until it passes.

---

## 3. Acceptance criteria (all must hold)

- [ ] `cargo build` and `cargo build --no-default-features` both succeed (slow_friend is CPU-only).
- [ ] `cargo test -p pesti-runner` — new unit tests pass; existing lib tests still pass (the known
      pre-existing compile errors in unrelated test targets are out of scope, but you must not add new ones).
- [ ] **Unit tests** in `slow_friend/state.rs`:
  - EMA converges: feeding a constant vector repeatedly → summary → that vector.
  - Boundedness: with bounded inputs and `alpha < 1`, `norm()` stays bounded (no blow-up over 10k steps).
  - `reset()` clears to zero.
  - Length mismatch is caught (`debug_assert!` / panic in debug builds).
- [ ] **Unit tests** in `slow_friend/divergence.rs`:
  - Identical vectors → divergence 0 (both metrics).
  - Orthogonal unit vectors (Cosine) → value ≈ 1.0.
  - Zero vector handled without NaN.
  - `RelL2` in [0,1] for random inputs.
- [ ] **Probe** (`slow_friend_drift.rs`) runs on a real GGUF model CPU-only and prints the table + a
      `PASS(G1)`/`FAIL(G1)` verdict with measured per-step cost.
- [ ] Per-step cost of `slow.update` + `divergence` reported in µs (should be negligible vs a decode step).

---

## 4. Done-when / handoff checklist

When you finish, update the tracking so the next agent can pick up G2:
1. Append results to `ROADMAP.md` → Phase 5: check off **G1** with the measured numbers (mean/max/tail
   divergence per seq_len + cost), and note PASS/FAIL.
2. Add a short subsection under **EDR-011** in `CHANGELOG.md` recording the G1 outcome (measured signal,
   commit hash) — keep it factual, no code.
3. If G1 **PASS**: leave G2 (scoped MoE routing) as the next nudge, unchanged.
4. If G1 **FAIL or inconclusive**: do NOT proceed to G2. Record the finding and what would change the
   verdict (e.g., "divergence flat across lengths; need FP8 path or longer context to see drift").

---

## 5. Constraints & cautions (from EDR-011)

- **Framing — peers, not ranked.** The fast friend and slow friend are *complementary poles* of a
  polarity, not "primary model + fallback" and not one being smarter/lesser. Fast = upper bound
  (reach); slow = lower bound (evidence floor). Never implement the slow friend as a hard veto or a
  degraded-mode backup — it's the *reference/receipt* (drift + re-anchor + user-credit audit trail),
  which is why it must be **stable and non-drifting**. Full framing: concept doc → "The Polarity"
  section. If you find yourself writing `if slow_disagrees { override }`, stop — that's a hard gate,
  the exact thing this design rejects. Use the bounded weight `w`.
- **Keep it bounded.** `alpha < 1.0` always. No unbounded accumulation in the summary.
- **CPU-only, feature-independent.** Must build under `--no-default-features`. No `cuda`, no `cudarc`.
- **No new heavy deps.** Use std + existing workspace deps (`half` if you need f16→f32, `thiserror` for errors). Do not add a linear-algebra crate for this — it's O(dim) elementwise.
- **Honest reporting.** If the signal isn't there, say so. A false PASS (tuned to look monotonic) defeats
  the whole point of the gate.
- **Don't touch the GPU e2e path** (Week 17). This is additive: a new module + one example + tests.
- **Determinism.** The probe must be reproducible — fixed seed sentence, no sampling randomness in the
  forward pass (greedy / single forward, no stochastic ops).

---

## 6. Suggested order of work (TDD-friendly)

1. Scaffold `slow_friend/{mod,state,divergence}.rs` + register in `kernel/mod.rs`. Write the **unit tests
   first** (state EMA, divergence metrics) — get them red, then green.
2. Confirm `cargo build --no-default-features` still clean.
3. Build `slow_friend_drift.rs` on top of `cpu_e2e_generate.rs`; wire `capture_per_layer` for the hidden
   states; run at seq_len=512 first to validate plumbing, then 1024/2048.
4. Read the table. If trend is there → PASS(G1). Write up numbers in ROADMAP + CHANGELOG.
5. Stop. G2 is a separate session (it needs a router to scope, which PESTI doesn't have yet — that's the
   next design decision).

---

## 7. Open questions for the implementer (decide + record, don't block)

- **Which hidden vector is "the precise path" at step t?** Default: final-layer pre-head state. If you find
  a cleaner anchor (e.g., the actual attention output), use it and note the choice in the CHANGELOG entry.
- **Summary dim**: default = final-layer hidden dim. If that's large, you may project to a smaller summary
  (fixed random projection, seeded) for speed — but record the projection so it's reproducible.
- **Update cadence**: per-step is fine for G1. If cost is non-negligible, update every N steps and note it.

---

*This spec is the contract for one coding session: build the slow friend + prove the drift signal (G1).
Everything past G2 is deliberately out of scope so the session stays small, testable, and reversible.*
