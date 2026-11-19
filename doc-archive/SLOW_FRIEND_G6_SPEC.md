# Slow-Friend G6 — Compaction / Re-anchor

**Status**: Ready to implement
**Date**: 2026-09-08
**Prerequisites**: G1 (drift signal) PASS, G3 (performance budget) PASS
**Reference**: `docs/concepts/SLOW_FRIEND_SUBSTRATE.md` §4 (Compaction), EDR-011

## Goal

Prove that periodic **re-anchoring** (compaction) of the slow-friend EMA summary keeps long-context drift bounded, compared to never compacting. The re-anchor trigger fires when divergence exceeds a threshold; on trigger, the slow-friend summary is reset toward the precise path rather than continuing to accumulate unbounded error.

## Design Principle

The slow friend is a **stable reference**, not a permanent accumulator. Like garbage collection, compaction periodically resets the bounded state to prevent drift from growing without limit. The governing rule: **bounded positive gates on anything that accumulates state** — compaction is the reset of the accumulation.

## Architecture

### New module: `pesti-runner/src/kernel/slow_friend/compaction.rs`

```rust
/// Compaction trigger based on divergence threshold.
pub struct CompactionTrigger {
    /// Divergence threshold above which re-anchor fires.
    pub threshold: f32,
}

impl CompactionTrigger {
    pub fn new(threshold: f32) -> Self;

    /// Should we re-anchor at this step? Returns true if divergence exceeds threshold.
    pub fn should_compact(&self, divergence_score: &DivergenceScore) -> bool;
}

/// Re-anchor the slow-friend summary toward the precise path.
/// Blends the current summary with the precise hidden state.
/// weight in [0,1]: 0 = no change, 1 = full reset to precise state.
pub fn reanchor(state: &mut SlowFriendState, precise: &[f32], weight: f32);
```

### Re-anchor update rule

```
summary = (1 - weight) * summary + weight * precise
```

- `weight` is the **compaction strength** — how aggressively to reset toward the precise path.
- Typical values: 0.5–1.0 (strong re-anchor). The slow friend should snap back close to the precise path when it drifts too far.
- This is a bounded write (weight < 1) that preserves some of the stable reference while correcting for accumulated drift.

### Probe example: `pesti-runner/examples/slow_friend_compaction.rs`

A CPU-only probe that:
1. Loads Qwen2.5-0.5B-Instruct-Q4_K_M.
2. Runs two forward passes at seq_len=1024:
   - **Baseline**: No compaction (never re-anchor). Measure final divergence.
   - **Compacted**: Re-anchor when divergence > threshold (e.g., 0.1). Measure final divergence and number of re-anchors.
3. Prints comparison table showing that compaction keeps drift bounded.

## Acceptance Criteria

- [ ] `cargo build` and `cargo build --no-default-features` succeed.
- [ ] Unit tests for CompactionTrigger: threshold logic correct, no false triggers below threshold.
- [ ] Unit test for reanchor(): weight=0 leaves summary unchanged, weight=1 fully resets to precise state.
- [ ] Probe runs on real GGUF model and prints baseline vs compacted divergence at seq_len=1024.
- [ ] **G6 pass criterion**: Compacted run has strictly lower final divergence than baseline (proves compaction reduces drift).
- [ ] Number of re-anchors reported (expect a small number, e.g., 1–5 over 1024 steps).

## Constraints

- CPU-only, feature-independent (`--no-default-features` must build).
- No new heavy deps.
- Deterministic probe (fixed seed sentence, greedy decode).
- Honest reporting — if compaction doesn't help, report it as FAIL(G6).

## Suggested Order of Work

1. Scaffold `compaction.rs` with CompactionTrigger and reanchor(). Write unit tests first.
2. Build `slow_friend_compaction.rs` probe on top of G1's drift example pattern.
3. Run at seq_len=512 first to validate plumbing, then 1024 for the actual test.
4. Report results in ROADMAP.md and CHANGELOG.md (EDR-011).

## Open Questions

- **Threshold selection**: Use G1's measured divergence values to pick a reasonable threshold. If G1 shows divergence growing from ~0.05 to ~0.1 over 1024 tokens, a threshold of 0.08 might trigger re-anchoring at the right moment.
- **Compaction weight**: Start with weight=1.0 (full reset). If that causes instability (divergence spikes after re-anchor), try weight=0.5–0.75 for softer re-anchoring.
