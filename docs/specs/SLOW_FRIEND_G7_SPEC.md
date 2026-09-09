# Slow-Friend G7 — End-to-End Integration

**Status**: 📋 Ready to implement
**Date**: 2026-09-08
**Prerequisites**: G1 (drift signal) PASS, G3 (performance budget) PASS, G5 (session logging) COMPLETE
**Purpose**: Integrate slow-friend into the actual decode loop and prove non-interference + measurable drift detection.

## What to Build

### 1. Decode Loop Integration (`pesti-runner/src/transformer/model.rs`)

Add an optional slow-friend hook to the generation loop:

```rust
// In LlamaModel or a wrapper/generator struct:
pub fn generate_with_slow_friend(
    &mut self,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sf_cfg: SlowFriendConfig,
) -> Result<GenerationResult, Error> {
    let mut slow = SlowFriendState::new(&sf_cfg);
    // ... generation loop with slow.update(h) and divergence scoring at each step
}
```

Key requirements:
- Slow-friend updates happen **after** the forward pass, using the **post-LN final-layer hidden state** (the tensor passed to the LM head for logits computation)
- Divergence score computed and logged per step (or every N steps for performance)
- Generation output (token sequence) must be **identical** to baseline generation without slow-friend — the slow-friend is observational only in G7
- No re-anchoring or intervention yet (that's G8+); just measure and report

### 2. Integration Example (`pesti-runner/examples/slow_friend_e2e.rs`)

A real end-to-end demo that:
1. Loads a model + tokenizer
2. Runs generation **without** slow-friend → baseline tokens (deterministic decoding: temperature=0)
3. Runs generation **with** slow-friend → should produce identical tokens (same deterministic settings)
4. Reports per-step divergence scores
5. **Separate drift detection test**: induces drift and shows the slow-friend detects it

Both runs use identical decoding parameters with deterministic token selection (greedy/temperature=0). Use Qwen2.5-0.5B-Instruct chat template for prompt formatting.

### 3. Drift Induction Test

Prove the slow-friend actually detects meaningful drift, not just noise:
- Run generation with slow-friend in **observational mode** (no intervention) → measure baseline divergence scores
- Separately, run a modified forward pass where we perturb the hidden state at certain steps (e.g., add Gaussian noise or scale by 1.05) → show that divergence scores are measurably higher during perturbed steps

This proves:
a) Slow-friend doesn't interfere with generation (tokens identical between baseline and slow-friend runs)
b) Slow-friend detects when the hidden state distribution actually shifts (perturbation test)

## Success Criteria

- Example compiles and runs without errors
- Token sequences are **bit-identical** between baseline and slow-friend-integrated runs (proves non-interference)
- Divergence scores are computed and reported for each step
- Mean divergence over 100+ steps is reported as a scalar metric
- Drift induction test shows measurably higher divergence (>2x mean baseline) during perturbed steps vs unperturbed

## Acceptance Criteria (PASS/FAIL)

**PASS(G7)** if:
1. Example compiles and runs without errors
2. Token sequences are bit-identical between baseline and slow-friend-integrated runs
3. Mean divergence over 100+ steps is reported as a scalar metric
4. Drift induction test shows measurably higher divergence (>2x mean baseline) during perturbed steps

**FAIL(G7)** if:
- Any assertion in the example fails
- Token sequences differ between baseline and slow-friend-integrated runs (non-interference violated)
- Divergence scores are not computed or reported
- Drift induction test does not show measurably higher divergence during perturbed steps

## Files to Create/Modify

- `pesti-runner/src/transformer/model.rs` — add `generate_with_slow_friend()` method
- `pesti-runner/examples/slow_friend_e2e.rs` — integration example
- Update `ROADMAP.md` with G7 status after implementation