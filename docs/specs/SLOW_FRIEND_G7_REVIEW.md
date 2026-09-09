# Adversarial Review: G7 E2E Integration Spec

**Target**: `docs/specs/SLOW_FRIEND_G7_SPEC.md`
**Reviewer**: Hermes Agent (self-review)
**Date**: 2026-09-08

## P0 — Design Flaws (Wrong Output / Broken Invariants)

### P0.1: "Bit-identical tokens" is unachievable with temperature sampling
The spec requires token sequence identity between baseline and slow-friend runs but doesn't mandate deterministic decoding. With any temperature > 0, floating-point non-determinism in softmax will produce different token sequences regardless of the slow-friend.

**Fix**: Mandate `temperature=0` (greedy) or explicit seed for top-k sampling. Add to acceptance criteria: "Both runs use identical decoding parameters with deterministic token selection."

### P0.2: Drift induction test conflates detection with interference
The spec proposes adding noise/injection to hidden states during generation and checking that the slow-friend detects it. But if you modify hidden states, tokens WILL differ from baseline — violating the non-interference claim simultaneously. You can't prove both "slow-friend doesn't change output" and "slow-friend detects modified computation" in the same run.

**Fix**: Split into two separate tests:
- Test A (non-interference): Run with slow-friend observing, compare tokens to baseline without slow-friend → must be identical
- Test B (detection): Two parallel runs — one clean, one with induced drift — show that slow-friend divergence score is higher in the degraded run. Tokens may differ between these two runs; that's expected and not a failure.

### P0.3: No quantitative threshold for "measurably higher divergence"
The acceptance criterion says "slow-friend reports measurably higher divergence" but doesn't define what constitutes measurable. A 0.001 increase in mean divergence might be numerical noise, not signal.

**Fix**: Add quantitative criterion: "Divergence score in degraded run exceeds baseline by ≥3σ of baseline per-step variance, or absolute difference > 0.05 (cosine)."

## P1 — Missing Edge Cases / Incomplete Spec

### P1.1: No EOS / early termination handling
What if the model generates an EOS token before max_tokens? The slow-friend should still be valid at that point. What about the divergence score computation when generation ends naturally vs is truncated?

**Fix**: Add acceptance criterion: "Test includes both natural EOS termination and max_tokens truncation; slow-friend state remains consistent in both cases."

### P1.2: No specification of which hidden states are used
The spec says "final-layer hidden state" but doesn't specify: pre-norm or post-norm? The residual stream before or after the layer norm? These produce different numerical ranges and may affect divergence scores significantly.

**Fix**: Specify exactly which tensor is fed to slow-friend (e.g., "post-LN final layer hidden state, matching what's passed to the LM head").

### P1.3: No prompt format / model variant specification
Qwen2.5-0.5B-Instruct uses chat templates with special tokens. Does the test use raw text or formatted chat? Different prompts produce very different generation patterns and drift characteristics.

**Fix**: Specify exact prompt construction method (chat template vs raw) and include a representative multi-turn example.

## P2 — Implementation Risks / Dead Code Potential

### P2.1: Slow-friend may be dead code in the non-interventional mode
If G7 only observes without intervening, the slow-friend's EMA summary is computed but never used to modify anything. This makes it a pure measurement tool, not a substrate. The real value proposition (using drift signal for routing/compaction) isn't tested until G8+.

**Note**: Acceptable for an integration test, but document that G7 proves plumbing works, not that the substrate provides value yet.

### P2.2: Performance measurement methodology unspecified
The spec says "overhead ≤ 5%" but doesn't specify how to measure it. Wall clock? CPU time? Does it include model loading? Tokenization? Just the generation loop?

**Fix**: Specify: "Measure only the autoregressive generation loop (token-by-token decoding), excluding prompt encoding and post-processing. Report mean and std dev across 3 runs."

## Recommendations Summary

1. **Add deterministic decoding requirement** (temperature=0 or fixed seed)
2. **Split drift test into two separate tests** (non-interference vs detection)
3. **Add quantitative threshold** for "measurably higher" divergence
4. **Specify exact hidden state tensor** used for slow-friend updates
5. **Specify prompt format** and include chat template handling
6. **Define performance measurement methodology** precisely

These are all fixable before implementation begins. The core design is sound — it's a good integration test that validates the plumbing without overreaching into unproven territory (like G2/G6 did).
