# Spike: Batched Generation for Parallel Prompts

## Problem
Current decode loop generates one token at a time per prompt. For N prompts, this means N sequential forward passes. We want to process multiple prompts in parallel (batch size > 1) to improve throughput.

## Current Architecture
- `generate()` in pesti-runner processes one prompt at a time
- Each step: embed → layers (attn + FFN) → lm_head → sample → repeat
- KV cache grows per position, separate per prompt

## Batched Generation Design

### Option A: True Batch Processing (Multiple Prompts)
Process N independent prompts simultaneously in one forward pass.

```rust
let prompts = ["Hello", "How are you", "What is"];
let batch_size = 3;

// Each prompt generates independently but shares the compute step
let outputs = runner.generate_batch(prompts, max_tokens=10);
// -> ["Hello world!", "I'm fine thanks", "The capital of"]
```

**Changes needed:**
1. Embedding layer: stack inputs [batch, seq] → [batch*seq, embed_dim]
2. Attention: already supports batch dimension (has `batch_size` param)
3. KV cache: need separate caches per prompt in batch
4. Sampling: independent per sequence
5. Early termination: some sequences finish before others

**Complexity:** High - requires tracking active/inactive sequences, padding, etc.

### Option B: Speculative Decoding (Parallel Token Generation)
Generate multiple tokens per step by predicting ahead and verifying.

```rust
// Instead of 1 token/step, predict k tokens, verify all at once
let draft = model.draft_k_tokens(prompt, k=4);
let accepted = model.verify(draft);  // forward pass on all k tokens
```

**Changes needed:**
1. Draft phase: generate k tokens autoregressively (fast, no verification)
2. Verify phase: single forward pass on [prompt + draft] to validate all at once
3. Accept prefix that matches, reject and retry from divergence point

**Complexity:** Medium - simpler than true batching but still non-trivial

### Option C: Chunked Prefill (Parallel Context Processing)
Process longer context windows in parallel chunks rather than one token at a time during prefill phase.

```rust
// Prefill 1024 tokens as 8 chunks of 128, processed in parallel
let chunks = context.chunks(128);
for chunk in chunks {
    // Forward pass on chunk (parallel within chunk via causal masking)
}
```

**Changes needed:**
1. Modify prefill loop to process chunks
2. Causal mask must account for chunk boundaries
3. KV cache writes per chunk

**Complexity:** Low - mostly optimization of existing code path

## Recommendation: Start with Option C (Chunked Prefill)

Lowest complexity, immediate speedup for long prompts, no architectural changes to decode loop.

Then explore Option B (Speculative Decoding) as it provides the biggest theoretical speedup without requiring multiple independent prompts.

Option A (True Batch Processing) is most complex and may not align with PESTI's single-prompt focus.

## Benchmark Results (Week 23, jambo RTX 4070 Ti SUPER)

Measured prefill throughput at increasing sequence lengths:

| Sequence Length | Prefill Time | Effective Throughput |
|-----------------|--------------|---------------------|
| 128 tokens      | 219.4s       | 0.6 tok/s           |
| 256+            | (timeout)    | < 0.3 tok/s         |

Baseline decode: ~81 tok/s (measured separately at seq=32). Prefill is **~135x slower** than decode per-token, confirming that the linear accumulation of per-token forward passes during prefill is the dominant cost at long sequences.

At production lengths (>1024 tokens), chunked prefill could provide 10-50x speedup by processing multiple positions per forward pass.

## Quick Experiment: Can we do speculative decoding?

Let's try a simple version: generate 2 tokens at once by computing logits for both positions simultaneously, then accepting the first if it matches what autoregressive sampling would produce.

```rust
// Draft: sample next token normally
let draft_token = sample(logits);

// Verify: forward pass on [prompt + draft] to get actual distribution
let verified_logits = model.forward(prompt + [draft_token]);

if sampled_from(verified_logits) == draft_token {
    accept(draft_token);
} else {
    // Reject and resample from verified distribution
    let corrected = sample(verified_logits);
    accept(corrected);
}
```

This is a simplified form of speculative decoding that should work with existing architecture.

## Next Steps
1. Implement chunked prefill (easy win)
2. Prototype speculative decoding with k=2
3. Benchmark both approaches vs baseline
