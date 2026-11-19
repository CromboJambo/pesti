# Progressive Context Retrieval — Start Compact, Expand When Needed

**Date:** November 19, 2026
**Status:** Draft
**Priority:** P3 (Optimization)
**North Star:** Don't load the whole library to find one book. Start with the summary, expand only when the summary isn't enough.

## Problem Statement

PESTI currently serves inference with either full context or a fixed-size context window. This is wasteful for easy queries ("What's the capital of France?" doesn't need 32K tokens of context) and potentially insufficient for hard queries that need more detail than the fixed window provides. Progressive context retrieval would start compact and expand only when confidence drops.

## Core Concept

Inference proceeds in **expansion rounds**:

1. **Round 0 (Compact)**: Start with a compressed summary of the context. Generate an answer.
2. **Confidence Check**: Measure confidence in the answer (logit spread, self-consistency, etc.)
3. **Expand if needed**: If confidence < threshold, add one level of detail from the context hierarchy.
4. **Repeat**: Continue expanding until confidence ≥ threshold or source is exhausted.

Each expansion round records:
- What was added to context
- How much compute it cost
- Whether confidence improved

This creates a natural trade-off curve: more context → higher confidence but more compute. The system finds the sweet spot per query.

## Architecture

### Context Hierarchy

Context is organized in layers of increasing detail:

```
Layer 0: Summary (1-2% of original)     — "What this document is about"
Layer 1: Outline (5-10%)                — "Key sections and their main points"
Layer 2: Key Details (20-30%)           — "Important facts, code snippets, examples"
Layer 3: Full Context (100%)            — "Everything"
```

Each layer is a **waypoint** in the transformation graph (see crabjar WAYPOINT-SYSTEM spec). Layer N contains references to the specific parts of Layer N+1 that produced it.

### Expansion Decision Model

```rust
pub struct ExpansionDecision {
    pub query_id: QueryId,
    pub current_layer: u32,              // Which layer we're currently at
    pub confidence_score: f32,           // Current confidence in answer
    pub threshold: f32,                  // Confidence needed to stop expanding
    
    pub decision: ExpansionAction,
    pub rationale: String,
    
    // Cost tracking
    pub compute_so_far: ComputeCost,     // Tokens processed so far
    pub estimated_cost_to_next: ComputeCost,  // What next expansion would cost
}

pub enum ExpansionAction {
    Stop { reason: StopReason },         // Confidence is sufficient
    Expand { target_layer: u32 },        // Add more detail
    GiveUp { reason: InsufficientContext },  // Even full context isn't enough
}

pub enum StopReason {
    ConfidenceMet,                       // Answer is good enough
    DiminishingReturns { 
        last_improvement: f32,           // Last expansion only improved confidence by X
        threshold: f32                   // Below this, not worth expanding further
    },
    BudgetExhausted { budget: ComputeCost },  // Hit compute/time limit
}
```

### Confidence Measurement

Confidence is measured using multiple signals:

```rust
pub struct ConfidenceSignal {
    pub logit_spread: f32,               // Gap between top-1 and top-2 logits
    pub self_consistency: f32,           // Do multiple samples agree?
    pub context_utilization: f32,        // Is the model actually using the context?
    pub answer_specificity: f32,         // Is the answer specific or vague?
}

pub fn compute_confidence(signals: &[ConfidenceSignal]) -> f32 {
    // Weighted combination of signals
    // Weights are task-category dependent (see RELEVANCE-PRESERVATION spec)
    weighted_average(signals)
}
```

### Progressive Retrieval API

```rust
pub trait ProgressiveRetriever {
    /// Get context at a specific layer
    fn get_layer(&self, source: &SourceId, layer: u32) -> Result<ContextLayer>;
    
    /// Get the difference between two layers (what would be added)
    fn layer_diff(&self, source: &SourceId, from: u32, to: u32) -> Result<ContextDelta>;
    
    /// Estimate confidence at a specific layer without full inference
    fn estimate_confidence(
        &self, 
        query: &Query, 
        source: &SourceId, 
        layer: u32
    ) -> Result<f32>;
    
    /// Execute progressive retrieval for a query
    fn retrieve_progressive(
        &self,
        query: &Query,
        source: &SourceId,
        max_layer: u32,                   // Don't expand beyond this
        confidence_threshold: f32,
        compute_budget: ComputeCost,
    ) -> Result<RetrievalResult>;
}

pub struct RetrievalResult {
    pub final_layer: u32,                 // Which layer we stopped at
    pub context_used: ContextLayer,       // The actual context that was used
    pub confidence: f32,                  // Final confidence score
    pub expansions_made: usize,           // How many times we expanded
    pub total_cost: ComputeCost,          // Total compute spent
    pub expansion_log: Vec<ExpansionLogEntry>,  // What happened at each step
}

pub struct ExpansionLogEntry {
    pub round: u32,
    pub from_layer: u32,
    pub to_layer: u32,
    pub confidence_before: f32,
    pub confidence_after: f32,
    pub compute_cost: ComputeCost,
    pub context_added_tokens: usize,
}
```

### Integration with KV Cache

Progressive retrieval interacts with the KV cache in interesting ways:

- **Layer 0 inference**: Small KV cache (summary only)
- **Expansion to Layer 1**: KV cache grows as new tokens are processed
- **Cache reuse**: Tokens from Layer 0 remain valid when expanding to Layer 1 (they're a subset)

This means the cost of expansion is **incremental**, not cumulative. You don't re-process the summary — you just add the new detail on top.

```rust
pub struct IncrementalKVCache {
    pub base_cache: KVCache,              // From previous layer
    pub delta_tokens: Vec<TokenId>,       // New tokens from expansion
    pub total_size: usize,
    
    /// Expand cache with new tokens without reprocessing old ones
    fn expand(&mut self, new_tokens: &[TokenId]) -> Result<()>;
}
```

## Integration Points

### With Relevance-Preservation Metrics
Relevance profiles tell us the expected confidence at each layer for different query categories. This lets us set appropriate confidence thresholds before starting retrieval.

### With Loss Accounting
Each context layer is a compressed representation with an associated LossProfile. When expanding from Layer 0 to Layer 1, we're recovering information that was lost in the Layer 0 compression. The LossProfile tells us what to expect to find.

### With Model Selection
Different models have different "expansion efficiencies" — how much confidence they gain per additional token of context. Some models are good at extracting signal from sparse context (good for early layers), others need more context to perform well. Progressive retrieval can be tuned per model.

## Constraints & Policies

### Maximum Expansion Depth
Queries cannot expand beyond Layer 3 (full context). If confidence is still below threshold at full context, the query is flagged as "insufficient context" rather than continuing to expand into noise.

### Compute Budget Enforcement
Each query has a compute budget (tokens processed × model size). Progressive retrieval stops when the budget is exhausted, even if confidence hasn't reached threshold. The result is marked as "budget-limited."

### Diminishing Returns Detection
If two consecutive expansions improve confidence by less than a configured threshold (e.g., 0.02), stop expanding. Further expansion is unlikely to help and just wastes compute.

## Success Criteria

1. Easy queries (factual lookup) complete at Layer 0 or 1 with < 20% of full-context compute
2. Hard queries that need detail reach appropriate layers without manual intervention
3. Expansion decisions are logged and auditable — we know why each query expanded (or didn't)
4. Total inference cost is reduced by 30-50% on mixed workloads compared to fixed full-context

## Open Questions

- How do we handle multi-document queries where different documents need different expansion depths?
- Can we predict expansion depth from query features before starting inference? (Learned predictor)
- What's the right confidence threshold for different task categories? (Probably task-specific)
- How does progressive retrieval interact with streaming responses? (Expand while generating?)
