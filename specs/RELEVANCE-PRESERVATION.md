# Relevance-Preservation Metrics — Compression That Preserves Answers

**Date:** November 19, 2026
**Status:** Draft
**Priority:** P2 (Quality Improvement)
**North Star:** The goal isn't "compress to X bytes." The goal is "minimize size subject to ΔR ≤ ε," where R(q,S) = P(correct answer | q, S).

## Problem Statement

PESTI optimizes inference for speed and memory: quantization levels (Q8→Q5→Q4), context window management, KV cache compression. These are all size/speed metrics. But the real question is: **does the compressed representation still produce correct answers?** A 3-bit quantized model might be 4x smaller but produce confidently wrong answers for certain query types. We need to measure *relevance preservation*, not just compression ratio.

## Core Concept

Define a **relevance-preservation metric** that measures how well a compressed representation preserves the probability of correct answers across different query types:

```
R(q, S) = P(correct answer | query q, context S)

Objective: minimize |C(S)| subject to R(q, C(S)) ≥ R(q, S) - ε for all q in Q_task
```

Where:
- `S` = original (full) context or weights
- `C(S)` = compressed representation
- `Q_task` = set of queries relevant to the task
- `ε` = acceptable reduction in answer quality
- `|C(S)|` = size of compressed representation

This reframes compression from "make it smaller" to "make it small enough while keeping answers correct."

## Architecture

### Relevance Profile Data Model

```rust
pub struct RelevanceProfile {
    pub id: ProfileId,
    pub model_id: ModelId,
    pub created_at: Timestamp,
    
    // What was measured
    pub quantization_level: QuantLevel,      // Q8, Q5, Q4, etc.
    pub context_size: usize,                 // Tokens in context window
    pub query_set: QuerySetRef,              // Which queries were tested
    
    // Results per query category
    pub category_scores: Vec<CategoryScore>,
    
    // Overall metrics
    pub mean_fidelity: f32,                  // Average R across all categories
    pub worst_category: CategoryScore,       // The weakest link
    pub confidence_interval: (f32, f32),     // 95% CI on mean fidelity
}

pub struct CategoryScore {
    pub category: QueryCategory,             // Factual, reasoning, coding, etc.
    pub original_fidelity: f32,              // R(q, S) with full context/weights
    pub compressed_fidelity: f32,            // R(q, C(S)) with compressed version
    pub delta: f32,                          // Reduction in fidelity
    pub samples: usize,                      // How many queries in this category
}

pub enum QueryCategory {
    FactualLookup,       // "What is X?"
    Reasoning,           // "Why does X happen?"
    CodeGeneration,      // "Write code that does X"
    CodeDebugging,       // "Fix this bug"
    Summarization,       // "Summarize this document"
    CreativeWriting,     // "Write a story about X"
    MathComputation,     // "Calculate X"
}
```

### Fidelity Measurement Protocol

```rust
pub struct FidelityMeasurement {
    pub query: String,
    pub category: QueryCategory,
    pub ground_truth: Option<String>,        // If available
    pub original_answer: String,             // From full-precision model
    pub compressed_answer: String,           // From compressed model
    pub similarity_score: f32,               // Semantic similarity between answers
    pub correctness_original: bool,          // Was the original answer correct?
    pub correctness_compressed: bool,        // Was the compressed answer correct?
    pub degradation_type: DegradationType,   // How did it get worse?
}

pub enum DegradationType {
    None,                    // Same quality
    MinorWording,            // Different words, same meaning
    PartialInformation,      // Missing some details but core is correct
    WrongDetail,             // Core is right but a specific fact is wrong
    StructurallyWrong,       // Answer has the wrong shape/structure
    CompletelyWrong,         // Answer is unrelated to the question
}
```

### Adaptive Quantization Decision

Use relevance profiles to choose quantization level per task:

```rust
pub struct QuantizationDecision {
    pub model_id: ModelId,
    pub task_category: QueryCategory,
    pub available_levels: Vec<QuantLevel>,   // Q8, Q5, Q4, ...
    pub selected_level: QuantLevel,
    pub rationale: String,                   // Why this level?
    
    // Trade-off analysis
    pub size_savings: f32,                   // % reduction vs. highest quality
    pub fidelity_cost: f32,                  // ΔR for this task category
    pub total_cost: f32,                     // C_storage + C_retrieval + C_error
}

pub fn choose_quantization(
    profile: &RelevanceProfile,
    task_category: QueryCategory,
    max_fidelity_loss: f32,                  // ε for this task
) -> QuantizationDecision {
    // Find the smallest model that still meets fidelity requirements
    let candidates = profile.category_scores
        .iter()
        .filter(|s| s.category == task_category && s.delta <= max_fidelity_loss);
    
    // Pick the one with minimum total cost
    candidates.min_by_total_cost()
}
```

### Total Cost Model

The cheapest representation isn't always the smallest. It's the one minimizing:

```
C_total = C_storage + C_retrieval + C_reconstruction + C_error

Where:
- C_storage = cost of storing the compressed representation
- C_retrieval = cost of loading/accessing it during inference
- C_reconstruction = cost of recovering details if needed (progressive retrieval)
- C_error = expected cost of wrong answers due to compression
```

A 4-bit model that's 4x smaller but produces 10% more wrong answers might have a higher C_total than a 5-bit model that's only 2.5x smaller but much more accurate.

## Integration Points

### With Inference Pipeline
Before serving a request, check the relevance profile for the model/quantization level being used. If the task category has low fidelity at the current level, either:
- Serve with the higher-quality model (if available)
- Flag the response as "lower confidence" to the caller

### With Model Selection
When choosing between models for a task, relevance profiles provide a quality dimension alongside speed and size. "Which 7B model should I use for code debugging?" becomes answerable with data.

### With Progressive Context Retrieval
Relevance profiles inform when to expand context: if fidelity drops below threshold for a query category, trigger progressive retrieval to add more context (see PROGRESSIVE-CONTEXT-RETRIEVAL spec).

## Constraints & Policies

### Minimum Fidelity Floors
Certain task categories have non-negotiable minimum fidelity:
- Code debugging: ΔR ≤ 0.05 (wrong debug advice is costly)
- Math computation: ΔR ≤ 0.01 (math errors compound)
- Creative writing: ΔR ≤ 0.20 (some variation is acceptable)

### Measurement Frequency
Relevance profiles should be re-measured when:
- Model weights change (fine-tuning, quantization update)
- Query distribution shifts significantly
- New query categories are added

### Unknown Categories
If a query doesn't match any measured category, use the worst-case fidelity from measured categories as a conservative estimate. Don't assume unknown = safe.

## Success Criteria

1. Relevance profiles exist for all deployed models at all quantization levels
2. Model selection considers fidelity, not just speed/size
3. Fidelity measurements are reproducible (same query set → same scores)
4. The total cost model actually influences deployment decisions

## Open Questions

- How do we get ground truth for open-ended queries where "correct" is subjective?
- Can we use LLM-as-judge for fidelity measurement, or do we need human evaluation?
- How often does the query distribution shift enough to require re-measurement?
- Should relevance profiles be shared across models (transferable knowledge about which categories are hard)?
