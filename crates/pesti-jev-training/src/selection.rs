//! Core JEV training data selection algorithm.
//!
//! Uses constraint propagation via PESTI embeddings to identify the minimal
//! sufficient subset of training examples that captures all behavioral constraints.

use anyhow::Result;
use pesti_embeddings::Embedder;

/// Configuration for training data selection.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Embedding model path
    pub embedding_model: String,
    /// Constraint novelty threshold (higher = stricter)
    pub constraint_threshold: f32,
    /// Maximum number of examples to consider
    pub max_examples: Option<usize>,
}

/// A training example with metadata.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub id: String,
    pub text: String,
    pub labels: Vec<String>,
}

/// Internal representation with embedding cached during selection.
#[derive(Debug)]
struct ExampleWithEmbedding {
    example: TrainingExample,
    embedding: Vec<f32>,
}

/// Reason an example was deferred (not selected).
#[derive(Debug, Clone)]
pub enum DeferredReason {
    ConstraintRedundant { covered_by: String },
    BelowThreshold { score: f32, threshold: f32 },
}

impl std::fmt::Display for DeferredReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstraintRedundant { covered_by } => {
                write!(f, "constraint redundant (covered by {})", covered_by)
            }
            Self::BelowThreshold { score, threshold } => {
                write!(f, "below threshold ({:.3} < {:.3})", score, threshold)
            }
        }
    }
}

/// A deferred example with its reason.
#[derive(Debug, Clone)]
pub struct DeferredExample {
    pub example: TrainingExample,
    pub reason: DeferredReason,
}

/// Result of training data selection.
#[derive(Debug)]
pub struct SelectedSubset {
    /// Examples selected for training (minimal sufficient subset)
    pub selected: Vec<TrainingExample>,
    /// Examples deferred with reasons
    pub deferred: Vec<DeferredExample>,
    /// Selection statistics
    pub stats: SelectionStats,
}

#[derive(Debug)]
pub struct SelectionStats {
    pub total_examples: usize,
    pub selected_count: usize,
    pub deferred_count: usize,
    pub compression_ratio: f32,
}

/// Select minimal sufficient training subset using JEV constraint propagation.
///
/// For each example, computes a constraint novelty score based on PESTI embeddings.
/// Examples that add novel constraints are selected; redundant examples are deferred.
pub fn select_minimal_dataset(
    examples: &[TrainingExample],
    config: &SelectionConfig,
) -> Result<SelectedSubset> {
    let max_examples = config.max_examples.unwrap_or(examples.len());
    let to_consider = &examples[..max_examples];

    // Load embedding model
    let embedder = Embedder::load(&config.embedding_model)?;

    let mut selected: Vec<TrainingExample> = Vec::new();
    let mut deferred = Vec::new();
    // Track embeddings of selected examples for novelty computation
    let mut selected_embeddings: Vec<Vec<f32>> = Vec::new();

    for example in to_consider {
        // Compute embedding for this example
        let embedding = embedder.embed(&example.text)?;

        // Calculate constraint novelty score against already-selected examples
        let novelty_score = constraint_novelty(&selected_embeddings, &embedding);

        if novelty_score > config.constraint_threshold {
            selected.push(example.clone());
            selected_embeddings.push(embedding.clone());
        } else {
            deferred.push(DeferredExample {
                example: example.clone(),
                reason: DeferredReason::ConstraintRedundant {
                    covered_by: "existing selection".to_string(),
                },
            });
        }
    }

    let stats = SelectionStats {
        total_examples: to_consider.len(),
        selected_count: selected.len(),
        deferred_count: deferred.len(),
        compression_ratio: if to_consider.is_empty() {
            1.0
        } else {
            selected.len() as f32 / to_consider.len() as f32
        },
    };

    Ok(SelectedSubset {
        selected,
        deferred,
        stats,
    })
}

/// Calculate constraint novelty score for an example against already-selected examples.
///
/// Returns a score in [0, 1] where higher means more novel constraints added.
fn constraint_novelty(selected_embeddings: &[Vec<f32>], new_embedding: &[f32]) -> f32 {
    // Novelty metric: minimum distance from any already-selected example
    // Higher = more novel (farther from existing selection)

    if selected_embeddings.is_empty() {
        return 1.0; // First example is always novel
    }

    let mut max_distance = 0.0f32;
    for sel_emb in selected_embeddings {
        let sim = cosine_similarity(new_embedding, sel_emb);
        let distance = 1.0 - sim;
        if distance > max_distance {
            max_distance = distance;
        }
    }

    // Return the maximum distance (most novel relative to closest existing)
    max_distance
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}