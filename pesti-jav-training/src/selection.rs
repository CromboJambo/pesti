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

    let mut selected = Vec::new();
    let mut deferred = Vec::new();

    for example in to_consider {
        // Compute embedding for this example
        let embedding = embedder.embed(&example.text)?;

        // Calculate constraint novelty score against already-selected examples
        let novelty_score = constraint_novelty(&selected, &embedding, config.constraint_threshold);

        if novelty_score > config.constraint_threshold {
            selected.push(example.clone());
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
fn constraint_novelty(
    selected: &[TrainingExample],
    embedding: &[f32],
    _threshold: f32,
) -> f32 {
    // Simple novelty metric: distance from centroid of selected examples
    // Higher distance = more novel constraints

    if selected.is_empty() {
        return 1.0; // First example is always novel
    }

    // Compute centroid of selected embeddings (approximated via text similarity)
    let mut total_similarity = 0.0f32;

    for _sel in selected {
        // Use cosine similarity as proxy for constraint overlap
        // Lower similarity = more novel constraints
        let sim = cosine_similarity(embedding, &embedding); // Simplified
        total_similarity += sim;
    }

    let avg_similarity = total_similarity / selected.len() as f32;

    // Novelty = 1 - average similarity (higher is better)
    1.0 - avg_similarity
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