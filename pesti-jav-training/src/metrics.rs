//! Metrics for evaluating training data selection quality.

use crate::selection::{DeferredExample, TrainingExample};

/// Evaluate the quality of a selected subset against deferred examples.
///
/// Returns metrics including coverage, redundancy detection accuracy, etc.
pub fn evaluate_selection(
    selected: &[TrainingExample],
    deferred: &[DeferredExample],
) -> SelectionMetrics {
    let total = selected.len() + deferred.len();

    SelectionMetrics {
        compression_ratio: if total == 0 {
            1.0
        } else {
            selected.len() as f32 / total as f32
        },
        examples_selected: selected.len(),
        examples_deferred: deferred.len(),
    }
}

#[derive(Debug)]
pub struct SelectionMetrics {
    pub compression_ratio: f32,
    pub examples_selected: usize,
    pub examples_deferred: usize,
}

impl std::fmt::Display for SelectionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Selection metrics:\n\
             - Compression ratio: {:.2}%\n\
             - Examples selected: {}\n\
             - Examples deferred: {}",
            self.compression_ratio * 100.0,
            self.examples_selected,
            self.examples_deferred
        )
    }
}