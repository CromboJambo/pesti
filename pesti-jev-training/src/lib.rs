//! JEV training data selection via constraint propagation.
//!
//! Finds the minimal sufficient subset of training examples that captures
//! all behavioral constraints in the full dataset. Uses PESTI structural
//! embeddings to identify novel vs redundant constraint contributions.

pub mod selection;
pub mod metrics;

pub use selection::{select_minimal_dataset, SelectionConfig, SelectedSubset, TrainingExample};