//! Semantic annotation layer for pesti-structural-tokenizer.
//!
//! This module analyzes Rust source code to extract semantic properties beyond
//! basic syntax: ownership modes, error handling strategies, and mutability state.
//! These annotations can be projected onto the token stream as parallel channels
//! or inline tags to provide richer context for LLM processing.

mod errors;
mod mutability;
mod ownership;
mod projection;

pub use errors::{analyze_errors, ErrorAnnotation, ErrorHandling};
pub use mutability::{analyze_mutability, MutKind, MutabilityAnnotation};
pub use ownership::{analyze_ownership, OwnershipAnnotation, OwnershipMode};
pub use projection::{annotate_all, project_annotations, SemanticAnnotations, SemanticTag};
