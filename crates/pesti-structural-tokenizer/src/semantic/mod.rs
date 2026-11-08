//! Semantic annotation layer for pesti-structural-tokenizer.
//!
//! This module analyzes Rust source code to extract semantic properties beyond
//! basic syntax: ownership modes, error handling strategies, mutability state,
//! and complexity metrics. These annotations can be projected onto the token
//! stream as parallel channels or inline tags to provide richer context for
//! LLM processing and JEV decision-making.
//!
//! # Stable API (v0.1.x)
//!
//! The following public items are part of the stable API and will not change
//! in minor or patch releases. Breaking changes only occur at major version bumps.
//!
//! ## Analyzers
//! - `analyze_ownership(src)` — Detects ownership transfer patterns (owned, borrowed, mutably_borrowed)
//! - `analyze_errors(src)` — Classifies error handling (Propagated, Recovered, Panicked, Ignored)
//! - `analyze_mutability(src)` — Identifies mutable vs immutable bindings and their mutation sites
//! - `analyze_complexity(src)` — Computes cyclomatic complexity, nesting depth, control flow count
//!
//! ## Projection
//! - `annotate_all(src)` — Run all analyzers and return combined SemanticAnnotations struct
//! - `project_annotations(tokens, annotations)` — Map annotations onto token indices
//! - `SemanticTag::encode(...)` / `decode(tag)` — Compact 32-bit encoding of annotation combinations
//!
//! # Breaking Change Policy
//!
//! - **Patch releases (0.1.x):** Bug fixes only. No API changes.
//! - **Minor releases (0.x.0):** New analyzers or features may be added. Existing APIs remain stable.
//! - **Major releases (x.0.0):** Breaking changes permitted. Migration guide published with release.
//!
//! # Analyzer Behavior
//!
//! All analyzers use lightweight string scanning (not AST parsing) for performance.
//! They operate on Rust source text and return annotations keyed by token index.
//! Confidence scores range from 0-100, with higher indicating stronger evidence.

mod complexity;
mod errors;
mod mutability;
mod ownership;
mod projection;

pub use complexity::{analyze_complexity, ComplexityAnnotation, ComplexityMetrics};
pub use errors::{analyze_errors, ErrorAnnotation, ErrorHandling};
pub use mutability::{analyze_mutability, MutKind, MutabilityAnnotation};
pub use ownership::{analyze_ownership, OwnershipAnnotation, OwnershipMode};
pub use projection::{annotate_all, project_annotations, SemanticAnnotations, SemanticTag};
