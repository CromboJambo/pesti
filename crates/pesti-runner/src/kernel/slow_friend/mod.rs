//! Slow-Friend Substrate: bounded low-pass reference for drift detection.
//!
//! Maintains a cheap, always-on, stable summary of the model's hidden states
//! and measures how far the precise path drifts from it as context grows.

pub mod compaction;
pub mod divergence;
pub mod logging;
pub mod scoping;
pub mod state;

pub use compaction::{CompactionTrigger, reanchor};
pub use divergence::{DivergenceMetric, DivergenceScore, divergence};
pub use logging::SessionLogger;
pub use scoping::{ExpertPrior, activation_pattern, apply_scoping, jaccard_similarity};
pub use state::{SlowFriendConfig, SlowFriendState};
