//! Slow-Friend Substrate: bounded low-pass reference for drift detection.
//!
//! Maintains a cheap, always-on, stable summary of the model's hidden states
//! and measures how far the precise path drifts from it as context grows.

pub mod state;
pub mod divergence;
pub mod scoping;
pub mod logging;

pub use state::{SlowFriendConfig, SlowFriendState};
pub use divergence::{DivergenceMetric, DivergenceScore, divergence};
pub use scoping::{ExpertPrior, apply_scoping, activation_pattern, jaccard_similarity};
pub use logging::SessionLogger;
