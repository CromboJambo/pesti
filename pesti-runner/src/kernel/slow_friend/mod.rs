//! Slow-Friend Substrate: bounded low-pass reference for drift detection.
//!
//! Maintains a cheap, always-on, stable summary of the model's hidden states
//! and measures how far the precise path drifts from it as context grows.

pub mod state;
pub mod divergence;

pub use state::{SlowFriendConfig, SlowFriendState};
pub use divergence::{DivergenceMetric, DivergenceScore, divergence};
