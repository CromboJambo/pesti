//! GPU forward pass with minimal H2D transfers.
//!
//! This validates that all transformer layers can run on GPU while avoiding
//! intermediate host-device round-trips (based on attention ...[truncated]