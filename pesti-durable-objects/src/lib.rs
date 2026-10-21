//! Durable objects layer with snapshot persistence, versioning, and migration handling.
//!
//! Extracted from herdr's session management architecture for use in crabjar's fleet coordination.
//!
//! # Design Principles
//!
//! - Snapshot-based persistence: capture entire state into serializable structs
//! - Versioned formats with migration functions for format evolution
//! - Object identity via UUIDs with lifecycle tracking
//! - Async operations for disk I/O

pub mod snapshot;
pub mod object_id;
pub mod migration;

// Re-export core types
pub use snapshot::{save_snapshot, load_snapshot, SnapshotError};
pub use object_id::ObjectId;