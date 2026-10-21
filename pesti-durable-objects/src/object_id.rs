//! Object identity management for durable objects.
//!
//! Mirrors herdr's workspace/tab/pane ID system: unique identifiers that persist
//! across session restarts and provide stable references to objects.

use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Unique identifier for a durable object.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId(String);

impl ObjectId {
    /// Generate a new unique object ID.
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create an object ID from an existing string.
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}