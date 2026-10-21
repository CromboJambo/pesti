//! Migration handling for durable object version upgrades.
//!
//! Mirrors herdr's snapshot migration pattern: when the serialization format
//! changes, migration functions convert old versions to new versions.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MigrationError {
    #[error("unknown version {0}")]
    UnknownVersion(u32),
    #[error("migration failed: {0}")]
    Failed(String),
}

/// Registry of migration functions for version upgrades.
pub struct MigrationRegistry {
    /// Map from version number to migration function name (for documentation)
    versions: Vec<u32>,
}

impl MigrationRegistry {
    pub fn new() -> Self {
        Self { versions: vec![] }
    }

    /// Register a migration from version n to n+1.
    pub fn register(&mut self, version: u32) -> &mut Self {
        self.versions.push(version);
        self
    }

    /// Get all registered migration versions.
    pub fn versions(&self) -> &[u32] {
        &self.versions
    }
}

/// Migrate snapshot data from an older version to the current version.
pub fn migrate_snapshot(
    data: serde_json::Value,
    from_version: u32,
    to_version: u32,
) -> Result<serde_json::Value, MigrationError> {
    let mut current = data;

    for version in from_version..to_version {
        match version {
            // Version 1 → 2 migration (example)
            1 => {
                // Apply migration logic here
            }
            _ => {
                return Err(MigrationError::UnknownVersion(version));
            }
        }
    }

    Ok(current)
}