//! Adapter registry records and lifecycle states.

use pesti_durable_objects::object_id::ObjectId;
use serde::{Deserialize, Serialize};

/// Lifecycle state for a registered adapter.
///
/// Represents where an adapter is in the knowledge validation pipeline:
/// Observed → Recorded → Examined → Tested → Accepted → Inherited → Revalidated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterState {
    /// Adapter exists but has not been examined or tested.
    Unverified,
    /// Adapter has been tested and produces expected behavior.
    Validated,
    /// Adapter is actively in use by inference sessions.
    Active,
    /// Base model changed; adapter needs re-examination before reuse.
    NeedsRevalidation,
    /// Adapter has been superseded or determined invalid.
    Deprecated,
}

impl std::fmt::Display for AdapterState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AdapterState::Unverified => "unverified",
            AdapterState::Validated => "validated",
            AdapterState::Active => "active",
            AdapterState::NeedsRevalidation => "needs_revalidation",
            AdapterState::Deprecated => "deprecated",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for AdapterState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unverified" => Ok(AdapterState::Unverified),
            "validated" => Ok(AdapterState::Validated),
            "active" => Ok(AdapterState::Active),
            "needs_revalidation" => Ok(AdapterState::NeedsRevalidation),
            "deprecated" => Ok(AdapterState::Deprecated),
            other => Err(format!("unknown adapter state: {other}")),
        }
    }
}

/// A saved adapter or edited model with full provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterRecord {
    /// Unique identifier (UUID via ObjectId).
    pub id: String,
    /// Base model this adapter was built on.
    pub base_model_id: String,
    /// Where the adapter/edited model was saved.
    pub output_path: String,
    /// Human-readable description of what this adapter encodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Conditions under which this adapter is considered valid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assumptions: Option<Vec<String>>,
    /// ISO 8601 timestamp of creation.
    pub created_at: String,
    /// Last validation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified_at: Option<String>,
    /// Current lifecycle state.
    pub state: AdapterState,
    /// Operation IDs that produced this adapter.
    #[serde(default)]
    pub operation_ids: Vec<i64>,
}

impl AdapterRecord {
    /// Create a new unverified adapter record.
    pub fn new(
        base_model_id: String,
        output_path: String,
        description: Option<String>,
        assumptions: Option<Vec<String>>,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: ObjectId::generate().as_str().to_string(),
            base_model_id,
            output_path,
            description,
            assumptions,
            created_at: now.clone(),
            verified_at: None,
            state: AdapterState::Unverified,
            operation_ids: Vec::new(),
        }
    }

    /// Create a new adapter record with specific operation IDs linked.
    pub fn with_operations(
        base_model_id: String,
        output_path: String,
        description: Option<String>,
        assumptions: Option<Vec<String>>,
        operation_ids: Vec<i64>,
    ) -> Self {
        let mut record = Self::new(base_model_id, output_path, description, assumptions);
        record.operation_ids = operation_ids;
        record
    }
}
