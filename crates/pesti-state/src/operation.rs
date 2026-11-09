//! Operation log types for weight-space audit trail.

use serde::{Deserialize, Serialize};

/// Types of operations tracked in the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationType {
    /// Weight-space session opened on a model.
    Open,
    /// Weight edits applied to tensors.
    Edit,
    /// Snapshot created for rollback.
    Snapshot,
    /// Rollback performed from snapshot.
    Rollback,
    /// Edited model saved to disk.
    Save,
    /// Weight-space session closed.
    Close,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            OperationType::Open => "open",
            OperationType::Edit => "edit",
            OperationType::Snapshot => "snapshot",
            OperationType::Rollback => "rollback",
            OperationType::Save => "save",
            OperationType::Close => "close",
        };
        write!(f, "{s}")
    }
}

/// Operation-specific payloads stored as JSON in the operations table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum OperationDetails {
    /// Session opened on a model.
    Open {
        session_id: String,
        model_path: String,
        tensor_count: usize,
    },
    /// Weight edits applied.
    Edit {
        edit_count: usize,
        tensors_modified: Vec<String>,
        snapshot_taken: bool,
    },
    /// Snapshot created.
    Snapshot { snapshot_index: usize },
    /// Rollback performed.
    Rollback { from_snapshot: usize },
    /// Model saved to disk.
    Save {
        output_path: String,
        format: String,
        adapter_id: Option<String>,
    },
    /// Session closed.
    Close {
        total_edits: usize,
        snapshots_created: usize,
        rollbacks_performed: usize,
        duration_secs: f64,
    },
}

/// A single entry in the append-only operation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationLog {
    /// Auto-incremented primary key (assigned by database).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    /// Model this operation pertains to.
    pub model_id: Option<String>,
    /// UUID identifying the WeightSpaceSession.
    pub session_id: String,
    /// Type of operation performed.
    pub op_type: OperationType,
    /// Operation-specific payload.
    pub details: OperationDetails,
    /// ISO 8601 timestamp.
    pub timestamp: String,
}

impl OperationLog {
    /// Create a new operation log entry with the current timestamp.
    pub fn new(
        model_id: Option<String>,
        session_id: String,
        op_type: OperationType,
        details: OperationDetails,
    ) -> Self {
        Self {
            id: None,
            model_id,
            session_id,
            op_type,
            details,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}
