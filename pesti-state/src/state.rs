//! Core state database: models, operations, adapters.
//!
//! All methods lock the internal connection mutex for the duration of the
//! operation, so callers can safely share a `PestiState` across threads
//! or sessions via cheap clones.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json;

use crate::adapter::{AdapterRecord, AdapterState};
use crate::error::{Result, StateError};
use crate::model::ModelRecord;
use crate::operation::{OperationDetails, OperationLog, OperationType};

/// Current database schema version.
const SCHEMA_VERSION: u32 = 1;

/// PESTI's persistent state layer.
///
/// Owns the SQLite connection (via `Arc<Mutex<Connection>>`) and provides
/// typed access to models, operations, and adapters. Cheaply cloneable —
/// all clones share the same underlying database connection.
///
/// # Thread safety
///
/// `PestiState` is `Send + Sync`. The inner connection is protected by a
/// mutex, so multiple sessions can safely share one `PestiState` instance.
pub struct PestiState {
    conn: Arc<Mutex<Connection>>,
    db_path: PathBuf,
}

impl Clone for PestiState {
    fn clone(&self) -> Self {
        Self {
            conn: Arc::clone(&self.conn),
            db_path: self.db_path.clone(),
        }
    }
}

impl PestiState {
    // ─── Construction ──────────────────────────────────────────────

    /// Open or create the state database at the given path.
    ///
    /// Creates parent directories if they don't exist. Enables WAL mode
    /// for concurrent readers with serialized writers. Runs schema
    /// migrations as needed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create parent directory
        if let Some(parent) = path.parent().filter(|p| !p.exists()) {
            std::fs::create_dir_all(parent).map_err(|e| StateError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        let conn = Connection::open(&path)?;

        // Enable WAL mode for concurrent access
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        // Reasonable busy timeout for multi-process scenarios
        conn.busy_timeout(std::time::Duration::from_secs(5))?;

        let state = Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: path,
        };
        state.migrate()?;
        Ok(state)
    }

    /// Default location: `~/.pesti/state.db`.
    pub fn open_default() -> Result<Self> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|e| StateError::Io {
                path: PathBuf::from("~"),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, e.to_string()),
            })?;
        Self::open(Path::new(&home).join(".pesti").join("state.db"))
    }

    /// Get the path to the state database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    // ─── Schema & Migration ────────────────────────────────────────

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;

        if version > SCHEMA_VERSION {
            return Err(StateError::SchemaVersionMismatch {
                found: version,
                expected: SCHEMA_VERSION,
            });
        }

        // Schema v1: initial tables
        if version < 1 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS models (
                    id              TEXT PRIMARY KEY,
                    path            TEXT NOT NULL,
                    architecture    TEXT NOT NULL,
                    quantization    TEXT NOT NULL,
                    param_count     INTEGER,
                    tensor_count    INTEGER NOT NULL,
                    first_seen      TEXT NOT NULL,
                    last_verified   TEXT
                );

                CREATE TABLE IF NOT EXISTS operations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id        TEXT REFERENCES models(id),
                    session_id      TEXT NOT NULL,
                    op_type         TEXT NOT NULL,
                    details         JSON NOT NULL,
                    timestamp       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS adapters (
                    id              TEXT PRIMARY KEY,
                    base_model_id   TEXT REFERENCES models(id),
                    output_path     TEXT NOT NULL,
                    description     TEXT,
                    assumptions     JSON,
                    created_at      TEXT NOT NULL,
                    verified_at     TEXT,
                    state           TEXT NOT NULL DEFAULT 'unverified',
                    operation_ids   JSON
                );

                CREATE INDEX IF NOT EXISTS idx_operations_model ON operations(model_id);
                CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id);
                CREATE INDEX IF NOT EXISTS idx_adapters_base ON adapters(base_model_id);
                CREATE INDEX IF NOT EXISTS idx_adapters_state ON adapters(state);
                "#,
            )?;

            conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
        }

        Ok(())
    }

    // ─── Model Registry ────────────────────────────────────────────

    /// Register a model (by GGUF header hash) if not already known.
    ///
    /// Returns the model ID. If the model exists but with a different
    /// path, updates the path and last_verified timestamp.
    pub fn register_model(&self, record: &ModelRecord) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        let now = chrono::Utc::now().to_rfc3339();

        // Check if model already exists
        let existing_path: Option<String> = conn
            .query_row(
                "SELECT path FROM models WHERE id = ?1",
                params![record.id],
                |row| row.get(0),
            )
            .optional()?;

        match existing_path {
            Some(path) if path == record.path => {
                // Same model, same path — just update last_verified
                conn.execute(
                    "UPDATE models SET last_verified = ?1 WHERE id = ?2",
                    params![now, record.id],
                )?;
            }
            Some(_) => {
                // Same model hash, different path — update path and timestamp
                conn.execute(
                    "UPDATE models SET path = ?1, last_verified = ?2 WHERE id = ?3",
                    params![record.path, now, record.id],
                )?;
            }
            None => {
                // New model — insert
                let first_seen = record.first_seen.clone().unwrap_or(now.clone());
                conn.execute(
                    "INSERT INTO models (id, path, architecture, quantization, param_count, tensor_count, first_seen, last_verified)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        record.id,
                        record.path,
                        record.architecture,
                        record.quantization,
                        record.param_count,
                        record.tensor_count as i64,
                        first_seen,
                        now,
                    ],
                )?;
            }
        }

        Ok(record.id.clone())
    }

    /// Look up a model by its content hash.
    pub fn get_model(&self, id: &str) -> Result<Option<ModelRecord>> {
        let conn = self.conn.lock().unwrap();

        let record = conn
            .query_row(
                "SELECT id, path, architecture, quantization, param_count, tensor_count, first_seen, last_verified
                 FROM models WHERE id = ?1",
                params![id],
                |row| {
                    Ok(ModelRecord {
                        id: row.get(0)?,
                        path: row.get(1)?,
                        architecture: row.get(2)?,
                        quantization: row.get(3)?,
                        param_count: row.get(4)?,
                        tensor_count: row.get::<_, i64>(5)? as usize,
                        first_seen: Some(row.get(6)?),
                        last_verified: row.get(7)?,
                    })
                },
            )
            .optional()?;

        Ok(record)
    }

    /// Get all models currently registered.
    pub fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, path, architecture, quantization, param_count, tensor_count, first_seen, last_verified
             FROM models ORDER BY first_seen DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(ModelRecord {
                id: row.get(0)?,
                path: row.get(1)?,
                architecture: row.get(2)?,
                quantization: row.get(3)?,
                param_count: row.get(4)?,
                tensor_count: row.get::<_, i64>(5)? as usize,
                first_seen: Some(row.get(6)?),
                last_verified: row.get(7)?,
            })
        })?;

        let mut models = Vec::new();
        for row in rows {
            models.push(row?);
        }
        Ok(models)
    }

    // ─── Operation Log ─────────────────────────────────────────────

    /// Append an operation to the audit log. Returns the assigned row ID.
    pub fn log_operation(&self, op: &OperationLog) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let details_json = serde_json::to_string(&op.details)?;

        conn.execute(
            "INSERT INTO operations (model_id, session_id, op_type, details, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                op.model_id,
                op.session_id,
                op.op_type.to_string(),
                details_json,
                op.timestamp,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Get all operations for a given session, in chronological order.
    pub fn session_operations(&self, session_id: &str) -> Result<Vec<OperationLog>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, model_id, session_id, op_type, details, timestamp
             FROM operations WHERE session_id = ?1 ORDER BY id ASC",
        )?;

        let rows = stmt.query_map(params![session_id], row_to_operation)?;

        let mut ops = Vec::new();
        for row in rows {
            ops.push(row?);
        }
        Ok(ops)
    }

    /// Get all operations for a given model (across all sessions).
    pub fn model_operations(&self, model_id: &str) -> Result<Vec<OperationLog>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, model_id, session_id, op_type, details, timestamp
             FROM operations WHERE model_id = ?1 ORDER BY id ASC",
        )?;

        let rows = stmt.query_map(params![model_id], row_to_operation)?;

        let mut ops = Vec::new();
        for row in rows {
            ops.push(row?);
        }
        Ok(ops)
    }

    // ─── Adapter Registry ──────────────────────────────────────────

    /// Register a new adapter with provenance metadata. Returns the adapter ID.
    pub fn register_adapter(&self, record: AdapterRecord) -> Result<String> {
        let conn = self.conn.lock().unwrap();

        let assumptions_json = record
            .assumptions
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let operation_ids_json = serde_json::to_string(&record.operation_ids)?;

        conn.execute(
            "INSERT INTO adapters (id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.id,
                record.base_model_id,
                record.output_path,
                record.description,
                assumptions_json,
                record.created_at,
                record.verified_at,
                record.state.to_string(),
                operation_ids_json,
            ],
        )?;

        Ok(record.id)
    }

    /// Update an adapter's lifecycle state.
    pub fn update_adapter_state(&self, adapter_id: &str, state: AdapterState) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let verified_at = if matches!(state, AdapterState::Validated | AdapterState::Active) {
            Some(chrono::Utc::now().to_rfc3339())
        } else {
            None
        };

        let rows = conn.execute(
            "UPDATE adapters SET state = ?1, verified_at = COALESCE(?2, verified_at) WHERE id = ?3",
            params![state.to_string(), verified_at, adapter_id],
        )?;

        if rows == 0 {
            return Err(StateError::AdapterNotFound(adapter_id.to_string()));
        }

        Ok(())
    }

    /// Get a single adapter by ID.
    pub fn get_adapter(&self, adapter_id: &str) -> Result<Option<AdapterRecord>> {
        let conn = self.conn.lock().unwrap();

        let record = conn
            .query_row(
                "SELECT id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids
                 FROM adapters WHERE id = ?1",
                params![adapter_id],
                row_to_adapter,
            )
            .optional()?;

        Ok(record)
    }

    /// Get all adapters built on a specific base model.
    pub fn adapters_for_model(&self, model_id: &str) -> Result<Vec<AdapterRecord>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids
             FROM adapters WHERE base_model_id = ?1 ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map(params![model_id], row_to_adapter)?;

        let mut adapters = Vec::new();
        for row in rows {
            adapters.push(row?);
        }
        Ok(adapters)
    }

    /// List all adapters, optionally filtered by lifecycle state.
    pub fn list_adapters(&self, state: Option<AdapterState>) -> Result<Vec<AdapterRecord>> {
        let conn = self.conn.lock().unwrap();

        let (sql, params_vec): (&str, Vec<String>) = match state {
            Some(s) => (
                "SELECT id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids
                 FROM adapters WHERE state = ?1 ORDER BY created_at DESC",
                vec![s.to_string()],
            ),
            None => (
                "SELECT id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids
                 FROM adapters ORDER BY created_at DESC",
                vec![],
            ),
        };

        let mut stmt = conn.prepare(sql)?;
        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        let rows = stmt.query_map(params_refs.as_slice(), row_to_adapter)?;

        let mut adapters = Vec::new();
        for row in rows {
            adapters.push(row?);
        }
        Ok(adapters)
    }

    // ─── Revalidation ──────────────────────────────────────────────

    /// When a base model's hash changes (or is re-verified as different),
    /// flag all adapters built on the old version for revalidation.
    ///
    /// Returns the count of adapters flagged.
    pub fn invalidate_adapters_for_model(&self, old_model_id: &str) -> Result<usize> {
        let conn = self.conn.lock().unwrap();

        let rows = conn.execute(
            "UPDATE adapters SET state = 'needs_revalidation'
             WHERE base_model_id = ?1 AND state NOT IN ('deprecated', 'needs_revalidation')",
            params![old_model_id],
        )?;
        Ok(rows)
    }

    /// Check if any registered adapters need revalidation.
    pub fn needs_revalidation(&self) -> Result<Vec<AdapterRecord>> {
        self.list_adapters(Some(AdapterState::NeedsRevalidation))
    }

    /// Get adapters for a model that are in an active/validated state.
    /// Used by the "no silent inheritance" check when opening a session.
    pub fn active_adapters_for_model(&self, model_id: &str) -> Result<Vec<AdapterRecord>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, base_model_id, output_path, description, assumptions, created_at, verified_at, state, operation_ids
             FROM adapters WHERE base_model_id = ?1 AND state IN ('validated', 'active') ORDER BY created_at DESC",
        )?;

        let rows = stmt.query_map(params![model_id], row_to_adapter)?;

        let mut adapters = Vec::new();
        for row in rows {
            adapters.push(row?);
        }
        Ok(adapters)
    }
}

// ─── Free functions for row parsing (no self needed) ──────────────

fn row_to_operation(row: &rusqlite::Row<'_>) -> rusqlite::Result<OperationLog> {
    let op_type_str: String = row.get(3)?;
    let details_json: String = row.get(4)?;

    let op_type = match op_type_str.as_str() {
        "open" => OperationType::Open,
        "edit" => OperationType::Edit,
        "snapshot" => OperationType::Snapshot,
        "rollback" => OperationType::Rollback,
        "save" => OperationType::Save,
        "close" => OperationType::Close,
        other => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                3,
                rusqlite::types::Type::Text,
                Box::<dyn std::error::Error + Send + Sync>::from(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown operation type: {other}"),
                )),
            ));
        }
    };

    let details: OperationDetails = serde_json::from_str(&details_json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(4, rusqlite::types::Type::Text, Box::new(e))
    })?;

    Ok(OperationLog {
        id: Some(row.get(0)?),
        model_id: row.get(1)?,
        session_id: row.get(2)?,
        op_type,
        details,
        timestamp: row.get(5)?,
    })
}

fn row_to_adapter(row: &rusqlite::Row<'_>) -> rusqlite::Result<AdapterRecord> {
    let assumptions_json: Option<String> = row.get(4)?;
    let state_str: String = row.get(7)?;
    let operation_ids_json: String = row.get(8)?;

    let assumptions = assumptions_json.and_then(|j| serde_json::from_str(&j).ok());
    let state: AdapterState = state_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            7,
            rusqlite::types::Type::Text,
            Box::<dyn std::error::Error + Send + Sync>::from(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e,
            )),
        )
    })?;
    let operation_ids: Vec<i64> = serde_json::from_str(&operation_ids_json).unwrap_or_default();

    Ok(AdapterRecord {
        id: row.get(0)?,
        base_model_id: row.get(1)?,
        output_path: row.get(2)?,
        description: row.get(3)?,
        assumptions,
        created_at: row.get(5)?,
        verified_at: row.get(6)?,
        state,
        operation_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelRecord;
    use crate::operation::{OperationDetails, OperationLog, OperationType};
    use std::time::SystemTime;

    fn test_state() -> PestiState {
        let tmp = tempfile::tempdir().unwrap();
        PestiState::open(tmp.path().join("test.db")).unwrap()
    }

    fn test_state_with_models() -> (PestiState, Vec<String>) {
        let state = test_state();
        let mut ids = Vec::new();
        for i in 1..=3 {
            let id = format!("model-{i}");
            state.register_model(&make_model(&id)).unwrap();
            ids.push(id);
        }
        (state, ids)
    }

    fn make_model(id: &str) -> ModelRecord {
        ModelRecord {
            id: id.to_string(),
            path: format!("/tmp/test-{id}.gguf"),
            architecture: "llama".to_string(),
            quantization: "Q4_K_M".to_string(),
            param_count: Some(1_000_000),
            tensor_count: 10,
            first_seen: None,
            last_verified: None,
        }
    }

    fn make_operation(
        session_id: &str,
        model_id: Option<&str>,
        op_type: OperationType,
    ) -> OperationLog {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        OperationLog {
            id: None,
            model_id: model_id.map(String::from),
            session_id: session_id.to_string(),
            op_type,
            details: match op_type {
                OperationType::Open => OperationDetails::Open {
                    session_id: session_id.to_string(),
                    model_path: "/tmp/test.gguf".to_string(),
                    tensor_count: 10,
                },
                OperationType::Edit => OperationDetails::Edit {
                    edit_count: 1,
                    tensors_modified: vec!["test".to_string()],
                    snapshot_taken: false,
                },
                OperationType::Snapshot => OperationDetails::Snapshot { snapshot_index: 0 },
                OperationType::Rollback => OperationDetails::Rollback { from_snapshot: 0 },
                OperationType::Save => OperationDetails::Save {
                    output_path: "/tmp/out.gguf".to_string(),
                    format: "gguf".to_string(),
                    adapter_id: None,
                },
                OperationType::Close => OperationDetails::Close {
                    total_edits: 0,
                    snapshots_created: 0,
                    rollbacks_performed: 0,
                    duration_secs: now,
                },
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[test]
    fn test_open_creates_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("subdir").join("state.db");
        let state = PestiState::open(&path).unwrap();
        assert!(path.exists());

        // Schema should be at version 1
        let conn = state.conn.lock().unwrap();
        let version: u32 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1);
    }

    #[test]
    fn test_register_model_inserts_new() {
        let state = test_state();
        let model = make_model("model-1");
        let id = state.register_model(&model).unwrap();
        assert_eq!(id, "model-1");

        let fetched = state.get_model("model-1").unwrap().unwrap();
        assert_eq!(fetched.id, "model-1");
        assert_eq!(fetched.architecture, "llama");
        assert_eq!(fetched.tensor_count, 10);
    }

    #[test]
    fn test_register_model_updates_existing() {
        let state = test_state();
        let model = make_model("model-1");
        state.register_model(&model).unwrap();

        // Re-register with same path — should update last_verified
        let model2 = ModelRecord {
            last_verified: None,
            ..make_model("model-1")
        };
        state.register_model(&model2).unwrap();

        let fetched = state.get_model("model-1").unwrap().unwrap();
        assert!(fetched.last_verified.is_some());
    }

    #[test]
    fn test_get_model_not_found() {
        let state = test_state();
        assert!(state.get_model("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_list_models_empty() {
        let state = test_state();
        assert!(state.list_models().unwrap().is_empty());
    }

    #[test]
    fn test_list_models_returns_all() {
        let state = test_state();
        state.register_model(&make_model("model-1")).unwrap();
        state.register_model(&make_model("model-2")).unwrap();
        assert_eq!(state.list_models().unwrap().len(), 2);
    }

    #[test]
    fn test_log_operation_assigns_id() {
        let (state, _) = test_state_with_models();
        let op = make_operation("sess-1", Some("model-1"), OperationType::Open);
        let id = state.log_operation(&op).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_session_operations_chronological() {
        let (state, _) = test_state_with_models();
        let op1 = make_operation("sess-1", Some("model-1"), OperationType::Open);
        let op2 = make_operation("sess-1", Some("model-1"), OperationType::Edit);
        let op3 = make_operation("sess-1", Some("model-1"), OperationType::Close);

        state.log_operation(&op1).unwrap();
        state.log_operation(&op2).unwrap();
        state.log_operation(&op3).unwrap();

        let ops = state.session_operations("sess-1").unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].op_type, OperationType::Open);
        assert_eq!(ops[1].op_type, OperationType::Edit);
        assert_eq!(ops[2].op_type, OperationType::Close);

        // Verify chronological order by ID
        for i in 1..ops.len() {
            assert!(ops[i].id.unwrap() > ops[i - 1].id.unwrap());
        }
    }

    #[test]
    fn test_model_operations_across_sessions() {
        let (state, _) = test_state_with_models();
        let op1 = make_operation("sess-1", Some("model-1"), OperationType::Open);
        let op2 = make_operation("sess-2", Some("model-1"), OperationType::Edit);
        let op3 = make_operation("sess-3", Some("model-2"), OperationType::Open);

        state.log_operation(&op1).unwrap();
        state.log_operation(&op2).unwrap();
        state.log_operation(&op3).unwrap();

        let ops = state.model_operations("model-1").unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].session_id, "sess-1");
        assert_eq!(ops[1].session_id, "sess-2");
    }

    #[test]
    fn test_register_and_get_adapter() {
        let (state, _) = test_state_with_models();
        let adapter = AdapterRecord {
            id: "adapter-1".to_string(),
            base_model_id: "model-1".to_string(),
            output_path: "/tmp/adapter.gguf".to_string(),
            description: Some("test adapter".to_string()),
            assumptions: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            verified_at: None,
            state: AdapterState::Unverified,
            operation_ids: vec![1, 2, 3],
        };

        let id = state.register_adapter(adapter).unwrap();
        assert_eq!(id, "adapter-1");

        let fetched = state.get_adapter("adapter-1").unwrap().unwrap();
        assert_eq!(fetched.base_model_id, "model-1");
        assert_eq!(fetched.state, AdapterState::Unverified);
        assert_eq!(fetched.operation_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_update_adapter_state() {
        let (state, _) = test_state_with_models();
        let adapter = AdapterRecord {
            id: "adapter-1".to_string(),
            base_model_id: "model-1".to_string(),
            output_path: "/tmp/adapter.gguf".to_string(),
            description: None,
            assumptions: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            verified_at: None,
            state: AdapterState::Unverified,
            operation_ids: vec![],
        };
        state.register_adapter(adapter).unwrap();

        state
            .update_adapter_state("adapter-1", AdapterState::Validated)
            .unwrap();
        let fetched = state.get_adapter("adapter-1").unwrap().unwrap();
        assert_eq!(fetched.state, AdapterState::Validated);
        assert!(fetched.verified_at.is_some());
    }

    #[test]
    fn test_update_adapter_state_not_found() {
        let state = test_state();
        let result = state.update_adapter_state("nonexistent", AdapterState::Validated);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalidate_adapters_for_model() {
        let (state, _) = test_state_with_models();
        for i in 1..=3 {
            let adapter = AdapterRecord {
                id: format!("adapter-{i}"),
                base_model_id: "model-1".to_string(),
                output_path: format!("/tmp/adapter-{i}.gguf"),
                description: None,
                assumptions: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                verified_at: None,
                state: AdapterState::Active,
                operation_ids: vec![],
            };
            state.register_adapter(adapter).unwrap();
        }

        // Add one adapter for a different model (should not be affected)
        let other = AdapterRecord {
            id: "adapter-other".to_string(),
            base_model_id: "model-2".to_string(),
            output_path: "/tmp/adapter-other.gguf".to_string(),
            description: None,
            assumptions: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            verified_at: None,
            state: AdapterState::Active,
            operation_ids: vec![],
        };
        state.register_adapter(other).unwrap();

        let count = state.invalidate_adapters_for_model("model-1").unwrap();
        assert_eq!(count, 3);

        let needs_reval = state.needs_revalidation().unwrap();
        assert_eq!(needs_reval.len(), 3);
        assert!(needs_reval.iter().all(|a| a.base_model_id == "model-1"));
    }

    #[test]
    fn test_invalidate_skips_deprecated() {
        let (state, _) = test_state_with_models();
        let adapter = AdapterRecord {
            id: "adapter-1".to_string(),
            base_model_id: "model-1".to_string(),
            output_path: "/tmp/adapter.gguf".to_string(),
            description: None,
            assumptions: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            verified_at: None,
            state: AdapterState::Deprecated,
            operation_ids: vec![],
        };
        state.register_adapter(adapter).unwrap();

        let count = state.invalidate_adapters_for_model("model-1").unwrap();
        assert_eq!(count, 0); // Deprecated adapters are not re-flagged
    }

    #[test]
    fn test_active_adapters_for_model() {
        let (state, _) = test_state_with_models();
        for (id, state_val) in [
            ("a1", AdapterState::Active),
            ("a2", AdapterState::Validated),
            ("a3", AdapterState::Unverified),
            ("a4", AdapterState::NeedsRevalidation),
        ] {
            let adapter = AdapterRecord {
                id: id.to_string(),
                base_model_id: "model-1".to_string(),
                output_path: format!("/tmp/{id}.gguf"),
                description: None,
                assumptions: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                verified_at: None,
                state: state_val,
                operation_ids: vec![],
            };
            state.register_adapter(adapter).unwrap();
        }

        let active = state.active_adapters_for_model("model-1").unwrap();
        assert_eq!(active.len(), 2); // Only Active and Validated
        let ids: Vec<&str> = active.iter().map(|a| a.id.as_str()).collect();
        assert!(ids.contains(&"a1"));
        assert!(ids.contains(&"a2"));
    }

    #[test]
    fn test_list_adapters_by_state() {
        let (state, _) = test_state_with_models();
        for (id, state_val) in [
            ("a1", AdapterState::Active),
            ("a2", AdapterState::Validated),
            ("a3", AdapterState::Unverified),
        ] {
            let adapter = AdapterRecord {
                id: id.to_string(),
                base_model_id: "model-1".to_string(),
                output_path: format!("/tmp/{id}.gguf"),
                description: None,
                assumptions: None,
                created_at: chrono::Utc::now().to_rfc3339(),
                verified_at: None,
                state: state_val,
                operation_ids: vec![],
            };
            state.register_adapter(adapter).unwrap();
        }

        let active = state.list_adapters(Some(AdapterState::Active)).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "a1");

        let all = state.list_adapters(None).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_state_is_cloneable_and_shared() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("shared.db");
        let state1 = PestiState::open(&path).unwrap();
        let state2 = state1.clone();

        // Both should operate on the same database
        state1.register_model(&make_model("model-1")).unwrap();
        let fetched = state2.get_model("model-1").unwrap();
        assert!(fetched.is_some());
    }
}
