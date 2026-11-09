use crate::error::SafetensorsSchemaError;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

/// Safetensors model weight storage DDL schema.
pub const SAFETENSORS_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS model_weights (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    repo_id TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    tensor_count INTEGER NOT NULL DEFAULT 0,
    dtype TEXT NOT NULL DEFAULT '',
    device TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    loaded_at INTEGER NOT NULL DEFAULT (unixepoch()),
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_weights_model ON model_weights(model_name);
CREATE INDEX IF NOT EXISTS idx_weights_repo ON model_weights(repo_id);
CREATE INDEX IF NOT EXISTS idx_weights_time ON model_weights(loaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_weights_active ON model_weights(active);

CREATE TABLE IF NOT EXISTS tensor_metadata (
    id TEXT PRIMARY KEY,
    weight_id TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    shape TEXT NOT NULL DEFAULT '',
    dtype TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (weight_id) REFERENCES model_weights(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tensor_weight ON tensor_metadata(weight_id);
CREATE INDEX IF NOT EXISTS idx_tensor_name ON tensor_metadata(tensor_name);

CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied TEXT NOT NULL DEFAULT (datetime('now')),
    note TEXT
);
"#;

pub fn init_db(conn: &Connection) -> Result<(), SafetensorsSchemaError> {
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    conn.execute_batch(SAFETENSORS_SCHEMA)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn insert_model_weights(
    conn: &Connection,
    model_name: &str,
    repo_id: &str,
    file_path: &str,
    tensor_count: i32,
    dtype: &str,
    device: &str,
    size_bytes: i64,
    checksum: &str,
    metadata: &str,
) -> Result<String, SafetensorsSchemaError> {
    let id = uuid::Uuid::new_v4().to_string();

    conn.execute(
        "INSERT INTO model_weights (id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            id,
            model_name,
            repo_id,
            file_path,
            tensor_count,
            dtype,
            device,
            size_bytes,
            checksum,
            metadata,
            chrono::Utc::now().timestamp(),
            chrono::Utc::now().timestamp(),
            1,
        ],
    )?;

    Ok(id)
}

pub fn insert_tensor_metadata(
    conn: &Connection,
    weight_id: &str,
    tensor_name: &str,
    shape: &str,
    dtype: &str,
    size_bytes: i64,
    checksum: &str,
) -> Result<(), SafetensorsSchemaError> {
    let id = uuid::Uuid::new_v4().to_string();

    conn.execute(
        "INSERT INTO tensor_metadata (id, weight_id, tensor_name, shape, dtype, size_bytes, checksum)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            id,
            weight_id,
            tensor_name,
            shape,
            dtype,
            size_bytes,
            checksum,
        ],
    )?;

    Ok(())
}

pub fn query_model_weights(
    conn: &Connection,
    model_name: &str,
    limit: usize,
) -> Result<Vec<ModelWeightRow>, SafetensorsSchemaError> {
    let mut stmt = conn.prepare(
        "SELECT id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active FROM model_weights
         WHERE model_name = ?1 AND active = 1
         ORDER BY loaded_at DESC LIMIT ?2",
    )?;

    let rows = stmt.query_map(params![model_name, limit as i64], |row| {
        let metadata_str: String = row.get(9)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap_or_default();

        Ok(ModelWeightRow {
            id: row.get(0)?,
            model_name: row.get(1)?,
            repo_id: row.get(2)?,
            file_path: row.get(3)?,
            tensor_count: row.get(4)?,
            dtype: row.get(5)?,
            device: row.get(6)?,
            size_bytes: row.get(7)?,
            checksum: row.get(8)?,
            metadata,
            loaded_at: row.get(10)?,
            created_at: row.get(11)?,
            active: row.get(12)?,
        })
    })?;

    let results = rows.collect::<Result<Vec<_>, _>>()?;

    Ok(results)
}

pub fn query_tensor_metadata(
    conn: &Connection,
    weight_id: &str,
) -> Result<Vec<TensorMetadataRow>, SafetensorsSchemaError> {
    let mut stmt = conn.prepare(
        "SELECT id, weight_id, tensor_name, shape, dtype, size_bytes, checksum FROM tensor_metadata
         WHERE weight_id = ?1",
    )?;

    let rows = stmt.query_map(params![weight_id], |row| {
        Ok(TensorMetadataRow {
            id: row.get(0)?,
            weight_id: row.get(1)?,
            tensor_name: row.get(2)?,
            shape: row.get(3)?,
            dtype: row.get(4)?,
            size_bytes: row.get(5)?,
            checksum: row.get(6)?,
        })
    })?;

    let results = rows.collect::<Result<Vec<_>, _>>()?;

    Ok(results)
}

pub fn verify_weight_checksum(
    conn: &Connection,
    weight_id: &str,
    expected_checksum: &str,
) -> Result<bool, SafetensorsSchemaError> {
    let mut stmt =
        conn.prepare("SELECT checksum FROM model_weights WHERE id = ?1 AND active = 1 LIMIT 1")?;

    let stored = stmt.query_row(params![weight_id], |row| row.get::<_, String>(0));

    match stored {
        Ok(s) => Ok(s == expected_checksum),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
        Err(err) => Err(SafetensorsSchemaError::Sqlite(err)),
    }
}

pub fn deactivate_weight(
    conn: &Connection,
    weight_id: &str,
) -> Result<usize, SafetensorsSchemaError> {
    let affected = conn.execute(
        "UPDATE model_weights SET active = 0 WHERE id = ?1",
        params![weight_id],
    )?;

    Ok(affected)
}

pub fn list_active_weights(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<ModelWeightRow>, SafetensorsSchemaError> {
    let mut stmt = conn.prepare(
        "SELECT id, model_name, repo_id, file_path, tensor_count, dtype, device, size_bytes, checksum, metadata, loaded_at, created_at, active FROM model_weights
         WHERE active = 1
         ORDER BY loaded_at DESC LIMIT ?1",
    )?;

    let rows = stmt.query_map(params![limit as i64], |row| {
        let metadata_str: String = row.get(9)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str).unwrap_or_default();

        Ok(ModelWeightRow {
            id: row.get(0)?,
            model_name: row.get(1)?,
            repo_id: row.get(2)?,
            file_path: row.get(3)?,
            tensor_count: row.get(4)?,
            dtype: row.get(5)?,
            device: row.get(6)?,
            size_bytes: row.get(7)?,
            checksum: row.get(8)?,
            metadata,
            loaded_at: row.get(10)?,
            created_at: row.get(11)?,
            active: row.get(12)?,
        })
    })?;

    let results = rows.collect::<Result<Vec<_>, _>>()?;

    Ok(results)
}

/// A single model weight row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeightRow {
    pub id: String,
    pub model_name: String,
    pub repo_id: String,
    pub file_path: String,
    pub tensor_count: i32,
    pub dtype: String,
    pub device: String,
    pub size_bytes: i64,
    pub checksum: String,
    pub metadata: serde_json::Value,
    pub loaded_at: i64,
    pub created_at: i64,
    pub active: i32,
}

/// A single tensor metadata row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadataRow {
    pub id: String,
    pub weight_id: String,
    pub tensor_name: String,
    pub shape: String,
    pub dtype: String,
    pub size_bytes: i64,
    pub checksum: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_init_db_creates_tables() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();
        assert!(db_path.exists());

        let count: i64 = conn
            .query_row("SELECT count(*) FROM model_weights", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_insert_model_weights() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();

        let id = insert_model_weights(
            &conn,
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "/tmp/model.safetensors",
            150,
            "F32",
            "CPU",
            2000000000,
            "abc123",
            "{}",
        )
        .unwrap();

        assert!(!id.is_empty());

        let rows = query_model_weights(&conn, "qwen3-4b", 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_name, "qwen3-4b");
        assert_eq!(rows[0].tensor_count, 150);
    }

    #[test]
    fn test_insert_tensor_metadata() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();

        let weight_id = insert_model_weights(
            &conn,
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "/tmp/model.safetensors",
            150,
            "F32",
            "CPU",
            2000000000,
            "abc123",
            "{}",
        )
        .unwrap();

        insert_tensor_metadata(
            &conn,
            &weight_id,
            "weight_fc1",
            "(784, 256)",
            "F32",
            800000,
            "hash1",
        )
        .unwrap();

        let rows = query_tensor_metadata(&conn, &weight_id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].tensor_name, "weight_fc1");
        assert_eq!(rows[0].shape, "(784, 256)");
    }

    #[test]
    fn test_verify_checksum() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();

        let weight_id = insert_model_weights(
            &conn,
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "/tmp/model.safetensors",
            150,
            "F32",
            "CPU",
            2000000000,
            "abc123",
            "{}",
        )
        .unwrap();

        let verified = verify_weight_checksum(&conn, &weight_id, "abc123").unwrap();
        assert!(verified);

        let verified = verify_weight_checksum(&conn, &weight_id, "wrong_hash").unwrap();
        assert!(!verified);
    }

    #[test]
    fn test_list_active_weights() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();

        insert_model_weights(
            &conn,
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "/tmp/model.safetensors",
            150,
            "F32",
            "CPU",
            2000000000,
            "abc123",
            "{}",
        )
        .unwrap();

        insert_model_weights(
            &conn,
            "mistral-7b",
            "mistralai/Mistral-7B",
            "/tmp/model.safetensors",
            200,
            "F32",
            "CPU",
            3000000000,
            "def456",
            "{}",
        )
        .unwrap();

        let rows = list_active_weights(&conn, 10).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_deactivate_weight() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        init_db(&conn).unwrap();

        let weight_id = insert_model_weights(
            &conn,
            "qwen3-4b",
            "Qwen/Qwen3-4B",
            "/tmp/model.safetensors",
            150,
            "F32",
            "CPU",
            2000000000,
            "abc123",
            "{}",
        )
        .unwrap();

        let affected = deactivate_weight(&conn, &weight_id).unwrap();
        assert_eq!(affected, 1);

        let rows = query_model_weights(&conn, "qwen3-4b", 10).unwrap();
        assert_eq!(rows.len(), 0);
    }
}
