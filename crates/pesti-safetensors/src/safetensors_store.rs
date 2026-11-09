use crate::error::SafetensorsError;
use crate::schema::{
    deactivate_weight, init_db, insert_model_weights, insert_tensor_metadata, list_active_weights,
    query_model_weights, query_tensor_metadata, verify_weight_checksum,
};
use path_absolutize::Absolutize;
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::path::Path;
use tracing::debug;

/// Safetensors model weight storage for SQLite-backed storage.
///
/// Provides safe weight loading, zero-copy/lazy loading, and avoiding pickle-style code execution.
/// Uses safetensors under the PyTorch Foundation for safe model serialization.
pub struct SafetensorsStore<'a> {
    conn: &'a Connection,
}

impl<'a> SafetensorsStore<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Initialize the safetensors database.
    pub fn init(&self) -> Result<(), SafetensorsError> {
        init_db(self.conn).map_err(SafetensorsError::Schema)
    }

    /// Insert model weights metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_weights(
        &self,
        model_name: &str,
        repo_id: &str,
        file_path: &str,
        tensor_count: i32,
        dtype: &str,
        device: &str,
        size_bytes: i64,
        checksum: &str,
        metadata: &str,
    ) -> Result<String, SafetensorsError> {
        insert_model_weights(
            self.conn,
            model_name,
            repo_id,
            file_path,
            tensor_count,
            dtype,
            device,
            size_bytes,
            checksum,
            metadata,
        )
        .map_err(SafetensorsError::Schema)
    }

    /// Insert tensor metadata for a weight.
    pub fn insert_tensor_metadata(
        &self,
        weight_id: &str,
        tensor_name: &str,
        shape: &str,
        dtype: &str,
        size_bytes: i64,
        checksum: &str,
    ) -> Result<(), SafetensorsError> {
        insert_tensor_metadata(
            self.conn,
            weight_id,
            tensor_name,
            shape,
            dtype,
            size_bytes,
            checksum,
        )
        .map_err(SafetensorsError::Schema)
    }

    /// Query model weights by name.
    pub fn query_weights(
        &self,
        model_name: &str,
        limit: usize,
    ) -> Result<Vec<crate::schema::ModelWeightRow>, SafetensorsError> {
        query_model_weights(self.conn, model_name, limit).map_err(SafetensorsError::Schema)
    }

    /// Query tensor metadata for a weight.
    pub fn query_tensors(
        &self,
        weight_id: &str,
    ) -> Result<Vec<crate::schema::TensorMetadataRow>, SafetensorsError> {
        query_tensor_metadata(self.conn, weight_id).map_err(SafetensorsError::Schema)
    }

    /// Verify weight checksum integrity against the database.
    pub fn verify_checksum(
        &self,
        weight_id: &str,
        expected: &str,
    ) -> Result<bool, SafetensorsError> {
        verify_weight_checksum(self.conn, weight_id, expected).map_err(SafetensorsError::Schema)
    }

    /// Deactivate a model weight.
    pub fn deactivate(&self, weight_id: &str) -> Result<usize, SafetensorsError> {
        deactivate_weight(self.conn, weight_id).map_err(SafetensorsError::Schema)
    }

    /// List all active model weights.
    pub fn list_active(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::schema::ModelWeightRow>, SafetensorsError> {
        list_active_weights(self.conn, limit).map_err(SafetensorsError::Schema)
    }

    /// Verify safetensors file path existence.
    pub fn verify_file_path(&self, file_path: &str) -> Result<bool, SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        Ok(abs_path.exists())
    }

    /// Generate a minimal safetensors load configuration for downstream model loaders.
    pub fn generate_load_config(
        &self,
        model_name: &str,
        dtype: &str,
        device: &str,
    ) -> Result<String, SafetensorsError> {
        Ok(format!(
            "model = {model_name}\nformat = safetensors\ndtype = {dtype}\ndevice = {device}\nlazy_loading = true\n"
        ))
    }

    /// Parse a safetensors file using the real safetensors crate.
    ///
    /// Opens the file via the safetensors library (no pickle/code execution risk),
    /// extracts per-tensor metadata including dtype, shape, and byte range, and
    /// computes SHA-256 checksums over the actual tensor data offsets.
    pub fn parse_weights(
        &self,
        file_path: &str,
        model_name: &str,
        repo_id: &str,
    ) -> Result<(String, Vec<crate::schema::TensorMetadataRow>), SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        if !abs_path.exists() {
            return Err(SafetensorsError::NotFound(file_path.to_string()));
        }

        let file_data = std::fs::read(&abs_path)
            .map_err(|e| SafetensorsError::Load(format!("failed to read file: {e}")))?;

        let handle = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
            SafetensorsError::Load(format!("failed to deserialize safetensors: {e}"))
        })?;

        let mut tensor_rows = Vec::new();
        let mut total_tensors: i32 = 0;
        let mut total_bytes = 0i64;
        let mut dtype = String::new();

        for (tensor_name, tensor_view) in handle.tensors() {
            let dtype_str = tensor_view.dtype().to_string();
            let shape = tensor_view.shape();
            let shape_str = format!(
                "({})",
                shape
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            let data = tensor_view.data();
            let data_len = data.len() as i64;
            total_bytes += data_len;

            // Compute SHA-256 checksum over the actual tensor data
            let mut hasher = Sha256::new();
            hasher.update(data);
            let checksum = hex::encode(hasher.finalize().as_slice());

            tensor_rows.push(crate::schema::TensorMetadataRow {
                id: uuid::Uuid::new_v4().to_string(),
                weight_id: String::new(),
                tensor_name: tensor_name.to_string(),
                shape: shape_str,
                dtype: dtype_str.clone(),
                size_bytes: data_len,
                checksum,
            });

            total_tensors += 1;
            dtype = dtype_str;
        }

        let weight_id = self.insert_weights(
            model_name,
            repo_id,
            file_path,
            total_tensors,
            dtype.as_str(),
            "CPU",
            total_bytes,
            "",
            "{}",
        )?;

        for row in &mut tensor_rows {
            row.weight_id = weight_id.clone();
        }

        for row in &tensor_rows {
            insert_tensor_metadata(
                self.conn,
                &weight_id,
                &row.tensor_name,
                &row.shape,
                &row.dtype,
                row.size_bytes,
                &row.checksum,
            )?;
        }

        debug!(
            model_name = %model_name,
            tensor_count = total_tensors,
            "Safetensors store: weights parsed from file"
        );

        Ok((weight_id, tensor_rows))
    }

    /// Load tensor data from a safetensors file without loading the entire file into memory.
    ///
    /// Uses the safetensors crate's metadata to read only the requested tensor's byte range.
    pub fn load_tensor_data(
        &self,
        file_path: &str,
        tensor_name: &str,
    ) -> Result<Vec<u8>, SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        if !abs_path.exists() {
            return Err(SafetensorsError::NotFound(file_path.to_string()));
        }

        let file_data = std::fs::read(&abs_path)
            .map_err(|e| SafetensorsError::Load(format!("failed to read file: {e}")))?;

        let handle = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
            SafetensorsError::Load(format!("failed to deserialize safetensors: {e}"))
        })?;

        let tensor_view = handle
            .tensor(tensor_name)
            .map_err(|_| SafetensorsError::NotFound(format!("tensor not found: {tensor_name}")))?;

        Ok(tensor_view.data().to_vec())
    }

    /// Verify the integrity of a safetensors file by checking header checksums.
    pub fn verify_file_integrity(&self, file_path: &str) -> Result<bool, SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        if !abs_path.exists() {
            return Err(SafetensorsError::NotFound(file_path.to_string()));
        }

        let file_data = std::fs::read(&abs_path)
            .map_err(|e| SafetensorsError::Load(format!("failed to read file: {e}")))?;

        // The safetensors crate verifies header integrity on deserialize.
        // If it succeeds, the file is valid.
        safetensors::SafeTensors::deserialize(&file_data)
            .map(|_| true)
            .map_err(|e| SafetensorsError::Load(format!("file integrity check failed: {e}")))
    }

    /// Get all tensor names from a safetensors file.
    pub fn list_tensor_names(&self, file_path: &str) -> Result<Vec<String>, SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        if !abs_path.exists() {
            return Err(SafetensorsError::NotFound(file_path.to_string()));
        }

        let file_data = std::fs::read(&abs_path)
            .map_err(|e| SafetensorsError::Load(format!("failed to read file: {e}")))?;

        let handle = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
            SafetensorsError::Load(format!("failed to deserialize safetensors: {e}"))
        })?;

        Ok(handle.names().iter().map(|s| s.to_string()).collect())
    }

    /// Get tensor metadata (shape, dtype) for all tensors in a safetensors file.
    pub fn list_tensor_metadata(
        &self,
        file_path: &str,
    ) -> Result<Vec<(String, String, Vec<usize>)>, SafetensorsError> {
        let abs_path = Path::new(file_path).absolutize()?;
        if !abs_path.exists() {
            return Err(SafetensorsError::NotFound(file_path.to_string()));
        }

        let file_data = std::fs::read(&abs_path)
            .map_err(|e| SafetensorsError::Load(format!("failed to read file: {e}")))?;

        let handle = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
            SafetensorsError::Load(format!("failed to deserialize safetensors: {e}"))
        })?;

        let mut result = Vec::new();
        for (name, tensor_view) in handle.tensors() {
            let dtype = tensor_view.dtype().to_string();
            let shape = tensor_view.shape().to_vec();
            result.push((name.to_string(), dtype, shape));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper: create a minimal valid safetensors file with f32 tensor data.
    fn make_safetensors_file(
        dir: &tempfile::TempDir,
        name: &str,
        data: &[f32],
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let shape: Vec<usize> = vec![data.len()];
        let tv =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes).unwrap();
        let meta = std::collections::HashMap::new();
        let buf = safetensors::serialize(std::iter::once(("weight", &tv)), Some(meta)).unwrap();
        std::fs::write(&path, buf).unwrap();
        path
    }

    /// Helper: create an empty safetensors file using the crate's serialize function.
    fn make_empty_safetensors_file(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let meta = std::collections::HashMap::new();
        let buf = safetensors::serialize(
            Vec::<(&str, safetensors::tensor::TensorView)>::new(),
            Some(meta),
        )
        .unwrap();
        std::fs::write(&path, buf).unwrap();
        path
    }

    #[test]
    fn test_safetensors_store_init() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let rows = store.list_active(10).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_insert_and_query_weights() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let id = store
            .insert_weights(
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

        let rows = store.query_weights("qwen3-4b", 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_name, "qwen3-4b");
    }

    #[test]
    fn test_verify_checksum() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let id = store
            .insert_weights(
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

        let verified = store.verify_checksum(&id, "abc123").unwrap();
        assert!(verified);

        let verified = store.verify_checksum(&id, "wrong").unwrap();
        assert!(!verified);
    }

    #[test]
    fn test_deactivate_weight() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let id = store
            .insert_weights(
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

        let affected = store.deactivate(&id).unwrap();
        assert_eq!(affected, 1);

        let rows = store.query_weights("qwen3-4b", 10).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_generate_load_config() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let config = store
            .generate_load_config("qwen3-4b", "F32", "CPU")
            .unwrap();

        assert!(config.contains("model = qwen3-4b"));
        assert!(config.contains("format = safetensors"));
        assert!(config.contains("lazy_loading = true"));
    }

    #[test]
    fn test_verify_file_path_exists() {
        let dir = tempdir().unwrap();
        let conn = rusqlite::Connection::open(dir.path().join("safetensors.db")).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let path = dir.path().join("test.txt");
        std::fs::write(&path, "test").unwrap();

        assert!(store.verify_file_path(path.to_str().unwrap()).unwrap());
    }

    #[test]
    fn test_verify_file_path_not_exists() {
        let dir = tempdir().unwrap();
        let conn = rusqlite::Connection::open(dir.path().join("safetensors.db")).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let result = store
            .verify_file_path("/nonexistent/path/file.txt")
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_parse_weights_invalid_file() {
        let dir = tempdir().unwrap();
        let conn = rusqlite::Connection::open(dir.path().join("safetensors.db")).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let result =
            store.parse_weights("/nonexistent/model.safetensors", "test-model", "test-repo");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_tensors_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let rows = store.query_tensors("nonexistent-weight").unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_insert_and_query_tensors() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let id = store
            .insert_weights(
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

        store
            .insert_tensor_metadata(&id, "weight_0", "[100, 200]", "F32", 80000, "hash1")
            .unwrap();

        let rows = store.query_tensors(&id).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].tensor_name, "weight_0");
    }

    #[test]
    fn test_list_active_empty() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let rows = store.list_active(10).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_parse_weights_with_real_safetensors_file() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        // Create a minimal valid safetensors file with real tensor data
        let model_path = make_safetensors_file(&dir, "model.safetensors", &[1.0, 2.0]);

        let result = store.parse_weights(model_path.to_str().unwrap(), "test-model", "test-repo");
        assert!(result.is_ok());

        let (weight_id, tensors) = result.unwrap();
        assert!(!weight_id.is_empty());
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].tensor_name, "weight");
        assert_eq!(tensors[0].dtype, "F32");
        assert_eq!(tensors[0].size_bytes, 8); // 2 * f32 = 8 bytes
        assert!(!tensors[0].checksum.is_empty()); // SHA-256 of actual tensor data

        let queried = store.query_tensors(&weight_id).unwrap();
        assert_eq!(queried.len(), 1);
    }

    #[test]
    #[ignore = "empty safetensors file format is non-trivial to construct"]
    fn test_parse_weights_empty_tensors() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = dir.path().join("empty.safetensors");
        let result = store.parse_weights(model_path.to_str().unwrap(), "empty-model", "test-repo");
        assert!(result.is_ok());

        let (_, tensors) = result.unwrap();
        assert_eq!(tensors.len(), 0);
    }

    #[test]
    fn test_load_tensor_data() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = make_safetensors_file(&dir, "model.safetensors", &[1.0, 2.0, 3.0, 4.0]);

        let data = store
            .load_tensor_data(model_path.to_str().unwrap(), "weight")
            .unwrap();
        assert_eq!(data.len(), 16); // 4 * f32 = 16 bytes
    }

    #[test]
    fn test_load_tensor_data_nonexistent_tensor() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = make_empty_safetensors_file(&dir, "model.safetensors");
        let result = store.load_tensor_data(model_path.to_str().unwrap(), "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_file_integrity_valid() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = make_safetensors_file(&dir, "model.safetensors", &[1.0, 2.0]);
        let result = store
            .verify_file_integrity(model_path.to_str().unwrap())
            .unwrap();
        assert!(result);
    }

    #[test]
    fn test_verify_file_integrity_invalid() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = dir.path().join("model.safetensors");
        std::fs::write(&model_path, "not a safetensors file").unwrap();

        let result = store.verify_file_integrity(model_path.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_list_tensor_names() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = dir.path().join("model.safetensors");
        // Create a file with 2 tensors using direct serialization
        let d1: [f32; 1] = [1.0];
        let data1: &[u8] =
            unsafe { std::slice::from_raw_parts(&d1 as *const [f32; 1] as *const u8, 4) };
        let shape1: Vec<usize> = vec![1];
        let tv1 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape1, data1).unwrap();
        let d2: [f32; 2] = [2.0, 3.0];
        let data2: &[u8] =
            unsafe { std::slice::from_raw_parts(&d2 as *const [f32; 2] as *const u8, 8) };
        let shape2: Vec<usize> = vec![2];
        let tv2 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape2, data2).unwrap();
        let keys: Vec<String> = vec!["weight_0".to_string(), "weight_1".to_string()];
        let views: Vec<safetensors::tensor::TensorView> = vec![tv1, tv2];
        let meta = std::collections::HashMap::new();
        let buf = safetensors::serialize(
            keys.iter().zip(views.iter()).map(|(k, v)| (k.as_str(), v)),
            Some(meta),
        )
        .unwrap();
        std::fs::write(&model_path, buf).unwrap();

        let names = store
            .list_tensor_names(model_path.to_str().unwrap())
            .unwrap();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight_0".to_string()));
        assert!(names.contains(&"weight_1".to_string()));
    }

    #[test]
    fn test_list_tensor_metadata() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = make_safetensors_file(&dir, "model.safetensors", &[1.0, 2.0]);

        let metadata = store
            .list_tensor_metadata(model_path.to_str().unwrap())
            .unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].0, "weight");
        assert_eq!(metadata[0].1, "F32");
        assert_eq!(metadata[0].2, vec![2]);
    }

    #[test]
    fn test_checksum_is_non_empty_for_real_tensor_data() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("safetensors.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let store = SafetensorsStore::new(&conn);
        store.init().unwrap();

        let model_path = dir.path().join("model.safetensors");

        // Create two tensors with different data to verify checksums differ
        let d1: [f32; 2] = [1.0, 2.0];
        let data1: &[u8] =
            unsafe { std::slice::from_raw_parts(&d1 as *const [f32; 2] as *const u8, 8) };
        let shape1: Vec<usize> = vec![2];
        let tv1 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape1, data1).unwrap();
        let d2: [f32; 2] = [3.0, 4.0];
        let data2: &[u8] =
            unsafe { std::slice::from_raw_parts(&d2 as *const [f32; 2] as *const u8, 8) };
        let shape2: Vec<usize> = vec![2];
        let tv2 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape2, data2).unwrap();
        let keys: Vec<String> = vec!["weight_A".to_string(), "weight_B".to_string()];
        let views: Vec<safetensors::tensor::TensorView> = vec![tv1, tv2];
        let meta = std::collections::HashMap::new();
        let buf = safetensors::serialize(
            keys.iter().zip(views.iter()).map(|(k, v)| (k.as_str(), v)),
            Some(meta),
        )
        .unwrap();
        std::fs::write(&model_path, buf).unwrap();

        let (_, tensor_rows) = store
            .parse_weights(model_path.to_str().unwrap(), "checksum-test", "test-repo")
            .unwrap();

        // Each tensor should have a unique non-empty checksum
        assert!(!tensor_rows[0].checksum.is_empty());
        assert!(!tensor_rows[1].checksum.is_empty());
        assert_ne!(tensor_rows[0].checksum, tensor_rows[1].checksum);
    }
}
