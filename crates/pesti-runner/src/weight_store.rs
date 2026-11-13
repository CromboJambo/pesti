//! Weight storage layering — Layer 0 (CoreTensor) and Layer 1 (TensorStructure).
//!
//! This module implements Phase 1 of TENSOR_LAYER_SEPARATION_SPEC:
//! - `CoreTensor`: name + raw bytes + shape + dtype (no layout interpretation)
//! - `TensorInfo`: per-tensor metadata (shape, dtype, byte offsets)
//! - `TensorStructure`: Layer 1 — the model's tensor graph as a lookup table
//! - `SafetensorsFile`: mmap wrapper with zero-copy tensor access
//! - `WeightStore`: minimal weight storage that loads Layers 0+1 only
//!
//! Design principles:
//! - Zero-copy: tensor data is a slice into the mmap'd file, not a copy
//! - No layout interpretation: CoreTensor doesn't know about row-major vs column-major
//! - No behavior change: existing `LlamaModel::load_safetensors()` continues to work
//! - Additive: this module can be adopted incrementally by consumers

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use tracing::debug;

use crate::error::{Result, RunnerError};

/// Data type of a tensor element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U8,
    I8,
    I16,
    I32,
    I64,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::I16 | Dtype::F16 => 2,
            Dtype::I32 => 4,
            Dtype::I64 => 8,
        }
    }

    /// Parse from safetensors dtype string.
    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F32" | "FLOAT_32" => Some(Dtype::F32),
            "F16" | "FLOAT_16" => Some(Dtype::F16),
            "BF16" | "BFLOAT_16" => Some(Dtype::BF16),
            "U8" | "UINT_8" => Some(Dtype::U8),
            "I8" | "INT_8" => Some(Dtype::I8),
            "I16" | "INT_16" => Some(Dtype::I16),
            "I32" | "INT_32" => Some(Dtype::I32),
            "I64" | "INT_64" => Some(Dtype::I64),
            _ => None,
        }
    }

    /// Convert to safetensors dtype string.
    pub fn to_safetensors(&self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
        }
    }
}

/// Per-tensor metadata (Layer 1).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: Dtype,
    /// Byte offset within the safetensors file (after header).
    pub byte_offset: usize,
    /// Total size in bytes.
    pub size_bytes: usize,
}

impl TensorInfo {
    /// Number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Verify that size_bytes matches shape × dtype size.
    pub fn is_consistent(&self) -> bool {
        self.size_bytes == self.element_count() * self.dtype.size()
    }
}

/// Layer 1: The model's tensor graph as a lookup table.
#[derive(Debug, Clone)]
pub struct TensorStructure {
    /// Tensors by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Architecture identifier (e.g. "llama", "qwen2").
    pub arch: String,
    /// Total number of parameters (sum of all element counts).
    pub total_parameters: usize,
}

impl TensorStructure {
    /// Get metadata for a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the structure is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Find all tensors matching a name prefix.
    pub fn with_prefix(&self, prefix: &str) -> Vec<&TensorInfo> {
        self.tensors.values().filter(|t| t.name.starts_with(prefix)).collect()
    }
}

/// Layer 0: A core tensor — name + raw bytes + shape + dtype.
///
/// No layout interpretation (row-major vs column-major is a Layer 2 concern).
/// Data is zero-copy from the mmap'd file.
#[derive(Debug)]
pub struct CoreTensor {
    /// Tensor name.
    pub name: String,
    /// Raw bytes (slice into mmap'd file — zero-copy).
    pub data: Arc<[u8]>,
    /// Shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: Dtype,
}

impl CoreTensor {
    /// Number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// View the data as f32 slices (only valid for F32 dtype).
    #[inline]
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != Dtype::F32 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, count) })
    }

    /// View the data as f16 slices (only valid for F16 dtype).
    #[inline]
    pub fn as_f16(&self) -> Option<&[half::f16]> {
        if self.dtype != Dtype::F16 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const half::f16, count) })
    }

    /// View the data as bf16 slices (only valid for BF16 dtype).
    #[inline]
    pub fn as_bf16(&self) -> Option<&[half::bf16]> {
        if self.dtype != Dtype::BF16 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const half::bf16, count) })
    }
}

/// Minimal interface for reading core tensors (Layer 2 contract).
pub trait TensorReader {
    /// Get a tensor by name. Returns None if not found.
    fn get_tensor(&self, name: &str) -> Option<CoreTensor>;

    /// Get tensor metadata without loading data.
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;

    /// List all tensor names.
    fn tensor_names(&self) -> Vec<String>;

    /// The model's tensor structure (Layer 1).
    fn structure(&self) -> &TensorStructure;
}

/// Memory-mapped safetensors file with zero-copy tensor access.
///
/// This is the Layer 0 storage backend. The file is mmap'd once, and tensor
/// data is accessed as slices into the mapping — no copying.
pub struct SafetensorsFile {
    /// Path to the safetensors file.
    pub path: PathBuf,
    /// Memory-mapped file contents (kept alive for the lifetime of this struct).
    mmap: Mmap,
    /// Layer 1: tensor structure parsed from the header.
    pub structure: TensorStructure,
}

impl SafetensorsFile {
    /// Open a safetensors file with mmap (zero-copy).
    ///
    /// Parses the header to build the TensorStructure. Tensor data is not
    /// loaded until accessed via `tensor_data()`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| RunnerError::ModelLoad(format!("Failed to open safetensors file: {e}")))?;

        // Memory-map the file (read-only).
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| RunnerError::ModelLoad(format!("Failed to mmap safetensors file: {e}")))?;

        // Parse the header. Safetensors format:
        // - u64 LE: header size
        // - JSON header (header_size bytes)
        // - tensor data
        if mmap.len() < 8 {
            return Err(RunnerError::ModelLoad(
                "Safetensors file too small".to_string(),
            ));
        }

        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if 8 + header_size > mmap.len() {
            return Err(RunnerError::ModelLoad(
                "Safetensors header size exceeds file size".to_string(),
            ));
        }

        let header_json = std::str::from_utf8(&mmap[8..8 + header_size])
            .map_err(|e| RunnerError::ModelLoad(format!("Invalid UTF-8 in header: {e}")))?;

        // Parse the JSON header. Format: {"__metadata__": {...}, "tensor_name": {"dtype": "...", "shape": [...], "data_offsets": [start, end]}, ...}
        let header: serde_json::Value = serde_json::from_str(header_json).map_err(|e| {
            RunnerError::ModelLoad(format!("Failed to parse safetensors header: {e}"))
        })?;

        // Extract metadata (architecture info).
        let arch = header
            .get("__metadata__")
            .and_then(|m| m.get("format"))
            .and_then(|f| f.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Build TensorStructure.
        let mut tensors = HashMap::new();
        let mut total_parameters = 0usize;

        for (name, value) in header.as_object().ok_or_else(|| {
            RunnerError::ModelLoad("Safetensors header is not an object".to_string())
        })? {
            if name == "__metadata__" {
                continue;
            }

            let dtype_str = value
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| RunnerError::ModelLoad(format!("Tensor '{name}' missing dtype")))?;

            let dtype = Dtype::from_safetensors(dtype_str).ok_or_else(|| {
                RunnerError::ModelLoad(format!(
                    "Unsupported dtype '{dtype_str}' for tensor '{name}'"
                ))
            })?;

            let shape: Vec<usize> = value
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| RunnerError::ModelLoad(format!("Tensor '{name}' missing shape")))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();

            let offsets = value
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("Tensor '{name}' missing data_offsets"))
                })?;

            let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_size;
            let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_size;

            if end > mmap.len() {
                return Err(RunnerError::ModelLoad(format!(
                    "Tensor '{name}' data extends beyond file size"
                )));
            }

            let size_bytes = end - start;
            let element_count: usize = shape.iter().product();

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    shape,
                    dtype,
                    byte_offset: start,
                    size_bytes,
                },
            );
            total_parameters += element_count;
        }

        let structure = TensorStructure {
            tensors,
            arch,
            total_parameters,
        };

        debug!(
            path = %path.display(),
            tensor_count = structure.len(),
            total_params = structure.total_parameters,
            "Opened safetensors file (mmap)"
        );

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            structure,
        })
    }

    /// Get zero-copy tensor data by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.structure.get(name)?;
        Some(&self.mmap[info.byte_offset..info.byte_offset + info.size_bytes])
    }

    /// Get a CoreTensor by name (zero-copy from mmap).
    pub fn core_tensor(&self, name: &str) -> Option<CoreTensor> {
        let info = self.structure.get(name)?.clone();
        let data = self.tensor_data(&info.name)?;
        Some(CoreTensor {
            name: info.name,
            data: Arc::from(data),
            shape: info.shape,
            dtype: info.dtype,
        })
    }

    /// Total file size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
}

impl TensorReader for SafetensorsFile {
    fn get_tensor(&self, name: &str) -> Option<CoreTensor> {
        self.core_tensor(name)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.structure.get(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.structure.names().cloned().collect()
    }

    fn structure(&self) -> &TensorStructure {
        &self.structure
    }
}

/// Minimal weight store — loads Layers 0+1 only (no architecture interpretation).
///
/// This is the entry point for Phase 1. Consumers can use this to:
/// - Inspect tensor shapes/dtypes without loading the full model
/// - Access raw tensor bytes via zero-copy mmap
/// - Verify tensor structure before committing to a full load
pub struct WeightStore {
    /// The underlying safetensors file (mmap'd).
    pub file: SafetensorsFile,
}

impl WeightStore {
    /// Load minimal weight store (Layers 0+1 only).
    ///
    /// This mmaps the file and parses the header. No tensor data is copied
    /// until accessed via `tensor_data()` or `core_tensor()`.
    pub fn load_minimal(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        Ok(Self { file })
    }

    /// Get zero-copy tensor data by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        self.file.tensor_data(name)
    }

    /// Get a CoreTensor by name (zero-copy from mmap).
    pub fn core_tensor(&self, name: &str) -> Option<CoreTensor> {
        self.file.core_tensor(name)
    }

    /// Get tensor metadata.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.file.tensor_info(name)
    }

    /// The model's tensor structure (Layer 1).
    pub fn structure(&self) -> &TensorStructure {
        &self.file.structure
    }

    /// Verify all tensors in the structure are consistent.
    pub fn verify_structure(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for info in self.file.structure.tensors.values() {
            if !info.is_consistent() {
                errors.push(format!(
                    "Tensor '{}' inconsistent: shape {:?} × dtype {:?} = {} bytes, but size_bytes = {}",
                    info.name,
                    info.shape,
                    info.dtype,
                    info.element_count() * info.dtype.size(),
                    info.size_bytes
                ));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::F32.size(), 4);
        assert_eq!(Dtype::F16.size(), 2);
        assert_eq!(Dtype::BF16.size(), 2);
        assert_eq!(Dtype::U8.size(), 1);
        assert_eq!(Dtype::I8.size(), 1);
        assert_eq!(Dtype::I16.size(), 2);
        assert_eq!(Dtype::I32.size(), 4);
        assert_eq!(Dtype::I64.size(), 8);
    }

    #[test]
    fn test_dtype_roundtrip() {
        for dtype in [
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::U8,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::I64,
        ] {
            let s = dtype.to_safetensors();
            let parsed = Dtype::from_safetensors(s).unwrap();
            assert_eq!(dtype, parsed, "Roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_tensor_info_consistency() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![4, 8],
            dtype: Dtype::F32,
            byte_offset: 0,
            size_bytes: 4 * 8 * 4, // 128 bytes
        };
        assert!(info.is_consistent());

        let bad = TensorInfo {
            name: "bad".to_string(),
            shape: vec![4, 8],
            dtype: Dtype::F32,
            byte_offset: 0,
            size_bytes: 100, // wrong size
        };
        assert!(!bad.is_consistent());
    }

    #[test]
    fn test_core_tensor_as_f32() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let data = Arc::from(bytes);
        let tensor = CoreTensor {
            name: "test".to_string(),
            data,
            shape: vec![2, 2],
            dtype: Dtype::F32,
        };

        let f32_view = tensor.as_f32().unwrap();
        assert_eq!(f32_view.len(), 4);
        assert_eq!(f32_view[0], 1.0);
        assert_eq!(f32_view[3], 4.0);
    }

    #[test]
    fn test_core_tensor_wrong_dtype() {
        let data = Arc::from(vec![1u8, 2, 3, 4]);
        let tensor = CoreTensor {
            name: "test".to_string(),
            data,
            shape: vec![4],
            dtype: Dtype::U8,
        };

        assert!(tensor.as_f32().is_none());
    }

    #[test]
    fn test_tensor_structure_lookup() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            TensorInfo {
                name: "model.layers.0.weight".to_string(),
                shape: vec![4, 8],
                dtype: Dtype::F32,
                byte_offset: 0,
                size_bytes: 128,
            },
        );
        tensors.insert(
            "model.layers.1.weight".to_string(),
            TensorInfo {
                name: "model.layers.1.weight".to_string(),
                shape: vec![4, 8],
                dtype: Dtype::F32,
                byte_offset: 128,
                size_bytes: 128,
            },
        );

        let structure = TensorStructure {
            tensors,
            arch: "test".to_string(),
            total_parameters: 64,
        };

        assert!(structure.contains("model.layers.0.weight"));
        assert!(!structure.contains("model.layers.99.weight"));
        assert_eq!(structure.len(), 2);
        assert_eq!(structure.with_prefix("model.layers.").len(), 2);
    }

    #[test]
    fn test_weight_store_with_real_file() {
        // Create a minimal valid safetensors file in memory.
        // Format: u64 header_size + JSON header + tensor data
        let values = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 f32 values
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "test_tensor": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, tensor_bytes.len()]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);

        // Write to a temp file.
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("pesti_test_safetensors.bin");
        std::fs::write(&tmp_path, &file_bytes).unwrap();

        // Load with WeightStore.
        let store = WeightStore::load_minimal(&tmp_path).unwrap();

        // Verify structure.
        assert!(store.structure().contains("test_tensor"));
        assert_eq!(store.structure().len(), 1);

        // Verify tensor info.
        let info = store.tensor_info("test_tensor").unwrap();
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(info.dtype, Dtype::F32);
        assert_eq!(info.size_bytes, 16);

        // Verify zero-copy data access.
        let data = store.tensor_data("test_tensor").unwrap();
        assert_eq!(data.len(), 16);

        // Verify CoreTensor creation.
        let tensor = store.core_tensor("test_tensor").unwrap();
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, Dtype::F32);

        let f32_view = tensor.as_f32().unwrap();
        assert_eq!(f32_view.len(), 4);
        assert_eq!(f32_view[0], 1.0);
        assert_eq!(f32_view[3], 4.0);

        // Verify structure consistency.
        assert!(store.verify_structure().is_empty());

        // Cleanup.
        std::fs::remove_file(&tmp_path).unwrap();
    }

    #[test]
    fn test_weight_store_missing_tensor() {
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("pesti_test_safetensors_missing.bin");

        // Create minimal file with one tensor.
        let values = vec![1.0f32];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "only_tensor": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);
        std::fs::write(&tmp_path, &file_bytes).unwrap();

        let store = WeightStore::load_minimal(&tmp_path).unwrap();

        assert!(store.tensor_data("nonexistent").is_none());
        assert!(store.core_tensor("nonexistent").is_none());
        assert!(store.tensor_info("nonexistent").is_none());

        std::fs::remove_file(&tmp_path).unwrap();
    }
}
