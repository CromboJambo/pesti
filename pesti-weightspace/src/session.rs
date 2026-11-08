//! Weight-space session manager.
//!
//! Provides atomic read/edit/snapshot/rollback operations on model weights
//! loaded from GGUF files. All edits are validated before application to
//! guarantee atomicity — either all edits in a patch succeed or none do.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use pesti_gguf::types::GgufHeader;
use pesti_runner::gguf_weight_loader::{load_gguf_weights, GgufWeights};
use serde::{Deserialize, Serialize};

use crate::error::{Result, WeightSpaceError};
use crate::stats::{StatType, WeightStatOp, WeightStatResult};

/// A single weight edit for atomic patching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEdit {
    /// Tensor name (e.g., "tok_embeddings.weight", "blk.0.attn_q.weight").
    pub param: String,
    /// Multi-dimensional index into the tensor.
    pub index: Vec<usize>,
    /// New weight value (f32).
    pub value: f32,
}

/// A weight-space operation descriptor.
#[derive(Debug, Clone)]
pub enum WeightOp {
    /// Read a specific weight value by tensor name and index.
    Read { param: String, index: Vec<usize> },
    /// Write a new weight value.
    Write {
        param: String,
        index: Vec<usize>,
        value: f32,
    },
    /// Apply a batch of weight edits atomically.
    Patch { edits: Vec<WeightEdit> },
}

/// A snapshot of model weights for rollback.
#[derive(Debug)]
pub struct Snapshot {
    /// Tensor name → (original f32 bytes) for all tensors at snapshot time.
    tensors: HashMap<String, Vec<u8>>,
}

impl Snapshot {
    fn new(weights: &GgufWeights) -> Self {
        let mut tensors = HashMap::with_capacity(weights.tensors.len());
        for (name, data) in &weights.tensors {
            tensors.insert(name.clone(), data.clone());
        }
        Self { tensors }
    }

    fn apply_to(&self, weights: &mut GgufWeights) {
        for (name, data) in &self.tensors {
            if let Some(slot) = weights.tensors.get_mut(name) {
                *slot = data.clone();
            }
        }
    }
}

/// Weight-space session manager.
///
/// Opens a GGUF model file, provides typed access to weight tensors, and
/// supports atomic edits with snapshot/rollback for safe experimentation.
pub struct WeightSpaceSession {
    /// Original model path (for provenance).
    model_path: PathBuf,
    /// Loaded model weights (mutable — edits modify this in place).
    weights: GgufWeights,
    /// Snapshot stack for rollback.
    snapshots: VecDeque<Snapshot>,
}

impl WeightSpaceSession {
    /// Open a weight-space session on a GGUF model file.
    ///
    /// Loads all tensors into memory (dequantized to f32). For large models
    /// this can use significant RAM — a 7B Q4_K_M model uses ~4GB when
    /// dequantized to f32 in memory.
    pub fn open(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(WeightSpaceError::ModelFile {
                path: path.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "model file does not exist",
                ),
            });
        }

        let weights = load_gguf_weights(path)?;
        Ok(Self {
            model_path: path.to_path_buf(),
            weights,
            snapshots: VecDeque::new(),
        })
    }

    /// Get the original model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// List all tensor names in the model.
    pub fn tensor_names(&self) -> Vec<String> {
        self.weights.tensors.keys().cloned().collect()
    }

    /// Get the shape of a tensor as a vector of dimensions.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        let shape = tensor_shape_from_header(&self.weights.header, name);
        if shape.is_empty() {
            return Err(WeightSpaceError::TensorNotFound(name.to_string()));
        }
        Ok(shape)
    }

    /// Read a single f32 value from a tensor at the given multi-dimensional index.
    pub fn read_value(&self, param: &str, index: &[usize]) -> Result<f32> {
        let data = self
            .weights
            .tensors
            .get(param)
            .ok_or_else(|| WeightSpaceError::TensorNotFound(param.to_string()))?;

        let shape = tensor_shape_from_header(&self.weights.header, param);
        let flat_idx = multi_to_flat_index(index, &shape, param)?;

        // Data is f32 bytes (4 bytes per element)
        let byte_start = flat_idx * 4;
        if byte_start + 4 > data.len() {
            return Err(WeightSpaceError::IndexOutOfBounds {
                tensor: param.to_string(),
                index: index.to_vec(),
                shape,
            });
        }

        let bytes = [
            data[byte_start],
            data[byte_start + 1],
            data[byte_start + 2],
            data[byte_start + 3],
        ];
        Ok(f32::from_le_bytes(bytes))
    }

    /// Read an entire tensor as a slice of f32 values.
    pub fn read_tensor(&self, param: &str) -> Result<&[f32]> {
        let data = self
            .weights
            .tensors
            .get(param)
            .ok_or_else(|| WeightSpaceError::TensorNotFound(param.to_string()))?;

        // Safety: data is always f32 bytes (4-byte aligned, length multiple of 4)
        // because load_gguf_weights dequantizes everything to f32.
        let ptr = data.as_ptr() as *const f32;
        let len = data.len() / std::mem::size_of::<f32>();
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Apply a batch of weight edits atomically.
    ///
    /// All edits are validated first (tensor exists, index in bounds, value
    /// finite). If any validation fails, no edits are applied and the error
    /// is returned. This guarantees atomicity.
    pub fn apply_edits(&mut self, edits: Vec<WeightEdit>) -> Result<()> {
        // Phase 1: Validate all edits
        for edit in &edits {
            if !edit.value.is_finite() {
                return Err(WeightSpaceError::InvalidEdit {
                    param: edit.param.clone(),
                    reason: format!("value {} is not finite", edit.value),
                });
            }

            let data = self
                .weights
                .tensors
                .get(&edit.param)
                .ok_or_else(|| WeightSpaceError::TensorNotFound(edit.param.clone()))?;

            let shape = tensor_shape_from_header(&self.weights.header, &edit.param);
            let flat_idx = multi_to_flat_index(&edit.index, &shape, &edit.param)?;

            let byte_start = flat_idx * 4;
            if byte_start + 4 > data.len() {
                return Err(WeightSpaceError::IndexOutOfBounds {
                    tensor: edit.param.clone(),
                    index: edit.index.clone(),
                    shape,
                });
            }
        }

        // Phase 2: Apply all edits (validation passed, this can't fail)
        for edit in &edits {
            let data = self
                .weights
                .tensors
                .get_mut(&edit.param)
                .expect("validated above");

            let shape = tensor_shape_from_header(&self.weights.header, &edit.param);
            let flat_idx =
                multi_to_flat_index(&edit.index, &shape, &edit.param).expect("validated above");

            let byte_start = flat_idx * 4;
            data[byte_start..byte_start + 4].copy_from_slice(&edit.value.to_le_bytes());
        }

        Ok(())
    }

    /// Create a snapshot of current weights for later rollback.
    pub fn snapshot(&mut self) {
        let snap = Snapshot::new(&self.weights);
        self.snapshots.push_back(snap);
    }

    /// Rollback to the most recent snapshot.
    pub fn rollback(&mut self) -> Result<()> {
        let snap = self
            .snapshots
            .pop_back()
            .ok_or(WeightSpaceError::NoSnapshot)?;
        snap.apply_to(&mut self.weights);
        Ok(())
    }

    /// Number of snapshots available for rollback.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Save edited weights back to a file in SafeTensors format.
    ///
    /// Note: This saves tensors as F32 regardless of original dtype.
    /// Quantized models will be upcast to full precision on save.
    pub fn save_edited_model(&self, output_path: impl AsRef<Path>) -> Result<()> {
        let path = output_path.as_ref();

        use pesti_safetensors::writer::{SafetensorTensor, SafetensorsWriter};

        let mut writer = SafetensorsWriter::new();

        for (name, data) in &self.weights.tensors {
            let shape = tensor_shape_from_header(&self.weights.header, name);
            let tensor = SafetensorTensor {
                name: name.clone(),
                dtype: "F32".to_string(),
                shape,
                data: data.clone(),
            };
            writer.add_tensor(tensor);
        }

        writer
            .write(path)
            .map_err(|e| WeightSpaceError::Gguf(format!("failed to write safetensors: {}", e)))?;
        Ok(())
    }

    /// Compute a statistic over all elements of a tensor.
    pub fn compute_stat(&self, op: &WeightStatOp) -> Result<WeightStatResult> {
        let values = self.read_tensor(&op.param)?;
        let count = values.len();

        if count == 0 {
            return Err(WeightSpaceError::InvalidEdit {
                param: op.param.clone(),
                reason: "tensor is empty".to_string(),
            });
        }

        let value: f64 = match op.stat_type {
            StatType::Mean => {
                let sum: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
                sum / count as f64
            }
            StatType::StdDev => {
                let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / count as f64;
                let variance: f64 =
                    values.iter().map(|&v| {
                        let d = v as f64 - mean;
                        d * d
                    }).sum::<f64>() / count as f64;
                variance.sqrt()
            }
            StatType::Min => {
                values.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min)
            }
            StatType::Max => {
                values.iter().map(|&v| v as f64).fold(f64::NEG_INFINITY, f64::max)
            }
            StatType::NonZeroCount => {
                values.iter().filter(|&&v| v != 0.0).count() as f64
            }
            StatType::L2Norm => {
                let sum_sq: f64 = values.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
                sum_sq.sqrt()
            }
        };

        Ok(WeightStatResult {
            param: op.param.clone(),
            stat_type: op.stat_type,
            value,
            element_count: count,
        })
    }

    /// Get the underlying GgufWeights for advanced access.
    pub fn weights(&self) -> &GgufWeights {
        &self.weights
    }

    /// Get mutable access to raw tensor bytes (for bulk operations).
    ///
    /// # Safety
    /// The caller must maintain the invariant that tensor data remains
    /// valid f32 byte arrays with length matching the original element count.
    pub fn weights_mut(&mut self) -> &mut GgufWeights {
        &mut self.weights
    }
}

/// Compute the flat (row-major) index from a multi-dimensional index and shape.
pub(crate) fn multi_to_flat_index(index: &[usize], shape: &[usize], tensor_name: &str) -> Result<usize> {
    if index.len() != shape.len() {
        return Err(WeightSpaceError::IndexOutOfBounds {
            tensor: tensor_name.to_string(),
            index: index.to_vec(),
            shape: shape.to_vec(),
        });
    }

    let mut flat = 0usize;
    for (i, (&idx, &dim)) in index.iter().zip(shape.iter()).enumerate() {
        if idx >= dim {
            return Err(WeightSpaceError::IndexOutOfBounds {
                tensor: tensor_name.to_string(),
                index: index.to_vec(),
                shape: shape.to_vec(),
            });
        }
        // Row-major: stride for dimension i is product of all subsequent dims
        let stride: usize = shape[i + 1..].iter().product();
        flat += idx * stride;
    }

    Ok(flat)
}

/// Extract tensor shape from GGUF header.
fn tensor_shape_from_header(header: &GgufHeader, name: &str) -> Vec<usize> {
    header
        .tensors
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.shape.iter().map(|&d| d as usize).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightEdit;

    const TEST_MODEL: &str = "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf";

    fn has_test_model() -> bool {
        std::path::Path::new(TEST_MODEL).exists()
    }

    #[test]
    fn test_multi_to_flat_index_1d() {
        let shape = vec![10];
        assert_eq!(multi_to_flat_index(&[3], &shape, "test").unwrap(), 3);
        assert_eq!(multi_to_flat_index(&[9], &shape, "test").unwrap(), 9);
    }

    #[test]
    fn test_multi_to_flat_index_2d() {
        let shape = vec![4, 5]; // 4 rows, 5 cols
        assert_eq!(multi_to_flat_index(&[0, 0], &shape, "test").unwrap(), 0);
        assert_eq!(multi_to_flat_index(&[1, 0], &shape, "test").unwrap(), 5);
        assert_eq!(multi_to_flat_index(&[0, 4], &shape, "test").unwrap(), 4);
        assert_eq!(multi_to_flat_index(&[3, 2], &shape, "test").unwrap(), 17);
    }

    #[test]
    fn test_multi_to_flat_index_3d() {
        let shape = vec![2, 3, 4]; // 2*3*4 = 24 elements
        assert_eq!(multi_to_flat_index(&[0, 0, 0], &shape, "test").unwrap(), 0);
        assert_eq!(multi_to_flat_index(&[1, 0, 0], &shape, "test").unwrap(), 12);
        assert_eq!(multi_to_flat_index(&[0, 1, 0], &shape, "test").unwrap(), 4);
        assert_eq!(multi_to_flat_index(&[0, 0, 3], &shape, "test").unwrap(), 3);
        assert_eq!(multi_to_flat_index(&[1, 2, 3], &shape, "test").unwrap(), 23);
    }

    #[test]
    fn test_multi_to_flat_index_out_of_bounds() {
        let shape = vec![4, 5];
        let err = multi_to_flat_index(&[4, 0], &shape, "test").unwrap_err();
        assert!(matches!(err, WeightSpaceError::IndexOutOfBounds { .. }));

        let err = multi_to_flat_index(&[0, 5], &shape, "test").unwrap_err();
        assert!(matches!(err, WeightSpaceError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_multi_to_flat_index_dim_mismatch() {
        let shape = vec![4, 5];
        let err = multi_to_flat_index(&[1], &shape, "test").unwrap_err();
        assert!(matches!(err, WeightSpaceError::IndexOutOfBounds { .. }));

        let err = multi_to_flat_index(&[1, 2, 3], &shape, "test").unwrap_err();
        assert!(matches!(err, WeightSpaceError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_open_nonexistent_file() {
        let result = WeightSpaceSession::open("/nonexistent/path/model.gguf");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, WeightSpaceError::ModelFile { .. }));
    }

    #[test]
    fn test_session_open_and_tensor_list() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found at {}", TEST_MODEL);
            return;
        }

        let session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open test model");

        // Should have tensors
        let names = session.tensor_names();
        assert!(!names.is_empty(), "Model should have tensors");

        // TinyLlama has block attention/ffn layers
        assert!(
            names.iter().any(|n| n.starts_with("blk.")),
            "Expected blk.* tensors for TinyLlama"
        );
    }

    #[test]
    fn test_read_value_roundtrip() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let mut session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");

        // Find a tensor with known shape
        let names = session.tensor_names();
        let target = names
            .iter()
            .find(|n| n.contains("norm"))
            .unwrap_or(&names[0]);
        let shape = session.tensor_shape(target).unwrap();

        // Read value at [0, 0] (or [0] for 1D)
        let index: Vec<usize> = vec![0; shape.len()];
        let original = session.read_value(target, &index).expect("Failed to read");

        // Verify it's finite
        assert!(original.is_finite(), "Weight value should be finite");

        // Write a new value and read it back
        session
            .apply_edits(vec![WeightEdit {
                param: target.to_string(),
                index: index.clone(),
                value: 0.42,
            }])
            .expect("Failed to apply edit");

        let updated = session
            .read_value(target, &index)
            .expect("Failed to read back");
        assert_eq!(updated, 0.42, "Value should match after edit");
    }

    #[test]
    fn test_apply_edits_atomicity() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let mut session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let names = session.tensor_names();
        let target = &names[0];
        let shape = session.tensor_shape(target).unwrap();
        let valid_index: Vec<usize> = vec![0; shape.len()];

        // First edit succeeds
        let original = session.read_value(target, &valid_index).unwrap();
        session
            .apply_edits(vec![WeightEdit {
                param: target.to_string(),
                index: valid_index.clone(),
                value: 99.0,
            }])
            .unwrap();
        assert_eq!(session.read_value(target, &valid_index).unwrap(), 99.0);

        // Second edit batch has one valid + one invalid (nonexistent tensor)
        let result = session.apply_edits(vec![
            WeightEdit {
                param: target.to_string(),
                index: valid_index.clone(),
                value: 50.0,
            },
            WeightEdit {
                param: "nonexistent.tensor".to_string(),
                index: vec![0],
                value: 1.0,
            },
        ]);

        // Should fail — and the valid edit should NOT have been applied
        assert!(result.is_err());
        let current = session.read_value(target, &valid_index).unwrap();
        assert_eq!(
            current, 99.0,
            "Atomicity: valid edit should not apply when batch fails"
        );

        // Restore original
        session
            .apply_edits(vec![WeightEdit {
                param: target.to_string(),
                index: valid_index.clone(),
                value: original,
            }])
            .unwrap();
    }

    #[test]
    fn test_snapshot_and_rollback() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let mut session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let names = session.tensor_names();
        let target = &names[0];
        let shape = session.tensor_shape(target).unwrap();
        let index: Vec<usize> = vec![0; shape.len()];

        let original = session.read_value(target, &index).unwrap();

        // Take snapshot
        session.snapshot();
        assert_eq!(session.snapshot_count(), 1);

        // Make edits
        session
            .apply_edits(vec![WeightEdit {
                param: target.to_string(),
                index: index.clone(),
                value: -123.456,
            }])
            .unwrap();
        assert_eq!(session.read_value(target, &index).unwrap(), -123.456);

        // Rollback
        session.rollback().expect("Rollback failed");
        let restored = session.read_value(target, &index).unwrap();
        assert!(
            (restored - original).abs() < 1e-6,
            "Value should be restored after rollback"
        );
        assert_eq!(session.snapshot_count(), 0);
    }

    #[test]
    fn test_rollback_no_snapshot() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let mut session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let result = session.rollback();
        assert!(matches!(result.unwrap_err(), WeightSpaceError::NoSnapshot));
    }

    #[test]
    fn test_tensor_not_found() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let result = session.read_value("does.not.exist", &[0]);
        assert!(matches!(
            result.unwrap_err(),
            WeightSpaceError::TensorNotFound(_)
        ));
    }

    #[test]
    fn test_compute_stat_mean() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let names = session.tensor_names();
        let target = &names[0];

        let op = crate::stats::WeightStatOp {
            param: target.to_string(),
            stat_type: crate::stats::StatType::Mean,
        };
        let result = session.compute_stat(&op).expect("Failed to compute mean");

        assert!(result.value.is_finite());
        assert!(result.element_count > 0);
    }

    #[test]
    fn test_compute_stat_l2_norm() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");
        let names = session.tensor_names();
        let target = &names[0];

        let op = crate::stats::WeightStatOp {
            param: target.to_string(),
            stat_type: crate::stats::StatType::L2Norm,
        };
        let result = session.compute_stat(&op).expect("Failed to compute L2 norm");

        assert!(result.value >= 0.0);
        assert!(result.value.is_finite());
    }
}
