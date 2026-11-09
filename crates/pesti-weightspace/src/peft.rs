//! PEFT/LoRA integration for weight-space operations.
//!
//! Bridges trained LoRA adapters with the weight-space API, enabling:
//! - Inspection of what weights a LoRA adapter would change
//! - Surgical application of only significant LoRA deltas
//! - Atomic merge/unmerge via weight-space sessions

use pesti_runner::peft::lora::LoRAAdapter;
use pesti_runner::peft::{AdapterConfig, AdapterType};

use crate::error::{Result, WeightSpaceError};
use crate::session::{WeightEdit, WeightSpaceSession};

/// Convert a LoRA adapter's delta into weight-space edits.
///
/// Computes ΔW = scaling * B @ A and returns one WeightEdit per element
/// whose absolute value meets the threshold. For large adapters this can
/// produce millions of edits; use `min_abs_value` to filter negligible changes.
///
/// # Arguments
/// - `adapter`: Trained LoRA adapter
/// - `param_name`: Target tensor name in the model (e.g., "blk.0.attn_q.weight")
/// - `min_abs_value`: Filter threshold — edits with |delta| below this are dropped
///
/// # Example
/// ```ignore
/// use pesti_weightspace::peft::lora_delta_to_edits;
///
/// // After training an adapter:
/// let edits = lora_delta_to_edits(&adapter, "blk.0.attn_q.weight", 1e-4)?;
/// println!("{} significant weight changes", edits.len());
/// ```
pub fn lora_delta_to_edits(
    adapter: &LoRAAdapter,
    param_name: &str,
    min_abs_value: f32,
) -> Result<Vec<WeightEdit>> {
    if adapter.in_features == 0 || adapter.out_features == 0 {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: "LoRA adapter has invalid dimensions".to_string(),
        });
    }

    let scale = adapter.effective_scaling();
    let in_features = adapter.in_features;
    let out_features = adapter.out_features;

    // Compute BA matrix using the built-in matmul: [out_features x in_features]
    let ba = adapter.matrix_b.matmul(&adapter.matrix_a);

    // If matrices aren't populated yet (training skeleton), no delta to apply
    if ba.is_empty() {
        return Ok(Vec::new());
    }

    let mut edits = Vec::new();
    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let delta = ba[out_idx * in_features + in_idx] * scale;
            if delta.abs() >= min_abs_value {
                edits.push(WeightEdit {
                    param: param_name.to_string(),
                    index: vec![out_idx, in_idx],
                    value: delta,
                });
            }
        }
    }

    Ok(edits)
}

/// Apply a LoRA adapter's delta to a weight-space session.
///
/// Reads the current base weights, adds the LoRA delta (scale * B @ A), and
/// writes back atomically. Equivalent to `adapter.merge_into()` but operates
/// through the weight-space session for snapshot/rollback support.
///
/// # Arguments
/// - `session`: Active weight-space session
/// - `adapter`: Trained LoRA adapter
/// - `param_name`: Target tensor name in the model
pub fn apply_lora_adapter(
    session: &mut WeightSpaceSession,
    adapter: &LoRAAdapter,
    param_name: &str,
) -> Result<()> {
    if adapter.in_features == 0 || adapter.out_features == 0 {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: "LoRA adapter has invalid dimensions".to_string(),
        });
    }

    let base_values: Vec<f32> = session.read_tensor(param_name)?.to_vec();
    let scale = adapter.effective_scaling();
    let in_features = adapter.in_features;
    let out_features = adapter.out_features;

    // Validate tensor size matches adapter dimensions
    if base_values.len() != out_features * in_features {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: format!(
                "tensor has {} elements, expected {} x {} = {}",
                base_values.len(),
                out_features,
                in_features,
                out_features * in_features
            ),
        });
    }

    // Compute BA matrix
    let ba = adapter.matrix_b.matmul(&adapter.matrix_a);

    if ba.is_empty() {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: "LoRA adapter matrices are not populated".to_string(),
        });
    }

    // Apply as atomic write: W' = W + scale * BA
    let mut edits = Vec::with_capacity(out_features * in_features);
    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let flat = out_idx * in_features + in_idx;
            edits.push(WeightEdit {
                param: param_name.to_string(),
                index: vec![out_idx, in_idx],
                value: base_values[flat] + scale * ba[flat],
            });
        }
    }

    session.apply_edits(edits)?;
    Ok(())
}

/// Remove a LoRA adapter's contribution from merged weights (inverse merge).
///
/// Computes W_base = W_merged - scale * B @ A and writes back atomically.
pub fn unmerge_lora_adapter(
    session: &mut WeightSpaceSession,
    adapter: &LoRAAdapter,
    param_name: &str,
) -> Result<()> {
    if adapter.in_features == 0 || adapter.out_features == 0 {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: "LoRA adapter has invalid dimensions".to_string(),
        });
    }

    let merged_values: Vec<f32> = session.read_tensor(param_name)?.to_vec();
    let scale = adapter.effective_scaling();
    let in_features = adapter.in_features;
    let out_features = adapter.out_features;

    if merged_values.len() != out_features * in_features {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: format!(
                "tensor has {} elements, expected {} x {} = {}",
                merged_values.len(),
                out_features,
                in_features,
                out_features * in_features
            ),
        });
    }

    // Compute BA matrix
    let ba = adapter.matrix_b.matmul(&adapter.matrix_a);

    if ba.is_empty() {
        return Err(WeightSpaceError::InvalidEdit {
            param: param_name.to_string(),
            reason: "LoRA adapter matrices are not populated".to_string(),
        });
    }

    // Apply inverse: W_base = W_merged - scale * BA
    let mut edits = Vec::with_capacity(out_features * in_features);
    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let flat = out_idx * in_features + in_idx;
            edits.push(WeightEdit {
                param: param_name.to_string(),
                index: vec![out_idx, in_idx],
                value: merged_values[flat] - scale * ba[flat],
            });
        }
    }

    session.apply_edits(edits)?;
    Ok(())
}

/// Create a default LoRA adapter config for weight-space operations.
pub fn default_lora_config(rank: usize) -> AdapterConfig {
    AdapterConfig {
        rank,
        scaling: 1.0,
        adapter_type: AdapterType::LoRA,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pesti_runner::peft::lora::LoRABuilder;

    const TEST_MODEL: &str = "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf";

    fn has_test_model() -> bool {
        std::path::Path::new(TEST_MODEL).exists()
    }

    #[test]
    fn test_lora_delta_to_edits_zero_adapter() {
        // Zero adapter should produce no edits above threshold
        let adapter = LoRABuilder::new().with_rank(4).build_zeros(8, 8).unwrap();

        let edits = lora_delta_to_edits(&adapter, "test.weight", 1e-6).unwrap();
        assert!(
            edits.is_empty(),
            "Zero adapter should have no non-zero deltas"
        );
    }

    #[test]
    fn test_lora_delta_to_edits_random_adapter() {
        // Random adapter should produce edits
        let adapter = LoRABuilder::new().with_rank(4).build_random(8, 8).unwrap();

        let edits = lora_delta_to_edits(&adapter, "test.weight", 0.0).unwrap();
        // With random init, most values should be non-zero
        assert!(
            !edits.is_empty(),
            "Random adapter should have non-zero deltas"
        );
        assert!(edits.len() <= 64, "At most 8x8 = 64 edits");

        // All edits should reference the correct param
        for edit in &edits {
            assert_eq!(edit.param, "test.weight");
            assert_eq!(edit.index.len(), 2);
        }
    }

    #[test]
    fn test_lora_delta_threshold_filtering() {
        let adapter = LoRABuilder::new().with_rank(4).build_random(8, 8).unwrap();

        // No threshold: all non-zero edits
        let all_edits = lora_delta_to_edits(&adapter, "test.weight", 0.0).unwrap();

        // High threshold: fewer edits
        let filtered = lora_delta_to_edits(&adapter, "test.weight", 1e6).unwrap();

        assert!(filtered.len() <= all_edits.len());
    }

    #[test]
    fn test_apply_and_unmerge_roundtrip() {
        if !has_test_model() {
            eprintln!("Skipping: test model not found");
            return;
        }

        let mut session = WeightSpaceSession::open(TEST_MODEL).expect("Failed to open");

        // Find a 2D tensor (matrices are stored flat, so we need shape info)
        // For simplicity, use attn_q which should be [out x in] and roughly square
        let names: Vec<String> = session.tensor_names().to_vec();
        let target = names
            .iter()
            .find(|n| n.contains("attn_q"))
            .cloned()
            .unwrap_or_else(|| names[0].clone());

        let values: Vec<f32> = session.read_tensor(&target).unwrap().to_vec();
        let size = values.len();

        // We need to know the 2D shape to create a matching adapter.
        // For TinyLlama attn_q: [n_heads*head_dim x n_embd] = [2048 x 2048]
        // But we can't easily get shape from the session API, so use sqrt heuristic
        // and verify it's exact. If not square, try to factor it.
        let side = (size as f64).sqrt() as usize;
        if side * side != size {
            eprintln!(
                "Skipping roundtrip: tensor {} size {} is not a perfect square",
                target, size
            );
            return;
        }

        // Use very small rank and scaling to avoid numerical overflow
        let adapter = LoRABuilder::new()
            .with_rank(1)
            .with_scaling(0.001)
            .build_random(side, side)
            .unwrap();

        // Verify the delta would be finite before applying
        let ba = adapter.matrix_b.matmul(&adapter.matrix_a);
        if !ba.iter().all(|v| v.is_finite()) {
            eprintln!("Skipping roundtrip: non-finite BA values");
            return;
        }

        // Apply LoRA
        apply_lora_adapter(&mut session, &adapter, &target).unwrap();

        // Verify weights changed
        let merged = session.read_tensor(&target).unwrap().to_vec();
        let diff: f64 = merged
            .iter()
            .zip(values.iter())
            .map(|(a, b)| (*a - *b) as f64)
            .map(f64::abs)
            .sum();
        assert!(diff > 0.0, "Weights should have changed after merge");

        // Unmerge
        unmerge_lora_adapter(&mut session, &adapter, &target).unwrap();

        // Verify weights restored (allow small floating point drift)
        let restored = session.read_tensor(&target).unwrap().to_vec();
        let restore_diff: f64 = restored
            .iter()
            .zip(values.iter())
            .map(|(a, b)| (*a - *b) as f64)
            .map(f64::abs)
            .sum();
        assert!(
            restore_diff < 1e-2,
            "Weights should be restored after unmerge (diff={})",
            restore_diff
        );
    }
}
