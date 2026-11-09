//! Integration test: full lifecycle with a real model file.
//!
//! Requires the Qwen2.5-0.5B GGUF model (small enough for CI).
//! Skips gracefully if the model file is not present.

use pesti_state::{AdapterState, OperationType, PestiState};
use std::path::Path;

const MODEL_PATH: &str = "../conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

fn skip_if_no_model() {
    if !Path::new(MODEL_PATH).exists() {
        eprintln!("SKIP: model file not found at {}", MODEL_PATH);
        std::process::exit(0);
    }
}

#[test]
fn test_full_lifecycle() {
    skip_if_no_model();

    let tmp = tempfile::tempdir().unwrap();
    let state_path = tmp.path().join("state.db");
    let output_path = tmp.path().join("adapter-output.gguf");

    // 1. Open state and tracked session
    let state = PestiState::open(&state_path).unwrap();
    let mut session =
        pesti_weightspace::WeightSpaceSession::open_tracked(MODEL_PATH, state.clone())
            .expect("open_tracked should succeed");

    let session_id = session.session_id().unwrap().to_string();
    let model_id = session.model_id().unwrap().to_string();

    // 2. Verify open operation was logged
    let ops = state.session_operations(&session_id).unwrap();
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].op_type, OperationType::Open);

    // 3. Apply some edits (modify a few values in the first tensor)
    let names = session.tensor_names();
    assert!(!names.is_empty(), "model should have tensors");

    let first_tensor = &names[0];
    let shape = session.tensor_shape(first_tensor).unwrap();
    let mut edits = Vec::new();

    // Set a few values to 0.0 (safe, reversible change)
    // Use proper multi-dimensional indices matching the tensor shape
    for i in 0..10 {
        if shape.len() == 2 {
            // 2D tensor: use [row, col] index
            let row = (i as usize) % shape[0];
            edits.push(pesti_weightspace::WeightEdit {
                param: first_tensor.clone(),
                index: vec![row, 0],
                value: 0.0,
            });
        } else if shape.len() == 1 {
            // 1D tensor: use flat index
            edits.push(pesti_weightspace::WeightEdit {
                param: first_tensor.clone(),
                index: vec![i as usize],
                value: 0.0,
            });
        } else {
            // Higher dimensional: flatten to first dimension
            let idx = (i as usize) % shape[0];
            edits.push(pesti_weightspace::WeightEdit {
                param: first_tensor.clone(),
                index: vec![idx],
                value: 0.0,
            });
        }
    }

    session.apply_edits(edits).unwrap();

    // Verify edit was logged
    let ops = state.session_operations(&session_id).unwrap();
    assert!(ops.iter().any(|op| op.op_type == OperationType::Edit));

    // 4. Take a snapshot
    session.snapshot();

    // Verify snapshot was logged
    let ops = state.session_operations(&session_id).unwrap();
    assert!(ops.iter().any(|op| op.op_type == OperationType::Snapshot));

    // 5. Save as adapter
    let adapter_id = session
        .save_as_adapter(
            "test-adapter",
            "Integration test adapter: zeroed first tensor values",
            &output_path,
        )
        .unwrap();

    assert!(!adapter_id.is_empty());

    // Verify adapter is registered
    let adapters = state.list_adapters(Some(AdapterState::Active)).unwrap();
    let adapter = adapters.iter().find(|a| a.id == adapter_id).unwrap();
    assert_eq!(adapter.base_model_id, model_id);
    assert_eq!(adapter.state, AdapterState::Active);

    // 6. Drop session (triggers close logging)
    drop(session);

    // 7. Verify close operation was logged
    let ops = state.session_operations(&session_id).unwrap();
    assert!(ops.iter().any(|op| op.op_type == OperationType::Close));

    // 8. Verify model is registered
    let model = state.get_model(&model_id).unwrap().unwrap();
    assert_eq!(model.id, model_id);
    assert!(model.tensor_count > 0);

    // 9. Verify operations are in chronological order
    for i in 1..ops.len() {
        assert!(
            ops[i].timestamp >= ops[i - 1].timestamp,
            "operations should be chronological"
        );
    }
}

#[test]
fn test_revalidation_flow() {
    skip_if_no_model();

    let tmp = tempfile::tempdir().unwrap();
    let state_path = tmp.path().join("state.db");

    let state = PestiState::open(&state_path).unwrap();
    let session = pesti_weightspace::WeightSpaceSession::open_tracked(MODEL_PATH, state.clone())
        .expect("open_tracked should succeed");

    let model_id = session.model_id().unwrap().to_string();
    drop(session);

    // Register an adapter for this model
    let adapter_id = state
        .register_adapter(pesti_state::AdapterRecord {
            id: pesti_durable_objects::object_id::ObjectId::generate().to_string(),
            base_model_id: model_id.clone(),
            output_path: "/tmp/test-adapter.gguf".to_string(),
            description: Some("test-adapter".to_string()),
            assumptions: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            verified_at: None,
            state: AdapterState::Validated,
            operation_ids: vec![],
        })
        .unwrap();

    // Verify adapter is in Validated state
    let adapters = state.list_adapters(Some(AdapterState::Validated)).unwrap();
    assert!(adapters.iter().any(|a| a.id == adapter_id));

    // Simulate model change: invalidate all adapters for this model
    let invalidated = state.invalidate_adapters_for_model(&model_id).unwrap();
    assert_eq!(invalidated, 1);

    // Verify adapter is now NeedsRevalidation
    let needs_reval = state.needs_revalidation().unwrap();
    assert_eq!(needs_reval.len(), 1);
    assert_eq!(needs_reval[0].id, adapter_id);
    assert_eq!(needs_reval[0].state, AdapterState::NeedsRevalidation);

    // Verify it's no longer in Validated
    let validated = state.list_adapters(Some(AdapterState::Validated)).unwrap();
    assert!(!validated.iter().any(|a| a.id == adapter_id));
}
