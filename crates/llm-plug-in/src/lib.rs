//! pesti-plug-in: Portable Execution Substrate for Transformer Inference — plug-in protocol for external model runtime integration.
//!
//! Provides weight manifest output from safetensors DB, inference request/response structs,
//! and runner config for plugging external inference engines into PESTI tool calls and skills.
//!
//! ## Protocol
//!
//! - WeightManifest: JSON output from safetensors DB → external runner consumes
//! - InferenceRequest: prompt + context + skill_refs → runner receives
//! - InferenceResponse: structured output → guard gate consumption
//! - RunnerConfig: external runner endpoint/protocol configuration

pub mod error;
pub mod manifest;
pub mod protocol;
pub mod templates;

pub use error::{PlugInError, Result};
pub use manifest::WeightManifest;
pub use protocol::{InferenceRequest, InferenceResponse, RunnerConfig};
pub use templates::{TemplateFamily, infer_template};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::generate_weight_manifest;
    use pesti_safetensors::schema;
    use tempfile::tempdir;

    // ── WeightManifest ─────────────────────────────────────────────────

    #[test]
    fn weight_manifest_serializes_and_deserializes() {
        let manifest = WeightManifest {
            weight_id: "w-1".into(),
            model_name: "qwen3".into(),
            repo_id: "Qwen/Qwen3".into(),
            file_path: "/tmp/qwen.safetensors".into(),
            tensor_count: 100,
            dtype: "F32".into(),
            device: "CPU".into(),
            size_bytes: 4_000_000_000,
            checksum: "sha256:abc123".into(),
            tensors: vec![],
            metadata: serde_json::json!({"author": "test"}),
            lazy_loading: true,
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let restored: WeightManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.weight_id, "w-1");
        assert_eq!(restored.model_name, "qwen3");
        assert_eq!(restored.tensor_count, 100);
        assert!(restored.lazy_loading);
    }

    #[test]
    fn weight_manifest_with_tensors_serializes() {
        let manifest = WeightManifest {
            weight_id: "w-2".into(),
            model_name: "llama".into(),
            repo_id: "meta/Llama".into(),
            file_path: "/tmp/llama.safetensors".into(),
            tensor_count: 2,
            dtype: "F16".into(),
            device: "cuda".into(),
            size_bytes: 8_000_000_000,
            checksum: "sha256:def456".into(),
            tensors: vec![crate::manifest::TensorMetadataRow {
                id: "t-1".into(),
                weight_id: "w-2".into(),
                tensor_name: "tok_embeddings.weight".into(),
                shape: "(4096, 4096)".into(),
                dtype: "F16".into(),
                size_bytes: 32_768,
                checksum: "sha256:tensor1".into(),
            }],
            metadata: serde_json::json!({}),
            lazy_loading: false,
        };

        let json = serde_json::to_string(&manifest).unwrap();
        let restored: WeightManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.tensors.len(), 1);
        assert_eq!(restored.tensors[0].tensor_name, "tok_embeddings.weight");
    }

    #[test]
    fn weight_manifest_deserialize_missing_fields_uses_defaults() {
        let json = r#"{"weight_id":"w-1","model_name":"m","repo_id":"","file_path":"",
            "tensor_count":0,"dtype":"","device":"","size_bytes":0,"checksum":"",
            "tensors":[],"metadata":{},"lazy_loading":false}"#;
        let manifest: WeightManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.weight_id, "w-1");
        assert!(manifest.tensors.is_empty());
    }

    // ── TensorMetadataRow ──────────────────────────────────────────────

    #[test]
    fn tensor_metadata_row_serializes() {
        let row = crate::manifest::TensorMetadataRow {
            id: "t-1".into(),
            weight_id: "w-1".into(),
            tensor_name: "w".into(),
            shape: "(1,1)".into(),
            dtype: "F32".into(),
            size_bytes: 4,
            checksum: "sha256:x".into(),
        };
        let json = serde_json::to_string(&row).unwrap();
        let restored: crate::manifest::TensorMetadataRow = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "t-1");
    }

    // ── ModelWeightRow ─────────────────────────────────────────────────

    #[test]
    fn model_weight_row_serializes() {
        let row = crate::manifest::ModelWeightRow {
            id: "w-1".into(),
            model_name: "qwen".into(),
            repo_id: "Qwen/Qwen".into(),
            file_path: "/tmp/qwen.safetensors".into(),
            tensor_count: 10,
            dtype: "F32".into(),
            device: "CPU".into(),
            size_bytes: 1000,
            checksum: "sha256:abc".into(),
            metadata: serde_json::json!({}),
            loaded_at: 1234567890,
            created_at: 1234567800,
            active: 1,
        };
        let json = serde_json::to_string(&row).unwrap();
        let restored: crate::manifest::ModelWeightRow = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, "w-1");
        assert_eq!(restored.loaded_at, 1234567890);
    }

    // ── generate_weight_manifest ───────────────────────────────────────

    #[test]
    fn generate_weight_manifest_no_such_model() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        schema::init_db(&conn).unwrap();

        let result = generate_weight_manifest(&conn, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn generate_weight_manifest_with_model() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        schema::init_db(&conn).unwrap();

        schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model.safetensors",
            5,
            "F32",
            "CPU",
            5000,
            "abc123",
            "{}",
        )
        .unwrap();

        let manifest = generate_weight_manifest(&conn, "test-model").unwrap();
        assert_eq!(manifest.model_name, "test-model");
        assert_eq!(manifest.tensor_count, 5);
        assert!(manifest.lazy_loading);
        assert_eq!(manifest.tensors.len(), 0);
    }

    #[test]
    fn generate_weight_manifest_with_tensors() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        schema::init_db(&conn).unwrap();

        let weight_id = schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model.safetensors",
            2,
            "F16",
            "cuda",
            10000,
            "def456",
            "{}",
        )
        .unwrap();

        schema::insert_tensor_metadata(
            &conn,
            &weight_id,
            "weight1",
            "(100, 200)",
            "F16",
            40000,
            "hash1",
        )
        .unwrap();

        let manifest = generate_weight_manifest(&conn, "test-model").unwrap();
        assert_eq!(manifest.tensors.len(), 1);
        assert_eq!(manifest.tensors[0].tensor_name, "weight1");
        assert_eq!(manifest.tensors[0].shape, "(100, 200)");
    }

    #[test]
    fn generate_weight_manifest_ignores_inactive() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        schema::init_db(&conn).unwrap();

        schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model.safetensors",
            1,
            "F32",
            "CPU",
            1000,
            "abc",
            "{}",
        )
        .unwrap();

        let weight_id = schema::insert_model_weights(
            &conn,
            "test-model",
            "test/repo",
            "/tmp/model2.safetensors",
            2,
            "F16",
            "cuda",
            2000,
            "def",
            "{}",
        )
        .unwrap();

        schema::deactivate_weight(&conn, &weight_id).unwrap();

        let manifest = generate_weight_manifest(&conn, "test-model").unwrap();
        assert_eq!(manifest.tensor_count, 1);
    }

    // ── InferenceRequest ───────────────────────────────────────────────

    #[test]
    fn inference_request_new_sets_defaults() {
        let req = InferenceRequest::new("prov-1", "gpt2", "hello");
        assert_eq!(req.provenance_id, "prov-1");
        assert_eq!(req.model_name, "gpt2");
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.weight_id, "");
        assert_eq!(req.max_tokens, 1024);
        assert_eq!(req.temperature, 0.7);
        assert_eq!(req.device, "CPU");
        assert_eq!(req.dtype, "F32");
        assert!(req.context.is_empty());
        assert!(req.skill_refs.is_empty());
    }

    #[test]
    fn inference_request_builder_weight_id() {
        let req = InferenceRequest::new("p", "m", "t").weight_id("w-1");
        assert_eq!(req.weight_id, "w-1");
    }

    #[test]
    fn inference_request_builder_context() {
        let req = InferenceRequest::new("p", "m", "t").context(vec!["a", "b"]);
        assert_eq!(req.context.len(), 2);
        assert_eq!(req.context[0], "a");
    }

    #[test]
    fn inference_request_builder_skill_refs() {
        let req = InferenceRequest::new("p", "m", "t").skill_refs(vec!["s1"]);
        assert_eq!(req.skill_refs.len(), 1);
        assert_eq!(req.skill_refs[0], "s1");
    }

    #[test]
    fn inference_request_builder_device() {
        let req = InferenceRequest::new("p", "m", "t").device("cuda");
        assert_eq!(req.device, "cuda");
    }

    #[test]
    fn inference_request_builder_dtype() {
        let req = InferenceRequest::new("p", "m", "t").dtype("F16");
        assert_eq!(req.dtype, "F16");
    }

    #[test]
    fn inference_request_builder_max_tokens() {
        let req = InferenceRequest::new("p", "m", "t").max_tokens(2048);
        assert_eq!(req.max_tokens, 2048);
    }

    #[test]
    fn inference_request_temperature_clamped() {
        let req = InferenceRequest::new("p", "m", "t").temperature(-1.0);
        assert_eq!(req.temperature, 0.0);

        let req = InferenceRequest::new("p", "m", "t").temperature(3.0);
        assert_eq!(req.temperature, 2.0);
    }

    #[test]
    fn inference_request_serializes() {
        let req = InferenceRequest::new("prov-1", "gpt2", "hello")
            .weight_id("w-1")
            .device("cuda")
            .dtype("F16")
            .max_tokens(512)
            .temperature(0.8);
        let json = serde_json::to_string(&req).unwrap();
        let restored: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.provenance_id, "prov-1");
        assert_eq!(restored.weight_id, "w-1");
        assert_eq!(restored.temperature, 0.8);
    }

    // ── InferenceResponse ──────────────────────────────────────────────

    #[test]
    fn inference_response_new_sets_defaults() {
        let resp = InferenceResponse::new("prov-1", "gpt2", "output");
        assert_eq!(resp.provenance_id, "prov-1");
        assert_eq!(resp.model_name, "gpt2");
        assert_eq!(resp.output, "output");
        assert_eq!(resp.confidence, 0.5);
        assert_eq!(resp.exit_code, 0);
        assert_eq!(resp.weight_id, "");
        assert!(resp.tokens.is_empty());
        assert_eq!(resp.output_hash, "");
        assert!(resp.skill_residue.is_none());
    }

    #[test]
    fn inference_response_builder_weight_id() {
        let resp = InferenceResponse::new("p", "m", "o").weight_id("w-1");
        assert_eq!(resp.weight_id, "w-1");
    }

    #[test]
    fn inference_response_builder_tokens() {
        let resp = InferenceResponse::new("p", "m", "o").tokens(vec!["a", "b"]);
        assert_eq!(resp.tokens.len(), 2);
    }

    #[test]
    fn inference_response_builder_confidence() {
        let resp = InferenceResponse::new("p", "m", "o").confidence(0.95);
        assert_eq!(resp.confidence, 0.95);
    }

    #[test]
    fn inference_response_builder_output_hash() {
        let resp = InferenceResponse::new("p", "m", "o").output_hash("hash123");
        assert_eq!(resp.output_hash, "hash123");
    }

    #[test]
    fn inference_response_builder_skill_residue() {
        let resp = InferenceResponse::new("p", "m", "o").skill_residue("residue");
        assert_eq!(resp.skill_residue, Some("residue".to_string()));
    }

    #[test]
    fn inference_response_builder_exit_code() {
        let resp = InferenceResponse::new("p", "m", "o").exit_code(1);
        assert_eq!(resp.exit_code, 1);
    }

    #[test]
    fn inference_response_confidence_clamped() {
        let resp = InferenceResponse::new("p", "m", "o").confidence(-1.0);
        assert_eq!(resp.confidence, 0.0);

        let resp = InferenceResponse::new("p", "m", "o").confidence(2.0);
        assert_eq!(resp.confidence, 1.0);
    }

    #[test]
    fn inference_response_serializes() {
        let resp = InferenceResponse::new("prov-1", "gpt2", "output")
            .weight_id("w-1")
            .confidence(0.9)
            .exit_code(0);
        let json = serde_json::to_string(&resp).unwrap();
        let restored: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.provenance_id, "prov-1");
        assert_eq!(restored.confidence, 0.9);
    }

    // ── RunnerConfig ───────────────────────────────────────────────────

    #[test]
    fn runner_config_default_values() {
        let config = RunnerConfig::default();
        assert_eq!(config.runner_name, "default");
        assert_eq!(config.runner_type, "external");
        assert!(config.endpoint.is_empty());
        assert_eq!(config.protocol, "json");
        assert_eq!(config.weight_manifest_format, "json");
        assert_eq!(config.inference_request_format, "json");
        assert_eq!(config.inference_response_format, "json");
        assert_eq!(config.device_preference, "CPU");
        assert_eq!(config.dtype_preference, "F32");
        assert_eq!(config.max_tokens_default, 1024);
        assert_eq!(config.temperature_default, 0.7);
    }

    #[test]
    fn runner_config_builder_with_runner_name() {
        let config = RunnerConfig::default().with_runner_name("custom");
        assert_eq!(config.runner_name, "custom");
    }

    #[test]
    fn runner_config_builder_with_runner_type() {
        let config = RunnerConfig::default().with_runner_type("http");
        assert_eq!(config.runner_type, "http");
    }

    #[test]
    fn runner_config_builder_with_endpoint() {
        let config = RunnerConfig::default().with_endpoint("http://localhost:3000");
        assert_eq!(config.endpoint, "http://localhost:3000");
    }

    #[test]
    fn runner_config_builder_with_protocol() {
        let config = RunnerConfig::default().with_protocol("grpc");
        assert_eq!(config.protocol, "grpc");
    }

    #[test]
    fn runner_config_builder_with_device() {
        let config = RunnerConfig::default().with_device("cuda");
        assert_eq!(config.device_preference, "cuda");
    }

    #[test]
    fn runner_config_builder_with_dtype() {
        let config = RunnerConfig::default().with_dtype("F16");
        assert_eq!(config.dtype_preference, "F16");
    }

    #[test]
    fn runner_config_serializes_and_deserializes() {
        let config = RunnerConfig {
            runner_name: "test".into(),
            runner_type: "http".into(),
            endpoint: "http://localhost:3000".into(),
            protocol: "json".into(),
            weight_manifest_format: "json".into(),
            inference_request_format: "json".into(),
            inference_response_format: "json".into(),
            device_preference: "cuda".into(),
            dtype_preference: "F16".into(),
            max_tokens_default: 2048,
            temperature_default: 0.9,
            configured_at: 12345,
            device_priority: vec!["cuda".into(), "cpu".into()],
            remote_endpoints: vec![],
        };
        let json = serde_json::to_string(&config).unwrap();
        let restored: RunnerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.runner_name, "test");
        assert_eq!(restored.max_tokens_default, 2048);
        assert_eq!(restored.temperature_default, 0.9);
    }

    #[test]
    fn runner_config_clone_works() {
        let config = RunnerConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.runner_name, config.runner_name);
    }

    // ── PlugInError ────────────────────────────────────────────────────

    #[test]
    fn plugin_error_not_found_message() {
        let err = PlugInError::NotFound("model x".into());
        assert!(err.to_string().contains("model x"));
    }

    #[test]
    fn plugin_error_config_invalid_message() {
        let err = PlugInError::ConfigInvalid("bad value".into());
        assert!(err.to_string().contains("bad value"));
    }

    #[test]
    fn plugin_error_protocol_mismatch_message() {
        let err = PlugInError::ProtocolMismatch("grpc vs json".into());
        assert!(err.to_string().contains("grpc vs json"));
    }

    #[test]
    fn plugin_error_internal_message() {
        let err = PlugInError::Internal("boom".into());
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn plugin_error_sqlite_from_rusqlite() {
        let err: PlugInError = rusqlite::Error::QueryReturnedNoRows.into();
        assert!(err.to_string().contains("sqlite") || !err.to_string().is_empty());
    }

    #[test]
    fn plugin_error_json_from_serde() {
        let err: PlugInError = serde_json::from_str::<serde_json::Value>("not json")
            .unwrap_err()
            .into();
        assert!(err.to_string().contains("JSON"));
    }
}
