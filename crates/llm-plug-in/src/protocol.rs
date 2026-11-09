use serde::{Deserialize, Serialize};

/// Inference request sent to external LLM runner.
///
/// prompt + context + skill_refs → runner produces InferenceResponse.
/// provenance_id tracked for guard gate consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub provenance_id: String,
    pub model_name: String,
    pub weight_id: String,
    pub prompt: String,
    pub context: Vec<String>,
    pub skill_refs: Vec<String>,
    pub max_tokens: u32,
    pub temperature: f64,
    pub device: String,
    pub dtype: String,
    pub requested_at: i64,
}

impl InferenceRequest {
    pub fn new(
        provenance_id: impl Into<String>,
        model_name: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            provenance_id: provenance_id.into(),
            model_name: model_name.into(),
            weight_id: String::new(),
            prompt: prompt.into(),
            context: Vec::new(),
            skill_refs: Vec::new(),
            max_tokens: 1024,
            temperature: 0.7,
            device: "CPU".to_string(),
            dtype: "F32".to_string(),
            requested_at: chrono::Utc::now().timestamp(),
        }
    }

    pub fn weight_id(mut self, weight_id: impl Into<String>) -> Self {
        self.weight_id = weight_id.into();
        self
    }

    pub fn context(mut self, context: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.context.extend(context.into_iter().map(Into::into));
        self
    }

    pub fn skill_refs(mut self, refs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.skill_refs.extend(refs.into_iter().map(Into::into));
        self
    }

    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    pub fn dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = dtype.into();
        self
    }

    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp.clamp(0.0, 2.0);
        self
    }
}

/// Inference response from external LLM runner.
///
/// Structured output for guard gate consumption.
/// not raw text — structured JSON fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub provenance_id: String,
    pub model_name: String,
    pub weight_id: String,
    pub tokens: Vec<String>,
    pub output: String,
    pub confidence: f64,
    pub exit_code: i32,
    pub output_hash: String,
    pub skill_residue: Option<String>,
    pub created_at: i64,
}

impl InferenceResponse {
    pub fn new(
        provenance_id: impl Into<String>,
        model_name: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            provenance_id: provenance_id.into(),
            model_name: model_name.into(),
            weight_id: String::new(),
            tokens: Vec::new(),
            output: output.into(),
            confidence: 0.5,
            exit_code: 0,
            output_hash: String::new(),
            skill_residue: None,
            created_at: chrono::Utc::now().timestamp(),
        }
    }

    pub fn weight_id(mut self, weight_id: impl Into<String>) -> Self {
        self.weight_id = weight_id.into();
        self
    }

    pub fn tokens(mut self, tokens: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tokens.extend(tokens.into_iter().map(Into::into));
        self
    }

    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn output_hash(mut self, hash: impl Into<String>) -> Self {
        self.output_hash = hash.into();
        self
    }

    pub fn skill_residue(mut self, residue: impl Into<String>) -> Self {
        self.skill_residue = Some(residue.into());
        self
    }

    pub fn exit_code(mut self, code: i32) -> Self {
        self.exit_code = code;
        self
    }
}

/// Runner config for external LLM runtime.
///
/// external runner endpoint/protocol configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    pub runner_name: String,
    pub runner_type: String,
    pub endpoint: String,
    pub protocol: String,
    pub weight_manifest_format: String,
    pub inference_request_format: String,
    pub inference_response_format: String,
    pub device_preference: String,
    pub dtype_preference: String,
    pub max_tokens_default: u32,
    pub temperature_default: f64,
    pub configured_at: i64,
    /// Additional remote endpoints for hybrid inference.
    pub remote_endpoints: Vec<String>,
    /// Device priority list (e.g., ["cuda:0", "cuda:1", "remote", "cpu"]).
    pub device_priority: Vec<String>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            runner_name: "default".to_string(),
            runner_type: "external".to_string(),
            endpoint: String::new(),
            protocol: "json".to_string(),
            weight_manifest_format: "json".to_string(),
            inference_request_format: "json".to_string(),
            inference_response_format: "json".to_string(),
            device_preference: "CPU".to_string(),
            dtype_preference: "F32".to_string(),
            max_tokens_default: 1024,
            temperature_default: 0.7,
            configured_at: chrono::Utc::now().timestamp(),
            remote_endpoints: Vec::new(),
            device_priority: Vec::new(),
        }
    }
}

impl RunnerConfig {
    pub fn with_runner_name(mut self, name: impl Into<String>) -> Self {
        self.runner_name = name.into();
        self
    }

    pub fn with_runner_type(mut self, type_: impl Into<String>) -> Self {
        self.runner_type = type_.into();
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_protocol(mut self, protocol: impl Into<String>) -> Self {
        self.protocol = protocol.into();
        self
    }

    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device_preference = device.into();
        self
    }

    pub fn with_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype_preference = dtype.into();
        self
    }

    pub fn with_remote_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.remote_endpoints.push(endpoint.into());
        self
    }

    pub fn with_remote_endpoints(mut self, endpoints: Vec<String>) -> Self {
        self.remote_endpoints = endpoints;
        self
    }

    pub fn with_device_priority(mut self, priority: Vec<String>) -> Self {
        self.device_priority = priority;
        self
    }
}
