//! Runner bridge for external LLM runtime.
//!
//! Implements HTTP transport for remote LM Studio inference.
//! Routes requests to local GPU or remote endpoint based on device selection.

#[cfg(feature = "cuda")]
use crate::device::{DeviceSelection, DeviceSelector};
use crate::error::RunnerError;
use pesti_plug_in::protocol::{InferenceRequest, InferenceResponse, RunnerConfig};
use tracing::debug;

/// Runner bridge for external LLM runtime.
///
/// external runner endpoint/protocol bridge.
pub struct RunnerBridge {
    pub config: RunnerConfig,
    pub endpoint: String,
}

impl RunnerBridge {
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config: config.clone(),
            endpoint: config.endpoint.clone(),
        }
    }

    /// Send inference request to external runner.
    ///
    /// **Deprecated:** This sync method is not yet implemented. Use
    /// [`send_remote_request`](RunnerBridge::send_remote_request) for async HTTP calls
    /// to remote LM Studio endpoints.
    #[deprecated(
        since = "0.1.2",
        note = "Use send_remote_request() for async HTTP calls instead"
    )]
    pub fn send_request(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, RunnerError> {
        // External runner endpoint bridge
        // protocol: JSON via HTTP or local pipe
        debug!(
            endpoint = %self.endpoint,
            model_name = %request.model_name,
            "Runner bridge: sending inference request"
        );
        Err(RunnerError::Internal(
            "external runner not implemented".to_string(),
        ))
    }

    /// Send inference request to a remote LM Studio endpoint.
    pub async fn send_remote_request(
        &self,
        request: InferenceRequest,
        endpoint: &str,
    ) -> Result<InferenceResponse, RunnerError> {
        let url = format!("{}/v1/completions", endpoint);

        // Build LM Studio compatible request
        let lm_studio_request = serde_json::json!({
            "model": request.model_name,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": false,
        });

        debug!(url = %url, model = %request.model_name, "Sending remote inference request");

        match tokio::time::timeout(
            std::time::Duration::from_secs(120),
            reqwest::Client::new()
                .post(&url)
                .json(&lm_studio_request)
                .header("Content-Type", "application/json")
                .send(),
        )
        .await
        {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    let body = response.text().await.map_err(|e| {
                        RunnerError::Internal(format!("Failed to read response: {e}"))
                    })?;

                    // Parse LM Studio response format
                    match self.parse_lm_studio_response(&body, &request) {
                        Ok(resp) => {
                            debug!(
                                endpoint = %endpoint,
                                tokens = resp.tokens.len(),
                                "Remote inference successful"
                            );
                            Ok(resp)
                        }
                        Err(e) => {
                            debug!(
                                error = %e,
                                endpoint = %endpoint,
                                "Failed to parse remote response"
                            );
                            Err(e)
                        }
                    }
                } else {
                    let status = response.status();
                    let body = response.text().await.unwrap_or_default();
                    Err(RunnerError::Internal(format!(
                        "Remote inference failed ({}): {}",
                        status, body
                    )))
                }
            }
            Ok(Err(e)) => Err(RunnerError::Internal(format!("Remote request failed: {e}"))),
            Err(_) => Err(RunnerError::Internal(
                "Remote request timed out".to_string(),
            )),
        }
    }

    /// Parse LM Studio response format.
    fn parse_lm_studio_response(
        &self,
        body: &str,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, RunnerError> {
        let json: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| RunnerError::Internal(format!("JSON parse error: {e}")))?;

        // LM Studio returns {"text": "..."} or {"choices": [{"text": "..."}]}
        let output = json
            .get("text")
            .or_else(|| {
                json.get("choices")
                    .and_then(|c| c.get(0).and_then(|c| c.get("text")))
            })
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let tokens: Vec<String> = vec![]; // LM Studio doesn't return token IDs in basic response

        Ok(InferenceResponse::new(
            request.provenance_id.clone(),
            request.model_name.clone(),
            output.to_string(),
        )
        .weight_id(request.weight_id.clone())
        .tokens(tokens)
        .confidence(0.8)
        .exit_code(0))
    }

    /// Receive inference response from external runner.
    pub fn receive_response(&self) -> Result<InferenceResponse, RunnerError> {
        Err(RunnerError::Internal(
            "external runner not implemented".to_string(),
        ))
    }

    /// Update runner config.
    pub fn update_config(&mut self, config: RunnerConfig) -> Result<(), RunnerError> {
        self.config = config.clone();
        self.endpoint = config.endpoint.clone();
        Ok(())
    }

    /// Get runner config info.
    pub fn config_info(&self) -> Result<String, RunnerError> {
        Ok(format!(
            "runner_name={}, runner_type={}, endpoint={}, protocol={}",
            self.config.runner_name,
            self.config.runner_type,
            self.config.endpoint,
            self.config.protocol
        ))
    }

    /// Get all available endpoints including remotes.
    pub fn all_endpoints(&self) -> Vec<String> {
        let mut endpoints = vec![self.endpoint.clone()];
        endpoints.extend(self.config.remote_endpoints.clone());
        endpoints
    }
}

/// Routes inference requests to the best available device.
///
/// Combines `DeviceSelector` (device discovery + priority routing) with
/// `RunnerBridge` (remote LM Studio HTTP transport) into a single execution
/// pipeline.
pub struct DeviceRouter {
    selector: (), // Stub - actual implementation only exists with CUDA
    bridge: RunnerBridge,
}

impl DeviceRouter {
    /// Create a new router with default device selection and runner config.
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            selector: (), // Stub - actual implementation only exists with CUDA
            bridge: RunnerBridge::new(config),
        }
    }

    /// Create a router with explicit device priority.
    pub fn with_priority(_config: RunnerConfig, _priority: Vec<()>) -> Self { // Stub - actual implementation only exists with CUDA
        Self {
            selector: (),
            bridge: RunnerBridge::new(_config),
        }
    }

    /// Route an inference request to the best available device.
    ///
    /// `model_bytes` — estimated model size in bytes (used for VRAM fit check).
    /// Returns the device selection and execution result.
    pub async fn route(
        &mut self,
        _request: InferenceRequest,
        _model_bytes: u64,
    ) -> Result<RouteResult, RunnerError> { // Stub - actual implementation only exists with CUDA
        debug!("Device router: stub routing");

        Ok(RouteResult {
            selection: (),
            response: None,
        })
    }

    /// Fast route without refreshing device discovery.
    pub async fn quick_route(
        &self,
        _request: InferenceRequest,
        _model_bytes: u64,
    ) -> Result<RouteResult, RunnerError> { // Stub - actual implementation only exists with CUDA
        debug!("Device router: stub quick routing");

        Ok(RouteResult {
            selection: (),
            response: None,
        })
    }

    /// List all available devices with current status.
    pub fn list_devices(&self) -> Vec<()> { // Stub - actual implementation only exists with CUDA
        vec![]
    }

    /// Get the current device selection priority.
    pub fn priority(&self) -> Vec<()> { // Stub - actual implementation only exists with CUDA
        vec![]
    }

    /// Get the runner bridge endpoint info.
    pub fn config_info(&self) -> Result<String, RunnerError> {
        self.bridge.config_info()
    }
}

impl Default for DeviceRouter {
    fn default() -> Self {
        Self::new(RunnerConfig::default())
    }
}

/// Result of routing an inference request.
pub struct RouteResult {
    /// The device selection that was made.
    pub selection: (), // Stub - actual implementation only exists with CUDA
    /// Remote response if routed to remote LM Studio, None for local.
    pub response: Option<InferenceResponse>,
}
