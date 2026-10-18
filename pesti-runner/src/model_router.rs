//! JEV-powered model routing for pesti-runner.
//!
//! Observes incoming inference request characteristics and calls the crabjar-conductor's
//! /decide/evaluate endpoint to determine the optimal model for each task.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Request body for conductor's /decide/evaluate endpoint.
#[derive(Debug, Serialize)]
struct EvaluateRequest {
    observation: String,
    criteria_file: String,
}

/// Response from conductor's /decide/evaluate endpoint.
#[derive(Debug, Deserialize)]
struct EvaluateResponse {
    success: bool,
    option: String,
    confidence: f64,
    model: Option<String>,
    rationale: String,
}

/// Model routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub option_id: String,
    pub model_name: String,
    pub confidence: f64,
    pub rationale: String,
}

/// JEV-powered model router.
pub struct ModelRouter {
    conductor_url: String,
    criteria_file: String,
    client: reqwest::Client,
    /// Fallback model if conductor is unavailable.
    fallback_model: String,
}

impl ModelRouter {
    /// Create a new model router.
    pub fn new(conductor_url: &str, criteria_file: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_default();

        Self {
            conductor_url: conductor_url.to_string(),
            criteria_file: criteria_file.to_string(),
            client,
            fallback_model: "Qwen2.5-0.5B-Instruct-Q4_K_M".to_string(),
        }
    }

    /// Set the fallback model to use when conductor is unavailable.
    pub fn with_fallback(mut self, model: &str) -> Self {
        self.fallback_model = model.to_string();
        self
    }

    /// Route an inference request based on its characteristics.
    ///
    /// Constructs a JEV observation from the task parameters and calls
    /// the conductor's evaluation endpoint.
    pub async fn route(
        &self,
        task_type: &str,
        input_tokens: usize,
        max_output_tokens: usize,
        complexity_hints: &[&str],
    ) -> Result<RoutingDecision> {
        // Construct JEV observation from task characteristics
        let mut observation = format!(
            "Inference request: {} with {} input tokens, max {} output tokens",
            task_type, input_tokens, max_output_tokens
        );

        if !complexity_hints.is_empty() {
            observation.push_str(&format!(" Complexity hints: {}", complexity_hints.join(", ")));
        }

        debug!(observation = %observation, "Routing inference request via JEV");

        // Call conductor's /decide/evaluate endpoint
        let url = format!("{}/decide/evaluate", self.conductor_url);
        let body = EvaluateRequest {
            observation: observation.clone(),
            criteria_file: self.criteria_file.clone(),
        };

        match self.client.post(&url).json(&body).send().await {
            Ok(response) => {
                if !response.status().is_success() {
                    warn!(
                        status = response.status().as_u16(),
                        "Conductor evaluation endpoint returned non-success"
                    );
                    return Ok(self.fallback_decision());
                }

                let result: EvaluateResponse = response
                    .json()
                    .await
                    .context("Failed to parse evaluation response")?;

                if !result.success {
                    warn!(rationale = %result.rationale, "JEV evaluation failed");
                    return Ok(self.fallback_decision());
                }

                let model_name = result.model.unwrap_or_else(|| self.fallback_model.clone());

                info!(
                    option = %result.option,
                    confidence = result.confidence,
                    model = %model_name,
                    "JEV routed inference request"
                );

                Ok(RoutingDecision {
                    option_id: result.option,
                    model_name,
                    confidence: result.confidence,
                    rationale: result.rationale,
                })
            }
            Err(e) => {
                warn!(error = %e, "Failed to reach conductor for JEV evaluation");
                Ok(self.fallback_decision())
            }
        }
    }

    /// Create a fallback routing decision when conductor is unavailable.
    fn fallback_decision(&self) -> RoutingDecision {
        debug!("Using fallback model: {}", self.fallback_model);
        RoutingDecision {
            option_id: "fallback".to_string(),
            model_name: self.fallback_model.clone(),
            confidence: 0.5,
            rationale: "Conductor unavailable, using fallback model".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_construction() {
        let router = ModelRouter::new("http://localhost:8091", "/opt/crabjar/criteria/model-selection.json");
        assert_eq!(router.conductor_url, "http://localhost:8091");
        assert_eq!(router.criteria_file, "/opt/crabjar/criteria/model-selection.json");
    }

    #[test]
    fn fallback_decision() {
        let router = ModelRouter::new("http://localhost:8091", "/opt/crabjar/criteria/model-selection.json")
            .with_fallback("Qwen2.5-7B-Instruct-Q4_K_M");
        let decision = router.fallback_decision();
        assert_eq!(decision.option_id, "fallback");
        assert_eq!(decision.model_name, "Qwen2.5-7B-Instruct-Q4_K_M");
        assert_eq!(decision.confidence, 0.5);
    }
}
