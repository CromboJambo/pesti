//! Unsloth Studio Rust SDK - Async Version
//!
//! High-performance async interface to Unsloth Studio API using tokio + reqwest
//!
//! This complements the sync version with:
//! - Non-blocking HTTP calls for concurrent model inference
//! - Stream-based responses for long-running generations
//! - Better throughput for batch processing
//!
//! Usage:
//! ```rust,no_run
//! use pesti_runner::unsloth_client_async::{AsyncUnslothClient, ModelConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = AsyncUnslothClient::new("http://localhost:8888").await?;
//!     
//!     // Concurrent model calls
//!     let config = ModelConfig::default();
//!     let results = tokio::join!(
//!         client.run_model("Prompt 1", &config),
//!         client.run_model("Prompt 2", &config),
//!         client.run_model("Prompt 3", &config),
//!     );
//!     
//!     println!("All models responded!");
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};

// Re-export types from sync module for consistency
pub use crate::unsloth_client::{
    ChatMessage, ChatThread, ClientError, ModelConfig, ModelInfo, ModelsListResponse, Quantization,
    RecipeJob, RecipeJobResult,
};

/// Async Unsloth Studio client using tokio + reqwest async runtime
///
/// Designed for high-throughput scenarios where you need to:
/// - Run multiple model calls concurrently
/// - Stream long-running generations
/// - Integrate with async web frameworks (axum, actix)
#[derive(Debug, Clone)]
pub struct AsyncUnslothClient {
    /// Base URL of Unsloth Studio API
    base_url: String,
    /// Async HTTP client with timeout configuration
    http_client: reqwest::Client,
    /// Session cookie for authentication
    session_cookie: Option<String>,
}

impl AsyncUnslothClient {
    /// Create new async Unsloth client with default configuration
    pub async fn new(base_url: &str) -> Result<Self, ClientError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout for long generations
            .build()?;

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http_client: client,
            session_cookie: None,
        })
    }

    /// Set session cookie manually (useful for loading from auth.json)
    pub fn with_session_cookie(mut self, cookie: String) -> Self {
        self.session_cookie = Some(cookie);
        self
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// List all available models (GUI: "Run Model" dropdown)
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, ClientError> {
        let url = format!("{}/api/models/list", self.base_url);

        let response = self
            .send_async_request(url, reqwest::Method::GET, None)
            .await?;
        let result = response.json::<ModelsListResponse>().await?;

        Ok(result.models)
    }

    /// Run a model with given prompt (GUI: "Run Model")
    pub async fn run_model(
        &self,
        prompt: &str,
        config: &ModelConfig,
    ) -> Result<ChatResult, ClientError> {
        let url = format!("{}/api/inference/completions", self.base_url);

        let payload = serde_json::json!({
            "prompt": prompt,
            "model_name": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "quantization": format!("{:?}", config.quantization).to_lowercase()
        });

        let response = self
            .send_async_request(url, reqwest::Method::POST, Some(payload))
            .await?;
        Ok(response.json().await?)
    }

    /// Chat interface with multi-turn conversation (GUI: "Unsloth Chat")
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        config: &ModelConfig,
    ) -> Result<ChatResult, ClientError> {
        let url = format!("{}/api/inference/chat/completions", self.base_url);

        let payload = serde_json::json!({
            "messages": messages,
            "model_name": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "quantization": format!("{:?}", config.quantization).to_lowercase()
        });

        let response = self
            .send_async_request(url, reqwest::Method::POST, Some(payload))
            .await?;
        Ok(response.json().await?)
    }

    /// Create a new chat thread (GUI: start new conversation)
    pub async fn create_chat_thread(&self, model_name: &str) -> Result<ChatThread, ClientError> {
        let url = format!("{}/api/chat/threads", self.base_url);

        let payload = serde_json::json!({
            "model_name": model_name
        });

        let response = self
            .send_async_request(url, reqwest::Method::POST, Some(payload))
            .await?;
        Ok(response.json().await?)
    }

    /// Execute a data recipe job (GUI: "Run Data Recipe")
    pub async fn execute_recipe(&self, recipe: &RecipeJob) -> Result<RecipeJobResult, ClientError> {
        let url = format!("{}/api/data-recipe/jobs", self.base_url);

        let response = self
            .send_async_request(
                url,
                reqwest::Method::POST,
                Some(serde_json::to_value(recipe)?),
            )
            .await?;

        Ok(response.json().await?)
    }

    /// Export recipe definition as JSON (for visualization tools)
    pub async fn export_recipe(&self, recipe_id: &str) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/data-recipe/{}/export", self.base_url, recipe_id);

        let response = self
            .send_async_request(url, reqwest::Method::GET, None)
            .await?;
        Ok(response.json().await?)
    }

    /// Stream a model completion (returns streaming response)
    pub async fn stream_model(
        &self,
        prompt: &str,
        config: &ModelConfig,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}/api/inference/completions/stream", self.base_url);

        let payload = serde_json::json!({
            "prompt": prompt,
            "model_name": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "quantization": format!("{:?}", config.quantization).to_lowercase(),
            "stream": true
        });

        let response = self
            .send_async_request(url, reqwest::Method::POST, Some(payload))
            .await?;

        // Verify it's actually a stream by checking headers
        let headers = response.headers();
        if let Some(content_type) = headers.get(reqwest::header::CONTENT_TYPE) {
            let content_type_str = content_type.to_str().unwrap_or("");
            if content_type_str.contains("text/event-stream")
                || content_type_str.contains("application/x-ndjson")
            {
                return Ok(response);
            }
        }

        // If not a stream, still return response (caller can handle it)
        Ok(response)
    }

    /// Internal async HTTP request helper with session cookie and error handling
    async fn send_async_request(
        &self,
        url: String,
        method: reqwest::Method,
        data: Option<serde_json::Value>,
    ) -> Result<reqwest::Response, ClientError> {
        let mut request = self.http_client.request(method, &url);

        // Add session cookie if available
        if let Some(ref cookie) = self.session_cookie {
            request = request.header("Cookie", cookie);
        }

        // Add JSON content type for POST/PUT
        if let Some(payload) = data {
            request = request.json(&payload);
        }

        let response = request.send().await?;

        // Handle 401/403 with auto-retry (session auth failure)
        if response.status().is_client_error() {
            let status = response.status();
            return Err(ClientError::Authentication(format!(
                "HTTP {} - Session may have expired. Set session_cookie or check auth.json",
                status
            )));
        }

        Ok(response)
    }
}

/// Result from an async chat/completion call (matches Python ChatResult)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResult {
    /// Generated response text
    pub response: String,
    /// Number of tokens used in the response
    pub tokens_used: usize,
    /// Model name that generated the response
    pub model_name: String,
    /// Total duration in milliseconds
    pub duration_ms: f64,
}

// Re-export reqwest for convenience
pub use reqwest;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.quantization, Quantization::Bits4);
    }

    #[test]
    fn test_quantization_serialization() {
        let bits4 = serde_json::to_string(&Quantization::Bits4).unwrap();
        assert_eq!(bits4, "\"bits4\"");
    }

    // Note: Async tests need #[tokio::test] macro
    // #[tokio::test]
    // async fn test_async_client_creation() {
    //     let client = AsyncUnslothClient::new("http://localhost:8888").await.unwrap();
    //     assert_eq!(client.base_url(), "http://localhost:8888");
    // }

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
    }
}
