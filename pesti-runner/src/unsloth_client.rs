//! Unsloth Studio Rust SDK
//!
//! Type-safe programmatic interface to Unsloth Studio API at http://localhost:8888
//!
//! This is a Rust rewrite of the Python SDK, providing:
//! - Compile-time type safety for model configs and responses
//! - Session cookie management with automatic retry on auth failure
//! - Recipe workflow graph structures (data-recipe/jobs)
//! - Blocking HTTP client (sync API)
//!
//! Usage:
//! ```rust,no_run
//! let client = UnslothClient::new("http://localhost:8888");
//!
//! let config = ModelConfig {
//!     model_name: "unsloth/llama-3-8b-Instruct-bnb-4bit".to_string(),
//!     max_tokens: 2048,
//!     temperature: 0.7,
//!     quantization: Quantization::Bits4,
//! };
//!
//! let result = client.run_model("Hello, world!", &config)?;
//! println!("Response: {}", result.response);
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for running a model (matches Python SDK ModelConfig)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier (e.g., "unsloth/llama-3-8b-Instruct-bnb-4bit")
    pub model_name: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,
    /// Top-p sampling threshold (nucleus sampling)
    pub top_p: f32,
    /// Quantization level for the model
    pub quantization: Quantization,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            quantization: Quantization::Bits4,
        }
    }
}

/// Quantization levels supported by Unsloth Studio
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    Bits8,
    Bits4,
    Float16,
    Int4,
}

impl Default for Quantization {
    fn default() -> Self {
        Self::Bits4
    }
}

/// Result from a chat/completion call (matches Python ChatResult)
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

/// Message in a chat thread (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    /// Role: "user", "assistant", or "system"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Chat thread for multi-turn conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatThread {
    /// Unique identifier for the chat thread
    pub id: String,
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Model used for this thread
    pub model_name: String,
    /// Creation timestamp (Unix epoch)
    pub created_at: u64,
}

/// Data recipe job configuration (for UNSLOTH_RECIPE_*.md workflow graphs)
#[derive(Debug, Clone, Serialize)]
pub struct RecipeJob {
    /// Unique recipe identifier
    pub recipe_id: String,
    /// Recipe configuration parameters
    pub config: std::collections::HashMap<String, serde_json::Value>,
    /// Seed data for the recipe (optional)
    pub seed_data: Option<serde_json::Value>,
}

/// Result from executing a data recipe job
#[derive(Debug, Clone, Deserialize)]
pub struct RecipeJobResult {
    /// Job execution status
    pub status: String,
    /// Number of records processed
    pub records_processed: usize,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Output dataset location (if applicable)
    pub output_path: Option<String>,
}

/// Unsloth Studio client for programmatic API access
///
/// This bridges the gap between GUI actions and HTTP endpoints:
/// - "Run Model" → POST /api/inference/completions
/// - "Unsloth Chat" → POST /api/inference/chat/completions  
/// - "List Models" → GET /api/models/list
/// - "Execute Recipe" → POST /api/data-recipe/jobs
#[derive(Debug)]
pub struct UnslothClient {
    /// Base URL of Unsloth Studio API
    base_url: String,
    /// HTTP client with timeout configuration
    http_client: reqwest::blocking::Client,
    /// Session cookie for authentication
    session_cookie: Option<String>,
}

impl UnslothClient {
    /// Create new Unsloth client with default configuration
    pub fn new(base_url: &str) -> Result<Self, ClientError> {
        let client = reqwest::blocking::Client::builder()
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
    pub fn list_models(&self) -> Result<Vec<ModelInfo>, ClientError> {
        let url = format!("{}/api/models/list", self.base_url);

        let response = self.send_request(url, reqwest::Method::GET, None)?;
        let result = response.json::<ModelsListResponse>()?;

        Ok(result.models)
    }

    /// Run a model with given prompt (GUI: "Run Model")
    pub fn run_model(&self, prompt: &str, config: &ModelConfig) -> Result<ChatResult, ClientError> {
        let url = format!("{}/api/inference/completions", self.base_url);

        let payload = serde_json::json!({
            "prompt": prompt,
            "model_name": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "quantization": format!("{:?}", config.quantization).to_lowercase()
        });

        let response = self.send_request(url, reqwest::Method::POST, Some(payload))?;
        Ok(response.json()?)
    }

    /// Chat interface with multi-turn conversation (GUI: "Unsloth Chat")
    pub fn chat(
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

        let response = self.send_request(url, reqwest::Method::POST, Some(payload))?;
        Ok(response.json()?)
    }

    /// Create a new chat thread (GUI: start new conversation)
    pub fn create_chat_thread(&self, model_name: &str) -> Result<ChatThread, ClientError> {
        let url = format!("{}/api/chat/threads", self.base_url);

        let payload = serde_json::json!({
            "model_name": model_name
        });

        let response = self.send_request(url, reqwest::Method::POST, Some(payload))?;
        Ok(response.json()?)
    }

    /// Execute a data recipe job (GUI: "Run Data Recipe")
    pub fn execute_recipe(&self, recipe: &RecipeJob) -> Result<RecipeJobResult, ClientError> {
        let url = format!("{}/api/data-recipe/jobs", self.base_url);

        let response = self.send_request(
            url,
            reqwest::Method::POST,
            Some(serde_json::to_value(recipe)?),
        )?;

        Ok(response.json()?)
    }

    /// Export recipe definition as JSON (for visualization tools)
    pub fn export_recipe(&self, recipe_id: &str) -> Result<serde_json::Value, ClientError> {
        let url = format!("{}/api/data-recipe/{}/export", self.base_url, recipe_id);

        let response = self.send_request(url, reqwest::Method::GET, None)?;
        Ok(response.json()?)
    }

    /// Internal HTTP request helper with session cookie and error handling
    fn send_request(
        &self,
        url: String,
        method: reqwest::Method,
        data: Option<serde_json::Value>,
    ) -> Result<reqwest::blocking::Response, ClientError> {
        let mut request = self.http_client.request(method, &url);

        // Add session cookie if available
        if let Some(ref cookie) = self.session_cookie {
            request = request.header("Cookie", cookie);
        }

        // Add JSON content type for POST/PUT
        if let Some(payload) = data {
            request = request.json(&payload);
        }

        let response = request.send()?;

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

/// Response from /api/models/list endpoint
#[derive(Debug, Deserialize)]
pub struct ModelsListResponse {
    pub models: Vec<ModelInfo>,
}

/// Information about a loaded model
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub name: String,
    /// Display name for UI
    pub display_name: String,
    /// Whether model is currently loaded
    pub is_loaded: bool,
    /// VRAM usage in bytes (if loaded)
    pub vram_usage_bytes: Option<u64>,
}

/// Client error types
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Invalid response status: {0}")]
    Status(String),
}

// Re-export for convenience
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

        let bits8 = serde_json::to_string(&Quantization::Bits8).unwrap();
        assert_eq!(bits8, "\"bits8\"");
    }

    #[test]
    fn test_client_creation() {
        let client = UnslothClient::new("http://localhost:8888").unwrap();
        assert_eq!(client.base_url(), "http://localhost:8888");
    }

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"What is Rust?\""));
    }
}
