//! pesti-server: OpenAI-compatible HTTP server for pesti-runner inference.
//!
//! Uses spawn_blocking with nested tokio runtimes to handle blocking llama.cpp
//! inference off the async runtime's thread pool.

use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use pesti_runner::llama::SamplingConfig;
use pesti_runner::runtime::Runtime;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing_subscriber::{EnvFilter, fmt};
use uuid::Uuid;

// ── OpenAI API request/response types ────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_p")]
    top_p: f64,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn default_temperature() -> f64 { 0.7 }
fn default_top_p() -> f64 { 1.0 }
fn default_max_tokens() -> u32 { 512 }

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: i32,
    completion_tokens: i32,
    total_tokens: i32,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: i64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

// ── Server state ─────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {}

// ── Handlers ────────────────────────────────────────────────────────

async fn list_models(
    State(_app): State<AppState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<serde_json::Value>)> {
    let available = tokio::task::spawn_blocking(|| {
        // Create a new runtime on this thread for model discovery
        let rt = Runtime::new();
        rt.list_available()
    }).await.map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "Blocking task failed"
        })))
    })?;

    let data: Vec<ModelInfo> = available
        .into_iter()
        .map(|name| ModelInfo {
            id: name.clone(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "pesti".to_string(),
        })
        .collect();

    Ok(Json(ModelsResponse {
        object: "list".to_string(),
        data,
    }))
}

async fn chat_completions(
    State(_app): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<serde_json::Value>)> {
    // Build prompt from messages on async thread
    let mut prompt = String::new();
    for msg in &req.messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("[SYSTEM]\n{}\n\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("[USER]\n{}\n\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("[ASSISTANT]\n{}\n\n", msg.content));
            }
            other => {
                prompt.push_str(&format!("[{}]\n{}\n\n", other, msg.content));
            }
        }
    }

    let model = req.model.clone();
    let temperature = req.temperature;
    let top_p = req.top_p;
    let max_tokens = req.max_tokens;

    // Spawn blocking task for inference (llama.cpp is not thread-safe)
    let result = tokio::task::spawn_blocking(move || {
        // Create a fresh tokio runtime on this thread to drive async pesti operations
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime");

        let pesti_rt = Runtime::new();

        // Load model
        match rt.block_on(pesti_rt.load_model(&model)) {
            Ok(_) => {}
            Err(e) => {
                return Err(format!("Failed to load model '{}': {}", model, e));
            }
        }

        // Generate
        let config = SamplingConfig {
            temperature,
            top_p,
            max_tokens,
            ..Default::default()
        };

        match pesti_rt.generate(&prompt, &config) {
            Ok(generation) => Ok((generation.text, generation.prompt_tokens, generation.generated_tokens)),
            Err(e) => Err(format!("Generation failed: {}", e)),
        }
    }).await.map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "Blocking task failed"
        })))
    })?;

    match result {
        Ok((text, prompt_tokens, generated_tokens)) => {
            let id = format!("chatcmpl-{}", Uuid::new_v4().to_string().chars().take(8).collect::<String>());
            let response = ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created: Utc::now().timestamp(),
                model: req.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: text,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Usage {
                    prompt_tokens: prompt_tokens as i32,
                    completion_tokens: generated_tokens as i32,
                    total_tokens: (prompt_tokens + generated_tokens) as i32,
                },
            };
            Ok(Json(response))
        }
        Err(msg) => {
            Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": msg
            }))))
        }
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "pesti-server"
    }))
}

// ── Main ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Initialize logging
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info")
            .add_directive("pesti_server=debug".parse().unwrap())
            .add_directive("axum=info".parse().unwrap())
            .add_directive("tower=info".parse().unwrap())
    });

    fmt()
        .with_writer(std::io::stderr)
        .pretty()
        .init();

    // Parse command-line args for port
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 8080;

    if let Ok(p) = std::env::var("PESTI_SERVER_PORT") {
        if let Ok(parsed) = p.parse::<u16>() {
            port = parsed;
        }
    }

    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    if let Ok(parsed) = args[i + 1].parse::<u16>() {
                        port = parsed;
                    }
                }
            }
            _ => {}
        }
    }

    let app_state = AppState {};

    // Build router matching OpenAI API paths
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health_check))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(addr = %addr, "pesti-server listening");

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .await
        .expect("server failed");
}
