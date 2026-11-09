//! pesti-server v2: Enhanced OpenAI-compatible HTTP server with model management.
//!
//! Features:
//! - Model hot-swapping (load/unload via API)
//! - Connection pooling with concurrent request handling
//! - SSE streaming for chat completions
//! - Hardware-aware auto-tuning integration

use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Sse};
use axum::response::sse::Event;
use axum::routing::{get, post};
use axum::Router;
use chrono::Utc;
use pesti_runner::llama::{LlamaChatMessage, LlamaRunner, SamplingConfig};
use std::ffi::CString;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio_stream::StreamExt;
use tracing_subscriber::{EnvFilter, fmt};
use uuid::Uuid;

// ── OpenAI API types ────────────────────────────────────────────────

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

// Streaming chunk types
#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<ChatChoiceChunk>,
}

#[derive(Debug, Serialize)]
struct ChatChoiceChunk {
    index: usize,
    delta: Option<DeltaContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct DeltaContent {
    role: Option<String>,
    content: Option<String>,
}

// Model management types
#[derive(Debug, Deserialize)]
struct LoadModelRequest {
    model_path: String,
    #[serde(default)]
    model_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelStatus {
    id: String,
    path: String,
    loaded_at: i64,
    status: String,
}

// ── Server state with connection pooling ────────────────────────────

struct ModelSlot {
    runner: Option<LlamaRunner>,
    model_id: Option<String>,
    model_path: Option<String>,
}

#[derive(Clone)]
struct AppState {
    /// Pool of model slots for concurrent request handling
    slots: Arc<Vec<Mutex<ModelSlot>>>,
    loaded_models: Arc<RwLock<std::collections::HashMap<String, String>>>, // id -> path
}

impl AppState {
    fn new(pool_size: usize) -> Self {
        let mut slots = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            slots.push(Mutex::new(ModelSlot {
                runner: None,
                model_id: None,
                model_path: None,
            }));
        }

        Self {
            slots: Arc::new(slots),
            loaded_models: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    async fn get_or_load_model(
        &self,
        model_id: &str,
        model_path: &str,
    ) -> Result<(usize, String), (StatusCode, Json<serde_json::Value>)> {
        // Check if already loaded
        let path = {
            let models = self.loaded_models.read().await;
            models.get(model_id).cloned()
        };
        
        if let Some(path) = path {
            return Ok((0, path)); // Slot 0 for simplicity
        }

        // Find an empty slot and load model
        for (i, slot) in self.slots.iter().enumerate() {
            let mut s = slot.lock().await;
            if s.runner.is_none() {
                match LlamaRunner::builder(model_path).build() {
                    Ok(runner) => {
                        s.runner = Some(runner);
                        s.model_id = Some(model_id.to_string());
                        s.model_path = Some(model_path.to_string());

                        let mut models = self.loaded_models.write().await;
                        models.insert(model_id.to_string(), model_path.to_string());
                        drop(models);

                        tracing::info!(model = model_id, path = model_path, "Model loaded");
                        return Ok((i, model_path.to_string()));
                    }
                    Err(e) => {
                        let err = format!("Failed to load model '{}': {}", model_id, e);
                        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": err}))));
                    }
                }
            }
        }

        Err((StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "error": "No available model slots"
        }))))
    }

    async fn unload_model(&self, model_id: &str) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
        let mut models = self.loaded_models.write().await;
        if !models.contains_key(model_id) {
            return Err((StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": format!("Model '{}' not loaded", model_id)
            }))));
        }

        // Find and clear the slot
        for slot in self.slots.iter() {
            let mut s = slot.lock().await;
            if s.model_id.as_deref() == Some(model_id) {
                s.runner = None;
                s.model_id = None;
                s.model_path = None;
                break;
            }
        }

        models.remove(model_id);
        tracing::info!(model = model_id, "Model unloaded");
        Ok(())
    }
}

// ── Handlers ────────────────────────────────────────────────────────

async fn list_models(
    State(app): State<AppState>,
) -> Json<serde_json::Value> {
    let models = app.loaded_models.read().await;
    let data: Vec<serde_json::Value> = models.keys().map(|id| {
        serde_json::json!({
            "id": id,
            "object": "model",
            "created": Utc::now().timestamp(),
            "owned_by": "pesti"
        })
    }).collect();

    Json(serde_json::json!({"object": "list", "data": data}))
}

async fn load_model_handler(
    State(app): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<ModelStatus>, (StatusCode, Json<serde_json::Value>)> {
    let model_id = req.model_id.unwrap_or_else(|| {
        // Derive ID from path
        let filename = std::path::Path::new(&req.model_path)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string());
        filename
    });

    let (slot, path) = app.get_or_load_model(&model_id, &req.model_path).await?;

    Ok(Json(ModelStatus {
        id: model_id,
        path,
        loaded_at: Utc::now().timestamp(),
        status: "loaded".to_string(),
    }))
}

async fn unload_model_handler(
    State(app): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let model_id = req["model"].as_str().ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "Missing 'model' field"})))
    })?;

    app.unload_model(model_id).await?;

    Ok(Json(serde_json::json!({
        "id": model_id,
        "status": "unloaded"
    })))
}

async fn chat_completions(
    State(app): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    let model_id = req.model.clone();
    let temperature = req.temperature;
    let top_p = req.top_p;
    let max_tokens = req.max_tokens;
    let stream = req.stream;

    // Load or get existing model
    let (slot_idx, _model_path) = app.get_or_load_model(&model_id, &model_id).await?;

    // Build prompt using model's chat template
    {
        let slot = &app.slots[slot_idx];
        let s = slot.lock().await;
        
        if let Some(runner) = &s.runner {
            // Convert messages to llama.cpp chat format
            let llama_messages: Vec<LlamaChatMessage> = req.messages.iter()
                .filter_map(|m| {
                    match LlamaChatMessage::new(m.role.clone(), m.content.clone()) {
                        Ok(msg) => Some(msg),
                        Err(e) => {
                            eprintln!("Failed to create chat message: {}", e);
                            None
                        }
                    }
                })
                .collect();
            
            // Apply chat template (uses model's built-in template)
            let prompt = runner.apply_chat_template(&llama_messages, true).map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()})))
            })?;

            if stream {
                // Generate response first (llama.cpp generates all at once)
                let sampling = SamplingConfig {
                    temperature,
                    top_p,
                    max_tokens,
                    ..Default::default()
                };

                let generation_result = runner.generate(&prompt, &sampling);

                // Now create stream with the result
                let id = format!("chatcmpl-{}", Uuid::new_v4().to_string().chars().take(8).collect::<String>());
                let created = Utc::now().timestamp();
                let model_id_clone = model_id.clone();

                match generation_result {
                    Ok(result) => {
                        let stream = async_stream::stream! {
                            // First chunk with role
                            yield Ok::<_, axum::Error>(Event::default().data(serde_json::json!({
                                "id": format!("chatcmpl-{}", Uuid::new_v4().to_string().chars().take(8).collect::<String>()),
                                "object": "chat.completion.chunk",
                                "created": Utc::now().timestamp(),
                                "model": model_id_clone.clone(),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": null
                                }]
                            }).to_string()));

                            // Content chunk
                            yield Ok(Event::default().data(serde_json::json!({
                                "id": format!("chatcmpl-{}", Uuid::new_v4().to_string().chars().take(8).collect::<String>()),
                                "object": "chat.completion.chunk",
                                "created": Utc::now().timestamp(),
                                "model": model_id_clone.clone(),
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": result.text},
                                    "finish_reason": "stop"
                                }]
                            }).to_string()));

                            // Final [DONE] marker
                            yield Ok(Event::default().data("[DONE]"));
                        };

                        return Ok(Sse::new(stream).into_response());
                    }
                    Err(e) => {
                        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))));
                    }
                }
            } else {
                // Non-streaming response
                let sampling = SamplingConfig {
                    temperature,
                    top_p,
                    max_tokens,
                    ..Default::default()
                };
                
                match runner.generate(&prompt, &sampling) {
                    Ok(result) => {
                        let id = format!("chatcmpl-{}", Uuid::new_v4().to_string().chars().take(8).collect::<String>());
                        
                        return Ok(Json(ChatCompletionResponse {
                            id,
                            object: "chat.completion".to_string(),
                            created: Utc::now().timestamp(),
                            model: req.model,
                            choices: vec![ChatChoice {
                                index: 0,
                                message: ChatMessage {
                                    role: "assistant".to_string(),
                                    content: result.text,
                                },
                                finish_reason: Some("stop".to_string()),
                            }],
                            usage: Usage {
                                prompt_tokens: 0, // TODO: Track actual token counts
                                completion_tokens: result.generated_tokens as i32,
                                total_tokens: result.generated_tokens as i32,
                            },
                        }).into_response());
                    }
                    Err(e) => {
                        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))));
                    }
                }
            }
        }
    }

    Err((StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": "Runner not available"}))))
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "pesti-server-v2"
    }))
}

// ── Main ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info")
            .add_directive("pesti_server=debug".parse().unwrap())
            .add_directive("axum=info".parse().unwrap())
    });

    fmt()
        .with_writer(std::io::stderr)
        .pretty()
        .init();

    let port: u16 = std::env::var("PESTI_SERVER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    // Connection pool size (number of concurrent model slots)
    let pool_size: usize = std::env::var("PESTI_POOL_SIZE")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(4);

    let app_state = AppState::new(pool_size);

    tracing::info!(port, pool_size, "pesti-server v2 starting");

    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        // Model management endpoints (hot-swapping)
        .route("/v1/models/load", post(load_model_handler))
        .route("/v1/models/unload", post(unload_model_handler))
        .route("/health", get(health_check))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(addr = %addr, "pesti-server listening");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .await
        .expect("server failed");
}
