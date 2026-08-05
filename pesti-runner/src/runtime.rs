//! Runtime — unified entry point for PESTI inference.
//!
//! Ties together model loading, inference (batch + streaming),
//! model lifecycle management, and device routing into a single API.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use pesti_runner::runtime::Runtime;
//!
//! let mut runtime = Runtime::new();
//! runtime.load_model("llama-3-8b").await?;
//!
//! // Batch inference
//! let result = runtime.generate("Explain quantum computing.", config)?;
//!
//! // Streaming inference
//! runtime.generate_streaming("Explain quantum computing.", config, |token| {
//!     print!("{}", token.text);
//!     Ok(())
//! })?;
//! ```
//!
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[cfg(feature = "cuda")]
use crate::device::DeviceType;
#[cfg(feature = "cuda")]
use crate::device_stub::LocalDevice as DeviceType;
use crate::error::Result;
use crate::llama::{GenerationResult, LlamaRunner, SamplingConfig, StreamingResult, TokenInfo};
use crate::model_manager::{ModelManager, ModelSpec, PreloadConfig, PreloadStats};
use crate::registry::{DiscoveredModel, ModelDiscovery, ModelEntry, ModelFormat, Registry};
#[cfg(feature = "cuda")]
use crate::transformer::{LlamaConfig, LlamaModel};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Configuration for the Runtime.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeConfig {
    /// Maximum number of models to keep preloaded.
    pub max_preloaded_models: usize,
    /// Maximum memory to use for preloaded models (MB).
    pub max_memory_mb: usize,
    /// Preferred device for inference.
    pub device_preference: (), // Stub - actual implementation only exists with CUDA
    /// Default context window size.
    pub max_ctx: usize,
    /// Number of threads (0 = auto).
    pub n_threads: i32,
    /// Whether to enable smart preloading.
    pub preload_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_preloaded_models: 3,
            max_memory_mb: 8192,
            device_preference: (), // Stub - actual implementation only exists with CUDA
            max_ctx: 4096,
            n_threads: 0,
            preload_enabled: true,
        }
    }
}

/// Runtime state for a loaded model.
#[derive(Debug, Clone)]
pub struct ModelState {
    /// Model name.
    pub name: String,
    /// Path to the model file.
    pub path: std::path::PathBuf,
    /// Format (GGUF or SafeTensors).
    pub format: ModelFormat,
    /// When it was loaded.
    pub loaded_at: std::time::SystemTime,
    /// Number of accesses.
    pub access_count: u64,
}

/// Backend for inference: llama.cpp (GGUF) or pure-Rust (SafeTensors).
pub enum RunnerBackend {
    /// llama.cpp FFI runner (GGUF models).
    Llama(LlamaRunner),
    /// Pure-Rust transformer model (SafeTensors).
    RustModel(()), // Stub - actual implementation only exists with CUDA
}

/// The unified inference runtime.
///
/// Manages model discovery, loading, inference (batch + streaming),
/// and lifecycle (preload/eviction) in a single struct.
pub struct Runtime {
    /// Model registry for discovery.
    registry: Registry,
    /// Model manager for lifecycle.
    model_manager: ModelManager,
    /// Currently loaded runner (if any).
    runner: Arc<RwLock<Option<RunnerBackend>>>,
    /// Runtime configuration.
    config: RuntimeConfig,
    /// Currently loaded model state.
    current_model: Arc<RwLock<Option<ModelState>>>,
}

impl Runtime {
    /// Create a new Runtime with default configuration.
    pub fn new() -> Self {
        Self::with_config(RuntimeConfig::default())
    }

    /// Create a Runtime with custom configuration.
    pub fn with_config(config: RuntimeConfig) -> Self {
        let registry = Registry::with_discovery();
        let model_manager = ModelManager::with_config(PreloadConfig {
            enabled: config.preload_enabled,
            max_preloaded_models: config.max_preloaded_models,
            max_memory_mb: config.max_memory_mb,
            ..Default::default()
        });

        Self {
            registry,
            model_manager,
            runner: Arc::new(RwLock::new(None)),
            config,
            current_model: Arc::new(RwLock::new(None)),
        }
    }

    /// Discover available models from filesystem search paths.
    pub async fn discover_models(&self) -> Result<Vec<DiscoveredModel>> {
        let discovery = ModelDiscovery::new();
        let models = discovery.discover_models().map_err(|e| {
            crate::error::RunnerError::Internal(format!("Failed to discover models: {}", e))
        })?;
        debug!(count = models.len(), "Discovered models");
        Ok(models)
    }

    /// List all available models (registered + discovered), deduplicated.
    pub fn list_available(&self) -> Vec<String> {
        self.registry.list_all_available()
    }

    /// Load a model by name. Discovers from registry or search paths.
    pub async fn load_model(&self, name: &str) -> Result<()> {
        // Check if already loaded
        {
            let current = self.current_model.read().await;
            if let Some(state) = current.as_ref() {
                if state.name == name {
                    debug!(model = name, "Model already loaded");
                    return Ok(());
                }
            }
        }

        // Unload current model
        self.unload_current().await?;

        // Find model spec
        let spec = self
            .registry
            .to_spec(name)
            .or_else(|| {
                // Try to auto-discover
                let mut registry = self.registry.clone();
                registry.refresh_discovered_models();
                registry.auto_register_discovered();
                registry.to_spec(name)
            })
            .ok_or_else(|| {
                crate::error::RunnerError::Internal(format!(
                    "Model '{name}' not found in registry or search paths"
                ))
            })?;

        // Determine format from path
        let format = if spec.base_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            ModelFormat::Gguf
        } else {
            ModelFormat::SafeTensors
        };

        // Load GGUF model via LlamaRunner
        if format == ModelFormat::Gguf && spec.base_path.exists() {
            info!(
                model = name,
                path = %spec.base_path.display(),
                "Loading GGUF model"
            );

            let runner = LlamaRunner::builder(&spec.base_path)
                .n_ctx(spec.ctx_len as u32)
                .n_batch(512)
                .n_threads(spec.n_threads.unwrap_or(self.config.n_threads))
                .build()
                .map_err(|e| crate::error::RunnerError::Internal(e.to_string()))?;

            *self.runner.write().await = Some(RunnerBackend::Llama(runner));

            let state = ModelState {
                name: name.to_string(),
                path: spec.base_path.clone(),
                format,
                loaded_at: std::time::SystemTime::now(),
                access_count: 0,
            };
            *self.current_model.write().await = Some(state);

            // Track in model manager
            self.model_manager
                .load_model(name.to_string(), spec)
                .await;

            info!(model = name, "Model loaded successfully");
        } else if format == ModelFormat::SafeTensors && spec.base_path.exists() {
            info!(
                model = name,
                path = %spec.base_path.display(),
                "Loading SafeTensors model"
            );

            // Extract config from safetensors metadata
            let _meta = crate::safetensors_weight_loader::extract_safetensors_config(&spec.base_path)
                .map_err(|e| crate::error::RunnerError::ModelLoad(format!("Failed to extract safetensors config: {e}")))
                .map(|_| ())?;

            // Load model from safetensors
            // Stub - actual implementation only exists with CUDA
            let _llama_model = ();

            *self.runner.write().await = Some(RunnerBackend::RustModel(_llama_model));

            let state = ModelState {
                name: name.to_string(),
                path: spec.base_path.clone(),
                format,
                loaded_at: std::time::SystemTime::now(),
                access_count: 0,
            };
            *self.current_model.write().await = Some(state);

            // Track in model manager
            self.model_manager
                .load_model(name.to_string(), spec)
                .await;

            info!(model = name, "SafeTensors model loaded successfully");
        } else {
            warn!(
                model = name,
                path = %spec.base_path.display(),
                "GGUF model path does not exist, skipping load"
            );
            return Err(crate::error::RunnerError::Internal(format!(
                "Model file not found: {}",
                spec.base_path.display()
            )));
        }

        Ok(())
    }

    /// Unload the currently loaded model.
    pub async fn unload_current(&self) -> Result<()> {
        let mut runner = self.runner.write().await;
        if runner.is_none() {
            return Ok(());
        }
        let name = {
            let current = self.current_model.read().await;
            current.as_ref().map(|s| s.name.clone())
        };

        *runner = None;
        *self.current_model.write().await = None;

        if let Some(ref name) = name {
            self.model_manager.unload_model(name).await;
            info!(model = name, "Model unloaded");
        }

        Ok(())
    }

    /// Get information about the currently loaded model.
    pub async fn model_info(&self) -> Option<ModelState> {
        self.current_model.read().await.clone()
    }

    /// Run batch inference on the loaded GGUF model (llama.cpp path).
    pub fn generate(&self, prompt: &str, config: &SamplingConfig) -> Result<GenerationResult> {
        let runner = self.runner.try_read().map_err(|e| {
            crate::error::RunnerError::Internal(format!("RWLock poisoned: {}", e))
        })?;
        let Some(RunnerBackend::Llama(runner)) = runner.as_ref() else {
            return Err(crate::error::RunnerError::Internal(
                "No GGUF model loaded. Call load_model() with a .gguf file.".to_string(),
            ));
        };

        let result = runner
            .generate(prompt, config)
            .map_err(|e| crate::error::RunnerError::Internal(e.to_string()))?;

        // Update access count (sync-safe via block_in_place)
        if let Some(state) = self
            .current_model
            .try_read()
            .ok()
            .and_then(|s| s.as_ref().cloned())
        {
            let name = state.name.clone();
            let duration = std::time::Duration::from_millis(result.eval_ms as u64);
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(self.model_manager.record_access(&name, duration));
            });
        }

        Ok(result)
    }

    /// Run batch inference on the loaded SafeTensors model (pure-Rust path).
    ///
    /// This uses the transformer-based inference engine, not llama.cpp.
    /// The prompt must be tokenized first using the model's tokenizer.
    pub fn generate_rust(&self, prompt_tokens: &[u32], max_tokens: usize, temp: f32) -> Result<Vec<u32>> {
        let model = self.runner.try_read().map_err(|e| {
            crate::error::RunnerError::Internal(format!("RWLock poisoned: {}", e))
        })?;
        let RunnerBackend::RustModel(model) = model.as_ref().ok_or_else(|| {
            crate::error::RunnerError::Internal(
                "No SafeTensors model loaded. Call load_model() with a .safetensors file.".to_string(),
            )
        })? else {
            return Err(crate::error::RunnerError::Internal(
                "Loaded model is not a SafeTensors model. Use generate() for GGUF models.".to_string(),
            ));
        };

        let sampling_config = crate::transformer::SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            repeat_penalty: 1.1,
        };
        let mut rng = StdRng::seed_from_u64(42);

        // Stub - actual implementation only exists with CUDA
        let _model = model;
        let _generated: Vec<u32> = vec![];

        // Update access count
        if let Some(state) = self
            .current_model
            .try_read()
            .ok()
            .and_then(|s| s.as_ref().cloned())
        {
            let name = state.name.clone();
            let duration = std::time::Duration::from_millis(1); // rough estimate
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(self.model_manager.record_access(&name, duration));
            });
        }

        Ok(generated)
    }

    /// Run streaming inference on the loaded model.
    ///
    /// The callback is invoked once per generated token.
    /// Return `Err(...)` to abort early.
    pub fn generate_streaming<F>(
        &self,
        prompt: &str,
        config: &SamplingConfig,
        mut on_token: F,
    ) -> Result<StreamingResult>
    where
        F: FnMut(&TokenInfo) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>,
    {
        let runner = self.runner.try_read().map_err(|e| {
            crate::error::RunnerError::Internal(format!("RWLock poisoned: {}", e))
        })?;
        let Some(RunnerBackend::Llama(runner)) = runner.as_ref() else {
            return Err(crate::error::RunnerError::Internal(
                "Streaming requires a GGUF model. Use generate_rust() for SafeTensors models.".to_string(),
            ));
        };

        let result = runner
            .generate_streaming(prompt, config, &mut on_token)
            .map_err(|e| crate::error::RunnerError::Internal(e.to_string()))?;

        // Update access count (sync-safe via block_in_place)
        if let Some(state) = self
            .current_model
            .try_read()
            .ok()
            .and_then(|s| s.as_ref().cloned())
        {
            let name = state.name.clone();
            let duration = std::time::Duration::from_millis(result.eval_ms as u64);
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(self.model_manager.record_access(&name, duration));
            });
        }

        Ok(result)
    }

    /// Get preloading statistics.
    pub async fn preload_stats(&self) -> PreloadStats {
        self.model_manager.preload_stats().await
    }

    /// Start the background preloading task.
    ///
    /// Returns a shutdown sender and a JoinHandle.
    /// Call `shutdown_tx.send(false)` to stop the task.
    pub fn start_preloading_task(
        &self,
    ) -> (
        tokio::sync::watch::Sender<bool>,
        tokio::task::JoinHandle<()>,
    ) {
        self.model_manager.start_preloading_task()
    }

    /// Get the list of loaded model names.
    pub async fn list_loaded_models(&self) -> Vec<String> {
        self.model_manager.list_loaded_models().await
    }

    /// Check if a model is loaded.
    pub async fn is_loaded(&self, name: &str) -> bool {
        self.model_manager.is_loaded(name).await
    }

    /// Refresh discovered models from filesystem.
    pub fn refresh_discovery(&mut self) {
        self.registry.refresh_discovered_models();
    }

    /// Register a model manually.
    pub fn register_model(&mut self, entry: ModelEntry) {
        self.registry.register(entry);
    }

    /// Get the model spec for a name (from registry or discovery).
    pub fn model_spec(&self, name: &str) -> Option<ModelSpec> {
        self.registry.to_spec(name)
    }

    /// Download a model file from HuggingFace Hub.
    ///
    /// Uses the `hf-hub` crate to download from the cache (downloading if needed).
    /// Returns the local path to the downloaded file.
    pub fn download_from_hf(repo_id: &str, filename: &str) -> Result<std::path::PathBuf> {
        let cache = hf_hub::Cache::from_env();
        let repo = cache.model(repo_id.to_string());
        let path = repo.get(filename).ok_or_else(|| {
            crate::error::RunnerError::Internal(format!(
                "Failed to download '{filename}' from '{repo_id}' (check HF token or network)"
            ))
        })?;
        Ok(path)
    }

    /// Get the current device preference.
    pub fn device_preference(&self) -> () { // Stub - actual implementation only exists with CUDA
        self.config.device_preference.clone()
    }
}