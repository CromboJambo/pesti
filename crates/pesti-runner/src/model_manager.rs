//! Model lifecycle management with smart preloading and popularity tracking.
//!
//! Inspired by shimmy's ModelManager. Manages model loading, unloading, and
//! background preloading based on usage patterns.
//!
//! ## Architecture
//!
//! ```text
//! ModelManager
//!   ├── loaded_models: HashMap<String, ModelLoadInfo>
//!   ├── usage_stats: HashMap<String, ModelUsageStats>
//!   ├── preload_queue: VecDeque<String>
//!   ├── preload_config: PreloadConfig
//!   ├── load_model() / unload_model()
//!   ├── record_access() → popularity scoring
//!   ├── evaluate_preloading() → queue candidates
//!   ├── start_preloading_task() → background task
//!   └── cleanup_old_models() → free memory
//! ```
//!
//! ## Popularity Scoring
//!
//! `popularity = ln(total_requests + 1) * (1 / (1 + hours_since_last_use / 3600))`
//!
//! Frequency factor grows logarithmically. Recency factor decays over hours.
//! Models exceeding `preload_threshold_score` and `min_usage_for_preload`
//! are queued for background preloading.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{RwLock, watch};
use tracing::{debug, info, warn};

/// Model specification for loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub name: String,
    pub base_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub template: Option<String>,
    pub ctx_len: usize,
    pub n_threads: Option<i32>,
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelLoadInfo {
    pub name: String,
    pub spec: ModelSpec,
    pub loaded_at: std::time::SystemTime,
    pub last_accessed: std::time::SystemTime,
    pub access_count: u64,
}

/// Usage statistics for popularity scoring.
#[derive(Debug, Clone)]
pub struct ModelUsageStats {
    pub model_name: String,
    pub total_requests: u64,
    pub last_used: std::time::SystemTime,
    pub average_response_time: Duration,
    pub popularity_score: f64,
}

/// Configuration for smart preloading behavior.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreloadConfig {
    pub enabled: bool,
    pub max_preloaded_models: usize,
    pub max_memory_mb: usize,
    pub preload_threshold_score: f64,
    pub min_usage_for_preload: u64,
    #[serde(default = "default_cleanup_secs")]
    pub cleanup_interval_secs: u64,
}

fn default_cleanup_secs() -> u64 {
    300
}

impl PreloadConfig {
    /// Get the cleanup interval as a [`Duration`].
    pub fn cleanup_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.cleanup_interval_secs)
    }
}

impl Default for PreloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_preloaded_models: 3,
            max_memory_mb: 8192,
            preload_threshold_score: 0.5,
            min_usage_for_preload: 2,
            cleanup_interval_secs: 300,
        }
    }
}

/// Preloading statistics for monitoring.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreloadStats {
    pub loaded_models: usize,
    pub max_models: usize,
    pub queue_length: usize,
    pub total_tracked_models: usize,
    pub memory_limit_mb: usize,
    pub preloading_enabled: bool,
}

/// Model lifecycle manager with smart preloading.
///
/// Tracks model access patterns, scores popularity, and manages a background
/// preload queue to keep frequently-used models ready.
#[derive(Clone)]
pub struct ModelManager {
    loaded_models: Arc<RwLock<HashMap<String, ModelLoadInfo>>>,
    usage_stats: Arc<RwLock<HashMap<String, ModelUsageStats>>>,
    preload_config: PreloadConfig,
    preload_queue: Arc<RwLock<VecDeque<String>>>,
    shutdown_tx: Arc<watch::Sender<bool>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::with_config(PreloadConfig::default())
    }

    pub fn with_config(config: PreloadConfig) -> Self {
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
            preload_config: config,
            preload_queue: Arc::new(RwLock::new(VecDeque::new())),
            shutdown_tx: Arc::new(shutdown_tx),
        }
    }

    /// Load a model into the manager's registry.
    ///
    /// Does not actually load the model into GPU memory — that is handled by
    /// the inference engine. This tracks the model as "loaded" for lifecycle
    /// purposes.
    pub async fn load_model(&self, name: String, spec: ModelSpec) {
        let now = std::time::SystemTime::now();

        let info = ModelLoadInfo {
            name: name.clone(),
            spec,
            loaded_at: now,
            last_accessed: now,
            access_count: 1,
        };

        {
            let mut models = self.loaded_models.write().await;
            models.insert(name.clone(), info);
        }

        info!("Model '{}' registered in manager", name);

        // Initialize usage stats
        self.update_usage_stats(&name, Duration::from_millis(100))
            .await;

        // Trigger preloading evaluation
        if self.preload_config.enabled {
            self.evaluate_preloading().await;
        }
    }

    /// Record a model access for usage tracking and popularity scoring.
    pub async fn record_access(&self, name: &str, response_time: Duration) {
        {
            let mut models = self.loaded_models.write().await;
            if let Some(info) = models.get_mut(name) {
                info.last_accessed = std::time::SystemTime::now();
                info.access_count += 1;
            }
        }

        self.update_usage_stats(name, response_time).await;
    }

    /// Update usage statistics for a model.
    async fn update_usage_stats(&self, name: &str, response_time: Duration) {
        let mut stats = self.usage_stats.write().await;

        if let Some(entry) = stats.get_mut(name) {
            entry.total_requests += 1;
            entry.last_used = std::time::SystemTime::now();

            let current_avg_ms = entry.average_response_time.as_millis() as f64;
            let new_response_ms = response_time.as_millis() as f64;
            let new_avg_ms = (current_avg_ms * (entry.total_requests - 1) as f64 + new_response_ms)
                / entry.total_requests as f64;
            entry.average_response_time = Duration::from_millis(new_avg_ms as u64);

            let time_since_last_use = std::time::SystemTime::now()
                .duration_since(entry.last_used)
                .unwrap_or_default()
                .as_secs() as f64;
            let recency_factor = 1.0 / (1.0 + time_since_last_use / 3600.0);
            let frequency_factor = (entry.total_requests as f64).ln() + 1.0;
            entry.popularity_score = frequency_factor * recency_factor;

            debug!(
                model = name,
                requests = entry.total_requests,
                popularity = entry.popularity_score,
                "Updated usage stats"
            );
        } else {
            stats.insert(
                name.to_string(),
                ModelUsageStats {
                    model_name: name.to_string(),
                    total_requests: 1,
                    last_used: std::time::SystemTime::now(),
                    average_response_time: response_time,
                    popularity_score: 1.0,
                },
            );

            debug!(
                model = name,
                requests = 1,
                popularity = 1.0,
                "Created new usage stats"
            );
        }
    }

    /// Evaluate which models should be preloaded based on popularity.
    async fn evaluate_preloading(&self) {
        if !self.preload_config.enabled {
            return;
        }

        let (candidates_to_queue, current_loaded) = {
            let stats = self.usage_stats.read().await;
            let loaded_models = self.loaded_models.read().await;

            let candidates_vec: Vec<_> = stats
                .iter()
                .filter(|(name, stat)| {
                    stat.total_requests >= self.preload_config.min_usage_for_preload
                        && stat.popularity_score >= self.preload_config.preload_threshold_score
                        && !loaded_models.contains_key(*name)
                })
                .map(|(name, stat)| (name.clone(), stat.popularity_score))
                .collect();

            let current_loaded = loaded_models.len();
            (candidates_vec, current_loaded)
        };

        let mut queue = self.preload_queue.write().await;
        let slots_available = self
            .preload_config
            .max_preloaded_models
            .saturating_sub(current_loaded);

        for (model_name, score) in candidates_to_queue.iter().take(slots_available) {
            if !queue.iter().any(|n| n.as_str() == model_name.as_str()) {
                queue.push_back(model_name.clone());
                info!(
                    model = model_name.as_str(),
                    score = score,
                    "Queued model for preloading"
                );
            }
        }
    }

    /// Start the background preloading task.
    ///
    /// Spawns a tokio task that periodically processes the preload queue.
    /// Returns a shutdown channel sender and a JoinHandle. Call `shutdown_tx.send(false)` to stop the task.
    pub fn start_preloading_task(
        &self,
    ) -> (
        tokio::sync::watch::Sender<bool>,
        tokio::task::JoinHandle<()>,
    ) {
        let manager = Arc::new(self.clone());
        let cleanup_interval = self.preload_config.cleanup_interval();
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {}
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                }

                let model_to_preload = {
                    let mut queue = manager.preload_queue.write().await;
                    queue.pop_front()
                };

                if let Some(model_name) = model_to_preload {
                    let current_count = manager.model_count().await;
                    if current_count < manager.preload_config.max_preloaded_models {
                        debug!(model = model_name, "Processing preload queue");
                    } else {
                        warn!(
                            limit = manager.preload_config.max_preloaded_models,
                            current = current_count,
                            "Memory limit reached, re-queuing model"
                        );
                        let mut queue = manager.preload_queue.write().await;
                        queue.push_front(model_name);
                    }
                }

                manager.cleanup_old_models().await;
            }

            debug!("Preloading task shut down");
        });

        (shutdown_tx, handle)
    }

    /// Clean up old/unused models to free memory.
    async fn cleanup_old_models(&self) {
        let current_count = self.model_count().await;
        if current_count <= self.preload_config.max_preloaded_models {
            return;
        }

        let cutoff_time = std::time::SystemTime::now() - Duration::from_secs(3600);

        let mut models = self.loaded_models.write().await;
        let mut candidates: Vec<_> = models
            .iter()
            .enumerate()
            .filter(|(_, (_, info))| info.last_accessed < cutoff_time && info.access_count < 5)
            .map(|(idx, (name, info))| (idx, name.clone(), info.last_accessed, info.access_count))
            .collect();

        candidates.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.3.cmp(&b.3)));

        let to_remove = current_count.saturating_sub(self.preload_config.max_preloaded_models);
        for (_idx, name, _, _) in candidates.iter().take(to_remove) {
            models.remove(name);
            info!(model = name, "Cleaned up unused model");
        }
    }

    /// Get preloading statistics.
    pub async fn preload_stats(&self) -> PreloadStats {
        let models = self.loaded_models.read().await;
        let stats = self.usage_stats.read().await;
        let queue = self.preload_queue.read().await;

        PreloadStats {
            loaded_models: models.len(),
            max_models: self.preload_config.max_preloaded_models,
            queue_length: queue.len(),
            total_tracked_models: stats.len(),
            memory_limit_mb: self.preload_config.max_memory_mb,
            preloading_enabled: self.preload_config.enabled,
        }
    }

    /// Unload a model from the manager.
    pub async fn unload_model(&self, name: &str) -> bool {
        let mut models = self.loaded_models.write().await;
        let had = models.contains_key(name);
        if had {
            models.remove(name);
            info!(model = name, "Model unloaded from manager");
        }
        had
    }

    /// Get information about a loaded model.
    pub async fn model_info(&self, name: &str) -> Option<ModelLoadInfo> {
        let models = self.loaded_models.read().await;
        models.get(name).cloned()
    }

    /// List all loaded model names.
    pub async fn list_loaded_models(&self) -> Vec<String> {
        let models = self.loaded_models.read().await;
        models.keys().cloned().collect()
    }

    /// Check if a model is loaded.
    pub async fn is_loaded(&self, name: &str) -> bool {
        let models = self.loaded_models.read().await;
        models.contains_key(name)
    }

    /// Count loaded models.
    pub async fn model_count(&self) -> usize {
        let models = self.loaded_models.read().await;
        models.len()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}
