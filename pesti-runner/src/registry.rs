//! Model registry with in-memory HashMap and filesystem auto-discovery.
//!
//! Inspired by shimmy's Registry. Combines manually registered models with
//! auto-discovered models from filesystem search paths.
//!

#![allow(clippy::if_same_then_else, clippy::collapsible_if)]
//! ## Architecture
//!
//! ```text
//! Registry
//!   ├── inner: HashMap<String, ModelEntry> (manually registered)
//!   ├── discovered_models: HashMap<String, DiscoveredModel> (filesystem scan)
//!   ├── register() / get() / list()
//!   ├── refresh_discovered_models() / auto_register_discovered()
//!   ├── infer_template() (ChatML vs Llama3 auto-routing)
//!   └── to_spec() (convert entry → ModelSpec)
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::model_manager::ModelSpec;

/// Discovered model from filesystem scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub name: String,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub size_bytes: Option<u64>,
    pub model_type: Option<String>,
    pub parameter_count: Option<String>,
}

/// Model file format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
}

/// Manually registered model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub base_path: PathBuf,
    pub lora_path: Option<PathBuf>,
    pub template: Option<String>,
    pub ctx_len: Option<usize>,
    pub n_threads: Option<i32>,
}

/// Model registry: in-memory HashMap + filesystem auto-discovery.
#[derive(Default, Clone)]
pub struct Registry {
    inner: HashMap<String, ModelEntry>,
    pub discovered_models: HashMap<String, DiscoveredModel>,
}

/// Read SHIMMY_MAX_CTX env var for default context window size.
pub fn registry_ctx_len() -> usize {
    std::env::var("CRABJAR_MAX_CTX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&c| (512..=131_072).contains(&c))
        .unwrap_or(2048)
}

impl Registry {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
            discovered_models: HashMap::new(),
        }
    }

    /// Create registry with auto-discovery of models from common paths.
    pub fn with_discovery() -> Self {
        let mut registry = Self::new();
        registry.refresh_discovered_models();
        registry
    }

    /// Scan filesystem for discovered models.
    pub fn refresh_discovered_models(&mut self) {
        let discovery = ModelDiscovery::new();
        if let Ok(models) = discovery.discover_models() {
            self.discovered_models.clear();
            for model in models {
                self.discovered_models.insert(model.name.clone(), model);
            }
            debug!(
                discovered = self.discovered_models.len(),
                "Refreshed discovered models"
            );
        }
    }

    /// Auto-register discovered models that aren't already registered.
    pub fn auto_register_discovered(&mut self) {
        for (name, discovered) in &self.discovered_models {
            if !self.inner.contains_key(name) {
                let entry = ModelEntry {
                    name: name.clone(),
                    base_path: discovered.path.clone(),
                    lora_path: None,
                    template: Some(self.infer_template(name)),
                    ctx_len: None,
                    n_threads: None,
                };
                self.inner.insert(name.clone(), entry);
                debug!(model = name, "Auto-registered discovered model");
            }
        }
    }

    /// Infer template family from model name.
    pub fn infer_template(&self, model_name: &str) -> String {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("llama-3")
            || name_lower.contains("llama3")
            || name_lower.contains("meta-llama-3")
        {
            "llama3".to_string()
        } else {
            "chatml".to_string()
        }
    }

    /// Register a model entry.
    pub fn register(&mut self, entry: ModelEntry) {
        self.inner.insert(entry.name.clone(), entry);
    }

    /// Get a registered model entry.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.inner.get(name)
    }

    /// List all registered model entries.
    pub fn list(&self) -> Vec<&ModelEntry> {
        self.inner.values().collect()
    }

    /// List all available models (registered + discovered), deduplicated.
    pub fn list_all_available(&self) -> Vec<String> {
        let mut available = Vec::new();
        available.extend(self.inner.keys().cloned());
        available.extend(self.discovered_models.keys().cloned());
        available.sort();
        available.dedup();
        available
    }

    /// Convert a model name to a ModelSpec.
    ///
    /// Checks registered models first, then discovered models.
    pub fn to_spec(&self, name: &str) -> Option<ModelSpec> {
        if let Some(e) = self.inner.get(name) {
            return Some(ModelSpec {
                name: e.name.clone(),
                base_path: e.base_path.clone(),
                lora_path: e.lora_path.clone(),
                template: e.template.clone(),
                ctx_len: e.ctx_len.unwrap_or_else(registry_ctx_len),
                n_threads: e.n_threads,
            });
        }

        if let Some(discovered) = self.discovered_models.get(name) {
            return Some(ModelSpec {
                name: discovered.name.clone(),
                base_path: discovered.path.clone(),
                lora_path: None,
                template: Some(self.infer_template(&discovered.name)),
                ctx_len: registry_ctx_len(),
                n_threads: None,
            });
        }

        None
    }

    /// Check if a model is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    /// Number of registered models.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether no models are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Filesystem model discovery.
///
/// Scans common model directories for GGUF and SafeTensors files.
#[derive(Clone)]
pub struct ModelDiscovery {
    search_paths: Vec<PathBuf>,
}

impl Default for ModelDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelDiscovery {
    pub fn new() -> Self {
        Self {
            search_paths: Vec::new(),
        }
    }

    /// Create discovery from environment variables and common paths.
    #[allow(clippy::if_same_then_else)]
    pub fn from_env() -> Self {
        let mut discovery = Self::new();

        if let Ok(base_path) = std::env::var("CRABJAR_BASE_GGUF") {
            if let Some(parent) = PathBuf::from(&base_path).parent() {
                discovery.add_search_path(parent.to_path_buf());
            }
        }

        #[allow(clippy::if_same_then_else)]
        if let Ok(custom_dirs) = std::env::var("CRABJAR_MODEL_PATHS") {
            for dir in custom_dirs.split(';').filter(|s| !s.is_empty()) {
                discovery.add_search_path(PathBuf::from(dir));
            }
        }

        #[allow(clippy::if_same_then_else)]
        if let Ok(ollama_models) = std::env::var("OLLAMA_MODELS") {
            discovery.add_search_path(PathBuf::from(ollama_models));
        }

        if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
            let home_path = PathBuf::from(home);
            discovery.add_search_path(home_path.join(".cache/huggingface"));
            discovery.add_search_path(home_path.join(".ollama/models"));
            discovery.add_search_path(home_path.join(".cache/lm-studio/models"));
            discovery.add_search_path(home_path.join("models"));
        }

        discovery
    }

    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    /// Discover models from all search paths.
    pub fn discover_models(&self) -> Result<Vec<DiscoveredModel>, std::io::Error> {
        let mut models = Vec::new();
        for path in &self.search_paths {
            if path.exists() {
                Self::scan_directory(path, &mut models)?;
            }
        }
        Ok(models)
    }

    fn scan_directory(
        dir: &std::path::Path,
        models: &mut Vec<DiscoveredModel>,
    ) -> Result<(), std::io::Error> {
        let mut model_files = Vec::new();
        let mut subdirs = Vec::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                subdirs.push(path);
            } else if Self::is_model_file(&path) {
                model_files.push(path);
            }
        }

        // Group sharded models
        let grouped = Self::group_sharded_models(dir, &model_files)?;
        for model in grouped {
            models.push(model);
        }

        // Recurse into subdirectories
        for subdir in subdirs {
            Self::scan_directory(&subdir, models)?;
        }

        Ok(())
    }

    #[allow(clippy::if_same_then_else)]
    fn group_sharded_models(
        dir: &std::path::Path,
        model_files: &[PathBuf],
    ) -> Result<Vec<DiscoveredModel>, std::io::Error> {
        use std::collections::HashMap;
        use std::collections::HashSet;

        let mut grouped_models = Vec::new();
        let mut processed = HashSet::new();

        // Match model-XXXX-of-YYYY.ext pattern
        let shard_pattern = regex::Regex::new(r"^(.+)-\d{5}-of-\d{5}(\..+)$").unwrap();

        let mut shard_groups: HashMap<String, Vec<PathBuf>> = HashMap::new();

        for file_path in model_files {
            if let Some(filename) = file_path.file_name().and_then(|f| f.to_str()) {
                if let Some(captures) = shard_pattern.captures(filename) {
                    let base_name = captures.get(1).unwrap().as_str();
                    let extension = captures.get(2).unwrap().as_str();
                    let group_key = format!("{}{}", base_name, extension);
                    shard_groups
                        .entry(group_key)
                        .or_default()
                        .push(file_path.clone());
                    processed.insert(file_path.clone());
                }
            }
        }

        for (group_key, files) in shard_groups {
            if files.len() > 1 {
                let total_size: u64 = files
                    .iter()
                    .filter_map(|path| std::fs::metadata(path).ok().map(|m| m.len()))
                    .sum();

                let model_name = dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&group_key)
                    .to_string();

                let format = if group_key.ends_with(".safetensors") {
                    ModelFormat::SafeTensors
                } else {
                    ModelFormat::Gguf
                };

                grouped_models.push(DiscoveredModel {
                    name: model_name,
                    path: files[0].clone(),
                    format,
                    size_bytes: Some(total_size),
                    model_type: None,
                    parameter_count: None,
                });
            }
        }

        // Non-sharded models
        for file_path in model_files {
            if !processed.contains(file_path) {
                if let Ok(model) = Self::analyze_model_file(file_path) {
                    grouped_models.push(model);
                }
            }
        }

        Ok(grouped_models)
    }

    #[allow(clippy::if_same_then_else)]
    fn is_model_file(path: &std::path::Path) -> bool {
        if let Some(ext) = path.extension() {
            if matches!(ext.to_str(), Some("gguf") | Some("safetensors")) {
                return Self::is_llm_model(path);
            }
        }
        false
    }

    fn is_llm_model(path: &std::path::Path) -> bool {
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_lowercase();

        let non_llm_patterns = [
            "flux",
            "sd",
            "stable-diffusion",
            "sdxl",
            "dalle",
            "midjourney",
            "video",
            "vid",
            "animate",
            "motion",
            "whisper",
            "audio",
            "speech",
            "tts",
            "voice",
            "clip",
            "embed",
            "encoder",
            "vision",
            "vae",
            "unet",
            "controlnet",
            "lora",
            "adapter",
        ];

        if non_llm_patterns.iter().any(|p| filename.contains(p)) {
            return false;
        }

        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return true;
        }

        let llm_patterns = [
            "llama",
            "mistral",
            "qwen",
            "phi",
            "gemma",
            "codellama",
            "vicuna",
            "alpaca",
            "orca",
            "falcon",
            "mpt",
            "gpt",
            "claude",
            "chatglm",
            "baichuan",
            "yi",
            "deepseek",
            "mixtral",
            "solar",
            "openchat",
            "starling",
            "wizardlm",
            "dolphin",
            "nous",
            "hermes",
            "airoboros",
        ];

        llm_patterns.iter().any(|p| filename.contains(p)) || true
    }

    fn analyze_model_file(path: &std::path::Path) -> Result<DiscoveredModel, std::io::Error> {
        let format = match path.extension().and_then(|s| s.to_str()) {
            Some("gguf") => ModelFormat::Gguf,
            Some("safetensors") => ModelFormat::SafeTensors,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Unknown format",
                ));
            }
        };

        let size_bytes = std::fs::metadata(path).ok().map(|m| m.len());
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Extract metadata for SafeTensors models
        let (model_type, parameter_count) = if format == ModelFormat::SafeTensors {
            ModelDiscovery::extract_safetensors_metadata(path)
        } else {
            (None, None)
        };

        Ok(DiscoveredModel {
            name,
            path: path.to_path_buf(),
            format,
            size_bytes,
            model_type,
            parameter_count,
        })
    }

    /// Extract model metadata from a SafeTensors file.
    fn extract_safetensors_metadata(path: &std::path::Path) -> (Option<String>, Option<String>) {
        use std::collections::HashMap;

        let file_data = match std::fs::read(path) {
            Ok(data) => data,
            Err(_) => return (None, None),
        };

        let handle = match safetensors::SafeTensors::read_metadata(&file_data) {
            Ok((_header_size, metadata)) => metadata,
            Err(_) => return (None, None),
        };

        let meta_map = match handle.metadata() {
            Some(map) => map,
            None => return (None, None),
        };

        // Extract architecture/model type
        let model_type = meta_map
            .get("general.architecture")
            .or_else(|| meta_map.get("model_type"))
            .or_else(|| meta_map.get("architectures"))
            .map(|s| s.to_string());

        // Estimate parameter count from tensor sizes
        let param_count = meta_map
            .iter()
            .filter(|(k, _)| {
                // Look for weight tensors (not metadata)
                k.starts_with("model.") || 
                k.starts_with("layers.") ||
                k.contains(".weight") ||
                k.contains(".bias")
            })
            .map(|(_, v)| {
                if let Ok(map) = serde_json::from_str::<HashMap<String, serde_json::Value>>(v) {
                    if let Some(shape) = map.get("shape").and_then(|s| s.as_array()) {
                        shape.iter().filter_map(|v| v.as_u64()).product::<u64>()
                    } else {
                        0
                    }
                } else {
                    0
                }
            })
            .sum::<u64>();

        let parameter_count = if param_count > 0 {
            // Format as human-readable (e.g., "7B", "8B")
            let billions = param_count as f64 / 1_000_000_000.0;
            if billions >= 1.0 {
                Some(format!("{:.1}B", billions))
            } else {
                Some(format!("{}M", (param_count / 1_000_000) as u64))
            }
        } else {
            None
        };

        (model_type, parameter_count)
    }
}

