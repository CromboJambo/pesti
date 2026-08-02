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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_entry(name: &str, path: &str) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            base_path: PathBuf::from(path),
            lora_path: None,
            template: None,
            ctx_len: None,
            n_threads: None,
        }
    }

    // ── Registry Construction ──────────────────────────────────────────

    #[test]
    fn registry_new_is_empty() {
        let registry = Registry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.discovered_models.is_empty());
    }

    #[test]
    fn registry_default_is_empty() {
        let registry = Registry::default();
        assert!(registry.is_empty());
    }

    // ── Registration ───────────────────────────────────────────────────

    #[test]
    fn register_model() {
        let mut registry = Registry::new();
        registry.register(make_entry("test-model", "/models/test.gguf"));
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test-model"));
    }

    #[test]
    fn register_overwrites_same_name() {
        let mut registry = Registry::new();
        registry.register(make_entry("dup", "/models/a.gguf"));
        registry.register(make_entry("dup", "/models/b.gguf"));
        assert_eq!(registry.len(), 1);

        let entry = registry.get("dup").unwrap();
        assert_eq!(entry.base_path, PathBuf::from("/models/b.gguf"));
    }

    #[test]
    fn get_registered_model() {
        let mut registry = Registry::new();
        registry.register(make_entry("gettest", "/models/g.gguf"));
        let entry = registry.get("gettest").unwrap();
        assert_eq!(entry.name, "gettest");
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let registry = Registry::new();
        assert!(registry.get("ghost").is_none());
    }

    #[test]
    fn list_models() {
        let mut registry = Registry::new();
        registry.register(make_entry("alpha", "/models/a.gguf"));
        registry.register(make_entry("beta", "/models/b.gguf"));
        let entries = registry.list();
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().any(|e| e.name == "alpha"));
        assert!(entries.iter().any(|e| e.name == "beta"));
    }

    #[test]
    fn list_empty_returns_empty() {
        let registry = Registry::new();
        assert!(registry.list().is_empty());
    }

    // ── Template Inference ─────────────────────────────────────────────

    #[test]
    fn infer_template_llama3_variants() {
        let registry = Registry::new();
        assert_eq!(registry.infer_template("llama-3-8b"), "llama3");
        assert_eq!(registry.infer_template("llama3-70b"), "llama3");
        assert_eq!(registry.infer_template("meta-llama-3-instruct"), "llama3");
        assert_eq!(registry.infer_template("Meta-Llama-3.1-8B"), "llama3");
    }

    #[test]
    fn infer_template_chatml_fallback() {
        let registry = Registry::new();
        assert_eq!(registry.infer_template("tinyllama-1.1b"), "chatml");
        assert_eq!(registry.infer_template("mistral-7b"), "chatml");
        assert_eq!(registry.infer_template("phi-3-mini"), "chatml");
        assert_eq!(registry.infer_template("qwen2-7b"), "chatml");
        assert_eq!(registry.infer_template("llama-2-7b"), "chatml");
    }

    // ── to_spec ────────────────────────────────────────────────────────

    #[test]
    fn to_spec_registered_model() {
        let mut registry = Registry::new();
        let entry = ModelEntry {
            name: "myspec".to_string(),
            base_path: PathBuf::from("/models/myspec.gguf"),
            lora_path: Some(PathBuf::from("/models/myspec-lora.safetensors")),
            template: Some("chatml".to_string()),
            ctx_len: Some(4096),
            n_threads: Some(8),
        };
        registry.register(entry);

        let spec = registry.to_spec("myspec").unwrap();
        assert_eq!(spec.name, "myspec");
        assert_eq!(spec.ctx_len, 4096);
        assert_eq!(spec.n_threads, Some(8));
        assert!(spec.lora_path.is_some());
    }

    #[test]
    fn to_spec_missing_returns_none() {
        let registry = Registry::new();
        assert!(registry.to_spec("does-not-exist").is_none());
    }

    #[test]
    fn to_spec_ctx_len_defaults() {
        let mut registry = Registry::new();
        registry.register(ModelEntry {
            name: "ctx-default".to_string(),
            base_path: PathBuf::from("/models/ctx.gguf"),
            lora_path: None,
            template: None,
            ctx_len: None,
            n_threads: None,
        });

        let spec = registry.to_spec("ctx-default").unwrap();
        assert_eq!(spec.ctx_len, 2048); // default
    }

    // ── list_all_available ─────────────────────────────────────────────

    #[test]
    fn list_all_available_registered_only() {
        let mut registry = Registry::new();
        registry.register(make_entry("reg-a", "/models/a.gguf"));
        registry.register(make_entry("reg-b", "/models/b.gguf"));

        let all = registry.list_all_available();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&"reg-a".to_string()));
        assert!(all.contains(&"reg-b".to_string()));
    }

    #[test]
    fn list_all_available_deduplicates() {
        let mut registry = Registry::new();
        registry.register(make_entry("shared", "/models/s.gguf"));
        registry.discovered_models.insert(
            "shared".to_string(),
            DiscoveredModel {
                name: "shared".to_string(),
                path: PathBuf::from("/discovered/s.gguf"),
                format: ModelFormat::Gguf,
                size_bytes: Some(1000),
                model_type: None,
                parameter_count: None,
            },
        );

        let all = registry.list_all_available();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], "shared");
    }

    #[test]
    fn list_all_available_sorted() {
        let mut registry = Registry::new();
        registry.register(make_entry("zebra", "/models/z.gguf"));
        registry.register(make_entry("alpha", "/models/a.gguf"));

        let all = registry.list_all_available();
        assert_eq!(all[0], "alpha");
        assert_eq!(all[1], "zebra");
    }

    // ── Auto-discovery ─────────────────────────────────────────────────

    #[test]
    fn discovery_new_has_no_paths() {
        let discovery = ModelDiscovery::new();
        assert!(discovery.search_paths().is_empty());
    }

    #[test]
    fn discovery_add_search_path() {
        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(PathBuf::from("/test/path"));
        assert_eq!(discovery.search_paths().len(), 1);
    }

    #[test]
    fn discovery_discover_empty_returns_empty() {
        let discovery = ModelDiscovery::new();
        let models = discovery.discover_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn discovery_discover_nonexistent_paths() {
        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(PathBuf::from("/nonexistent/path"));
        let models = discovery.discover_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn discovery_discovers_gguf_and_safetensors() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        fs::write(temp_dir.path().join("model1.gguf"), "gguf content")?;
        fs::write(temp_dir.path().join("model2.safetensors"), "st content")?;
        fs::write(temp_dir.path().join("not_model.txt"), "text")?;

        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(temp_dir.path().to_path_buf());

        let models = discovery.discover_models()?;
        assert_eq!(models.len(), 2);

        let names: Vec<_> = models.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"model1"));
        assert!(names.contains(&"model2"));

        Ok(())
    }

    #[test]
    fn discovery_recursive_scan() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir)?;

        fs::write(temp_dir.path().join("root.gguf"), "root")?;
        fs::write(subdir.join("nested.gguf"), "nested")?;

        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(temp_dir.path().to_path_buf());

        let models = discovery.discover_models()?;
        assert_eq!(models.len(), 2);

        Ok(())
    }

    #[test]
    fn discovery_excludes_non_llm_models() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        fs::write(temp_dir.path().join("llama-2-7b.gguf"), "llm")?;
        fs::write(temp_dir.path().join("flux-dev.gguf"), "image")?;
        fs::write(temp_dir.path().join("whisper-large.gguf"), "audio")?;
        fs::write(temp_dir.path().join("unknown-model.gguf"), "unknown")?;

        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(temp_dir.path().to_path_buf());

        let models = discovery.discover_models()?;

        // llama-2-7b and unknown-model should be included; flux and whisper excluded
        let names: Vec<_> = models.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"llama-2-7b"));
        assert!(names.contains(&"unknown-model"));
        assert!(!names.contains(&"flux-dev"));
        assert!(!names.contains(&"whisper-large"));

        Ok(())
    }

    #[test]
    fn discovery_excludes_lora_adapters() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        fs::write(temp_dir.path().join("base-model.gguf"), "base")?;
        fs::write(temp_dir.path().join("adapter-lora.gguf"), "adapter")?;

        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(temp_dir.path().to_path_buf());

        let models = discovery.discover_models()?;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "base-model");

        Ok(())
    }

    #[test]
    fn discovery_analyze_model_file_format() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        let gguf_path = temp_dir.path().join("test.gguf");
        fs::write(&gguf_path, "gguf data")?;
        let model = ModelDiscovery::analyze_model_file(&gguf_path)?;
        assert!(matches!(model.format, ModelFormat::Gguf));
        assert_eq!(model.name, "test");

        let st_path = temp_dir.path().join("test.safetensors");
        fs::write(&st_path, "st data")?;
        let model = ModelDiscovery::analyze_model_file(&st_path)?;
        assert!(matches!(model.format, ModelFormat::SafeTensors));

        Ok(())
    }

    #[test]
    fn discovery_unknown_format_errors() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.unknown");
        fs::write(&path, "data").unwrap();
        let result = ModelDiscovery::analyze_model_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn discovery_multiple_search_paths() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir1 = TempDir::new()?;
        let temp_dir2 = TempDir::new()?;

        fs::write(temp_dir1.path().join("model1.gguf"), "m1")?;
        fs::write(temp_dir2.path().join("model2.safetensors"), "m2")?;

        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(temp_dir1.path().to_path_buf());
        discovery.add_search_path(temp_dir2.path().to_path_buf());

        let models = discovery.discover_models()?;
        assert_eq!(models.len(), 2);

        Ok(())
    }

    // ── ModelEntry Serialization ───────────────────────────────────────

    #[test]
    fn model_entry_serializes() {
        let entry = ModelEntry {
            name: "ser-test".to_string(),
            base_path: PathBuf::from("/models/ser.gguf"),
            lora_path: Some(PathBuf::from("/models/ser-lora.safetensors")),
            template: Some("chatml".to_string()),
            ctx_len: Some(4096),
            n_threads: Some(8),
        };

        let json = serde_json::to_string(&entry).unwrap();
        let restored: ModelEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "ser-test");
        assert_eq!(restored.ctx_len, Some(4096));
        assert_eq!(restored.n_threads, Some(8));
    }

    #[test]
    fn model_entry_defaults_serializes() {
        let entry = ModelEntry {
            name: "defaults".to_string(),
            base_path: PathBuf::from("/models/d.gguf"),
            lora_path: None,
            template: None,
            ctx_len: None,
            n_threads: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let restored: ModelEntry = serde_json::from_str(&json).unwrap();
        assert!(restored.lora_path.is_none());
        assert!(restored.template.is_none());
        assert!(restored.ctx_len.is_none());
        assert!(restored.n_threads.is_none());
    }

    // ── DiscoveredModel Serialization ──────────────────────────────────

    #[test]
    fn discovered_model_serializes() {
        let model = DiscoveredModel {
            name: "disc-test".to_string(),
            path: PathBuf::from("/models/disc.gguf"),
            format: ModelFormat::Gguf,
            size_bytes: Some(1024),
            model_type: Some("llm".to_string()),
            parameter_count: Some("7B".to_string()),
        };

        let json = serde_json::to_string(&model).unwrap();
        let restored: DiscoveredModel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "disc-test");
        assert!(matches!(restored.format, ModelFormat::Gguf));
        assert_eq!(restored.size_bytes, Some(1024));
        assert_eq!(restored.model_type, Some("llm".to_string()));
    }

    #[test]
    fn model_format_serialization() {
        let gguf = serde_json::to_string(&ModelFormat::Gguf).unwrap();
        assert!(gguf.contains("Gguf"));

        let st = serde_json::to_string(&ModelFormat::SafeTensors).unwrap();
        assert!(st.contains("SafeTensors"));

        let parsed_gguf: ModelFormat = serde_json::from_str(&gguf).unwrap();
        assert!(matches!(parsed_gguf, ModelFormat::Gguf));

        let parsed_st: ModelFormat = serde_json::from_str(&st).unwrap();
        assert!(matches!(parsed_st, ModelFormat::SafeTensors));
    }

    // ── Registry with Discovery ────────────────────────────────────────

    #[test]
    fn to_spec_discovered_model() {
        let mut registry = Registry::new();

        registry.discovered_models.insert(
            "disc-model".to_string(),
            DiscoveredModel {
                name: "disc-model".to_string(),
                path: PathBuf::from("/discovered/d.gguf"),
                format: ModelFormat::Gguf,
                size_bytes: Some(5000),
                model_type: None,
                parameter_count: None,
            },
        );

        let spec = registry.to_spec("disc-model").unwrap();
        assert_eq!(spec.name, "disc-model");
        assert_eq!(spec.ctx_len, 2048);
        assert_eq!(spec.template, Some("chatml".to_string()));
    }

    #[test]
    fn auto_register_discovered() {
        let mut registry = Registry::new();

        registry.discovered_models.insert(
            "auto-reg".to_string(),
            DiscoveredModel {
                name: "auto-reg".to_string(),
                path: PathBuf::from("/discovered/auto.gguf"),
                format: ModelFormat::Gguf,
                size_bytes: Some(3000),
                model_type: None,
                parameter_count: None,
            },
        );

        registry.auto_register_discovered();
        assert!(registry.contains("auto-reg"));

        let entry = registry.get("auto-reg").unwrap();
        assert_eq!(entry.template, Some("chatml".to_string()));
        assert!(entry.lora_path.is_none());
    }

    #[test]
    fn auto_register_skips_existing() {
        let mut registry = Registry::new();

        registry.register(ModelEntry {
            name: "existing".to_string(),
            base_path: PathBuf::from("/models/existing.gguf"),
            lora_path: None,
            template: Some("llama3".to_string()),
            ctx_len: Some(8192),
            n_threads: None,
        });

        registry.discovered_models.insert(
            "existing".to_string(),
            DiscoveredModel {
                name: "existing".to_string(),
                path: PathBuf::from("/discovered/existing.gguf"),
                format: ModelFormat::Gguf,
                size_bytes: Some(1000),
                model_type: None,
                parameter_count: None,
            },
        );

        registry.auto_register_discovered();

        // Should still be the manually registered version
        let entry = registry.get("existing").unwrap();
        assert_eq!(entry.template, Some("llama3".to_string()));
        assert_eq!(entry.ctx_len, Some(8192));
    }

    // ── Edge Cases ─────────────────────────────────────────────────────

    #[test]
    fn registry_clone_works() {
        let mut registry = Registry::new();
        registry.register(make_entry("clone-test", "/models/clone.gguf"));

        let cloned = registry.clone();
        assert_eq!(registry.len(), cloned.len());
        assert!(cloned.contains("clone-test"));
    }

    #[test]
    fn registry_list_all_available_empty() {
        let registry = Registry::new();
        assert!(registry.list_all_available().is_empty());
    }

    #[test]
    fn model_discovery_clone_works() {
        let mut discovery = ModelDiscovery::new();
        discovery.add_search_path(PathBuf::from("/test"));
        let cloned = discovery.clone();
        assert_eq!(discovery.search_paths().len(), cloned.search_paths().len());
    }
}
