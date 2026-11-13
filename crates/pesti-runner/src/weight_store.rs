//! Weight storage layering — Layers 0, 1, and 2.
//!
//! This module implements Phases 1-2 of TENSOR_LAYER_SEPARATION_SPEC:
//! - `CoreTensor`: name + raw bytes + shape + dtype (no layout interpretation) [Layer 0]
//! - `TensorInfo`: per-tensor metadata (shape, dtype, byte offsets) [Layer 1]
//! - `TensorStructure`: Layer 1 — the model's tensor graph as a lookup table
//! - `SafetensorsFile`: mmap wrapper with zero-copy tensor access
//! - `SystemFactors`: Layer 2 — inference-critical config (rope, norms, attention pattern)
//! - `WeightStore`: weight storage that loads Layers 0+1 (minimal) or 0+1+2 (inference)
//!
//! Design principles:
//! - Zero-copy: tensor data is a slice into the mmap'd file, not a copy
//! - No layout interpretation: CoreTensor doesn't know about row-major vs column-major
//! - No behavior change: existing `LlamaModel::load_safetensors()` continues to work
//! - Additive: this module can be adopted incrementally by consumers

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use tracing::debug;

use crate::error::{Result, RunnerError};
use crate::transformer::rope::RopeConfig;

/// Data type of a tensor element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U8,
    I8,
    I16,
    I32,
    I64,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::I64 => 8,
        }
    }

    /// Parse from safetensors dtype string.
    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "F32" | "FLOAT_32" => Some(Dtype::F32),
            "F16" | "FLOAT_16" => Some(Dtype::F16),
            "BF16" | "BFLOAT_16" => Some(Dtype::BF16),
            "U8" | "UINT_8" => Some(Dtype::U8),
            "I8" | "INT_8" => Some(Dtype::I8),
            "I16" | "INT_16" => Some(Dtype::I16),
            "I32" | "INT_32" => Some(Dtype::I32),
            "I64" | "INT_64" => Some(Dtype::I64),
            _ => None,
        }
    }

    /// Convert to safetensors dtype string.
    pub fn to_safetensors(&self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
        }
    }
}

/// Per-tensor metadata (Layer 1).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: Dtype,
    /// Byte offset within the safetensors file (after header).
    pub byte_offset: usize,
    /// Total size in bytes.
    pub size_bytes: usize,
}

impl TensorInfo {
    /// Number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Verify that size_bytes matches shape × dtype size.
    pub fn is_consistent(&self) -> bool {
        self.size_bytes == self.element_count() * self.dtype.size()
    }
}

/// Layer 1: The model's tensor graph as a lookup table.
#[derive(Debug, Clone)]
pub struct TensorStructure {
    /// Tensors by name.
    pub tensors: HashMap<String, TensorInfo>,
    /// Architecture identifier (e.g. "llama", "qwen2").
    pub arch: String,
    /// Total number of parameters (sum of all element counts).
    pub total_parameters: usize,
    /// Number of transformer layers detected from tensor names.
    pub num_layers: usize,
    /// Hidden size (embedding dimension), inferred from embedding weight shape.
    pub hidden_size: Option<usize>,
}

impl TensorStructure {
    /// Get metadata for a tensor by name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the structure is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Find all tensors matching a name prefix.
    pub fn with_prefix(&self, prefix: &str) -> Vec<&TensorInfo> {
        self.tensors
            .values()
            .filter(|t| t.name.starts_with(prefix))
            .collect()
    }

    /// Get the number of transformer layers detected in this structure.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the hidden size (embedding dimension).
    pub fn hidden_size(&self) -> Option<usize> {
        self.hidden_size
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Layer 2: System Factors
// ──────────────────────────────────────────────────────────────────────────────

/// Attention masking pattern for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPattern {
    /// Standard causal mask (lower triangular) — decoder-only models.
    Causal,
    /// No masking — encoder-style bidirectional attention.
    Bidirectional,
    /// Sliding window attention with the given window size in tokens.
    SlidingWindow(usize),
}

/// Normalization statistics for a layer.
#[derive(Debug, Clone)]
pub struct NormStats {
    /// Epsilon for RMS/Layer norm (typically 1e-5 or 1e-6).
    pub eps: f32,
    /// Optional per-channel weight (for learned norms like RMSNorm).
    pub weight: Option<Arc<[f32]>>,
}

/// Quantization scheme used by the model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// No quantization (F32).
    None,
    /// 16-bit floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// Q4_K quants (GGUF style).
    Q4K,
    /// Q5_K quants (GGUF style).
    Q5K,
    /// Q8_0 quants (GGUF style).
    Q8_0,
}

impl QuantScheme {
    /// Infer the quantization scheme from a safetensors dtype string.
    pub fn from_dtype(dtype: &str) -> Self {
        match dtype {
            "F16" | "FLOAT_16" => QuantScheme::F16,
            "BF16" | "BFLOAT_16" => QuantScheme::BF16,
            _ => QuantScheme::None,
        }
    }
}

/// Per-layer system factors.
#[derive(Debug, Clone)]
pub struct LayerSystemFactors {
    /// Normalization statistics for this layer.
    pub norm: NormStats,
    /// Optional attention bias (for models that use it).
    pub attention_bias: Option<Arc<[f32]>>,
}

/// Layer 2: System factors required for correct inference.
///
/// These are the model parameters that determine *how* tensors are used,
/// not *what* they contain. Without these, the same tensor bytes could be
/// interpreted incorrectly (wrong RoPE base, wrong norm epsilon, etc.).
#[derive(Debug, Clone)]
pub struct SystemFactors {
    /// RoPE configuration (base, dimension, scaling).
    pub rope: RopeConfig,
    /// Global normalization epsilon (used when per-layer stats aren't available).
    pub norm_eps: f32,
    /// Attention masking pattern.
    pub attention_pattern: AttentionPattern,
    /// Quantization scheme of the weights.
    pub quantization: QuantScheme,
    /// Per-layer system factors (norms, biases).
    pub per_layer: Vec<LayerSystemFactors>,
}

impl SystemFactors {
    /// Create default system factors for a standard causal LLM.
    pub fn new(head_dim: usize, rope_base: f32, max_position_embeddings: usize) -> Self {
        let rope = RopeConfig::new(head_dim, rope_base, max_position_embeddings);
        Self {
            rope,
            norm_eps: 1e-5,
            attention_pattern: AttentionPattern::Causal,
            quantization: QuantScheme::None,
            per_layer: Vec::new(),
        }
    }

    /// Get norm stats for a specific layer, falling back to global epsilon.
    pub fn norm_for_layer(&self, layer: usize) -> NormStats {
        self.per_layer
            .get(layer)
            .map(|lf| lf.norm.clone())
            .unwrap_or(NormStats {
                eps: self.norm_eps,
                weight: None,
            })
    }
}

/// Minimal interface for inference (Layer 0 + 1 + 2).
///
/// Types implementing this trait provide access to both tensor data
/// and the system factors needed to use them correctly.
pub trait InferenceReader: TensorReader {
    /// Get system factors (rope, norms, attention pattern).
    fn system_factors(&self) -> Option<&SystemFactors>;

    /// Get norm stats for a specific layer.
    fn norm_stats(&self, layer: usize) -> Option<NormStats>;

    /// Get RoPE configuration.
    fn rope_config(&self) -> Option<&RopeConfig>;

    /// Get the attention pattern.
    fn attention_pattern(&self) -> Option<AttentionPattern>;
}

/// Layer 0: A core tensor — name + raw bytes + shape + dtype.
///
/// No layout interpretation (row-major vs column-major is a Layer 2 concern).
/// Data is zero-copy from the mmap'd file.
#[derive(Debug)]
pub struct CoreTensor {
    /// Tensor name.
    pub name: String,
    /// Raw bytes (slice into mmap'd file — zero-copy).
    pub data: Arc<[u8]>,
    /// Shape as a list of dimensions.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: Dtype,
}

impl CoreTensor {
    /// Number of elements in this tensor.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// View the data as f32 slices (only valid for F32 dtype).
    #[inline]
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != Dtype::F32 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, count) })
    }

    /// View the data as f16 slices (only valid for F16 dtype).
    #[inline]
    pub fn as_f16(&self) -> Option<&[half::f16]> {
        if self.dtype != Dtype::F16 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const half::f16, count) })
    }

    /// View the data as bf16 slices (only valid for BF16 dtype).
    #[inline]
    pub fn as_bf16(&self) -> Option<&[half::bf16]> {
        if self.dtype != Dtype::BF16 {
            return None;
        }
        let count = self.element_count();
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const half::bf16, count) })
    }
}

/// Minimal interface for reading core tensors (Layer 2 contract).
pub trait TensorReader {
    /// Get a tensor by name. Returns None if not found.
    fn get_tensor(&self, name: &str) -> Option<CoreTensor>;

    /// Get tensor metadata without loading data.
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo>;

    /// List all tensor names.
    fn tensor_names(&self) -> Vec<String>;

    /// The model's tensor structure (Layer 1).
    fn structure(&self) -> &TensorStructure;
}

/// Memory-mapped safetensors file with zero-copy tensor access.
///
/// This is the Layer 0 storage backend. The file is mmap'd once, and tensor
/// data is accessed as slices into the mapping — no copying.
pub struct SafetensorsFile {
    /// Path to the safetensors file.
    pub path: PathBuf,
    /// Memory-mapped file contents (kept alive for the lifetime of this struct).
    mmap: Mmap,
    /// Layer 1: tensor structure parsed from the header.
    pub structure: TensorStructure,
    /// Metadata from the safetensors header (`__metadata__` field).
    metadata: HashMap<String, String>,
}

impl SafetensorsFile {
    /// Open a safetensors file with mmap (zero-copy).
    ///
    /// Parses the header to build the TensorStructure. Tensor data is not
    /// loaded until accessed via `tensor_data()`.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| RunnerError::ModelLoad(format!("Failed to open safetensors file: {e}")))?;

        // Memory-map the file (read-only).
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| RunnerError::ModelLoad(format!("Failed to mmap safetensors file: {e}")))?;

        // Parse the header. Safetensors format:
        // - u64 LE: header size
        // - JSON header (header_size bytes)
        // - tensor data
        if mmap.len() < 8 {
            return Err(RunnerError::ModelLoad(
                "Safetensors file too small".to_string(),
            ));
        }

        let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if 8 + header_size > mmap.len() {
            return Err(RunnerError::ModelLoad(
                "Safetensors header size exceeds file size".to_string(),
            ));
        }

        let header_json = std::str::from_utf8(&mmap[8..8 + header_size])
            .map_err(|e| RunnerError::ModelLoad(format!("Invalid UTF-8 in header: {e}")))?;

        // Parse the JSON header. Format: {"__metadata__": {...}, "tensor_name": {"dtype": "...", "shape": [...], "data_offsets": [start, end]}, ...}
        let header: serde_json::Value = serde_json::from_str(header_json).map_err(|e| {
            RunnerError::ModelLoad(format!("Failed to parse safetensors header: {e}"))
        })?;

        // Extract metadata (architecture info and other key-value pairs).
        let mut metadata = HashMap::new();
        let arch = header
            .get("__metadata__")
            .and_then(|m| m.get("format"))
            .and_then(|f| f.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Store all metadata string values for later use.
        if let Some(meta_obj) = header.get("__metadata__").and_then(|m| m.as_object()) {
            for (k, v) in meta_obj {
                if let Some(s) = v.as_str() {
                    metadata.insert(k.clone(), s.to_string());
                }
            }
        }

        // Build TensorStructure.
        let mut tensors = HashMap::new();
        let mut total_parameters = 0usize;
        let mut max_layer_idx = 0usize;
        let mut hidden_size = None;

        for (name, value) in header.as_object().ok_or_else(|| {
            RunnerError::ModelLoad("Safetensors header is not an object".to_string())
        })? {
            if name == "__metadata__" {
                continue;
            }

            let dtype_str = value
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| RunnerError::ModelLoad(format!("Tensor '{name}' missing dtype")))?;

            let dtype = Dtype::from_safetensors(dtype_str).ok_or_else(|| {
                RunnerError::ModelLoad(format!(
                    "Unsupported dtype '{dtype_str}' for tensor '{name}'"
                ))
            })?;

            let shape: Vec<usize> = value
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| RunnerError::ModelLoad(format!("Tensor '{name}' missing shape")))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();

            let offsets = value
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| {
                    RunnerError::ModelLoad(format!("Tensor '{name}' missing data_offsets"))
                })?;

            let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_size;
            let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_size;

            if end > mmap.len() {
                return Err(RunnerError::ModelLoad(format!(
                    "Tensor '{name}' data extends beyond file size"
                )));
            }

            let size_bytes = end - start;
            let element_count: usize = shape.iter().product();

            // Detect layer index from tensor names like "model.layers.37.self_attn.q_proj.weight"
            if let Some(layer_idx) = extract_layer_index(name) {
                max_layer_idx = max_layer_idx.max(layer_idx + 1);
            }

            // Infer hidden size from the embedding weight (first dimension of [hidden, vocab] or [vocab, hidden])
            if name.contains("embed") && shape.len() == 2 && hidden_size.is_none() {
                // Embedding is typically [vocab_size, hidden_size] or [hidden_size, vocab_size]
                // The smaller dimension is usually hidden_size for large vocab models
                hidden_size = Some(shape[1].min(shape[0]));
            }

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    shape,
                    dtype,
                    byte_offset: start,
                    size_bytes,
                },
            );
            total_parameters += element_count;
        }

        let structure = TensorStructure {
            tensors,
            arch,
            total_parameters,
            num_layers: max_layer_idx,
            hidden_size,
        };

        debug!(
            path = %path.display(),
            tensor_count = structure.len(),
            total_params = structure.total_parameters,
            "Opened safetensors file (mmap)"
        );

        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            structure,
            metadata,
        })
    }

    /// Get zero-copy tensor data by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.structure.get(name)?;
        Some(&self.mmap[info.byte_offset..info.byte_offset + info.size_bytes])
    }

    /// Get a CoreTensor by name (zero-copy from mmap).
    pub fn core_tensor(&self, name: &str) -> Option<CoreTensor> {
        let info = self.structure.get(name)?.clone();
        let data = self.tensor_data(&info.name)?;
        Some(CoreTensor {
            name: info.name,
            data: Arc::from(data),
            shape: info.shape,
            dtype: info.dtype,
        })
    }

    /// Get the metadata from the safetensors header.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Total file size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }
}

impl TensorReader for SafetensorsFile {
    fn get_tensor(&self, name: &str) -> Option<CoreTensor> {
        self.core_tensor(name)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.structure.get(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.structure.names().cloned().collect()
    }

    fn structure(&self) -> &TensorStructure {
        &self.structure
    }
}

/// Weight store — loads Layers 0+1 (minimal) or 0+1+2 (inference).
///
/// This is the main public API for weight access. Use:
/// - `load_minimal()` for Phase 1: tensor data + structure only
/// - `load_inference()` for Phase 2: adds system factors (rope, norms, attention pattern)
pub struct WeightStore {
    /// The underlying safetensors file (mmap'd).
    pub file: SafetensorsFile,
    /// Layer 2: System factors required for inference (None in minimal mode).
    system: Option<SystemFactors>,
}

impl WeightStore {
    /// Load minimal weight store (Layers 0+1 only).
    ///
    /// This mmaps the file and parses the header. No tensor data is copied
    /// until accessed via `tensor_data()` or `core_tensor()`.
    pub fn load_minimal(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        Ok(Self {
            file,
            system: None,
        })
    }

    /// Load weight store with full inference support (Layers 0+1+2).
    ///
    /// Parses system factors (rope config, norm stats, attention pattern) from
    /// either a HuggingFace config.json or safetensors metadata.
    pub fn load_inference(path: &Path, config_path: Option<&Path>) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;

        // Parse system factors from config.json or safetensors metadata
        let system = if let Some(cp) = config_path {
            parse_system_factors_from_config_json(cp, &file.structure)?
        } else {
            parse_system_factors_from_metadata(file.metadata(), &file.structure)?
        };

        debug!(
            path = %path.display(),
            tensor_count = file.structure.len(),
            num_layers = file.structure.num_layers,
            rope_base = system.rope.base,
            norm_eps = system.norm_eps,
            "Loaded inference weight store (Layers 0+1+2)"
        );

        Ok(Self {
            file,
            system: Some(system),
        })
    }

    /// Get zero-copy tensor data by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        self.file.tensor_data(name)
    }

    /// Get a CoreTensor by name (zero-copy from mmap).
    pub fn core_tensor(&self, name: &str) -> Option<CoreTensor> {
        self.file.core_tensor(name)
    }

    /// Get tensor metadata.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.file.tensor_info(name)
    }

    /// The model's tensor structure (Layer 1).
    pub fn structure(&self) -> &TensorStructure {
        &self.file.structure
    }

    /// Get system factors (Layer 2), if loaded in inference mode.
    pub fn system_factors(&self) -> Option<&SystemFactors> {
        self.system.as_ref()
    }

    /// Verify all tensors in the structure are consistent.
    pub fn verify_structure(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for info in self.file.structure.tensors.values() {
            if !info.is_consistent() {
                errors.push(format!(
                    "Tensor '{}' inconsistent: shape {:?} × dtype {:?} = {} bytes, but size_bytes = {}",
                    info.name,
                    info.shape,
                    info.dtype,
                    info.element_count() * info.dtype.size(),
                    info.size_bytes
                ));
            }
        }
        errors
    }
}

impl TensorReader for WeightStore {
    fn get_tensor(&self, name: &str) -> Option<CoreTensor> {
        self.core_tensor(name)
    }

    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.file.tensor_info(name)
    }

    fn tensor_names(&self) -> Vec<String> {
        self.file.structure.names().cloned().collect()
    }

    fn structure(&self) -> &TensorStructure {
        &self.file.structure
    }
}

impl InferenceReader for WeightStore {
    fn system_factors(&self) -> Option<&SystemFactors> {
        self.system.as_ref()
    }

    fn norm_stats(&self, layer: usize) -> Option<NormStats> {
        self.system.as_ref().map(|s| s.norm_for_layer(layer))
    }

    fn rope_config(&self) -> Option<&RopeConfig> {
        self.system.as_ref().map(|s| &s.rope)
    }

    fn attention_pattern(&self) -> Option<AttentionPattern> {
        self.system.as_ref().map(|s| s.attention_pattern)
    }
}

/// Extract the layer index from a tensor name like "model.layers.37.self_attn.q_proj.weight".
/// Returns None if the name doesn't match the expected pattern.
fn extract_layer_index(name: &str) -> Option<usize> {
    // Pattern: ...layers.{N}...
    let parts: Vec<&str> = name.split('.').collect();
    for i in 0..parts.len() {
        if parts[i] == "layers" && i + 1 < parts.len() {
            return parts[i + 1].parse::<usize>().ok();
        }
    }
    None
}

/// Parse system factors from a HuggingFace config.json file.
fn parse_system_factors_from_config_json(
    path: &Path,
    structure: &TensorStructure,
) -> Result<SystemFactors> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to read config.json: {e}")))?;

    let config: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| RunnerError::ModelLoad(format!("Failed to parse config.json: {e}")))?;

    // Extract rope_theta (default 10000.0)
    let rope_base = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0) as f32;

    // Extract max_position_embeddings (default 4096)
    let max_pos = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as usize;

    // Extract rms_norm_eps (default 1e-5)
    let norm_eps = config
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;

    // Determine head dimension from structure
    let hidden_size = structure.hidden_size.unwrap_or(4096);
    let num_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or((hidden_size as u64) / 128) as usize;
    let head_dim = hidden_size / num_heads;

    // Determine attention pattern
    let attention_pattern = if config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|m| m.contains("encoder"))
        .unwrap_or(false)
    {
        AttentionPattern::Bidirectional
    } else {
        AttentionPattern::Causal
    };

    // Determine quantization from torch_dtype
    let torch_dtype = config
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("float32");
    let quantization = match torch_dtype {
        "float16" | "fp16" => QuantScheme::F16,
        "bfloat16" | "bf16" => QuantScheme::BF16,
        _ => QuantScheme::None,
    };

    // Build per-layer factors (norms with default epsilon)
    let per_layer = (0..structure.num_layers)
        .map(|_| LayerSystemFactors {
            norm: NormStats {
                eps: norm_eps,
                weight: None,
            },
            attention_bias: None,
        })
        .collect();

    let rope = RopeConfig::new(head_dim, rope_base, max_pos);

    debug!(
        path = %path.display(),
        rope_base,
        max_pos,
        norm_eps,
        head_dim,
        num_layers = structure.num_layers,
        "Parsed system factors from config.json"
    );

    Ok(SystemFactors {
        rope,
        norm_eps,
        attention_pattern,
        quantization,
        per_layer,
    })
}

/// Parse system factors from safetensors metadata (fallback when no config.json).
fn parse_system_factors_from_metadata(
    metadata: &HashMap<String, String>,
    structure: &TensorStructure,
) -> Result<SystemFactors> {
    let rope_base = metadata
        .get("rope_theta")
        .and_then(|v| v.parse::<f32>().ok())
        .or_else(|| metadata.get("rope_base").and_then(|v| v.parse::<f32>().ok()))
        .unwrap_or(10000.0);

    let max_pos = metadata
        .get("max_position_embeddings")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| metadata.get("context_length").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(4096);

    let norm_eps = metadata
        .get("rms_norm_eps")
        .and_then(|v| v.parse::<f32>().ok())
        .or_else(|| metadata.get("norm_eps").and_then(|v| v.parse::<f32>().ok()))
        .unwrap_or(1e-5);

    let head_dim = 128; // Default head dimension

    let per_layer = (0..structure.num_layers)
        .map(|_| LayerSystemFactors {
            norm: NormStats {
                eps: norm_eps,
                weight: None,
            },
            attention_bias: None,
        })
        .collect();

    let rope = RopeConfig::new(head_dim, rope_base, max_pos);

    debug!(
        rope_base,
        max_pos,
        norm_eps,
        head_dim,
        num_layers = structure.num_layers,
        "Parsed system factors from safetensors metadata"
    );

    Ok(SystemFactors {
        rope,
        norm_eps,
        attention_pattern: AttentionPattern::Causal,
        quantization: QuantScheme::None,
        per_layer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::F32.size(), 4);
        assert_eq!(Dtype::F16.size(), 2);
        assert_eq!(Dtype::BF16.size(), 2);
        assert_eq!(Dtype::U8.size(), 1);
        assert_eq!(Dtype::I8.size(), 1);
        assert_eq!(Dtype::I16.size(), 2);
        assert_eq!(Dtype::I32.size(), 4);
        assert_eq!(Dtype::I64.size(), 8);
    }

    #[test]
    fn test_dtype_roundtrip() {
        for dtype in [
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::U8,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::I64,
        ] {
            let s = dtype.to_safetensors();
            let parsed = Dtype::from_safetensors(s).unwrap();
            assert_eq!(dtype, parsed, "Roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_tensor_info_consistency() {
        let info = TensorInfo {
            name: "test".to_string(),
            shape: vec![4, 8],
            dtype: Dtype::F32,
            byte_offset: 0,
            size_bytes: 4 * 8 * 4, // 128 bytes
        };
        assert!(info.is_consistent());

        let bad = TensorInfo {
            name: "bad".to_string(),
            shape: vec![4, 8],
            dtype: Dtype::F32,
            byte_offset: 0,
            size_bytes: 100, // wrong size
        };
        assert!(!bad.is_consistent());
    }

    #[test]
    fn test_core_tensor_as_f32() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let data = Arc::from(bytes);
        let tensor = CoreTensor {
            name: "test".to_string(),
            data,
            shape: vec![2, 2],
            dtype: Dtype::F32,
        };

        let f32_view = tensor.as_f32().unwrap();
        assert_eq!(f32_view.len(), 4);
        assert_eq!(f32_view[0], 1.0);
        assert_eq!(f32_view[3], 4.0);
    }

    #[test]
    fn test_core_tensor_wrong_dtype() {
        let data = Arc::from(vec![1u8, 2, 3, 4]);
        let tensor = CoreTensor {
            name: "test".to_string(),
            data,
            shape: vec![4],
            dtype: Dtype::U8,
        };

        assert!(tensor.as_f32().is_none());
    }

    #[test]
    fn test_tensor_structure_lookup() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            TensorInfo {
                name: "model.layers.0.weight".to_string(),
                shape: vec![4, 8],
                dtype: Dtype::F32,
                byte_offset: 0,
                size_bytes: 128,
            },
        );
        tensors.insert(
            "model.layers.1.weight".to_string(),
            TensorInfo {
                name: "model.layers.1.weight".to_string(),
                shape: vec![4, 8],
                dtype: Dtype::F32,
                byte_offset: 128,
                size_bytes: 128,
            },
        );

        let structure = TensorStructure {
            tensors,
            arch: "test".to_string(),
            total_parameters: 64,
            num_layers: 2,
            hidden_size: Some(8),
        };

        assert!(structure.contains("model.layers.0.weight"));
        assert!(!structure.contains("model.layers.99.weight"));
        assert_eq!(structure.len(), 2);
        assert_eq!(structure.with_prefix("model.layers.").len(), 2);
    }

    #[test]
    fn test_weight_store_with_real_file() {
        // Create a minimal valid safetensors file in memory.
        // Format: u64 header_size + JSON header + tensor data
        let values = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 f32 values
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "test_tensor": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, tensor_bytes.len()]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);

        // Write to a temp file.
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("pesti_test_safetensors.bin");
        std::fs::write(&tmp_path, &file_bytes).unwrap();

        // Load with WeightStore.
        let store = WeightStore::load_minimal(&tmp_path).unwrap();

        // Verify structure.
        assert!(store.structure().contains("test_tensor"));
        assert_eq!(store.structure().len(), 1);

        // Verify tensor info.
        let info = store.tensor_info("test_tensor").unwrap();
        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(info.dtype, Dtype::F32);
        assert_eq!(info.size_bytes, 16);

        // Verify zero-copy data access.
        let data = store.tensor_data("test_tensor").unwrap();
        assert_eq!(data.len(), 16);

        // Verify CoreTensor creation.
        let tensor = store.core_tensor("test_tensor").unwrap();
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, Dtype::F32);

        let f32_view = tensor.as_f32().unwrap();
        assert_eq!(f32_view.len(), 4);
        assert_eq!(f32_view[0], 1.0);
        assert_eq!(f32_view[3], 4.0);

        // Verify structure consistency.
        assert!(store.verify_structure().is_empty());

        // Cleanup.
        std::fs::remove_file(&tmp_path).unwrap();
    }

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(extract_layer_index("model.layers.37.mlp.gate_proj.weight"), Some(37));
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
        assert_eq!(extract_layer_index("layers.5.attn.v_proj.weight"), Some(5));
    }

    #[test]
    fn test_system_factors_default() {
        let factors = SystemFactors::new(128, 10000.0, 4096);
        assert_eq!(factors.rope.head_dim, 128);
        assert_eq!(factors.rope.base, 10000.0);
        assert_eq!(factors.rope.max_position_embeddings, 4096);
        assert!((factors.norm_eps - 1e-5).abs() < f32::EPSILON);
        assert_eq!(factors.attention_pattern, AttentionPattern::Causal);
        assert_eq!(factors.quantization, QuantScheme::None);
        assert!(factors.per_layer.is_empty());
    }

    #[test]
    fn test_system_factors_norm_fallback() {
        let mut factors = SystemFactors::new(128, 10000.0, 4096);
        factors.norm_eps = 1e-6;
        factors.per_layer.push(LayerSystemFactors {
            norm: NormStats {
                eps: 1e-4,
                weight: None,
            },
            attention_bias: None,
        });

        // Layer 0 has explicit stats.
        assert!((factors.norm_for_layer(0).eps - 1e-4).abs() < f32::EPSILON);
        // Layer 1 falls back to global epsilon.
        assert!((factors.norm_for_layer(1).eps - 1e-6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_system_factors_from_config_json() {
        let tmp_dir = std::env::temp_dir();
        let config_path = tmp_dir.join("pesti_test_config.json");

        // Write a mock HuggingFace config.json.
        let config = serde_json::json!({
            "model_type": "llama",
            "rope_theta": 1000000.0,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "num_attention_heads": 32,
            "torch_dtype": "float16"
        });
        std::fs::write(&config_path, config.to_string()).unwrap();

        // Create a mock structure with known properties.
        let mut tensors = HashMap::new();
        for i in 0..2 {
            tensors.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                TensorInfo {
                    name: format!("model.layers.{i}.self_attn.q_proj.weight"),
                    shape: vec![4096, 4096],
                    dtype: Dtype::F16,
                    byte_offset: i * 32 * 1024 * 1024,
                    size_bytes: 32 * 1024 * 1024,
                },
            );
        }
        let structure = TensorStructure {
            tensors,
            arch: "llama".to_string(),
            total_parameters: 2 * 4096 * 4096,
            num_layers: 2,
            hidden_size: Some(4096),
        };

        let factors = parse_system_factors_from_config_json(&config_path, &structure).unwrap();

        assert!((factors.rope.base - 1000000.0).abs() < f32::EPSILON);
        assert_eq!(factors.rope.max_position_embeddings, 8192);
        assert!((factors.norm_eps - 1e-5).abs() < f32::EPSILON);
        assert_eq!(factors.attention_pattern, AttentionPattern::Causal);
        assert_eq!(factors.quantization, QuantScheme::F16);
        assert_eq!(factors.per_layer.len(), 2);

        std::fs::remove_file(&config_path).unwrap();
    }

    #[test]
    fn test_parse_system_factors_from_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("rope_theta".to_string(), "500000.0".to_string());
        metadata.insert("max_position_embeddings".to_string(), "2048".to_string());
        metadata.insert("rms_norm_eps".to_string(), "1e-6".to_string());

        let structure = TensorStructure {
            tensors: HashMap::new(),
            arch: "test".to_string(),
            total_parameters: 0,
            num_layers: 3,
            hidden_size: Some(2048),
        };

        let factors = parse_system_factors_from_metadata(&metadata, &structure).unwrap();

        assert!((factors.rope.base - 500000.0).abs() < f32::EPSILON);
        assert_eq!(factors.rope.max_position_embeddings, 2048);
        assert!((factors.norm_eps - 1e-6).abs() < f32::EPSILON);
        assert_eq!(factors.attention_pattern, AttentionPattern::Causal);
        assert_eq!(factors.per_layer.len(), 3);
    }

    #[test]
    fn test_weight_store_inference_mode() {
        // Create a minimal valid safetensors file with model-like tensor names.
        let values = vec![1.0f32];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "__metadata__": {
                "format": "pt",
                "rope_theta": "10000.0",
                "max_position_embeddings": "4096",
                "rms_norm_eps": "1e-5"
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);

        let tmp_dir = std::env::temp_dir();
        let wt_path = tmp_dir.join("pesti_test_inference_weights.bin");
        std::fs::write(&wt_path, &file_bytes).unwrap();

        // Load in inference mode (no config.json → uses metadata).
        let store = WeightStore::load_inference(&wt_path, None).unwrap();

        // Verify structure.
        assert_eq!(store.structure().num_layers(), 1);
        assert!(store.structure().contains("model.layers.0.self_attn.q_proj.weight"));

        // Verify system factors are present.
        let system = store.system_factors().unwrap();
        assert!((system.rope.base - 10000.0).abs() < f32::EPSILON);
        assert_eq!(system.rope.max_position_embeddings, 4096);
        assert_eq!(system.attention_pattern, AttentionPattern::Causal);

        // Verify InferenceReader trait methods.
        let norm = store.norm_stats(0).unwrap();
        assert!((norm.eps - 1e-5).abs() < f32::EPSILON);

        let rope = store.rope_config().unwrap();
        assert_eq!(rope.base, system.rope.base);

        let pattern = store.attention_pattern().unwrap();
        assert_eq!(pattern, AttentionPattern::Causal);

        // Verify TensorReader trait still works.
        assert!(store.get_tensor("model.layers.0.self_attn.q_proj.weight").is_some());
        assert_eq!(store.tensor_names().len(), 1);

        std::fs::remove_file(&wt_path).unwrap();
    }

    #[test]
    fn test_weight_store_minimal_has_no_system() {
        // Create a minimal valid safetensors file.
        let values = vec![1.0f32];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "test_tensor": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);

        let tmp_dir = std::env::temp_dir();
        let wt_path = tmp_dir.join("pesti_test_minimal_weights.bin");
        std::fs::write(&wt_path, &file_bytes).unwrap();

        // Load in minimal mode.
        let store = WeightStore::load_minimal(&wt_path).unwrap();

        // System factors should be None in minimal mode.
        assert!(store.system_factors().is_none());
        assert!(store.norm_stats(0).is_none());
        assert!(store.rope_config().is_none());
        assert!(store.attention_pattern().is_none());

        std::fs::remove_file(&wt_path).unwrap();
    }

    #[test]
    fn test_weight_store_missing_tensor() {
        let tmp_dir = std::env::temp_dir();
        let tmp_path = tmp_dir.join("pesti_test_safetensors_missing.bin");

        // Create minimal file with one tensor.
        let values = vec![1.0f32];
        let tensor_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header_json = serde_json::json!({
            "only_tensor": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4]
            }
        })
        .to_string();

        let header_size = header_json.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(header_json.as_bytes());
        file_bytes.extend_from_slice(&tensor_bytes);
        std::fs::write(&tmp_path, &file_bytes).unwrap();

        let store = WeightStore::load_minimal(&tmp_path).unwrap();

        assert!(store.tensor_data("nonexistent").is_none());
        assert!(store.core_tensor("nonexistent").is_none());
        assert!(store.tensor_info("nonexistent").is_none());

        std::fs::remove_file(&tmp_path).unwrap();
    }
}
