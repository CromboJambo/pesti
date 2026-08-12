//! pesti-runner: Portable Execution Substrate for Transformer Inference.
//!
//! Separate workspace member that eventually becomes independent.
//! Interface boundary: consumes WeightManifest from safetensors, emits InferenceResponse to guard.
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

pub mod cpu_optimized; // Optimized CPU attention with SIMD and parallelism
pub mod cpu_optimized_ndarray; // ndarray-based implementation for comparison
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
pub mod memory_pool; // Pre-allocated device memory pool for efficiency
#[cfg(feature = "cuda")]
pub mod async_memory; // Async H2D/D2H transfers for 15-25% throughput gain
#[cfg(feature = "cuda")]
pub mod cuda_shim;
#[cfg(not(feature = "cuda"))]
pub mod cuda_stub;
pub mod dequantize;
#[cfg(feature = "cuda")]
pub mod gguf_dequant;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(feature = "cuda")]
pub mod device_discovery;
#[cfg(not(feature = "cuda"))]
pub mod device_stub;
pub mod error;
#[cfg(not(feature = "cuda"))]
pub mod error_stub;
pub mod gguf_weight_loader;
pub mod inertia; // Computational inertia subsystem
pub mod inference_engine;
pub mod kernel;
pub mod model;
pub mod model_loader;
pub mod model_manager;
pub mod peft; // Parameter-efficient fine-tuning adapters (LoRA, QLoRA)
pub mod plug_in;
pub mod quantized_linear; // Quantized linear layer using tile dequantization
pub mod registry;
#[cfg(feature = "cuda")]
pub mod remote_discovery;
pub mod runner;
pub mod runtime;
pub mod safetensors_tokenizer;
pub mod safetensors_weight_loader;
pub mod tier;
pub mod tile_dequant; // Tile-by-tile dequantization for memory efficiency
pub mod tokenizer;
#[cfg(feature = "cuda")]
pub mod transformer;
pub mod transformer_cpu;
#[cfg(not(feature = "cuda"))]
pub mod transformer_stub;
pub mod trl; // TRL-like training orchestrator
pub mod unsloth; // Unsloth-style efficient training optimizations
pub mod unsloth_client; // Unsloth Studio SDK bridge (sync version)
pub mod unsloth_client_async; // Unsloth Studio SDK bridge (async version with tokio) // CPU-only full forward pass implementation

// Re-export CPU transformer primitives for external use
#[cfg(not(feature = "cuda"))]
pub use transformer_cpu::{CpuTransformerModel, Linear, RmsNorm, TransformerConfig};

#[cfg(feature = "cuda")]
pub use cuda_runtime::{
    CudaDeviceInfo, CudaError, CudaRuntime, device_count, enumerate_devices, is_available,
    select_best_device,
};
#[cfg(feature = "cuda")]
pub use device::{DeviceBackend, DeviceInfo, DeviceSelection, DeviceSelector, DeviceType};
#[cfg(feature = "cuda")]
pub use device_discovery::LocalDevice;
#[cfg(feature = "cuda")]
pub use error::{Result, RunnerError};
#[cfg(not(feature = "cuda"))]
pub use error_stub::{Result, RunnerError};
pub use gguf_weight_loader::{GgufWeights, load_gguf_tensor, load_gguf_weights};
pub use inference_engine::InferenceEngine;
pub use kernel::{AttentionKernel, CpuAttentionKernel, GemmBuilder, GemmKernel};
#[cfg(feature = "cuda")]
pub use kernel::{DeviceBuffer, HostTmaDescriptor, Kvcache};
#[cfg(not(feature = "cuda"))]
pub use kernel::{DeviceBuffer, kvcache_stub::Kvcache};
#[cfg(not(feature = "cuda"))]
pub use model::CpuModel;
pub use model::{Model, ModelConfig};
pub use model_loader::ModelLoader;
pub use model_manager::{ModelManager, ModelSpec, PreloadConfig, PreloadStats};
pub use plug_in::PlugInProtocol;
pub use registry::{DiscoveredModel, ModelDiscovery, ModelEntry, ModelFormat, Registry};
#[cfg(feature = "cuda")]
pub use remote_discovery::{RemoteDevice, RemoteDiscoveryConfig};
pub use runner::{DeviceRouter, RouteResult, RunnerBridge};
pub use safetensors_tokenizer::{
    SafetensorsTokenizerConfig, load_tokenizer_for_model, load_tokenizer_from_safetensors,
};
pub use safetensors_weight_loader::{
    SafetensorsWeights, extract_safetensors_config, get_safetensors_tensor_count,
    get_safetensors_total_size, load_safetensors_tensor, load_safetensors_weights,
};
pub use tier::{LayerProfiler, Tier, TieredExecution};
pub use tokenizer::Tokenizer;
#[cfg(feature = "cuda")]
pub use transformer::{
    GgufTokenizerConfig, LlamaModel, SamplingConfig, argmax, load_tokenizer_from_gguf, sample,
};
#[cfg(not(feature = "cuda"))]
pub use transformer_stub::{
    GgufTokenizerConfig, LlamaModel, SamplingConfig, argmax, load_tokenizer_from_gguf, sample,
};

// ── llama.rs: High-level API over llama.cpp ──
pub mod llama;

pub use llama::{
    ContextConfig, GenerationResult, KvCacheType, LlamaRunner, LlamaRunnerBuilder, ModelInfo,
    SessionManager, StreamingResult, TokenCallback, TokenInfo,
};
pub use runtime::{ModelState, RunnerBackend, Runtime, RuntimeConfig};

// ── Mistral.rs backend (optional, enabled via `mistralrs` feature) ──
#[cfg(feature = "mistralrs")]
pub mod mistralrs_backend {
    pub use crate::kernel::mistralrs_backend::{
        MistralRsAttentionKernel, MistralRsBackend, MistralRsGemmKernel,
    };
}
