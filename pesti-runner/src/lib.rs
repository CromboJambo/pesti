//! pesti-runner: Portable Execution Substrate for Transformer Inference.
//!
//! Separate workspace member that eventually becomes independent.
//! Interface boundary: consumes WeightManifest from safetensors, emits InferenceResponse to guard.

#[cfg(feature = "cuda")]
pub mod cuda_runtime;
#[cfg(not(feature = "cuda"))]
pub mod cuda_stub;
pub mod dequantize;
#[cfg(feature = "cuda")]
pub mod device;
#[cfg(not(feature = "cuda"))]
pub mod device_stub;
#[cfg(feature = "cuda")]
pub mod device_discovery;
pub mod error;
#[cfg(not(feature = "cuda"))]
pub mod error_stub;
pub mod gguf_weight_loader;
pub mod inference_engine;
pub mod kernel;
pub mod model;
pub mod model_loader;
pub mod model_manager;
pub mod plug_in;
pub mod registry;
#[cfg(feature = "cuda")]
pub mod remote_discovery;
pub mod runtime;
pub mod runner;
pub mod safetensors_weight_loader;
pub mod safetensors_tokenizer;
pub mod tier;
pub mod tokenizer;
#[cfg(feature = "cuda")]
pub mod transformer;
#[cfg(not(feature = "cuda"))]
pub mod transformer_stub;

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
pub use safetensors_weight_loader::{
    extract_safetensors_config, get_safetensors_tensor_count, get_safetensors_total_size,
    load_safetensors_tensor, load_safetensors_weights, SafetensorsWeights,
};
pub use safetensors_tokenizer::{
    load_tokenizer_for_model, load_tokenizer_from_safetensors, SafetensorsTokenizerConfig,
};
pub use inference_engine::InferenceEngine;
pub use kernel::{AttentionKernel, CpuAttentionKernel, GemmBuilder, GemmKernel};
#[cfg(feature = "cuda")]
pub use kernel::{DeviceBuffer, HostTmaDescriptor, Kvcache};
#[cfg(not(feature = "cuda"))]
pub use kernel::{DeviceBuffer as DeviceBuffer, kvcache_stub::Kvcache};
pub use model::{CpuModel, Model, ModelConfig};
pub use model_loader::ModelLoader;
pub use model_manager::{ModelManager, ModelSpec, PreloadConfig, PreloadStats};
pub use plug_in::PlugInProtocol;
pub use registry::{DiscoveredModel, ModelDiscovery, ModelEntry, ModelFormat, Registry};
#[cfg(feature = "cuda")]
pub use remote_discovery::{RemoteDevice, RemoteDiscoveryConfig};
pub use runner::{DeviceRouter, RouteResult, RunnerBridge};
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
    GenerationResult, KvCacheType, SessionManager, StreamingResult, TokenCallback, TokenInfo,
    LlamaRunner, LlamaRunnerBuilder, ModelInfo, ContextConfig,
};
pub use runtime::{Runtime, RuntimeConfig, ModelState, RunnerBackend};

// ── Mistral.rs backend (optional, enabled via `mistralrs` feature) ──
#[cfg(feature = "mistralrs")]
pub mod mistralrs_backend {
    pub use crate::kernel::mistralrs_backend::{
        MistralRsBackend, MistralRsGemmKernel, MistralRsAttentionKernel,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    
    

    // ── DeviceBackend ──────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    #[test]
    fn device_backend_new_defaults_to_cpu() {
        let backend = DeviceBackend::new("cuda");
        assert_eq!(backend.preference, "cuda");
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_backend_select_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        assert!(matches!(backend.device, candle_core::Device::Cpu));
    }

    // ── InferenceEngine ────────────────────────────────────────────────

    #[test]
    fn inference_engine_new_sets_device_and_dtype() {
        let engine = InferenceEngine::new(candle_core::Device::Cpu, DType::F32);
        assert!(matches!(engine.device, candle_core::Device::Cpu));
        assert_eq!(engine.dtype, DType::F32);
    }

    // ── TieredExecution tests (Phase 5.3) ────────────────────────────────

    #[test]
    fn tiered_execution_cpu_baseline_start() {
        let execution = TieredExecution::new(Tier::CpuBaseline);
        assert_eq!(execution.current_tier(), Tier::CpuBaseline);
        assert!(!execution.current_tier().is_gpu());
    }

    #[test]
    fn tiered_execution_gpu_flash_attention_start() {
        let execution = TieredExecution::new(Tier::GpuFlashAttention);
        assert_eq!(execution.current_tier(), Tier::GpuFlashAttention);
        assert!(execution.current_tier().is_gpu());
    }

    #[test]
    fn tiered_execution_invocation_tracking() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        // First 99 invocations should not trigger tier-up
        for _ in 0..99 {
            assert_eq!(execution.current_tier(), Tier::CpuBaseline);
            assert!(execution.record_invocation().is_none());
        }

        // 100th invocation triggers tier-up to GPU flash attention
        let tier_up = execution.record_invocation().unwrap();
        assert_eq!(tier_up, Tier::GpuFlashAttention);
    }

    #[test]
    fn layer_profiler_basic() {
        let profiler = LayerProfiler::new("transformer.layer.0");

        assert_eq!(profiler.count(), 0);

        for i in 1..=51 {
            let count = profiler.record_forward();
            if i % 50 == 0 {
                // Should log at multiples of 50
                assert_eq!(count, i);
            }
        }

        assert_eq!(profiler.count(), 51);
    }

    #[test]
    fn tiered_execution_reset() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        for _ in 0..150 {
            execution.record_invocation();
        }

        assert!(execution.current_tier().is_gpu()); // Should have tiered up

        execution.reset_profile();
        assert_eq!(execution.invocation_count(), 0);
    }

    #[test]
    fn manual_tier_switch() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        for _ in 0..25 {
            execution.record_invocation();
        }

        assert_eq!(execution.current_tier(), Tier::CpuBaseline); // Still at CPU

        execution.set_tier(Tier::GpuFullBackend);
        assert_eq!(execution.current_tier(), Tier::GpuFullBackend);
        assert_eq!(execution.invocation_count(), 0); // Counter reset
    }

    #[test]
    fn tiered_execution_has_gpu() {
        let execution = TieredExecution::new(Tier::CpuBaseline);
        assert!(execution.has_gpu_available()); // MVP assumes GPU always available
    }
}
