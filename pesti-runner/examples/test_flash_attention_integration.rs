//! Test Flash Attention kernel integration with InferenceEngine.
//!
//! Verifies that Flash Attention PTX kernel loads successfully when the `flash-attention` feature is enabled.

use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::{AttentionKernel, FlashAttentionConfig, FlashAttentionKernel};
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_kernel_creation() {
    // Initialize CUDA runtime
    let cuda_runtime = CudaRuntime::new(0).expect("CUDA runtime should initialize");
    let stream = cuda_runtime.new_stream().expect("Stream creation should succeed");

    // Create Flash Attention config
    let config = FlashAttentionConfig::default();

    // Create memory backend
    let device_info = cuda_runtime.device_info().clone();
    let memory_backend = Arc::new(
        pesti_runner::kernel::memory::CudaMemoryBackend::with_device_info(stream.clone(), device_info),
    );

    // Try to create Flash Attention kernel
    match FlashAttentionKernel::new(
        cuda_runtime.context().clone(),
        stream,
        (*memory_backend).clone(),
        config,
    ) {
        Ok(kernel) => {
            assert!(kernel.is_ready());
            println!("✓ Flash Attention kernel created successfully");
            println!("  Config: {} heads, {} dim, max_seq={}", 
                     kernel.config().num_heads, 
                     kernel.config().head_dim, 
                     kernel.config().max_seq);
        }
        Err(e) => {
            eprintln!("✗ Flash Attention kernel creation failed: {}", e);
            panic!("Flash Attention kernel should load");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_flash_attention_kernel_arch() {
    // Initialize CUDA runtime
    let cuda_runtime = CudaRuntime::new(0).expect("CUDA runtime should initialize");
    let stream = cuda_runtime.new_stream().expect("Stream creation should succeed");

    // Create Flash Attention config
    let config = FlashAttentionConfig::default();

    // Create memory backend
    let device_info = cuda_runtime.device_info().clone();
    let memory_backend = Arc::new(
        pesti_runner::kernel::memory::CudaMemoryBackend::with_device_info(stream.clone(), device_info),
    );

    // Create kernel
    let kernel = FlashAttentionKernel::new(
        cuda_runtime.context().clone(),
        stream,
        (*memory_backend).clone(),
        config,
    ).expect("Flash Attention kernel should create");

    // Check architecture
    assert_eq!(kernel.arch(), pesti_runner::kernel::AttentionArch::Wgmma);
    println!("✓ Flash Attention uses WGMMA (tensor cores) architecture");
}
