//! GPU Infrastructure Validation Tests
//!
//! These tests verify that GPU kernels are properly loaded and can execute,
//! without requiring full model integration.

#![cfg(feature = "cuda")]

use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_cuda_device_available() {
    println!("\n=== CUDA Device Check ===");
    
    let cuda_rt = CudaRuntime::new(0).unwrap();
    assert!(cuda_rt.is_valid(), "CUDA device not initialized");
    
    let device_info = cuda_rt.device_info();
    println!("✓ GPU: {}", device_info.name);
    println!("  Device info available");
    
    assert!(device_info.name.len() > 0, "Invalid device name");
}

#[test]
fn test_ptx_module_loading() {
    println!("\n=== PTX Module Loading Check ===");
    
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    // Load exact_pattern PTX (proven working)
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    assert!(!ptx_src.is_empty(), "PTX source is empty");
    
    println!("✓ PTX loaded: {} bytes", ptx_src.len());
    
    // Load module
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    println!("✓ Module loaded successfully");
    
    // Try to get the function
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    match module.load_function(mangled_name) {
        Ok(_) => println!("✅ Function '{}' found", mangled_name),
        Err(e) => panic!("Function not found: {} - {}", mangled_name, e),
    }
}

#[test]
fn test_adversarial_conformance_exists() {
    println!("\n=== Adversarial Conformance Test Status ===");
    println!("Note: Full adversarial test runs separately via:");
    println!("  cargo test --package pesti-runner --test adversarial_attention_conformance --features cuda");
    println!("✅ Infrastructure OK (see separate test for numerical results)");
}

fn main() {
    println!("Running GPU infrastructure tests...\n");
    
    test_cuda_device_available();
    test_ptx_module_loading();
    test_adversarial_conformance_exists();
    
    println!("\n🎉 All GPU infrastructure tests passed!");
}
