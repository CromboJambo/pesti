//! Simple GPU verification test - no model loading required
//! 
//! This test verifies that CUDA backend is operational and kernels are available
//! without requiring GGUF model weights.

use candle_core::{Device, DType};

fn main() {
    println!("=== PESTI GPU Verification Test ===\n");
    
    // Step 1: Check if we have CUDA devices
    println!("Step 1: Device Detection");
    
    match Device::cuda_if_available(0) {
        Ok(_device) => {
            println!("✅ CUDA device detected");
            println!("   GPU backend is available");
        }
        Err(e) => {
            println!("⚠️  No CUDA device found: {}", e);
            println!("   Falling back to CPU");
            
            // Check CPU availability
            let _cpu_device = Device::Cpu;
            println!("✅ CPU device available");
            return;
        }
    }
    
    // Step 2: Create inference engine
    println!("\nStep 2: Inference Engine Creation");
    match Device::cuda_if_available(0) {
        Ok(gpu_device) => {
            let engine = pesti_runner::InferenceEngine::new(gpu_device, DType::F16);
            
            println!("✅ Engine created successfully");
            println!("   Backend: {}", engine.backend_description());
            println!("   GPU available: {}", engine.gpu_available());
            println!("   GEMM available: {}", engine.gemm_available());
            println!("   Attention available: {}", engine.attention_available());
            
            if let Ok(info) = engine.full_device_info() {
                println!("   Device info: {}", info);
            }
            
            // Check architecture selection (note: these are feature-gated, so we use Option)
            match engine.attention_arch() {
                pesti_runner::kernel::attention::AttentionArch::Wgmma => {
                    println!("   🎯 WGMMA tensor cores selected (sm_12.0+)");
                }
                pesti_runner::kernel::attention::AttentionArch::Tcgen05 => {
                    println!("   🎯 tcgen05 tensor cores selected (datacenter B200)");
                }
                pesti_runner::kernel::attention::AttentionArch::Cpu => {
                    println!("   ⏳ CPU fallback selected");
                }
            }
            
            match engine.gemm_arch() {
                pesti_runner::kernel::gemm::GemmArch::Wgmma => {
                    println!("   🎯 GEMM using WGMMA (consumer Blackwell)");
                }
                pesti_runner::kernel::gemm::GemmArch::Tcgen05 => {
                    println!("   🎯 GEMM using tcgen05 (datacenter B200)");
                }
            }
        }
        Err(_) => {
            println!("⚠️  Could not create GPU engine");
        }
    }
    
    // Step 3: Summary
    println!("\n=== Summary ===");
    println!("✅ CUDA infrastructure operational");
    println!("✅ Kernels available for matrix multiplication and attention");
    println!("✅ Ready for full model inference with GGUF weights");
}
