//! Flash attention end-to-end inference benchmark
//! Measures tokens/sec on Qwen2.5-0.5B using custom kernels vs llama.cpp baseline

use pesti_runner::cuda_runtime::CudaRuntime;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Flash Attention End-to-End Benchmark ===");
    
    // Initialize CUDA
    let cuda_rt = CudaRuntime::new(0)?;
    let stream = cuda_rt.new_stream()?;
    
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();
    
    // Build flash attention kernel
    println!("Building flash attention kernel...");
    let kernel = pesti_runner::kernel::flash_attention::build_flash_attention_kernel(
        pesti_runner::kernel::flash_attention::FlashAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    )?;
    
    println!("✅ Flash attention kernel built successfully");
    println!("  Architecture: {:?}", kernel.arch());
    println!();
    
    // TODO: Load model and run inference
    // For now, just verify the kernel loads and can be launched
    
    println!("=== Next Steps ===");
    println!("1. Integrate with model loader (gguf_weight_loader)");
    println!("2. Run full forward pass on Qwen2.5-0.5B");
    println!("3. Measure tokens/sec vs llama.cpp baseline (84.9 tok/s)");
    
    Ok(())
}
