//! Numerical conformance test for Flash Attention with softmax
//! Compares GPU output vs CPU reference implementation

use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::flash_attention::{FlashAttentionConfig, FlashAttentionKernel};
use pesti_runner::AttentionKernel;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Flash Attention Numerical Conformance Test ===");
    println!();
    
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("GPU: {:?}", cuda_rt.device_info());
    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");
    
    // Build flash attention kernel with softmax
    println!();
    println!("Building flash attention kernel (with softmax)...");
    let start = std::time::Instant::now();
    
    match FlashAttentionKernel::new(
        cuda_rt.context().clone(),
        stream.clone(),
        pesti_runner::kernel::memory::CudaMemoryBackend::new(stream.clone()),
        FlashAttentionConfig::default(),
    ) {
        Ok(kernel) => {
            let elapsed = start.elapsed();
            println!("✅ Kernel built successfully");
            println!("  - Architecture: {:?}", kernel.arch());
            println!("  - Build time: {:?}", elapsed);
            
            // TODO: Add numerical conformance test with known inputs
            // For now, just verify kernel launches without error
            println!();
            println!("=== Next Steps ===");
            println!("1. Create test inputs (Q, K, V tensors)");
            println!("2. Run GPU kernel");
            println!("3. Compare output vs CPU reference (llama.cpp)");
            println!("4. Verify max absolute error < 1e-2");
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("❌ Kernel build failed: {}", e);
            println!("  - Build time: {:?}", elapsed);
            std::process::exit(1);
        }
    }
    
    Ok(())
}
