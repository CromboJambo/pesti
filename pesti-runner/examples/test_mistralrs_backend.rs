//! Test mistral.rs backend integration (Option B - Hybrid fallback)

use pesti_runner::cuda_runtime::CudaRuntime;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => Arc::new(rt),
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Mistral.rs Backend Test (Option B - Hybrid) ===");
    println!("GPU: {:?}", cuda_rt.device_info());
    println!();

    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");

    // Try to build mistral.rs GEMM kernel
    println!("Testing mistral.rs backend...");
    
    #[cfg(feature = "mistralrs")]
    {
        use pesti_runner::kernel::mistralrs_backend::{MistralRsBackend, MistralRsGemmKernel};
        use pesti_runner::kernel::GemmArch;
        use pesti_runner::{AttentionKernel, GemmKernel};

        match MistralRsGemmKernel::try_new(GemmArch::Wgmma) {
            Some(kernel) => {
                println!("✅ MISTRAL.RS GEMM KERNEL AVAILABLE");
                println!("  - Architecture: {:?}", kernel.arch());
                println!("  - Available: {}", kernel.is_available());
                
                let device_info = cuda_rt.device_info();
                println!("  - Target GPU: {}", device_info.name);
                
                println!();
                println!("Expected performance: ~72 tok/s on Llama 3.1 8B Q4_K_M");
                println!("This is our production fallback if flash attention doesn't reach parity.");
            }
            None => {
                println!("⚠️  MISTRAL.RS GEMM KERNEL NOT AVAILABLE");
                println!("  - CUDA available but kernel creation failed");
                println!("  - This might be expected in debug mode");
            }
        }

        // Try to build mistral.rs attention kernel
        #[cfg(feature = "mistralrs")]
        match pesti_runner::kernel::mistralrs_backend::MistralRsAttentionKernel::try_new(
            pesti_runner::kernel::AttentionArch::Wgmma,
        ) {
            Some(kernel) => {
                println!("✅ MISTRAL.RS ATTENTION KERNEL AVAILABLE");
                println!("  - Architecture: {:?}", kernel.arch());
                println!("  - Available: {}", kernel.is_available());
            }
            None => {
                println!("⚠️  MISTRAL.RS ATTENTION KERNEL NOT AVAILABLE");
            }
        }

        // Show backend description
        let backend = MistralRsBackend::default();
        println!();
        println!("Backend selected: {}", backend.description());
        
    }

    #[cfg(not(feature = "mistralrs"))]
    {
        println!("⚠️  MISTRAL.RS FEATURE NOT ENABLED");
        println!("  Run with: cargo build --features cuda,mistralrs");
    }

    Ok(())
}
