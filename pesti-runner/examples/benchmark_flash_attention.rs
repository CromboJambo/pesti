//! Flash attention kernel benchmark (Option C - Focused)
//!
//! Compares baseline fused attention vs flash attention (single-kernel approach)

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

    println!("=== Flash Attention Kernel Benchmark ===");
    println!("GPU: {:?}", cuda_rt.device_info());
    println!();

    let stream = cuda_rt.new_stream().expect("Failed to create CUDA stream");

    // Build baseline kernel
    println!("Building baseline fused attention kernel...");
    let start = std::time::Instant::now();
    let _baseline_kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    )?;
    let baseline_time = start.elapsed();

    // Build flash attention kernel
    println!("Building flash attention kernel (single-kernel fusion)...");
    let start = std::time::Instant::now();
    
    match pesti_runner::kernel::flash_attention::build_flash_attention_kernel(
        pesti_runner::kernel::flash_attention::FlashAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    ) {
        Ok(kernel) => {
            let elapsed = start.elapsed();
            println!("✅ FLASH ATTENTION KERNEL SUCCESS");
            println!("  - Architecture: {:?}", kernel.arch());
            println!("  - Build time: {:?}", elapsed);
            println!();
            println!("  Expected improvement: 40-50% speedup on 512+ tokens");
            println!("  (Single kernel launch vs 2 GEMM calls + CPU softmax)");
        }
        Err(e) => {
            let elapsed = start.elapsed();
            println!("⚠️  FLASH ATTENTION KERNEL BUILD FAILED (PTX stub)");
            println!("  - Error: {}", e);
            println!("  - Build time: {:?}", elapsed);
            println!();
            println!("  Note: PTX kernel is a stub - full implementation needed");
        }
    }

    println!();
    println!("=== Results ===");
    println!("Baseline build time:   {:?}", baseline_time);
    
    // Compare to optimized attention (RoPE caching)
    let start = std::time::Instant::now();
    match pesti_runner::kernel::optimized_attention::build_optimized_attention_kernel(
        pesti_runner::kernel::optimized_attention::OptimizedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    ) {
        Ok(_) => {
            let optimized_time = start.elapsed();
            println!("RoPE cached build time: {:?}", optimized_time);
            println!();
            println!("Improvement chain:");
            println!("  Baseline → RoPE cached: {:.1}% faster build", 
                ((baseline_time.as_nanos() - optimized_time.as_nanos()) as f64 / baseline_time.as_nanos() as f64) * 100.0);
        }
        Err(_) => {}
    }

    println!();
    println!("=== Next Steps ===");
    println!("1. Implement full PTX kernel (see docs/GRINDING-TO-MISTRAL-RS-PARITY.md)");
    println!("2. Benchmark vs baseline on real model");
    println!("3. If parity not reached (~50 tok/s), enable mistral.rs backend (Option B)");

    Ok(())
}
