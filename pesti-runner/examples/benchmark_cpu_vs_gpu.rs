//! GPU vs CPU Performance Benchmark

use candle_core::{DType, Device};
use pesti_runner::InferenceEngine;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PESTI GPU vs CPU Benchmark ===\n");

    let model_path = Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );

    if !model_path.exists() {
        eprintln!("❌ Model not found: {:?}", model_path);
        return Err("Model file not found".into());
    }

    let metadata = std::fs::metadata(model_path)?;
    println!(
        "📦 Model: {:?} ({:.1} MB)",
        model_path.file_name().unwrap(),
        metadata.len() as f64 / 1024.0 / 1024.0
    );

    println!("\n🖥️  STEP 1: CPU Inference Benchmark");
    println!("{}", "-".repeat(50));

    let cpu_start = Instant::now();
    let cpu_device = Device::Cpu;
    let cpu_engine = InferenceEngine::new(cpu_device, DType::F32);

    let cpu_init_time = cpu_start.elapsed();
    println!(
        "✅ CPU Engine initialized in {:.3}s",
        cpu_init_time.as_secs_f64()
    );
    println!("   Backend: {}", cpu_engine.backend_description());
    println!(
        "   GEMM: {}, Attention: {}",
        cpu_engine.gemm_available(),
        cpu_engine.attention_available()
    );

    let gpu_metrics = if let Ok(gpu_device) = Device::cuda_if_available(0) {
        println!("\n🎮 STEP 2: GPU Inference Benchmark");
        println!("{}", "-".repeat(50));

        let gpu_start = Instant::now();
        let gpu_engine = InferenceEngine::new(gpu_device, DType::F16);

        let gpu_init_time = gpu_start.elapsed();
        println!(
            "✅ GPU Engine initialized in {:.3}s",
            gpu_init_time.as_secs_f64()
        );
        println!("   Backend: {}", gpu_engine.backend_description());
        println!(
            "   GPU available: {}, GEMM: {}, Attention: {}",
            gpu_engine.gpu_available(),
            gpu_engine.gemm_available(),
            gpu_engine.attention_available()
        );

        if let Ok(info) = gpu_engine.full_device_info() {
            println!("   Device: {}", info);
        }

        Some((gpu_init_time, gpu_engine))
    } else {
        println!("\n⏸️  STEP 2: GPU Benchmark (skipped - no CUDA available)");
        println!("{}", "-".repeat(50));
        None
    };

    println!("\n📊 STEP 3: Performance Summary");
    println!("{}", "-".repeat(50));

    println!("CPU Initialization: {:.3}s", cpu_init_time.as_secs_f64());

    if let Some((gpu_time, _)) = &gpu_metrics {
        let speedup = cpu_init_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("GPU Initialization: {:.3}s", gpu_time.as_secs_f64());
        println!("Speedup: {:.2}x (GPU faster)", speedup);

        if speedup > 1.0 {
            println!(
                "✅ GPU initialization is {}x faster than CPU",
                (speedup - 1.0) * 100.0
            );
        } else {
            println!(
                "⚠️  GPU is {:.2}x slower than CPU (expected for small models)",
                1.0 / speedup
            );
        }
    } else {
        println!("GPU: Not available");
    }

    println!("\n💡 NEXT STEPS FOR FULL BENCHMARKING");
    println!("{}", "-".repeat(50));
    println!("1. Run full inference with model loading:");
    println!(
        "   cargo run --example e2e_gpu_inference --features cuda"
    );
    println!();

    println!("2. Test different quantization levels:");
    for q in &["q2_k", "q3_k", "q4_k_m", "q5_k", "q6_k", "q8_0"] {
        let path = format!(
            "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-{}.gguf",
            q
        );
        println!(
            "   - {:?}",
            Path::new(&path).file_name().unwrap_or_default()
        );
    }

    Ok(())
}
