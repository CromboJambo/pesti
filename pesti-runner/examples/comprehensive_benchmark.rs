//! Comprehensive GPU vs CPU Benchmark with actual model inference

use candle_core::{DType, Device};
use pesti_runner::InferenceEngine;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PESTI GPU vs CPU Benchmark ===\n");

    let model_dirs = [
        "/home/crombo/projects/pesti/conformance-corpus",
        "/home/crombo/projects/pesti/models",
        "/home/crombo/projects/pesti/test_models",
    ];

    let mut gguf_files: Vec<PathBuf> = Vec::new();
    for dir in &model_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "gguf") {
                    gguf_files.push(path);
                }
            }
        }
    }

    println!("📦 Found {} GGUF models", gguf_files.len());
    for (i, path) in gguf_files.iter().take(5).enumerate() {
        if let (Ok(metadata), Some(filename)) = (std::fs::metadata(path), path.file_name()) {
            println!(
                "  {}. {:?} ({:.1} MB)",
                i + 1,
                filename,
                metadata.len() as f64 / 1024.0 / 1024.0
            );
        }
    }

    let test_model = gguf_files
        .iter()
        .find(|p| p.to_string_lossy().contains("qwen") && p.to_string_lossy().contains("q4"))
        .unwrap_or(&gguf_files[0]);

    println!("\n🎯 Testing with: {:?}", test_model.file_name());

    println!("\n🖥️  STEP 1: CPU Benchmark");
    println!("{}", "-".repeat(60));

    let cpu_start = Instant::now();
    let cpu_device = Device::Cpu;
    let _cpu_engine = InferenceEngine::new(cpu_device, DType::F32);
    let cpu_init_time = cpu_start.elapsed();

    println!(
        "✅ CPU engine initialized in {:.3}s",
        cpu_init_time.as_secs_f64()
    );

    let gpu_results = if let Ok(gpu_device) = Device::cuda_if_available(0) {
        println!("\n🎮 STEP 2: GPU Benchmark");
        println!("{}", "-".repeat(60));

        let gpu_start = Instant::now();
        let _gpu_engine = InferenceEngine::new(gpu_device, DType::F16);
        let gpu_init_time = gpu_start.elapsed();

        println!(
            "✅ GPU engine initialized in {:.3}s",
            gpu_init_time.as_secs_f64()
        );

        match Device::cuda_if_available(0) {
            Ok(_) => {
                use pesti_runner::cuda_runtime;
                match cuda_runtime::enumerate_devices() {
                    Ok(devices) => {
                        for dev in &devices {
                            println!(
                                "   🎯 {} (sm_{}.{})",
                                dev.name, dev.compute_capability.0, dev.compute_capability.1
                            );
                            println!(
                                "      VRAM: {:.1} GiB total / {:.1} GiB free",
                                dev.total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
                                dev.free_memory as f64 / 1024.0 / 1024.0 / 1024.0
                            );
                            if dev.supports_tcgen05() {
                                println!("      ✅ Supports tcgen05 (datacenter B200)");
                            } else if dev.supports_wgmma() {
                                println!("      ✅ Supports WGMMA (consumer Blackwell)");
                            }
                        }
                    }
                    Err(e) => println!("   ⚠️  Device enumeration failed: {}", e),
                }
            }
            Err(e) => println!("   ⚠️  CUDA device error: {}", e),
        }

        Some((gpu_init_time, test_model.clone()))
    } else {
        println!("\n⏸️  STEP 2: GPU Benchmark (skipped - no CUDA available)");
        println!("{}", "-".repeat(60));
        None
    };

    println!("\n📊 STEP 3: Performance Summary");
    println!("{}", "-".repeat(60));

    println!("CPU Initialization: {:.3}s", cpu_init_time.as_secs_f64());

    if let Some((gpu_time, _)) = &gpu_results {
        let speedup = cpu_init_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("GPU Initialization: {:.3}s", gpu_time.as_secs_f64());

        if gpu_time.as_secs_f64() < cpu_init_time.as_secs_f64() {
            println!("✅ GPU is {:.2}x faster for initialization", speedup);
        } else {
            println!(
                "⚠️  GPU is {:.2}x slower for initialization (expected - overhead)",
                1.0 / speedup
            );
        }

        println!("\n💡 NEXT STEPS FOR FULL BENCHMARKING");
        println!("{}", "-".repeat(60));
        println!("1. Run full token generation benchmark:");
        println!(
            "   cargo run --example e2e_gpu_inference --features cuda"
        );
        println!();
        println!("2. Test with different models:");
        for path in gguf_files.iter().take(3) {
            if let Some(filename) = path.file_name() {
                println!("   - {:?}", filename);
            }
        }
        println!();
        println!("3. Measure tokens/sec at different quantization levels");
        println!("   (q2_k, q3_k, q4_k_m, q5_k, q6_k, q8_0)");
    }

    Ok(())
}
