//! Real GPU vs CPU inference benchmark via the llama.cpp FFI backend.
//! Includes kernel timing profiler for GEMM/attention breakdown.
//!
//! Loads the same Qwen2.5-0.5B Q4_K_M model twice:
//!   - n_gpu_layers=0  → pure CPU inference
//!   - n_gpu_layers=-1 → full GPU offload (llama.cpp must be built with CUDA)
//!
//! Measures model load time and token generation throughput for both.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda,llama-cpp-2/cuda \
//!     --example llama_gpu_vs_cpu

use pesti_runner::LlamaRunner;
use pesti_runner::llama::SamplingConfig;
use pesti_runner::profiler;
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf";
const PROMPT: &str = "The quick brown fox jumps over the lazy dog. Summarize:";

fn bench(label: &str, n_gpu_layers: i32) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {label} (n_gpu_layers={n_gpu_layers}) ===");
    println!("{}", "-".repeat(50));

    let load_start = Instant::now();
    let runner = LlamaRunner::builder(Path::new(MODEL))
        .n_gpu_layers(n_gpu_layers)
        .n_ctx(2048)
        .build()?;
    let load_time = load_start.elapsed();
    println!("Model load: {:.3}s", load_time.as_secs_f64());

    let info = runner.model_info();
    println!(
        "Model: {} params, {} layers, {} vocab",
        info.n_params, info.n_layer, info.n_vocab
    );

    let mut config = SamplingConfig::default();
    config.temperature = 0.8;
    config.top_k = 40;
    config.max_tokens = 64;

    // Reset profiler before generation
    profiler::reset();

    let gen_start = Instant::now();
    let result = runner.generate(PROMPT, &config)?;
    let gen_time = gen_start.elapsed();
    let n_tokens = result.generated_tokens;
    let tok_s = n_tokens as f64 / gen_time.as_secs_f64();

    println!(
        "Generated {n_tokens} tokens in {:.3}s",
        gen_time.as_secs_f64()
    );
    println!("Throughput: {:.1} tok/s", tok_s);
    println!(
        "Output: \"{}...\"",
        &result.text[..result.text.len().min(80)]
    );

    // Print kernel timing breakdown
    profiler::print_report();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PESTI llama.cpp GPU vs CPU Benchmark (Week 26 Profiling) ===\n");
    println!("Model: {MODEL}");
    println!("Prompt: \"{PROMPT}\"");

    // Initialize profiler
    let _profiler = profiler::init();

    bench("CPU", 0)?;
    bench("GPU (full offload)", -1)?;

    println!("\n=== Done ===");
    Ok(())
}
