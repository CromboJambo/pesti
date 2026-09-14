//! Benchmark: pesti's pure Rust inference stack vs llama.cpp FFI wrapper.
//! Uses the same model and prompt for fair comparison. Both paths use GPU
//! (RTX 4070 Ti SUPER) to isolate the CPU overhead of the FFI boundary.
//!
//! NOTE: Runs sequentially with explicit cleanup between runs to avoid OOM.

use std::path::Path;
use std::time::Instant;
use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::SeedableRng;

const MODEL_PATH: &str = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const PROMPT: &str = "Write a short story about a robot learning to cook.";
const NUM_TOKENS: usize = 64;

fn benchmark_pure_rust() -> f64 {
    println!("
=== Pure Rust Path (pesti-runner native) ===");
    let start = Instant::now();

    // Load model using pesti's own GGUF loader + transformer implementation
    let mut model = LlamaModel::load_gguf(Path::new(MODEL_PATH)).unwrap();

    // Tokenize using pesti's tokenizer (loaded from GGUF)
    let tokenizer = model.tokenizer.as_ref().expect("Tokenizer not loaded");
    let input_ids = tokenizer.encode(PROMPT).unwrap();
    println!("Prompt tokens: {}", input_ids.len());

    // Generate tokens (pure Rust forward pass + pesti CUDA kernels)
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        seed: Some(42),
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let generated = model.generate(&input_ids, NUM_TOKENS, &sampling, &mut rng, &[]).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    println!("Generated {} tokens in {:.3}s", generated.len(), elapsed);
    let tok_per_sec = generated.len() as f64 / elapsed;
    println!("Throughput: {:.2} tok/s (pure Rust)", tok_per_sec);
    
    // Drop model to free GPU memory before FFI run
    drop(model);
    tok_per_sec
}

fn benchmark_ffi() -> f64 {
    println!("
=== FFI Path (llama.cpp via LlamaRunner) ===");
    let start = Instant::now();

    // Use pesti-runner's llama.cpp wrapper (FFI path)
    let runner = pesti_runner::llama::LlamaRunnerBuilder::new(MODEL_PATH)
        .n_gpu_layers(-1) // Full GPU offload
        .build()
        .unwrap();

    // Generate tokens (FFI path) - use default sampling config for comparison
    let result = runner.generate(PROMPT, &pesti_runner::llama::SamplingConfig::default()).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    println!("Generated {} tokens in {:.3}s", result.generated_tokens, elapsed);
    let tok_per_sec = result.generated_tokens as f64 / elapsed;
    println!("Throughput: {:.2} tok/s (FFI)", tok_per_sec);
    
    // Drop runner to free GPU memory
    drop(runner);
    tok_per_sec
}

fn main() {
    println!("=== pesti-runner: Pure Rust vs FFI Benchmark ===");
    println!("Model: Qwen2.5-0.5B-Instruct-Q4_K_M");
    println!("Hardware: RTX 4070 Ti SUPER (CUDA)");
    println!("Tokens to generate: {}", NUM_TOKENS);

    // Run pure Rust first, then FFI (sequential to avoid OOM)
    let pure_rust_tps = benchmark_pure_rust();
    std::thread::sleep(std::time::Duration::from_secs(1)); // Let GPU free up
    let ffi_tps = benchmark_ffi();

    println!("
=== RESULTS ===");
    println!("Pure Rust (pesti-runner): {:.2} tok/s", pure_rust_tps);
    println!("FFI (llama.cpp wrapper):  {:.2} tok/s", ffi_tps);
    println!("Ratio: {:.3}x", ffi_tps / pure_rust_tps);

    if ffi_tps < pure_rust_tps * 1.5 {
        println!("✓ Pure Rust is within 1.5x of FFI");
    } else {
        println!("✗ Pure Rust is >1.5x slower than FFI");
    }
}
