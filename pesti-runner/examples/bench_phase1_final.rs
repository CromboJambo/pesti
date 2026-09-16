use std::time::Instant;
use pesti_runner::{LlamaRunnerBuilder};
use pesti_runner::llama::SamplingConfig;

fn main() {
    let model_path = "./conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("=== PESTI tok/s Benchmark (Phase 1 CPU GEMM) ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m");
    println!();

    // Warmup: load model and run one generation to warm caches
    println!("Loading model for warmup...");
    let runner_warmup = LlamaRunnerBuilder::new(model_path)
        .n_ctx(1024)
        .build()
        .expect("Failed to load model for warmup");
    
    let warmup_config = SamplingConfig {
        max_tokens: 5,
        ..Default::default()
    };
    println!("Running warmup generation...");
    let _warmup = runner_warmup.generate("Write a short story about ", &warmup_config);
    drop(runner_warmup); // Free warmup model

    // Fresh model for actual benchmark (avoid KV cache position issues)
    println!("Loading fresh model for benchmark...");
    let t_load = Instant::now();
    let runner = LlamaRunnerBuilder::new(model_path)
        .n_ctx(1024)
        .build()
        .expect("Failed to load model for benchmark");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Timed run - generate tokens
    let config = SamplingConfig {
        max_tokens: 10,
        ..Default::default()
    };
    
    println!("Running benchmark generation...");
    let t_gen = Instant::now();
    let result = runner.generate("Write a short story about ", &config)
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();

    println!();
    println!("=== Results ===");
    println!("Generated {} tokens in {:.2}s", result.generated_tokens, gen_time);
    
    if result.generated_tokens > 0 {
        let tok_per_sec = result.generated_tokens as f64 / gen_time;
        println!("Decode speed: {:.2} tok/s", tok_per_sec);
        println!("Avg time per token: {:.0}ms", 1000.0 / tok_per_sec);
    }

    println!();
    let display_len = result.text.len().min(200);
    println!("Output: {}", &result.text[..display_len]);
    println!();
    println!("=== Benchmark Complete ===");
}
