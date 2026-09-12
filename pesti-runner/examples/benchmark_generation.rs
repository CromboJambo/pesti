//! Benchmark pesti-runner inference using built-in llama.cpp timing API.
//! Measures prompt eval, token generation, and overall throughput.

use pesti_runner::llama::{LlamaRunner, SamplingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        return Err("Model file not found".into());
    }

    println!("=== pesti-runner Benchmark ===");
    println!("Model: Qwen2.5-0.5B-Instruct-Q4_K_M");
    println!("Hardware: RTX 3070 Ti (ftw3)");
    println!();

    // Build runner with GPU offload
    let runner = LlamaRunner::builder(model_path)
        .n_ctx(2048)
        .n_batch(512)
        .n_gpu_layers(-1) // All layers on GPU
        .build()?;

    println!("Model loaded: {} params", runner.model_info().n_params);
    println!();

    // Test prompt
    let prompt = "Explain the concept of recursion in programming with a simple example.";

    // Run generation and measure
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        max_tokens: 128,
        ..Default::default()
    };

    println!("Generating response...");
    let result = runner.generate(prompt, &config)?;

    // Extract timing info from result
    println!();
    println!("=== Benchmark Results ===");
    println!("Prompt tokens: {}", result.prompt_tokens);
    println!("Generated tokens: {}", result.generated_tokens);
    println!("Load time: {:.2} ms", result.load_time_ms);
    println!("Prompt eval time: {:.2} ms ({:.1} tok/s)", 
        result.prompt_eval_ms,
        if result.prompt_eval_ms > 0.0 {
            result.prompt_tokens as f64 / (result.prompt_eval_ms / 1000.0)
        } else { 0.0 }
    );
    println!("Token eval time: {:.2} ms", result.eval_ms);
    println!("Generation speed: {:.2} tok/s",
        if result.eval_ms > 0.0 {
            result.generated_tokens as f64 / (result.eval_ms / 1000.0)
        } else { 0.0 }
    );

    // Calculate total throughput
    let total_time = result.prompt_eval_ms + result.eval_ms;
    let total_tokens = result.prompt_tokens + result.generated_tokens;
    println!("Total time: {:.2} ms", total_time);
    println!("Overall throughput: {:.1} tok/s", 
        if total_time > 0.0 {
            total_tokens as f64 / (total_time / 1000.0)
        } else { 0.0 }
    );

    Ok(())
}
