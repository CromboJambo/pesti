//! Benchmark: Speculative decoding vs standard autoregressive decoding.
//!
//! Compares throughput (tokens/sec) between the two approaches on identical prompts.

use pesti_runner::speculative::{generate_speculative, SpeculativeParams};
use pesti_runner::{GenerationResult, ModelLoader, RuntimeConfig, SamplingConfig};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: speculative_bench <model_path>");

    println!("=== PESTI Speculative Decoding Benchmark ===");
    println!("Model: {}", model_path);
    println!();

    // Load model once for both tests
    println!("Loading model...");
    let start = Instant::now();
    let loader = ModelLoader::new(model_path, RuntimeConfig::default());
    let mut runtime = loader.load().expect("Failed to load model");
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!();

    let prompt = "Write a short story about a robot who learns to paint.";
    let max_tokens = 50;

    // Test 1: Standard autoregressive decoding
    println!("Test 1: Standard autoregressive decoding");
    println!("----------------------------------------");
    let config = SamplingConfig {
        temperature: 0.7,
        top_p: 1.0,
        seed: Some(42),
    };
    let start = Instant::now();
    let result1 = runtime.generate(prompt, &config).expect("Generation failed");
    let elapsed1 = start.elapsed();
    println!("Tokens generated: {}", result1.num_tokens);
    println!("Time: {:.3}s", elapsed1.as_secs_f64());
    println!(
        "Throughput: {:.2} tokens/sec",
        result1.num_tokens as f64 / elapsed1.as_secs_f64()
    );
    println!("Output: {}", &result1.text[..result1.text.len().min(200)]);
    println!();

    // Test 2: Speculative decoding (k=2 draft)
    println!("Test 2: Speculative decoding (k=2 draft)");
    println!("----------------------------------------");
    let spec_params = SpeculativeParams {
        draft_size: 2,
        temperature: 0.7,
        top_p: 1.0,
    };
    let start = Instant::now();
    let result2 = generate_speculative(&mut runtime, prompt, &spec_params).expect("Spec generation failed");
    let elapsed2 = start.elapsed();
    println!("Tokens generated: {}", result2.num_tokens);
    println!("Time: {:.3}s", elapsed2.as_secs_f64());
    println!(
        "Throughput: {:.2} tokens/sec",
        result2.num_tokens as f64 / elapsed2.as_secs_f64()
    );
    println!("Output: {}", &result2.text[..result2.text.len().min(200)]);
    println!();

    // Comparison
    println!("=== Results ===");
    let speedup = elapsed1.as_secs_f64() / elapsed2.as_secs_f64();
    println!(
        "Standard: {:.3}s ({:.2} tok/s)",
        elapsed1.as_secs_f64(),
        result1.num_tokens as f64 / elapsed1.as_secs_f64()
    );
    println!(
        "Speculative: {:.3}s ({:.2} tok/s)",
        elapsed2.as_secs_f64(),
        result2.num_tokens as f64 / elapsed2.as_secs_f64()
    );
    if speedup > 1.0 {
        println!("Speedup: {:.2}x faster", speedup);
    } else {
        println!("Slower by {:.2}x", 1.0 / speedup);
    }
}
