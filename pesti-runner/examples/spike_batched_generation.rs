//! Spike: batched generation - measure kernel launch overhead vs compute time.
//!
//! Runs the same prompt N times sequentially, measuring per-iteration time.
//! If kernel launches are expensive relative to compute, batching multiple
//! sequences into a single forward pass would be beneficial.

use std::path::Path;
use std::time::Instant;
use pesti_runner::{load_gguf_weights, LlamaModel};
use pesti_runner::transformer::tokenizer::{TokenizerBackend, load_tokenizer_from_gguf};
use rand::SeedableRng;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The capital of France is";
    let max_tokens = 8;
    let iterations = 10;

    println!("=== Batched Generation Spike ===");
    println!("Model: {}", model_path);
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens per generation: {}", max_tokens);
    println!("Iterations: {}\n", iterations);

    // Load model once
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Load tokenizer
    let (_config, tokenizer) = load_tokenizer_from_gguf(Path::new(model_path), TokenizerBackend::MistralRs)
        .expect("Failed to load tokenizer");

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt).expect("Failed to encode");
    println!("Prompt tokens: {}", prompt_tokens.len());

    // Sampling config (greedy for deterministic output)
    let sampling = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };

    // Measure sequential generations
    println!("\nRunning {} sequential generations...", iterations);
    let t_total = Instant::now();
    
    for i in 0..iterations {
        let t_iter = Instant::now();
        
        let result = model.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &mut rand::rngs::StdRng::seed_from_u64(42 + i as u64),
            &[151663], // EOS token for Qwen2.5
        );

        let iter_time = t_iter.elapsed().as_secs_f64();
        
        match result {
            Ok(tokens) => {
                if i == 0 || i == iterations - 1 {
                    let text = tokenizer.decode(&tokens).expect("Failed to decode");
                    println!("  [{}] {:.2}s - {} tokens: \"{}...\"", 
                        i, iter_time, tokens.len(), &text[..text.len().min(40)]);
                } else if i == iterations / 2 {
                    println!("  [{}] {:.2}s - {} tokens", i, iter_time, tokens.len());
                }
            }
            Err(e) => {
                println!("  [{}] Error: {}", i, e);
            }
        }
    }

    let total_time = t_total.elapsed().as_secs_f64();
    let avg_time = total_time / iterations as f64;
    
    println!("\n=== Results ===");
    println!("Total time for {} generations: {:.2}s", iterations, total_time);
    println!("Average per generation: {:.3}s ({:.0}ms)", avg_time, avg_time * 1000.0);
    println!("Effective throughput: {:.2} tok/s (across all generations)", 
        (iterations * max_tokens) as f64 / total_time);
    
    // Estimate batched performance
    println!("\nIf batched into single forward pass (theoretical):");
    println!("  - Would process {} sequences in ~{:.2}s", iterations, avg_time);
    println!("  - Speedup: {:.1}x vs sequential", total_time / avg_time);
    
    println!("\n=== Spike Complete ===");
}