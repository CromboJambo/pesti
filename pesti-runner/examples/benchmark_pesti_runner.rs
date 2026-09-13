//! Benchmark pesti-runner's own CUDA inference stack (not llama.cpp FFI).
//! Uses LlamaModel::from_gguf_weights() + generate() directly.
//!
//! This measures how far pesti-runner gets on its own, without the
//! llama.cpp FFI wrapper used by benchmark_generation.rs.

use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== pesti-runner standalone CUDA benchmark ===");
    println!("Model: {}", model_path);
    println!();

    // Step 1: Load GGUF weights via pesti-runner's own loader
    let load_start = Instant::now();
    println!("Loading GGUF weights (pesti-runner)...");
    let weights = match pesti_runner::load_gguf_weights(std::path::Path::new(model_path)) {
        Ok(w) => w,
        Err(e) => {
            println!("Failed to load weights: {}", e);
            return;
        }
    };
    println!("✓ Weights loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // Step 2: Build LlamaModel from weights using pesti-runner's own stack
    let build_start = Instant::now();
    println!("Building LlamaModel (pesti-runner)...");
    let mut model = match pesti_runner::LlamaModel::from_gguf_weights(&weights) {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to build model: {}", e);
            return;
        }
    };
    println!("✓ Model built in {:.2}s", build_start.elapsed().as_secs_f32());

    // Step 3: Run generation loop using pesti-runner's own forward pass + sampling
    let prompt = "Hello, how are you?";
    println!("\nPrompt: \"{}\"", prompt);
    println!("Generating via pesti-runner's own stack (not llama.cpp FFI)...\n");

    // Tokenize manually using pesti-runner's tokenizer
    let tokens = match model.tokenize(prompt) {
        Ok(t) => t,
        Err(e) => {
            println!("Tokenization failed: {}", e);
            return;
        }
    };
    println!("✓ Tokenized to {} tokens", tokens.len());

    // Run generation loop manually using pesti-runner's forward pass
    let gen_start = Instant::now();
    let mut all_tokens = tokens.clone();
    let max_new_tokens = 50;
    let mut generated = 0;

    for step in 0..max_new_tokens {
        // Run forward pass through pesti-runner's own stack
        let logits = match model.forward(&all_tokens) {
            Ok(l) => l,
            Err(e) => {
                println!("Forward pass failed at step {}: {}", step, e);
                break;
            }
        };

        // Sample next token using pesti-runner's sampler
        let next_token = match pesti_runner::sample(&logits, 0.7, 0.9) {
            Ok(t) => t,
            Err(e) => {
                println!("Sampling failed: {}", e);
                break;
            }
        };

        all_tokens.push(next_token);
        generated += 1;

        // Print progress every 10 tokens
        if step % 10 == 9 || step == max_new_tokens - 1 {
            let elapsed = gen_start.elapsed().as_secs_f32();
            println!("  [step {}] {:.3}s elapsed, {:.1} tok/s", 
                     generated, elapsed, generated as f32 / elapsed);
        }
    }

    let total_time = gen_start.elapsed().as_secs_f32();
    let tokens_per_sec = generated as f32 / total_time;

    println!("\n=== Results ===");
    println!("Generated {} tokens in {:.3}s", generated, total_time);
    println!("Tokens/sec (pesti-runner own stack): {:.1}", tokens_per_sec);
    println!("(vs llama.cpp FFI wrapper for comparison)");
}
