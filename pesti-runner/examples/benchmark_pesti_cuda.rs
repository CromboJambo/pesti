//! Profile pesti-runner's own GPU inference path.
//! Measures GEMM vs attention time split at production sequence lengths
//! to identify the softmax host-transfer bottleneck.
//!
//! Uses LlamaModel::load_gguf() + forward_with_dispatch() directly,
//! NOT llama.cpp FFI.

use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    };

    println!("=== pesti-runner GPU Inference Profile ===");
    println!("Model: {}", model_path);
    println!();

    // Load model through pesti-runner's own path (not llama.cpp FFI)
    let load_start = Instant::now();
    let mut model = match pesti_runner::transformer::model::LlamaModel::load_gguf(Path::new(model_path)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };
    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}s", load_time.as_secs_f64());

    // Tokenize prompt - returns (config, tokenizer) tuple
    let (_config, tokenizer) = match pesti_runner::transformer::tokenizer::load_tokenizer_from_gguf(
        Path::new(model_path),
        pesti_runner::transformer::tokenizer::TokenizerBackend::Pesti,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load tokenizer: {}", e);
            std::process::exit(1);
        }
    };

    let prompt = "The quick brown fox jumps over the lazy dog. ";
    let tokens = match tokenizer.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to encode: {}", e);
            std::process::exit(1);
        }
    };

    println!("Prompt: '{}'", prompt);
    println!("Tokens: {} ({:?})", tokens.len(), tokens);
    println!();

    // Run generation loop manually to measure each step
    let mut all_tokens = tokens.clone();
    let max_new_tokens = 32;

    for step in 0..max_new_tokens {
        let seq_len = all_tokens.len() - 1;
        let start_pos = seq_len;

        // Forward pass through pesti-runner's own CUDA kernels
        let forward_start = Instant::now();
        let logits = match model.forward_with_dispatch(&all_tokens, start_pos) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Forward failed at step {}: {}", step, e);
                break;
            }
        };
        let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

        // Sample next token (argmax for determinism)
        let last_logits = &logits[seq_len];
        let next_token = pesti_runner::transformer::sampling::argmax(last_logits);

        all_tokens.push(next_token);

        if step < 3 || step >= max_new_tokens - 3 {
            println!("Step {}: {:.1}ms", step, forward_ms);
        }
    }

    // Decode output
    let generated = match tokenizer.decode(&all_tokens) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to decode: {}", e);
            std::process::exit(1);
        }
    };

    println!("\nGenerated: '{}'", generated);
}