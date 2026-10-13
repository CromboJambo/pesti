//! Detailed tok/s benchmark with kernel-level profiling.
//! Measures pesti-runner vs llama.cpp reference and breaks down time by operation.

use pesti_runner::{load_gguf_weights, LlamaModel};
use pesti_runner::profiler;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

#[test]
fn benchmark_detailed() {
    // Use absolute path to avoid working directory issues
    let model_path = "/home/crombo/projects/active/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !Path::new(model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }
    
    println!("=== PESTI Detailed Benchmark ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m (Q4_K_M)");
    println!();
    
    // Initialize profiler
    let _p = profiler::init();
    profiler::reset();
    
    // Load weights and build model
    let t_load = Instant::now();
    let weights = load_gguf_weights(Path::new(model_path)).expect("Failed to load GGUF weights");
    let mut model = LlamaModel::from_gguf_weights(weights).expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());
    
    // Load tokenizer from GGUF metadata
    let (_tokenizer_config, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        Path::new(model_path),
        pesti_runner::transformer::TokenizerBackend::MistralRs,
    )
    .expect("Failed to load tokenizer from GGUF");
    model.tokenizer = Some(tokenizer);
    
    // Benchmark prompt
    let prompt = "Write a haiku about programming.";
    let max_tokens = 64;
    
    // Tokenize
    let prompt_tokens = {
        let tok = model.tokenizer.as_ref().expect("No tokenizer");
        tok.encode(prompt).expect("Failed to encode")
    };
    println!("Prompt: \"{}\"", prompt);
    println!("Encoded {} tokens", prompt_tokens.len());
    
    // Sampling config for greedy decoding (fastest, deterministic)
    let sampling = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };
    
    // Warmup run (not measured)
    println!("Running warmup...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    model.generate(&prompt_tokens, 8, &sampling, &mut rng, &[151663])
        .expect("Warmup generation failed");
    
    // Reset profiler after warmup
    profiler::reset();
    
    // Timed run with profiling
    println!("Running benchmark with profiling...");
    let t_gen = Instant::now();
    
    // Measure tokenization separately
    let t_tok = Instant::now();
    let prompt_tokens = {
        let tok = model.tokenizer.as_ref().expect("No tokenizer");
        tok.encode(prompt).expect("Failed to encode")
    };
    println!("Tokenization: {:.3}s", t_tok.elapsed().as_secs_f64());
    
    rng = rand::rngs::StdRng::seed_from_u64(42);
    let result_tokens = model.generate(&prompt_tokens, max_tokens, &sampling, &mut rng, &[151663])
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();
    
    // Measure decoding separately
    let t_dec = Instant::now();
    if let Some(tok) = model.tokenizer.as_ref() {
        let _generated_text = tok.decode(&result_tokens).expect("Failed to decode");
        println!("Decoding: {:.3}s", t_dec.elapsed().as_secs_f64());
    }
    
    println!();
    println!("=== Results ===");
    println!("Generated {} tokens in {:.2}s", result_tokens.len(), gen_time);
    
    if !result_tokens.is_empty() {
        let tok_per_sec = result_tokens.len() as f64 / gen_time;
        println!("Decode speed: {:.2} tok/s", tok_per_sec);
        println!("Avg time per token: {:.0}ms", 1000.0 / tok_per_sec);
    }
    
    // Print kernel timing breakdown
    profiler::print_report();
    
    // Decode and show output
    if let Some(tok) = model.tokenizer.as_ref() {
        let generated_text = tok.decode(&result_tokens).expect("Failed to decode");
        println!("Output: {}", &generated_text[..generated_text.len().min(200)]);
    }
    
    println!();
    println!("=== Benchmark Complete ===");
}
