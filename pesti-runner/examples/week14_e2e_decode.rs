//! Week 14 end-to-end decode benchmark with real transformer forward pass.
//!
//! This benchmark exercises the full autoregressive inference pipeline:
//! 1. Load GGUF weights (Qwen2.5-0.5B)
//! 2. Tokenize prompt using GgufTokenizer
//! 3. Prefill: process prompt tokens through transformer layers
//! 4. Decode loop: generate new tokens one at a time with KV caching
//! 5. Measure actual tokens/sec throughput

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("=== Week 14: Real E2E Decode Benchmark ===");
    println!("Model: {}", model_path);
    
    // Load weights
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let load_time = t0.elapsed();
    println!("Loaded weights in {:.2}s", load_time.as_secs_f32());
    
    println!(
        "tensors={}, bytes={}",
        weights.tensors.len(),
        weights.tensors.values().map(|v| v.len()).sum::<usize>()
    );
    
    // Load model
    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    let model_load_time = t1.elapsed();
    println!("Built model in {:.2}s", model_load_time.as_secs_f32());
    
    // Load tokenizer
    let (tokenizer_config, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path))?;
    println!("Loaded tokenizer with {} tokens", tokenizer_config.vocab_size);
    
    // Tokenize prompt
    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("\nPrompt: {}", prompt);
    
    let t2 = Instant::now();
    let prompt_tokens = tokenizer.encode(prompt)?;
    let encode_time = t2.elapsed();
    println!("Encoded {} tokens in {:.3}ms", prompt_tokens.len(), encode_time.as_secs_f32() * 1000.0);
    
    // Reset KV caches before generation
    model.reset_cpu_kv_caches();
    
    // Generate tokens
    let max_tokens = 64;
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.0, // Greedy decoding for reproducibility
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.0,
    };
    
    let t3 = Instant::now();
    let generated_tokens = model.generate(
        &prompt_tokens,
        max_tokens,
        &sampling_config,
        &mut rand::rngs::StdRng::seed_from_u64(42),
        &[0], // No stop tokens
    )?;
    let gen_time = t3.elapsed();
    
    println!("\n=== Results ===");
    println!("Generated tokens:   {}", generated_tokens.len());
    println!("Generation time:    {:.3}s", gen_time.as_secs_f32());
    println!("Throughput:         {:.2} tok/s", generated_tokens.len() as f64 / gen_time.as_secs_f32());
    
    // Decode and print first few generated tokens
    let decoded = tokenizer.decode(&generated_tokens)?;
    println!("\nGenerated text (first 200 chars):");
    println!("{}", decoded.chars().take(200).collect::<String>());
    
    // Estimate total time including all phases
    let total_time = load_time + model_load_time + encode_time + gen_time;
    println!("\n=== Timeline ===");
    println!("Weight loading:     {:.2}s", load_time.as_secs_f32());
    println!("Model build:        {:.2}s", model_load_time.as_secs_f32());
    println!("Tokenization:       {:.3}ms", encode_time.as_secs_f32() * 1000.0);
    println!("Generation:         {:.3}s", gen_time.as_secs_f32());
    println!("Total time:         {:.3}s", total_time.as_secs_f32());
    
    Ok(())
}
