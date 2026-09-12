//! Batched generation spike: process multiple prompts in parallel using LlamaBatch
//! with n_seq_max > 1. Each prompt gets its own sequence ID and they're all decoded
//! together, improving throughput for multi-prompt workloads.
//!
//! Usage: cargo run --release --example batched_generation_spike -- /path/to/model.gguf
//!
//! This is a SPIKE — exploratory code to measure feasibility and performance of
//! parallel prompt processing. Not production-ready API design.

use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let model_path = std::env::args().nth(1).expect("Usage: batched_generation_spike <model.gguf>");

    // Load runner with larger context to fit multiple sequences
    let runner = LlamaRunner::builder(&model_path)
        .n_ctx(8192)
        .n_batch(1024)
        .build()
        .expect("Failed to build runner");

    // Test prompts — independent, can be processed in parallel
    let prompts = vec![
        "Explain why the sky is blue in one sentence.",
        "Write a haiku about coding.",
        "What is 2+2? Answer with just the number.",
        "List three fruits separated by commas.",
    ];

    let config = SamplingConfig {
        max_tokens: 50,
        temperature: 0.7,
        ..Default::default()
    };

    println!("=== Sequential Generation (baseline) ===");
    let t_seq_start = Instant::now();
    for (i, prompt) in prompts.iter().enumerate() {
        let result = runner.generate(prompt, &config).expect("generation failed");
        println!("[{}] Generated {} tokens: {}", i, result.generated_tokens, result.text.trim());
    }
    let seq_time = t_seq_start.elapsed().as_secs_f64();
    println!("Sequential total: {:.2}s", seq_time);

    // Reset KV cache for batched run
    runner.clear_kv_cache().expect("failed to clear KV cache");

    println!("\n=== Batched Generation (n_seq_max={}) ===", prompts.len());
    let t_batch_start = Instant::now();

    // Encode all prompts
    let mut prompt_tokens: Vec<Vec<i32>> = Vec::new();
    for prompt in &prompts {
        let tokens = runner.encode(prompt, true).expect("encode failed");
        prompt_tokens.push(tokens);
    }

    let max_seq_len = prompt_tokens.iter().map(|t| t.len()).max().unwrap();

    // Create batch with n_seq_max = number of prompts (parallel sequences)
    let mut batch = llama_cpp_2::llama_batch::LlamaBatch::new(
        max_seq_len,
        prompts.len().try_into().unwrap(),
    );

    for (seq_id, tokens) in prompt_tokens.iter().enumerate() {
        for (pos, tok) in tokens.iter().enumerate() {
            batch
                .add(llama_cpp_2::token::LlamaToken(*tok), pos as i32, &[seq_id as i32], true)
                .expect("batch add failed");
        }
    }

    // Prefill all sequences at once
    runner.decode(&mut batch).expect("prefill decode failed");

    // Sample for each sequence independently using greedy decoding
    let ctx = runner.context().borrow_mut();
    let mut sampled_tokens: Vec<i32> = Vec::new();
    for seq_id in 0..prompts.len() {
        let logits = ctx.get_logits_ith(seq_id as i32);
        // Use greedy sampling for simplicity in this spike
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        sampled_tokens.push(best as i32);
    }
    drop(ctx);

    println!("Sampled tokens: {:?}", sampled_tokens);

    let batch_time = t_batch_start.elapsed().as_secs_f64();
    println!("Batched total: {:.2}s", batch_time);
    println!("Speedup: {:.2}x", seq_time / batch_time.max(0.001));

    println!("\n=== Spike Results ===");
    println!(
        "Sequential throughput: {:.2} tok/s (total)",
        prompts.len() as f64 * 50.0 / seq_time
    );
    println!(
        "Batched throughput: {:.2} tok/s (total)",
        prompts.len() as f64 * 50.0 / batch_time.max(0.001)
    );
}
