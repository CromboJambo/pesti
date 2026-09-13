//! Benchmark pesti-runner's CUDA inference stack with fused attention kernel.
//! Uses forward_with_dispatch() for GPU autoregressive decoding.

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("/home/crombo/projects/pesti/models/Qwen2-7B-Instruct-Q4_K_M.gguf");
    if !model_path.exists() {
        eprintln!("Model not found at {}", model_path.display());
        std::process::exit(1);
    }

    println!("=== pesti-runner CUDA Benchmark ===");
    println!("Model: Qwen2-7B-Instruct-Q4_K_M.gguf");
    println!("Inference stack: pesti-runner (own CUDA kernels)");
    println!("Attention kernel: fused attention (softmax on GPU, eliminates D2H/H2D transfers)");
    println!();

    // Initialize llama model via pesti-runner API
    let mut model = pesti_runner::llama_cpp::LlamaModel::new();
    println!("Loading model...");
    let load_start = Instant::now();
    model.load_gguf(model_path)?;
    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    // Enable fused attention kernel (softmax on GPU, no host transfer)
    use pesti_runner::kernel::fused_attention_conformant::fused_attention;
    model.set_fused_kernel(fused_attention);
    println!("Fused attention kernel enabled");

    // Use hardcoded token IDs from llama.cpp oracle: "What is 2+2?"
    // Token IDs: [50280, 374, 2610, 9219, 374, 2706, 4333]
    let input_ids: Vec<pesti_runner::llama_cpp::LlamaToken> = vec![50280, 374, 2610, 9219, 374, 2706, 4333];

    println!("\nRunning warmup pass...");
    let mut seq_len = 0;
    for &tid in &input_ids {
        let emb = model.embed(tid)?;
        let logits = model.forward_with_dispatch(&emb, seq_len)?;
        seq_len += 1;
    }
    println!("Warmup complete");

    // Benchmark: decode 64 tokens
    const NUM_TOKENS: usize = 64;
    println!("\nBenchmarking {} token generation...", NUM_TOKENS);

    let bench_start = Instant::now();
    seq_len = 0;
    for &tid in &input_ids {
        let emb = model.embed(tid)?;
        let logits = model.forward_with_dispatch(&emb, seq_len)?;
        seq_len += 1;
    }

    let mut last_token_id = input_ids[input_ids.len() - 1];
    for _ in 0..NUM_TOKENS {
        let emb = model.embed(last_token_id)?;
        let logits = model.forward_with_dispatch(&emb, seq_len)?;
        // Argmax for greedy decoding
        let (best_idx, _) = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        last_token_id = best_idx as pesti_runner::llama_cpp::LlamaToken;
        seq_len += 1;
    }

    let elapsed = bench_start.elapsed();
    let total_tokens = input_ids.len() + NUM_TOKENS;

    println!("\n=== Results ===");
    println!("Total tokens processed: {}", total_tokens);
    println!("Elapsed time: {:.3}s", elapsed.as_secs_f64());
    println!(
        "Measured throughput: {:.2} tok/s",
        total_tokens as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}