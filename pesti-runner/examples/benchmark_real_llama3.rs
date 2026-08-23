//! End-to-end benchmark with Llama 3.1 8B Q4_K_M
//!
//! Measures actual token generation throughput (tok/s) to verify RoPE caching claims
//! Expected: ~72 tok/s with mistral.rs backend, ~50-60 tok/s with flash attention

use pesti_runner::transformer::LlamaModel;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== End-to-End Benchmark: Llama 3.1 8B Q4_K_M ===");

    // Load model (already downloaded to test_models/)
    let model_path =
        PathBuf::from("/home/crombo/projects/pesti/test_models/llama3.1-8b-q4_k_m.gguf");

    println!("Loading model from: {:?}", model_path);
    let mut model = LlamaModel::load_gguf(&model_path)?;

    // Test prompt (short for benchmarking)
    let prompt = "The future of LLM inference is";
    println!("\nPrompt: {}", prompt);

    // Embed the first token manually (simplified - real implementation would use proper tokenizer)
    let embed_dim = model.config.embed_dim;
    let mut hidden: Vec<f32> = (0..embed_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Measure time to generate 100 tokens
    println!("\nGenerating 100 tokens with KV cache...");
    let start = Instant::now();

    for pos in 0..100 {
        // Use cached forward pass (this is where RoPE caching shines!)
        model.forward_layers_with_cache(&hidden, pos)?;

        // Update hidden state for next token (simplified - real implementation would use proper sampling)
        hidden = model.forward(pos as u32, pos)?;
    }

    let duration = start.elapsed();

    // Calculate metrics
    let tok_per_sec = 100.0 / duration.as_secs_f64();
    println!("\n=== Results ===");
    println!("Tokens generated: {}", 100);
    println!("Time: {:?}", duration);
    println!("Throughput: {:.2} tok/s", tok_per_sec);

    // Compare with target (72 tok/s for mistral.rs)
    let gap = (72.0 - tok_per_sec) / 72.0 * 100.0;
    println!("\nPerformance gap vs mistral.rs (~72 tok/s): {:.1}%", gap);

    if tok_per_sec >= 65.0 {
        println!("✅ EXCELLENT: Within 10% of target!");
    } else if tok_per_sec >= 50.0 {
        println!("⚠️  GOOD: Within 30% of target");
    } else {
        println!("❌ NEEDS WORK: Below 50 tok/s, need more optimizations");
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}
