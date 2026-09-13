//! Benchmark pesti-runner's own CUDA inference path (not llama.cpp FFI).
//! Uses LlamaModel::from_gguf_weights() + fused attention kernel.
//!
//! Usage: cargo run --release --features cuda --example benchmark_pesti_cuda

use pesti_runner::kernel::dispatch::{DispatchContext, InferenceEngine};
use pesti_runner::{LlamaModel, SamplingConfig};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== pesti-runner CUDA Benchmark (own kernels) ===");
    println!("Model: Qwen2.5-0.5B-Instruct-Q4_K_M");
    println!("Path: pesti-runner transformer stack with fused attention");
    println!();

    // Load weights through pesti-runner's own stack
    let start = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = LlamaModel::from_gguf_weights(weights)?;
    let load_time = start.elapsed().as_secs_f64();
    println!("✓ Model loaded in {:.2}s", load_time);
    println!("  Architecture: {:?}", model.config.arch);
    println!("  Layers: {}", model.config.num_layers);
    println!("  Embed dim: {}", model.config.embed_dim);
    println!("  Heads: {} (KV: {})", model.config.num_heads, model.config.num_kv_heads);
    println!();

    // Initialize CUDA dispatch context
    let ctx = DispatchContext::from_engine(InferenceEngine::Cuda);
    if !ctx.gpu_available() {
        return Err("CUDA not available".into());
    }
    println!("✓ CUDA initialized: {}", ctx.device_info());

    // Build and set fused attention kernel on each layer's dispatch
    let head_dim = model.config.head_dim as i32;
    let num_heads = model.config.num_heads as i32;
    let num_kv_heads = model.config.num_kv_heads as i32;

    println!("Building fused attention kernel...");
    let start = Instant::now();
    let fused_kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        head_dim, num_heads, num_kv_heads, 4096, None,
    )?;
    println!("✓ Fused attention kernel built in {:.2}s", start.elapsed().as_secs_f64());

    // Set fused kernel on each layer's dispatch
    if let Some(dispatch_layers) = model.dispatch_layers.as_mut() {
        for (i, layer_dispatch) in dispatch_layers.iter_mut().enumerate() {
            layer_dispatch.attention.set_fused_kernel(fused_kernel.clone(), head_dim, num_heads, num_kv_heads);
            if i == 0 {
                println!("✓ Fused kernel set on layer {} (total: {})", i, model.config.num_layers);
            }
        }
    }

    // Tokenize prompt using pesti-runner's tokenizer
    let start = Instant::now();
    let (_config, tokenizer) = pesti_runner::transformer::tokenizer::load_tokenizer_from_gguf(
        Path::new(model_path),
        pesti_runner::transformer::tokenizer::TokenizerBackend::Pesti,
    )?;
    println!("✓ Tokenizer loaded in {:.2}s", start.elapsed().as_secs_f64());

    let prompt = "Explain the concept of recursion in programming with a simple example.";
    let tokens = tokenizer.encode(prompt)?;
    println!("  Prompt: {} tokens", tokens.len());
    println!();

    // Reset KV caches for clean benchmark
    model.reset_cpu_kv_caches();

    // Run generation through pesti-runner's own stack
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        max_tokens: 128,
        ..Default::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    println!("Generating response through pesti-runner CUDA kernels...");
    let start = Instant::now();
    let generated = model.generate(&tokens, 128, &sampling, &mut rng, &[0])?;
    let total_time = start.elapsed().as_secs_f64();

    // Decode output
    let output_text = tokenizer.decode(&generated)?;

    println!();
    println!("=== Results ===");
    println!("Generated tokens: {}", generated.len());
    println!("Total time: {:.2}s", total_time);
    if generated.len() > 0 {
        let tok_per_sec = generated.len() as f64 / total_time;
        println!("Throughput: {:.1} tok/s", tok_per_sec);
        println!();
        println!("=== Generated Text ===");
        println!("{}", output_text);
    }

    Ok(())
}