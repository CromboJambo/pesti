//! KV Cache benchmark: compares uncached vs cached forward pass.
//!
//! The uncached path recomputes ALL attention over the full sequence every token.
//! The cached path only computes attention over the KV cache + new position.
//!
//! Usage: cargo run --package pesti-runner --features cuda --example kv_cache_bench

use pesti_runner::transformer::LlamaModel;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== KV Cache Benchmark ===\n");

    let model_path = find_model()?;
    println!("Model: {}", model_path.display());

    let load_start = Instant::now();
    let mut model = LlamaModel::load_gguf(&model_path)?;
    println!(
        "Loaded in {:.1}ms ({} layers, embed_dim={}, heads={}, kv_heads={}, head_dim={})",
        load_start.elapsed().as_secs_f64() * 1000.0,
        model.config.num_layers,
        model.config.embed_dim,
        model.config.num_heads,
        model.config.num_kv_heads,
        model.config.head_dim,
    );

    let embed_dim = model.config.embed_dim;
    let hidden: Vec<f32> = (0..embed_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let num_tokens = 50;

    // ── Uncached: recompute everything each token ──
    println!("\n--- Uncached (forward_layers) ---");
    let start = Instant::now();
    for pos in 0..num_tokens {
        model.cpu_kv_caches = None;
        let _ = model.forward_layers(&hidden, pos);
    }
    let uncached_total = start.elapsed();
    let uncached_ms = uncached_total.as_secs_f64() * 1000.0 / num_tokens as f64;
    let uncached_tok_s = num_tokens as f32 / uncached_total.as_secs_f32();
    println!(
        "  {} tokens: total={:.1}ms, avg={:.1}ms/token, {:.1} tok/s",
        num_tokens,
        uncached_total.as_secs_f64() * 1000.0,
        uncached_ms,
        uncached_tok_s,
    );

    // ── Cached: incremental decode with KV cache ──
    println!("\n--- Cached (forward_layers_with_cache) ---");
    model.reset_cpu_kv_caches();
    let start = Instant::now();
    for pos in 0..num_tokens {
        let _ = model.forward_layers_with_cache(&hidden, pos);
    }
    let cached_total = start.elapsed();
    let cached_ms = cached_total.as_secs_f64() * 1000.0 / num_tokens as f64;
    let cached_tok_s = num_tokens as f32 / cached_total.as_secs_f32();
    println!(
        "  {} tokens: total={:.1}ms, avg={:.1}ms/token, {:.1} tok/s",
        num_tokens,
        cached_total.as_secs_f64() * 1000.0,
        cached_ms,
        cached_tok_s,
    );

    // ── Summary ──
    println!("\n--- Summary ---");
    let speedup = uncached_total.as_secs_f32() / cached_total.as_secs_f32();
    println!(
        "  Uncached: {:.1} tok/s  |  Cached: {:.1} tok/s  |  Speedup: {:.1}x",
        uncached_tok_s, cached_tok_s, speedup,
    );

    if speedup > 1.0 {
        println!(
            "  ✅ KV cache provides {:.1}x speedup over {} tokens",
            speedup, num_tokens
        );
    } else {
        println!("  ⚠️  KV cache is slower at this sequence length (overhead dominates)");
        println!("  Try longer sequences to see the crossover point.");
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

fn find_model() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let candidates = vec![
        "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
        "../conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
    ];

    if let Ok(path) = std::env::var("PESTI_MODEL") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Ok(p);
        }
    }

    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }

    Err("No GGUF model found. Set PESTI_MODEL env var or place model in conformance-corpus/".into())
}
