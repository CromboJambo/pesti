//! Benchmark module for measuring inference performance across configurations.

use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};

/// Configuration variant to benchmark.
#[derive(Debug, Clone)]
pub struct ConfigVariant {
    pub name: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub n_gpu_layers: i32,
}

/// Results from benchmarking a single config variant.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub config_name: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
    pub n_gpu_layers: i32,
    pub load_time_ms: f64,
    pub prompt_eval_ms: f64,
    pub eval_ms: f64,
    pub tokens_per_second: f64,
    pub generated_tokens: usize,
}

/// Generate config variants to benchmark based on hardware.
pub fn generate_variants(n_gpu_layers: i32) -> Vec<ConfigVariant> {
    vec![
        ConfigVariant {
            name: "small-batch".into(),
            n_ctx: 4096,
            n_batch: 128,
            n_threads: 0, // auto
            n_gpu_layers,
        },
        ConfigVariant {
            name: "medium-batch".into(),
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 0,
            n_gpu_layers,
        },
        ConfigVariant {
            name: "large-batch".into(),
            n_ctx: 8192,
            n_batch: 1024,
            n_threads: 0,
            n_gpu_layers,
        },
    ]
}

/// Benchmark a single configuration.
pub fn benchmark_variant(
    model_path: &str,
    variant: &ConfigVariant,
) -> anyhow::Result<BenchmarkResult> {
    let start = Instant::now();

    // Build runner with this config
    let runner = LlamaRunner::builder(model_path)
        .n_ctx(variant.n_ctx)
        .n_batch(variant.n_batch)
        .n_threads(variant.n_threads)
        .n_gpu_layers(variant.n_gpu_layers)
        .build()?;

    let load_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Standard benchmark prompt (short, deterministic)
    let prompt = "What is the capital of France? Answer in one word.";
    let sampling = SamplingConfig::greedy();

    let result = runner.generate(prompt, &sampling)?;

    let tokens_per_second = if result.eval_ms > 0.0 {
        result.generated_tokens as f64 / (result.eval_ms / 1000.0)
    } else {
        0.0
    };

    Ok(BenchmarkResult {
        config_name: variant.name.clone(),
        n_ctx: variant.n_ctx,
        n_batch: variant.n_batch,
        n_threads: variant.n_threads,
        n_gpu_layers: variant.n_gpu_layers,
        load_time_ms,
        prompt_eval_ms: result.prompt_eval_ms,
        eval_ms: result.eval_ms,
        tokens_per_second,
        generated_tokens: result.generated_tokens,
    })
}

/// Run all variants and return results sorted by tok/s (best first).
pub fn run_benchmarks(
    model_path: &str,
    n_gpu_layers: i32,
) -> anyhow::Result<Vec<BenchmarkResult>> {
    let variants = generate_variants(n_gpu_layers);
    let mut results = Vec::new();

    for variant in &variants {
        println!("\nBenchmarking: {} (ctx={}, batch={})", variant.name, variant.n_ctx, variant.n_batch);
        match benchmark_variant(model_path, variant) {
            Ok(result) => {
                println!("  → {:.2} tok/s", result.tokens_per_second);
                results.push(result);
            }
            Err(e) => {
                println!("  → Failed: {}", e);
            }
        }
    }

    // Sort by tokens per second (best first)
    results.sort_by(|a, b| b.tokens_per_second.partial_cmp(&a.tokens_per_second).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Print benchmark results table.
pub fn print_results(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));
    println!("{:<16} {:>7} {:>7} {:>8} {:>8} {:>10}", "Config", "ctx", "batch", "gpu_l", "load_ms", "tok/s");
    println!("{}", "-".repeat(80));

    for (i, result) in results.iter().enumerate() {
        let marker = if i == 0 { " ← BEST" } else { "" };
        println!(
            "{:<16} {:>7} {:>7} {:>8} {:>8.0} {:>10.2}{}",
            result.config_name,
            result.n_ctx,
            result.n_batch,
            result.n_gpu_layers,
            result.load_time_ms,
            result.tokens_per_second,
            marker
        );
    }

    if let Some(best) = results.first() {
        println!("{}", "-".repeat(80));
        println!("Recommended: {} (n_ctx={}, n_batch={})", best.config_name, best.n_ctx, best.n_batch);
    }
}
