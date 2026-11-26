//! Spectral profiling demo: real inference-time attention weight capture.
//!
//! Demonstrates Phase 2 integration: the profiler is hooked into
//! `Attention::forward()` so it captures actual softmax weights from
//! live inference rather than synthetic patterns.
//!
//! Usage:
//!   PESTI_SPECTRAL_PROFILE=1 cargo run --example spectral_profile
//!
//! Without the env var, profiling is disabled (zero overhead).

use pesti_runner::diagnostics::spectral::{DegeneracySeverity, SpectralProfilerCollector};
use pesti_runner::transformer::linear::Linear;
use pesti_runner::transformer::layer::Attention;
use std::sync::{Arc, Mutex};

/// Create a small synthetic linear layer with deterministic weights.
fn make_linear(in_dim: usize, out_dim: usize, seed: u64) -> Linear {
    // Deterministic pseudo-random initialization
    let mut state = seed;
    let rand_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let weights: Vec<f32> = (0..(in_dim * out_dim)).map(|_| rand_f32(&mut state)).collect();
    Linear::new(weights, None, in_dim, out_dim)
}

fn main() {
    println!("PESTI Spectral Profiler — Phase 2 (real attention dispatch)\n");

    // Check if profiling is enabled
    let enabled = SpectralProfilerCollector::is_profiling_enabled();
    if !enabled {
        println!("Profiling disabled. Set PESTI_SPECTRAL_PROFILE=1 to enable.");
        println!("Continuing with demo (profiler will be inactive).\n");
    }

    // Model dimensions (small for demo)
    let seq_len = 64;
    let num_heads = 8;
    let head_dim = 32;
    let embed_dim = num_heads * head_dim;
    let num_kv_heads = 2; // GQA: 4 query heads per KV head
    let kv_dim = num_kv_heads * head_dim;

    println!("Config:");
    println!("  seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}");
    println!("  GQA: {num_heads} query heads / {num_kv_heads} KV heads\n");

    // Build attention layer with deterministic weights
    let wq = make_linear(embed_dim, embed_dim, 42);
    let wk = make_linear(embed_dim, kv_dim, 137);
    let wv = make_linear(embed_dim, kv_dim, 256);
    let wo = make_linear(embed_dim, embed_dim, 999);

    let mut attention = Attention::new(wq, wk, wv, wo, head_dim, num_heads, num_kv_heads, 10000.0);

    // Create and attach profiler
    let collector = SpectralProfilerCollector::new(
        "demo-attention-layer".into(),
        seq_len,
        1,      // single layer
        num_heads,
    );
    attention = attention.with_profiler(Arc::new(Mutex::new(collector)));

    // Generate deterministic input (simulates token embeddings after embedding + position)
    let mut state = 12345u64;
    let rand_f32 = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let input: Vec<f32> = (0..(seq_len * embed_dim)).map(|_| rand_f32(&mut state)).collect();

    println!("Running attention forward pass with {seq_len} tokens...");
    let start = std::time::Instant::now();
    let output = attention.forward(&input, 1, seq_len, 0);
    let elapsed = start.elapsed();
    println!("  Completed in {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Output shape: [{}] (batch=1, seq={seq_len}, dim={embed_dim})\n", output.len());

    // Generate report if profiling was active
    let collector = attention.profiler.as_ref().unwrap();
    let mut collector = collector.lock().unwrap();

    if !collector.is_enabled() {
        println!("Profiling was inactive (PESTI_SPECTRAL_PROFILE not set). No report.");
        return;
    }

    match collector.finish() {
        Some(report) => {
            println!("Spectral Profile Report:");
            println!("  Model: {}", report.model);
            println!("  Heads profiled: {} / {}", report.summary.total_heads, num_heads);
            println!("  Degenerate heads: {} ({:.1}%)",
                report.summary.degenerate_heads,
                report.summary.degeneracy_rate * 100.0);
            println!();

            // Show per-head results
            for head in &report.heads {
                let severity_str = match head.severity {
                    DegeneracySeverity::Healthy => "HEALTHY",
                    DegeneracySeverity::ModerateDegeneration => "MODERATE",
                    DegeneracySeverity::SevereDegeneration => "SEVERE",
                };
                println!("  Head {:>2}: λ_max={:.3}  effective_rank={:.1}  participation_ratio={:.3}  [{}]",
                    head.head_id,
                    head.spectrum.largest_eigenvalue,
                    head.effective_rank,
                    head.participation_ratio,
                    severity_str);
            }

            // Top recommendations
            if !report.recommendations.is_empty() {
                println!("\nRecommendations:");
                for rec in &report.recommendations {
                    println!("  - [{}] {}", rec.severity, rec.message);
                }
            }
        }
        None => {
            println!("No spectral data collected (profiler disabled or no heads analyzed).\n");
            println!("Re-run with: PESTI_SPECTRAL_PROFILE=1 cargo run --example spectral_profile");
        }
    }
}
