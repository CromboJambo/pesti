//! Spectral profiling example: analyze attention matrices for degeneracy.
//!
//! Demonstrates the RMT-based spectral analysis on synthetic attention patterns
//! that represent different health states.

use ndarray::Array2;
use pesti_runner::diagnostics::spectral::{SpectralProfiler, SpectralProfilerCollector};

fn main() {
    println!("=== Attention Spectral Profiling Demo ===");
    println!();

    let profiler = SpectralProfiler::new();

    // 1. Rank-1 degenerate attention (all rows identical)
    println!("--- Degenerate: Rank-1 Attention ---");
    let n = 32;
    let uniform = Array2::from_elem((n, n), 1.0f32 / n as f32);
    let spec = profiler.analyze_attention(0, 0, &uniform);
    println!("Effective rank: {:.4}", spec.effective_rank);
    println!("Numerical rank: {}", spec.numerical_rank);
    println!("Degeneracy score: {:.4}", spec.degeneracy_score);
    println!("Top eigenvalue dominance: {:.4}", spec.top_eigenvalue_dominance());
    println!();

    // 2. Local/banded attention (healthy pattern)
    println!("--- Healthy: Local Attention ---");
    let bandwidth = 5;
    let mut local = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let dist = (i as i32 - j as i32).abs() as usize;
            if dist <= bandwidth {
                local[[i, j]] = 1.0f32 / (2 * bandwidth + 1) as f32;
            }
        }
    }
    let spec = profiler.analyze_attention(0, 1, &local);
    println!("Effective rank: {:.4}", spec.effective_rank);
    println!("Numerical rank: {}", spec.numerical_rank);
    println!("Degeneracy score: {:.4}", spec.degeneracy_score);
    println!();

    // 3. Sparse spike attention (one dominant key)
    println!("--- Degenerate: Spike Attention ---");
    let mut spike = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        spike[[i, 0]] = 1.0;  // All queries attend to first key
    }
    let spec = profiler.analyze_attention(0, 2, &spike);
    println!("Effective rank: {:.4}", spec.effective_rank);
    println!("Numerical rank: {}", spec.numerical_rank);
    println!("Degeneracy score: {:.4}", spec.degeneracy_score);
    println!();

    // 4. Multi-head profiling with collector
    println!("--- Multi-Head Profiling ---");
    let mut collector = SpectralProfilerCollector::new(
        "demo-model".to_string(),
        12,  // num_layers
        8,   // num_heads
        64,  // head_dim
    );

    if collector.enabled {
        println!("Profiling enabled (PESTI_SPECTRAL_PROFILE=1)");
        for layer in 0..3 {
            for head in 0..4 {
                let pattern = match head % 4 {
                    0 => uniform.clone(),   // Degenerate
                    1 => local.clone(),     // Healthy
                    2 => spike.clone(),     // Degenerate
                    _ => {                  // Mixed
                        let mut mixed = Array2::<f32>::zeros((n, n));
                        for i in 0..n {
                            for j in 0..n {
                                mixed[[i, j]] = 1.0f32 / n as f32;
                            }
                            mixed[[i, i % 4]] += 0.5;  // Add diagonal preference
                        }
                        // Normalize rows
                        for i in 0..n {
                            let sum: f32 = mixed.row(i).sum();
                            if sum > 0.0 {
                                mixed.row_mut(i).mapv_inplace(|x| x / sum);
                            }
                        }
                        mixed
                    }
                };
                collector.profile_head(layer, head, &pattern);
            }
        }

        let report = collector.report();
        println!("Total heads profiled: {}", report.total_heads_profiled);
        println!("Degenerate heads (>0.7): {}", report.degenerate_heads.len());
        for (key, spec) in &report.degenerate_heads {
            println!("  Layer {} Head {}: score={:.4}, rank={}",
                key.layer, key.head, spec.degeneracy_score, spec.numerical_rank);
        }
        println!();

        // Show JSON output
        let json = serde_json::to_string_pretty(&report).unwrap();
        println!("JSON report (truncated):");
        println!("{}", &json[..std::cmp::min(500, json.len())]);
    } else {
        println!("Profiling disabled. Set PESTI_SPECTRAL_PROFILE=1 to enable.");
        println!();
        println!("(Running with profiling enabled for demo...)");

        // Force-enable for demo purposes
        unsafe { std::env::set_var("PESTI_SPECTRAL_PROFILE", "1") };
        let mut collector = SpectralProfilerCollector::new(
            "demo-model".to_string(), 12, 8, 64,
        );
        for layer in 0..3 {
            for head in 0..4 {
                let pattern = match head % 4 {
                    0 => uniform.clone(),
                    1 => local.clone(),
                    2 => spike.clone(),
                    _ => {
                        let mut mixed = Array2::<f32>::zeros((n, n));
                        for i in 0..n {
                            for j in 0..n {
                                mixed[[i, j]] = 1.0f32 / n as f32;
                            }
                            mixed[[i, i % 4]] += 0.5;
                        }
                        for i in 0..n {
                            let sum: f32 = mixed.row(i).sum();
                            if sum > 0.0 {
                                mixed.row_mut(i).mapv_inplace(|x| x / sum);
                            }
                        }
                        mixed
                    }
                };
                collector.profile_head(layer, head, &pattern);
            }
        }
        let report = collector.report();
        println!("Total heads profiled: {}", report.total_heads_profiled);
        println!("Degenerate heads (>0.7): {}", report.degenerate_heads.len());
    }

    println!();
    println!("=== Demo Complete ===");
}
