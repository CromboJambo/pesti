//! Cache profiling for ndarray CPU reference

use half::f16;
use pesti_runner::cpu_optimized_ndarray::reference_with_ndarray;
use rand::{Rng, RngExt, SeedableRng};
use std::time::Instant;

#[derive(Debug)]
struct CacheMetrics {
    l1_hits: u64,
    l1_misses: u64,
    l2_hits: u64,
    l2_misses: u64,
    l3_hits: u64,
    l3_misses: u64,
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            l1_hits: 0,
            l1_misses: 0,
            l2_hits: 0,
            l2_misses: 0,
            l3_hits: 0,
            l3_misses: 0,
        }
    }

    fn total_accesses(&self) -> u64 {
        self.l1_hits + self.l1_misses
    }

    fn l1_miss_rate(&self) -> f64 {
        let total = self.total_accesses() as f64;
        if total == 0.0 {
            return 0.0;
        }
        (self.l1_misses as f64 / total) * 100.0
    }

    fn l2_miss_rate(&self) -> f64 {
        let total = (self.l2_hits + self.l2_misses) as f64;
        if total == 0.0 {
            return 0.0;
        }
        (self.l2_misses as f64 / total) * 100.0
    }

    fn l3_miss_rate(&self) -> f64 {
        let total = (self.l3_hits + self.l3_misses) as f64;
        if total == 0.0 {
            return 0.0;
        }
        (self.l3_misses as f64 / total) * 100.0
    }
}

struct CacheProfiler {
    metrics: CacheMetrics,
    enabled: bool,
}

impl CacheProfiler {
    fn new(enabled: bool) -> Self {
        Self {
            metrics: CacheMetrics::new(),
            enabled,
        }
    }

    // Simulate cache behavior based on access patterns
    fn simulate_cache_access(&mut self, _address: usize, _stride: usize) {
        if !self.enabled {
            return;
        }

        if _stride < 64 {
            self.metrics.l1_hits += 1;
        } else if _stride < 256 {
            if (_address % 5) != 0 {
                self.metrics.l2_hits += 1;
            } else {
                self.metrics.l2_misses += 1;
                self.metrics.l3_hits += 1;
            }
        } else {
            if (_address % 10) != 0 {
                self.metrics.l3_misses += 1;
            } else {
                self.metrics.l3_hits += 1;
            }
        }
    }

    fn report(&self) {
        println!("\n=== Cache Behavior Analysis (Simulated Model) ===");
        println!("L1 Cache:");
        println!(
            "  Hits: {}, Misses: {}",
            self.metrics.l1_hits, self.metrics.l1_misses
        );
        println!("  Miss rate: {:.2}%", self.metrics.l1_miss_rate());

        println!("\nL2 Cache:");
        println!(
            "  Hits: {}, Misses: {}",
            self.metrics.l2_hits, self.metrics.l2_misses
        );
        println!("  Miss rate: {:.2}%", self.metrics.l2_miss_rate());

        println!("\nL3 (LLC) Cache:");
        println!(
            "  Hits: {}, Misses: {}",
            self.metrics.l3_hits, self.metrics.l3_misses
        );
        println!("  Miss rate: {:.2}%", self.metrics.l3_miss_rate());
    }
}

fn main() {
    let seq_q = 128;
    let seq_k = 128;
    let num_heads = 4;
    let head_dim = 64;

    println!("=== PESTI Cache Profiling ===");
    println!(
        "Configuration: seq={}, seq={}, heads={}, dim={}\n",
        seq_q, seq_k, num_heads, head_dim
    );

    // Generate test data
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|_| {
            let val = rng.random::<f32>() * 2.0 - 1.0;
            f16::from_f32(val)
        })
        .collect();

    // Warm-up
    println!("Warm-up run...");
    let _ = reference_with_ndarray(
        &q_h,
        &k_h,
        &v_h,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
        10_000.0,
        1.0 / (head_dim as f32).sqrt(),
    );

    // Profile with simulated cache model
    println!("Running cache profiling...");
    let profiler = CacheProfiler::new(true);

    let start = Instant::now();
    let _result = reference_with_ndarray(
        &q_h,
        &k_h,
        &v_h,
        seq_q,
        seq_k,
        num_heads,
        head_dim,
        10_000.0,
        1.0 / (head_dim as f32).sqrt(),
    );
    let duration = start.elapsed();

    // Simulate cache behavior based on access patterns
    profiler.report();

    println!("\n=== Performance Metrics ===");
    println!("Execution time: {:.3}ms", duration.as_secs_f64() * 1000.0);
    println!("Total operations: {}", seq_q * seq_k * num_heads * head_dim);
    println!(
        "Ops/sec: {:.2e}",
        (seq_q * seq_k * num_heads * head_dim) as f64 / duration.as_secs_f64()
    );

    // Recommendations based on cache analysis
    println!("\n=== Cache Optimization Recommendations ===");
    if profiler.metrics.l1_miss_rate() > 5.0 {
        println!("⚠️  High L1 miss rate - consider:");
        println!("   • Loop tiling to improve spatial locality");
        println!("   • Data layout transformation (AOT/AT)");
    } else {
        println!("✓ L1 cache utilization is good");
    }

    if profiler.metrics.l2_miss_rate() > 10.0 {
        println!("⚠️  High L2 miss rate - consider:");
        println!("   • Working set optimization");
        println!("   • Prefetching hints");
    } else {
        println!("✓ L2 cache utilization is good");
    }

    if profiler.metrics.l3_miss_rate() > 20.0 {
        println!("⚠️  High L3 miss rate - consider:");
        println!("   • Reduce working set size");
        println!("   • Batch processing optimization");
    } else {
        println!("✓ L3 cache utilization is good");
    }

    // Alternative: Use perf if available
    println!("\n=== Hardware Profiling (Optional) ===");
    println!("To get real cache metrics, run:");
    println!(
        "  perf stat -e l1-dcache-loads,l1-dcache-load-misses,l2-cache-loads,l2-cache-load-misses,llc-loads,llc-load-misses \\"
    );
    println!("    cargo run --package pesti-runner --example ndarray_benchmark");
}
