//! Kernel-level timing profiler for pesti-runner.
//!
//! Measures wall-clock time of GPU kernel execution (including sync overhead)
//! to compute the GEMM vs attention time split at production sequence lengths.
//!
//! Usage:
//!   let _timer = Profiler::time(KernelCategory::Attention);
//!   // ... kernel call ...
//!   drop(_timer);  // automatically stops and records

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Kernel categories for time breakdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelCategory {
    Gemm,
    Attention,
    Rope,
    LayerNorm,
    Other,
}

impl std::fmt::Display for KernelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            KernelCategory::Gemm => write!(f, "gemm"),
            KernelCategory::Attention => write!(f, "attention"),
            KernelCategory::Rope => write!(f, "rope"),
            KernelCategory::LayerNorm => write!(f, "layernorm"),
            KernelCategory::Other => write!(f, "other"),
        }
    }
}

/// Per-category timing accumulator.
#[derive(Default)]
struct CategoryStats {
    total_ms: f64,
    count: u64,
}

/// Global profiler state — thread-safe, accessible from any dispatch call.
pub struct ProfilerState {
    enabled: bool,
    categories: HashMap<KernelCategory, CategoryStats>,
}

static PROFILER: std::sync::OnceLock<Arc<Mutex<ProfilerState>>> = std::sync::OnceLock::new();

/// RAII timer — starts on creation, records elapsed time on drop.
pub struct KernelTimer {
    category: KernelCategory,
    start: Instant,
}

impl KernelTimer {
    pub fn new(category: KernelCategory) -> Self {
        let _ = init();
        Self {
            category,
            start: Instant::now(),
        }
    }
}

impl Drop for KernelTimer {
    fn drop(&mut self) {
        let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;

        let Some(p) = PROFILER.get() else {
            return;
        };
        let mut state = p.lock().unwrap();
        if !state.enabled {
            return;
        }

        let stats = state.categories.entry(self.category).or_default();
        stats.total_ms += elapsed_ms;
        stats.count += 1;
    }
}

/// Initialize the profiler. Call once at startup.
pub fn init() -> Arc<Mutex<ProfilerState>> {
    PROFILER
        .get_or_init(|| {
            Arc::new(Mutex::new(ProfilerState {
                enabled: true,
                categories: HashMap::new(),
            }))
        })
        .clone()
}

/// Start timing a kernel operation. Returns a timer that records on drop.
pub fn time(category: KernelCategory) -> KernelTimer {
    KernelTimer::new(category)
}

/// Print timing report to stdout. Call after generation completes.
pub fn print_report() {
    let Some(p) = PROFILER.get() else {
        return;
    };

    let state = p.lock().unwrap();
    if !state.enabled {
        return;
    }

    println!("\n=== PESTI Kernel Timing Report ===");
    let mut total_ms = 0.0f64;
    for (cat, stats) in &state.categories {
        println!(
            "  {:>12}  {:>8} calls  {:>10.3} ms",
            cat.to_string(),
            stats.count,
            stats.total_ms
        );
        total_ms += stats.total_ms;
    }
    println!("----------------------------------");
    println!("  Total:      {:>10.3} ms", total_ms);

    // Compute percentages
    if total_ms > 0.0 {
        println!("\n  Time distribution:");
        for (cat, stats) in &state.categories {
            let pct = (stats.total_ms / total_ms) * 100.0;
            println!("    {:>12}: {:>6.1}%", cat.to_string(), pct);
        }
    }
    println!("=================================\n");
}

/// Check if profiler is enabled.
pub fn is_enabled() -> bool {
    match PROFILER.get() {
        Some(p) => p.lock().unwrap().enabled,
        None => false,
    }
}
