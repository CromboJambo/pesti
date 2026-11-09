//! Kernel-level GPU timing profiler using CUDA events.
//!
//! Records wall-clock time of individual kernel launches (GEMM, attention, etc.)
//! via cuda_event_record/cudaEventElapsedTime to compute the GEMM vs attention
//! time split at production sequence lengths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Raw CUDA event API via FFI — no cudarc Event type dependency.
extern "C" {
    fn cudaEventCreate(event: *mut u64) -> i32;
    fn cudaEventDestroy(event: u64) -> i32;
    fn cudaEventRecord(event: u64, stream: u64) -> i32;
    fn cudaEventSynchronize(event: u64) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: u64, stop: u64) -> i32;
}

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
static PROFILER: std::sync::OnceLock<Arc<Mutex<ProfilerState>>> = std::sync::OnceLock::new();

struct ProfilerState {
    enabled: bool,
    categories: HashMap<KernelCategory, CategoryStats>,
}

/// Initialize the profiler. Call once at startup.
pub fn init() {
    let _ = PROFILER.get_or_init(|| {
        Arc::new(Mutex::new(ProfilerState {
            enabled: true,
            categories: HashMap::new(),
        }))
    });
}

/// Start timing a kernel operation. Returns opaque handle to pass to stop().
pub fn start(category: KernelCategory) -> u64 {
    let ev = unsafe {
        let mut e = 0u64;
        cudaEventCreate(&mut e);
        cudaEventRecord(e, 0); // stream 0 = default
        e
    };
    init();
    let p = PROFILER.as_ref().unwrap();
    let mut state = p.lock().unwrap();
    let stats = state.categories.entry(category).or_default();
    stats.count += 1;
    ev
}

/// Stop timing and record elapsed time for the category.
pub fn stop(category: KernelCategory, start_event: u64) {
    unsafe {
        let mut end_event = 0u64;
        cudaEventCreate(&mut end_event);
        cudaEventRecord(end_event, 0);
        cudaEventSynchronize(end_event);

        let mut elapsed = 0.0f32;
        cudaEventElapsedTime(&mut elapsed, start_event, end_event);

        let p = PROFILER.as_ref().unwrap();
        let mut state = p.lock().unwrap();
        let stats = state.categories.entry(category).or_default();
        stats.total_ms += elapsed as f64;

        cudaEventDestroy(start_event);
        cudaEventDestroy(end_event);
    }
}

/// Print timing report to stdout. Call after generation completes.
pub fn print_report() {
    let p = PROFILER.as_ref().unwrap();
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
