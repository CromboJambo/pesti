//! Tiered execution infrastructure for profile-driven optimization paths.
//!
//! Inspired by lumen's tier-0→tier-1→tier-2 model, but optimized for GPU-first workflows:
//! - **Tier 1 (CPU baseline)**: Pure-Rust transformer kernels (correctness oracle)
//! - **Tier 2 (GPU flash attention)**: cuda-oxide WGMMA + tcgen05 optimizations
//! - **Tier 3 (llama.cpp FFI CUDA)**: Full model offload to optimized backend

use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::debug;

/// Execution tier with profile-driven tier-up thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// CPU baseline — generic kernels, always available
    CpuBaseline = 0,
    /// GPU flash attention — cuda-oxide WGMMA kernels, requires VRAM
    GpuFlashAttention = 1,
    /// Full CUDA backend — llama.cpp FFI with optimized kernels
    GpuFullBackend = 2,
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuBaseline => "cpu_baseline",
            Self::GpuFlashAttention => "gpu_flash_attention",
            Self::GpuFullBackend => "gpu_full_backend",
        }
    }

    /// Is this tier GPU-accelerated?
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::GpuFlashAttention | Self::GpuFullBackend)
    }

    /// Get the default tier-up threshold (invocations before tiering up).
    pub fn tier_up_threshold(&self) -> usize {
        match self {
            Self::CpuBaseline => 100,      // Tier up after 100 invocations
            Self::GpuFlashAttention => 500, // Stay at GPU attention until 500
            Self::GpuFullBackend => usize::MAX, // Never tier up (already max)
        }
    }

    /// Convert from numeric ID.
    pub fn from_id(id: u32) -> Tier {
        match id {
            0 => Tier::CpuBaseline,
            1 => Tier::GpuFlashAttention,
            _ => Tier::GpuFullBackend,
        }
    }

    /// Get numeric ID.
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }

    /// Get tier-up threshold by numeric ID (for helper functions).
    fn threshold_by_id(id: u32) -> usize {
        match id {
            0 => 100,      // CPU baseline → GPU flash attention
            1 => 500,      // GPU flash attention → full backend
            _ => usize::MAX, // Already at max tier
        }
    }
}

/// Profile-driven execution state with call-count tracking.
pub struct TieredExecution {
    current_tier: AtomicUsize, // Numeric ID (u32 cast to usize for atomic ops)
    call_count: AtomicUsize,   // Cumulative invocations per tier
    tier_up_threshold: usize,  // Threshold for tier-up decision
}

impl TieredExecution {
    pub fn new(initial_tier: Tier) -> Self {
        Self {
            current_tier: AtomicUsize::new(initial_tier.as_u32() as usize),
            call_count: AtomicUsize::new(0),
            tier_up_threshold: initial_tier.tier_up_threshold(),
        }
    }

    /// Get the current execution tier.
    pub fn current_tier(&self) -> Tier {
        let val = self.current_tier.load(Ordering::Relaxed);
        Tier::from_id(val as u32)
    }

    /// Record a layer invocation and check if tier-up is needed.
    pub fn record_invocation(&self) -> Option<Tier> {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Check if we should tier up based on threshold for current tier
        let current_tier_id = self.current_tier.load(Ordering::Relaxed);
        let threshold = Tier::threshold_by_id(current_tier_id as u32);

        debug!(
            invocation_count = count,
            threshold = threshold,
            "Tiered execution: recorded invocation"
        );

        if count >= threshold {
            // Tier up logic (simplified for Phase 5.3 MVP)
            let new_tier_id = current_tier_id + 1;

            // Reset call counter on tier transition
            self.call_count.store(0, Ordering::Relaxed);
            self.current_tier.store(new_tier_id, Ordering::Relaxed);

            debug!(
                from_tier = ?Tier::from_id(current_tier_id as u32),
                to_tier = ?Tier::from_id(new_tier_id as u32),
                "Tiered execution: tier-up triggered"
            );

            Some(Tier::from_id(new_tier_id as u32))
        } else {
            None
        }
    }

    /// Manually set the current tier (for testing or forced switching).
    pub fn set_tier(&self, tier: Tier) {
        self.current_tier
            .store(tier.as_u32() as usize, Ordering::Relaxed);
        self.call_count.store(0, Ordering::Relaxed); // Reset counter on tier change
    }

    /// Get the invocation count for current tier.
    pub fn invocation_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }

    /// Check if GPU execution is available (for tier-up decision).
    pub fn has_gpu_available(&self) -> bool {
        // Phase 5.3 MVP: assume GPU always available
        // TODO: integrate with DeviceSelector for real VRAM checks
        true
    }

    /// Reset profile counters (e.g., after model unload or session timeout).
    pub fn reset_profile(&self) {
        self.call_count.store(0, Ordering::Relaxed);
    }
}

/// Layer-level profiling hook for tier-up decisions.
pub struct LayerProfiler {
    layer_id: String,
    call_count: AtomicUsize,
}

impl LayerProfiler {
    pub fn new(layer_name: &str) -> Self {
        Self {
            layer_id: layer_name.to_string(),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Record a forward pass invocation for this layer.
    pub fn record_forward(&self) -> usize {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;

        if count % 50 == 0 { // Log every 50 invocations
            debug!(layer = %self.layer_id, count = count, "Layer profiler: invocation recorded");
        }

        count
    }

    /// Get current invocation count.
    pub fn count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

