//! Local GPU device discovery.
//!
//! Enumerates available CUDA GPUs with per-device VRAM info via NVML.
//! Falls back to CUDA driver API if NVML is unavailable.
//!
//! TODO: Phase 2 — implement full GPU kernel path with tcgen05/WGMMA
//! TODO: Add compute capability detection via cuDeviceGetAttribute
//! TODO: Add persistence mode detection via cuDeviceGetAttribute
//! TODO: Add ECC error detection via cuDeviceGetEccStatus

use serde::{Deserialize, Serialize};
use std::os::raw::c_int;
use tracing::debug;

/// A discovered local GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDevice {
    /// CUDA device ordinal.
    pub ordinal: u32,
    /// GPU name (e.g., "NVIDIA GeForce RTX 4070").
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram: u64,
    /// Free VRAM in bytes (approximate, may change).
    pub free_vram: u64,
    /// Compute capability (major.minor).
    pub compute_capability: String,
    /// Whether the device is currently available.
    pub available: bool,
    /// Used VRAM in bytes (if detectable).
    pub used_vram: u64,
}

impl LocalDevice {
    /// Free VRAM as a percentage of total.
    pub fn free_vram_pct(&self) -> f32 {
        if self.total_vram == 0 {
            return 0.0;
        }
        (self.free_vram as f32 / self.total_vram as f32) * 100.0
    }

    /// Model size in bytes that can fit on this device.
    ///
    /// Conservative: reserves 2GiB for CUDA context and kernels.
    pub fn max_model_bytes(&self) -> u64 {
        let reserve = 2 * 1024 * 1024 * 1024u64;
        self.free_vram.saturating_sub(reserve)
    }

    /// Whether this device can hold a model of the given size.
    pub fn can_hold_model(&self, model_bytes: u64) -> bool {
        self.max_model_bytes() >= model_bytes
    }
}

/// Discover all local CUDA devices.
///
/// Returns an empty vec if no CUDA-capable GPU is available.
pub fn discover_local_devices() -> Vec<LocalDevice> {
    let mut devices = Vec::new();

    match init_cuda() {
        Ok(_) => {
            if let Ok(count) = get_device_count() {
                for ordinal in 0..count {
                    if let Some(device) = get_device_info(ordinal) {
                        let name = device.name.clone();
                        devices.push(device);
                        debug!(ordinal, name = %name, "Discovered local CUDA device");
                    }
                }
            }
        }
        Err(e) => {
            debug!(error = %e, "CUDA initialization failed, no local GPUs discovered");
        }
    }

    devices.push(LocalDevice {
        ordinal: u32::MAX,
        name: "CPU (fallback)".to_string(),
        total_vram: 0,
        free_vram: 0,
        compute_capability: "N/A".to_string(),
        available: true,
        used_vram: 0,
    });

    debug!(count = devices.len(), "Local device discovery complete");
    devices
}

/// Initialize CUDA driver API.
fn init_cuda() -> Result<(), String> {
    cudarc::driver::result::init().map_err(|e| format!("CUDA init failed: {e}"))
}

/// Get the number of CUDA devices.
fn get_device_count() -> Result<c_int, String> {
    cudarc::driver::result::device::get_count().map_err(|e| format!("Device count failed: {e}"))
}

/// Get info for a specific CUDA device.
fn get_device_info(ordinal: i32) -> Option<LocalDevice> {
    let dev = match cudarc::driver::result::device::get(ordinal) {
        Ok(d) => d,
        Err(_) => return None,
    };

    let name = match cudarc::driver::result::device::get_name(dev) {
        Ok(n) => n,
        Err(_) => return None,
    };

    // TODO: Phase 2 — get compute capability via cuDeviceGetAttribute
    let compute_capability = "stubbed".to_string();

    // Use NVML for per-device VRAM info (cuMemGetInfo is global, not per-device).
    let (total_vram, free_vram) = match get_nvml_memory(ordinal) {
        Some((total, free)) => (total, free),
        None => {
            debug!(
                ordinal,
                "NVML unavailable, using global cuMemGetInfo as fallback"
            );
            match cudarc::driver::result::mem_get_info() {
                Ok((total, free)) => (total as u64, free as u64),
                Err(_) => (0, 0),
            }
        }
    };

    let used_vram = total_vram.saturating_sub(free_vram);

    // TODO: Phase 2 — check persistence mode via cuDeviceGetAttribute
    let available = true;

    Some(LocalDevice {
        ordinal: ordinal as u32,
        name,
        total_vram,
        free_vram,
        compute_capability,
        available,
        used_vram,
    })
}

/// Get per-device VRAM info from NVML.
///
/// Returns (total_bytes, free_bytes) or None if NVML is unavailable.
fn get_nvml_memory(ordinal: i32) -> Option<(u64, u64)> {
    use nvml_wrapper::Nvml;

    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return None,
    };

    // Get device by index (ordinal)
    let dev = match nvml.device_by_index(ordinal as u32) {
        Ok(d) => d,
        Err(_) => return None,
    };

    // Get memory info for this device
    match dev.memory_info() {
        Ok(mem_info) => Some((mem_info.total, mem_info.free)),
        Err(_) => None,
    }
}

/// Get the best available local GPU for inference.
///
/// Returns the device with the most free VRAM that can hold the model,
/// or None if no GPU can fit the model.
pub fn select_best_gpu(model_bytes: u64) -> Option<LocalDevice> {
    let devices = discover_local_devices();
    let gpus: Vec<_> = devices
        .into_iter()
        .filter(|d| d.ordinal != u32::MAX)
        .filter(|d| d.can_hold_model(model_bytes))
        .collect();

    gpus.into_iter().max_by_key(|d| d.free_vram)
}

