//! CUDA runtime: context management, device enumeration, compute capability detection.
//!
//! Wraps `cudarc` to provide a stable interface for the inference
//! engine's GPU path. Handles initialization, device discovery, and error propagation.
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::driver::sys;
use std::sync::Arc;
use tracing::warn;

/// Error type for CUDA runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver not initialized: {0}")]
    NotInitialized(String),

    #[error("CUDA device unavailable: ordinal={ordinal}")]
    DeviceUnavailable { ordinal: usize },

    #[error("CUDA context creation failed: {0}")]
    ContextCreation(String),

    #[error("CUDA compute capability unsupported: sm_{major}.{minor} < sm_100")]
    ComputeCapabilityUnsupported { major: i32, minor: i32 },

    #[error("CUDA library load failed: {0}")]
    LibraryLoad(String),

    #[error("CUDA error: {0}")]
    DriverError(String),

    #[error("CUDA not available on this system")]
    NotAvailable,
}

impl From<cudarc::driver::result::DriverError> for CudaError {
    fn from(e: cudarc::driver::result::DriverError) -> Self {
        CudaError::DriverError(format!("{e:?}"))
    }
}

/// Information about a single CUDA device.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Zero-based device ordinal.
    pub ordinal: usize,
    /// Device name (e.g., "NVIDIA GeForce RTX 5060 Ti").
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (i32, i32),
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Free device memory in bytes.
    pub free_memory: u64,
}

impl CudaDeviceInfo {
    /// Whether this device can hold a model of the given size.
    pub fn can_hold_model(&self, model_bytes: u64) -> bool {
        // Reserve 2 GiB for overhead (KV cache, intermediate buffers, PTX JIT)
        self.free_memory > model_bytes + 2 * 1024 * 1024 * 1024
    }

    /// Default info for consumer Blackwell RTX 50-series GPUs (sm_12.0).
    pub fn default_for_consumer() -> Self {
        Self {
            ordinal: 0,                                     // default device
            name: "NVIDIA GeForce RTX 5060 Ti".to_string(), // example model
            compute_capability: (12, 0),                    // sm_12.0 for Blackwell consumer GPUs
            total_memory: 8 * 1024 * 1024 * 1024,           // 8 GiB typical
            free_memory: 7 * 1024 * 1024 * 1024,            // ~7 GiB available
        }
    }

    /// Whether this device supports tcgen05 (sm_100a/sm_103a, datacenter Blackwell B200/B300).
    /// Consumer Blackwell (sm_120, RTX 50-series) has no tensor memory and no tcgen05 — ptxas rejects it for sm_120a.
    /// Ada Lovelace (sm_8.9, RTX 40-series) uses mma.sync tensor cores instead of tcgen05.
    pub fn supports_tcgen05(&self) -> bool {
        let (major, minor) = self.compute_capability;
        // Blackwell datacenter: sm_10.0 or sm_10.3
        (major, minor) == (10, 0) || (major, minor) == (10, 3)
    }

    /// Whether this device supports Ada Lovelace tensor cores (sm_8.9).
    /// Uses mma.sync instructions instead of tcgen05.
    pub fn supports_adalovelace_tensor_cores(&self) -> bool {
        let (major, minor) = self.compute_capability;
        // Ada Lovelace: sm_8.9 (RTX 40-series consumer GPUs)
        (major, minor) == (8, 9)
    }

    /// Whether this device supports WGMMA (sm_90a, Hopper H100/H200 only).
    /// Consumer Blackwell (sm_120) does NOT support wgmma — ptxas rejects
    /// `wgmma.mma_async` for sm_120a; llama.cpp uses mma.sync there instead.
    pub fn supports_wgmma(&self) -> bool {
        let (major, minor) = self.compute_capability;
        (major, minor) == (9, 0)
    }
}

/// A live CUDA runtime for a specific device.
///
/// Wraps `Arc<CudaContext>` and tracks the device ordinal for routing.
#[derive(Debug, Clone)]
pub struct CudaRuntime {
    /// The underlying cudarc context.
    ctx: Arc<CudaContext>,
    /// Device ordinal.
    ordinal: usize,
    /// Device info (cached at creation).
    device_info: CudaDeviceInfo,
}

impl CudaRuntime {
    /// Create a new CUDA runtime for the device at `ordinal`.
    ///
    /// Initializes the CUDA driver, obtains the primary context, and queries
    /// device properties. Returns `CudaError::NotAvailable` if the device
    /// cannot be found or the driver fails to initialize.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        // Create context for the specified device
        let ctx =
            CudaContext::new(ordinal).map_err(|e| CudaError::ContextCreation(format!("{e:?}")))?;

        // Get device name
        let name = ctx
            .name()
            .map_err(|e| CudaError::DeviceUnavailable { ordinal })?;

        // Get compute capability
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| CudaError::DeviceUnavailable { ordinal })?;

        // Get memory info
        let (free_memory, total_memory) = ctx
            .mem_get_info()
            .map_err(|e| CudaError::DeviceUnavailable { ordinal })?;

        let device_info = CudaDeviceInfo {
            ordinal,
            name: name.clone(),
            compute_capability: (major, minor),
            total_memory: total_memory as u64,
            free_memory: free_memory as u64,
        };

        tracing::debug!(
            ordinal,
            name = %device_info.name,
            cc = format!("{}.{}", major, minor),
            "CUDA runtime: initialized device"
        );

        Ok(Self {
            ctx,
            ordinal,
            device_info,
        })
    }

    /// Create a CUDA runtime for the first available device (ordinal 0).
    pub fn for_default_device() -> Result<Self, CudaError> {
        Self::new(0)
    }

    /// Returns the underlying cudarc context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns cached device info.
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    /// Create a new non-blocking stream in this context.
    pub fn new_stream(&self) -> Result<Arc<CudaStream>, CudaError> {
        self.ctx
            .new_stream()
            .map_err(|e| CudaError::ContextCreation(format!("{e:?}")))
    }

    /// Synchronize the context (blocks until all pending work completes).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // cudarc streams handle synchronization
        self.ctx
            .synchronize()
            .map_err(|e| CudaError::ContextCreation(format!("{e:?}")))
    }

    /// Check if this runtime is still valid (context not destroyed).
    pub fn is_valid(&self) -> bool {
        // cudarc handles this via Arc strong count
        Arc::strong_count(&self.ctx) > 0
    }
}

/// Estimate compute capability from device name.
fn estimate_compute_capability_from_name(name: &str) -> (i32, i32) {
    let name_lower = name.to_lowercase();

    if name_lower.contains("50") {
        // RTX 50 series - likely sm_12+
        (12, 0)
    } else if name_lower.contains("40") {
        // RTX 40 series - Ada Lovelace = sm_8.9
        (8, 9)
    } else if name_lower.contains("30") {
        // RTX 30 series - Ampere = sm_8.6
        (8, 6)
    } else if name_lower.contains("20") {
        // RTX 20 series - Turing = sm_7.5
        (7, 5)
    } else if name_lower.contains("10") {
        // GTX 10 series - Pascal = sm_6.1
        (6, 1)
    } else {
        // Unknown, default to conservative estimate
        (8, 0)
    }
}

/// Initialize CUDA and enumerate available devices.
///
/// Returns a list of `CudaDeviceInfo` for all devices that can be queried.
/// Returns an empty list if CUDA is not available or no devices are found.
pub fn enumerate_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
    // First try NVML (more reliable when context is in use)
    #[cfg(feature = "cuda")]
    {
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(device_count) = nvml.device_count() {
                if device_count > 0 {
                    let mut devices = Vec::with_capacity(device_count as usize);

                    for ordinal in 0..device_count as usize {
                        if let Ok(device) = nvml.device_by_index(ordinal as u32) {
                            // Get name
                            if let Ok(name) = device.name() {
                                // Get memory info
                                if let Ok(mem_info) = device.memory_info() {
                                    let total_memory = mem_info.total;
                                    let free_memory = mem_info.free;

                                    // Estimate compute capability from name
                                    let cc = estimate_compute_capability_from_name(&name);

                                    devices.push(CudaDeviceInfo {
                                        ordinal,
                                        name: name.to_string(),
                                        compute_capability: cc,
                                        total_memory,
                                        free_memory,
                                    });
                                }
                            }
                        }
                    }

                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }
        }
    }

    // Fallback to cudarc context API
    #[cfg(feature = "cuda")]
    {
        let mut devices = Vec::new();
        let mut ordinal = 0;

        loop {
            match CudaContext::new(ordinal) {
                Ok(ctx) => {
                    let name = ctx.name().unwrap_or_default();
                    let cc = ctx.compute_capability().unwrap_or((0, 0));
                    let (free, total) = ctx.mem_get_info().unwrap_or((0, 0));

                    devices.push(CudaDeviceInfo {
                        ordinal,
                        name,
                        compute_capability: cc,
                        total_memory: total as u64,
                        free_memory: free as u64,
                    });
                    ordinal += 1;
                }
                Err(_) => break,
            }
        }

        if !devices.is_empty() {
            return Ok(devices);
        }
    }

    Ok(Vec::new())
}

/// Select the best device for a model of the given size.
///
/// Prioritizes:
/// 1. Devices with enough free VRAM
/// 2. Higher compute capability (tcgen05 > WGMMA > CPU)
/// 3. More free memory as tiebreaker
pub fn select_best_device(model_bytes: u64) -> Option<CudaDeviceInfo> {
    let devices = enumerate_devices().unwrap_or_default();

    let mut candidates: Vec<&CudaDeviceInfo> = devices
        .iter()
        .filter(|d| d.can_hold_model(model_bytes))
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Sort by: tcgen05 support > WGMMA support > free memory descending
    candidates.sort_by(|a, b| {
        let a_tc = if a.supports_tcgen05() { 3 } else { 0 };
        let a_wg = if a.supports_wgmma() { 2 } else { 0 };
        let b_tc = if b.supports_tcgen05() { 3 } else { 0 };
        let b_wg = if b.supports_wgmma() { 2 } else { 0 };

        let a_score = a_tc + a_wg + (a.free_memory as i64 / (1024 * 1024 * 1024));
        let b_score = b_tc + b_wg + (b.free_memory as i64 / (1024 * 1024 * 1024));

        b_score.cmp(&a_score)
    });

    candidates.first().cloned().cloned()
}

/// Check if CUDA is available on this system.
///
/// Uses NVML (NVIDIA Management Library) for more reliable device detection,
/// especially when another process already has the CUDA context.
pub fn is_available() -> bool {
    // First try NVML (more reliable when context is in use)
    #[cfg(feature = "cuda")]
    {
        if let Ok(nvml) = nvml_wrapper::Nvml::init() {
            if let Ok(count) = nvml.device_count() {
                if count > 0 {
                    return true;
                }
            }
        }
    }

    // Fallback to cudarc
    enumerate_devices().map(|d| !d.is_empty()).unwrap_or(false)
}

/// Get the number of CUDA devices.
pub fn device_count() -> usize {
    let mut count: i32 = 0;
    let result = unsafe { sys::cuDeviceGetCount(&mut count) };
    if result == sys::CUresult::CUDA_SUCCESS {
        count as usize
    } else {
        0
    }
}

// =============================================================================
// Memory Allocation Wrappers (Phase 2)
// =============================================================================

/// Allocate device memory.
///
/// Wraps `cuMemAlloc_v2` to provide a safe interface for GPU buffer allocation.
pub fn allocate_device_memory(size_in_bytes: usize) -> Result<*mut u8, CudaError> {
    let mut dptr: sys::CUdeviceptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut dptr, size_in_bytes)
            .result()
            .map_err(|e| CudaError::DriverError(format!("{e:?}")))?;
        Ok(dptr as *mut u8)
    }
}

/// Free device memory.
///
/// Wraps `cuMemFree_v2` to release previously allocated GPU buffers.
pub fn free_device_memory(ptr: *mut u8) -> Result<(), CudaError> {
    unsafe {
        sys::cuMemFree_v2(ptr as sys::CUdeviceptr)
            .result()
            .map_err(|e| CudaError::DriverError(format!("{e:?}")))?;
        Ok(())
    }
}

/// Copy data from host to device.
///
/// Wraps `cuMemcpyHtoD_v2` for synchronous H2D transfers.
pub fn copy_host_to_device(
    dst_device_ptr: *mut u8,
    src_host_ptr: *const u8,
    size_in_bytes: usize,
) -> Result<(), CudaError> {
    use std::ffi::c_void;
    unsafe {
        sys::cuMemcpyHtoD_v2(
            dst_device_ptr as sys::CUdeviceptr,
            src_host_ptr as *const c_void,
            size_in_bytes,
        )
        .result()
        .map_err(|e| CudaError::DriverError(format!("{e:?}")))?;
        Ok(())
    }
}

/// Copy data from device to host.
///
/// Wraps `cuMemcpyDtoH_v2` for synchronous D2H transfers.
pub fn copy_device_to_host(
    dst_host_ptr: *mut u8,
    src_device_ptr: *const u8,
    size_in_bytes: usize,
) -> Result<(), CudaError> {
    use std::ffi::c_void;
    unsafe {
        sys::cuMemcpyDtoH_v2(
            dst_host_ptr as *mut c_void,
            src_device_ptr as sys::CUdeviceptr,
            size_in_bytes,
        )
        .result()
        .map_err(|e| CudaError::DriverError(format!("{e:?}")))?;
        Ok(())
    }
}

/// Trait extension for CUresult to provide `.result()` method.
/// This mirrors cuda-oxide's `IntoResult` trait.
/// Re-exported from cuda_shim.
pub use crate::cuda_shim::IntoResult;
