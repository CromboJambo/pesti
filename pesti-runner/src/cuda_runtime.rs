//! CUDA runtime: context management, device enumeration, compute capability detection.
//!
//! Wraps cuda-oxide's `CudaContext` to provide a stable interface for the inference
//! engine's GPU path. Handles initialization, device discovery, and error propagation.

use cuda_core::{CudaContext, IntoResult};
use std::sync::Arc;
use tracing::{debug, warn};

// Re-export cuda_bindings through cuda_core
use cuda_core::sys as cuda_sys;

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

    /// Whether this device supports tcgen05 (sm_100a/sm_103a, datacenter
    /// Blackwell B200/B300). Consumer Blackwell (sm_120, RTX 50-series) has
    /// no tensor memory and no tcgen05 — ptxas rejects it for sm_120a.
    pub fn supports_tcgen05(&self) -> bool {
        let (major, minor) = self.compute_capability;
        (major, minor) == (10, 0) || (major, minor) == (10, 3)
    }

    /// Whether this device supports WGMMA (sm_90a, Hopper H100/H200 only).
    /// Consumer Blackwell (sm_120) does NOT support wgmma — ptxas rejects
    /// `wgmma.mma_async` for sm_120a; llama.cpp uses mma.sync there instead.
    pub fn supports_wgmma(&self) -> bool {
        let (major, minor) = self.compute_capability;
        (major, minor) == (9, 0)
    }
}

/// A live CUDA context for a specific device.
///
/// Wraps `Arc<CudaContext>` and tracks the device ordinal for routing.
#[derive(Debug, Clone)]
pub struct CudaRuntime {
    /// The underlying cuda-oxide context.
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
        // Initialize CUDA driver
        unsafe {
            cuda_core::init(0).map_err(|e| CudaError::NotInitialized(e.to_string()))?;
        };

        // Get device handle
        let cu_device = unsafe {
            let mut device = std::mem::MaybeUninit::uninit();
            cuda_sys::cuDeviceGet(device.as_mut_ptr(), ordinal as i32)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            device.assume_init()
        };

        // Get device name
        let mut name_buf = [0i8; 256];
        unsafe {
            cuda_sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, cu_device)
        };
        let name: String = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();

        // Get compute capability
        let mut major = std::mem::MaybeUninit::uninit();
        let mut minor = std::mem::MaybeUninit::uninit();
        unsafe {
            cuda_sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            cuda_sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
        }
        let (major, minor) = unsafe { (major.assume_init(), minor.assume_init()) };

        // Retain primary context and bind it to this thread BEFORE querying
        // device memory: cuMemGetInfo_v2 requires a current context.
        let ctx =
            CudaContext::new(ordinal).map_err(|e| CudaError::ContextCreation(e.to_string()))?;

        // Get memory info (valid only after a context is current on this thread)
        let (free_memory, total_memory) = unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            cuda_sys::cuMemGetInfo_v2(&mut free, &mut total)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            (free as u64, total as u64)
        };

        let device_info = CudaDeviceInfo {
            ordinal,
            name: name.clone(),
            compute_capability: (major, minor),
            total_memory,
            free_memory,
        };

        debug!(
            ordinal,
            name = %device_info.name,
            cc = "%d.%d", major, minor,
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

    /// Returns the underlying cuda-oxide context.
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
    pub fn new_stream(&self) -> Result<Arc<cuda_core::CudaStream>, CudaError> {
        self.ctx
            .new_stream()
            .map_err(|e| CudaError::ContextCreation(e.to_string()))
    }

    /// Synchronize the context (blocks until all pending work completes).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx
            .synchronize()
            .map_err(|e| CudaError::ContextCreation(e.to_string()))
    }

    /// Check if this runtime is still valid (context not destroyed).
    pub fn is_valid(&self) -> bool {
        !self.ctx.cu_ctx().is_null()
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
    
    // Fallback to driver API if NVML fails or returns empty
    #[cfg(feature = "cuda")]
    {
        unsafe {
            cuda_core::init(0).map_err(|_| CudaError::NotAvailable)?;
        };

        let mut device_count: i32 = 0;
        unsafe {
            cuda_sys::cuDeviceGetCount(&mut device_count)
                .result()
                .map_err(|_| CudaError::NotAvailable)?;
        };

        if device_count == 0 {
            return Ok(Vec::new());
        }

        let mut devices = Vec::with_capacity(device_count as usize);

        for ordinal in 0..device_count {
            // Get device handle
            let cu_device = match unsafe {
                let mut device = std::mem::MaybeUninit::uninit();
                cuda_sys::cuDeviceGet(device.as_mut_ptr(), ordinal)
                    .result()
                    .map_err(|_| CudaError::DeviceUnavailable {
                        ordinal: ordinal as usize,
                    })?;
                Ok::<i32, CudaError>(device.assume_init())
            } {
                Ok(d) => d,
                Err(e) => {
                    warn!(ordinal, "CUDA device enumeration skipped: {e}");
                    continue;
                }
            };

            // Get device name
            let mut name_buf = [0i8; 256];
            unsafe {
                cuda_sys::cuDeviceGetName(name_buf.as_mut_ptr(), name_buf.len() as i32, cu_device)
            };
            let name: String = name_buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect::<Vec<u8>>()
                .into_iter()
                .map(|b| b as char)
                .collect();

            // Get compute capability
            let (major, minor) = {
                let mut m = std::mem::MaybeUninit::uninit();
                let mut n = std::mem::MaybeUninit::uninit();
                unsafe {
                    cuda_sys::cuDeviceGetAttribute(
                        m.as_mut_ptr(),
                        cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                        cu_device,
                    )
                    .result()
                    .map_err(|_| CudaError::DeviceUnavailable { ordinal: ordinal as usize })?;
                    cuda_sys::cuDeviceGetAttribute(
                        n.as_mut_ptr(),
                        cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                        cu_device,
                    )
                    .result()
                    .map_err(|_| CudaError::DeviceUnavailable { ordinal: ordinal as usize })?;
                    (m.assume_init(), n.assume_init())
                }
            };

            // Get memory info
            let (free_memory, total_memory) = unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                cuda_sys::cuMemGetInfo_v2(&mut free, &mut total)
                    .result()
                    .map_err(|_| CudaError::DeviceUnavailable {
                        ordinal: ordinal as usize,
                    })?;
                (free as u64, total as u64)
            };

            devices.push(CudaDeviceInfo {
                ordinal: ordinal as usize,
                name,
                compute_capability: (major, minor),
                total_memory,
                free_memory,
            });
        }

        Ok(devices)
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        Ok(Vec::new())
    }
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
    
    // Fallback to driver API if NVML fails
    enumerate_devices().is_ok() && !enumerate_devices().unwrap_or_default().is_empty()
}

/// Get the number of CUDA devices.
pub fn device_count() -> usize {
    let mut count: i32 = 0;
    match unsafe { cuda_sys::cuDeviceGetCount(&mut count).result() } {
        Ok(_) => count as usize,
        Err(_) => 0,
    }
}

// =============================================================================
// Memory Allocation Wrappers (Phase 2)
// =============================================================================

/// Allocate device memory.
///
/// Wraps `cuMemAlloc_v2` to provide a safe interface for GPU buffer allocation.
pub fn allocate_device_memory(size_in_bytes: usize) -> Result<*mut u8, CudaError> {
    let mut dptr: u64 = 0;
    unsafe {
        cuda_sys::cuMemAlloc_v2(&mut dptr, size_in_bytes)
            .result()
            .map_err(|e| CudaError::DriverError(format!("{:?}", e)))?;
        Ok(dptr as *mut u8)
    }
}

/// Free device memory.
///
/// Wraps `cuMemFree_v2` to release previously allocated GPU buffers.
pub fn free_device_memory(ptr: *mut u8) -> Result<(), CudaError> {
    unsafe {
        cuda_sys::cuMemFree_v2(ptr as u64)
            .result()
            .map_err(|e| CudaError::DriverError(format!("{:?}", e)))?;
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
    unsafe {
        cuda_sys::cuMemcpyHtoD_v2(
            dst_device_ptr as u64,
            src_host_ptr as *const std::os::raw::c_void,
            size_in_bytes,
        )
        .result()
        .map_err(|e| CudaError::DriverError(format!("{:?}", e)))?;
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
    unsafe {
        cuda_sys::cuMemcpyDtoH_v2(
            dst_host_ptr as *mut std::os::raw::c_void,
            src_device_ptr as u64,
            size_in_bytes,
        )
        .result()
        .map_err(|e| CudaError::DriverError(format!("{:?}", e)))?;
        Ok(())
    }
}
