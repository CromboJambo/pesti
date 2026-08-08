//! Memory backend abstraction for PESTI.
//!
//! MemoryBackend operates in bytes, not T. DeviceBuffer<T> is a thin typed view
//! over a RawHandle — the backend owns allocation/lifecycle, DeviceBuffer just
//! provides type info and bounds.
//!
//! Three-layer separation:
//! 1. MemoryBackend — byte-level allocate/free/transfer, operates on RawHandle
//! 2. DeviceBuffer<T> — RawHandle + element count, knows T for size/alignment
//! 3. TensorView<T> (future) — shape + stride on top of a DeviceBuffer<T>
//!
//! RawHandle is a u64 newtype. For CPU it's a slab index, for CUDA it's the
//! device pointer cast to u64. The backend impl knows how to interpret it.

use cudarc::driver::result::{self, DriverError};
use cudarc::driver::safe::CudaStream;
use cudarc::driver::sys;
use std::sync::Arc;
use std::sync::Mutex;

/// Opaque handle to memory managed by a MemoryBackend.
///
/// For CPU: index into the slab allocator.
/// For CUDA: device pointer cast to u64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RawHandle(pub u64);

impl RawHandle {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.0 as *mut std::ffi::c_void
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Error type for memory backend operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("allocation failed: requested {requested} bytes (max {max})")]
    AllocationFailed { requested: usize, max: usize },

    #[error("invalid handle: {0:?}")]
    InvalidHandle(RawHandle),

    #[error("transfer failed: {0}")]
    Transfer(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("sync failed: {0}")]
    Sync(String),
}

impl From<DriverError> for MemoryError {
    fn from(e: DriverError) -> Self {
        MemoryError::Cuda(format!("{e:?}"))
    }
}

/// Byte-level memory backend.
///
/// All operations work on raw bytes. DeviceBuffer<T> wraps a RawHandle
/// to provide typed access without the backend needing to know about T.
pub trait MemoryBackend: Send + Sync {
    /// Allocate `bytes` bytes. Returns a handle to the allocated memory.
    fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError>;

    /// Free previously allocated memory.
    fn free(&self, handle: RawHandle) -> Result<(), MemoryError>;

    /// Copy `src` host bytes to device memory at `dst`.
    fn h2d(&self, src: &[u8], dst: RawHandle) -> Result<(), MemoryError>;

    /// Copy `bytes` from device memory at `src` to host buffer `dst`.
    fn d2h(&self, src: RawHandle, dst: &mut [u8]) -> Result<(), MemoryError>;

    /// Copy `bytes` from device memory at `src` to device memory at `dst`.
    fn d2d(&self, src: RawHandle, dst: RawHandle, bytes: usize) -> Result<(), MemoryError>;

    /// Synchronize the backend (ensure all pending operations complete).
    fn sync(&self) -> Result<(), MemoryError>;
}

/// CPU-backed memory using a slab allocator over Vec<u8>.
///
/// Each allocation gets a slot in the slab. Freeing marks the slot
/// as available for reuse. Handles are u64 indices into the slab.
pub struct CpuMemoryBackend {
    slab: Mutex<Vec<SlabEntry>>,
    capacity: usize,
}

struct SlabEntry {
    allocated: bool,
    data: Vec<u8>,
}

impl CpuMemoryBackend {
    pub fn new(capacity: usize) -> Self {
        Self {
            slab: Mutex::new(Vec::new()),
            capacity,
        }
    }

    /// Total bytes allocated across all live allocations.
    pub fn used_bytes(&self) -> usize {
        self.slab
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.allocated)
            .map(|e| e.data.len())
            .sum()
    }
}

impl MemoryBackend for CpuMemoryBackend {
    fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError> {
        let mut slab = self.slab.lock().unwrap();

        // Try to reuse a free slot first
        if let Some(idx) = slab.iter().position(|e| !e.allocated) {
            slab[idx].allocated = true;
            slab[idx].data = vec![0u8; bytes];
            return Ok(RawHandle(idx as u64));
        }

        let total_used: usize = slab
            .iter()
            .filter(|e| e.allocated)
            .map(|e| e.data.len())
            .sum();
        if total_used + bytes > self.capacity {
            return Err(MemoryError::AllocationFailed {
                requested: bytes,
                max: self.capacity,
            });
        }

        let idx = slab.len();
        slab.push(SlabEntry {
            allocated: true,
            data: vec![0u8; bytes],
        });
        Ok(RawHandle(idx as u64))
    }

    fn free(&self, handle: RawHandle) -> Result<(), MemoryError> {
        let idx = handle.0 as usize;
        let mut slab = self.slab.lock().unwrap();
        if idx >= slab.len() || !slab[idx].allocated {
            return Err(MemoryError::InvalidHandle(handle));
        }
        slab[idx].allocated = false;
        Ok(())
    }

    fn h2d(&self, src: &[u8], dst: RawHandle) -> Result<(), MemoryError> {
        let idx = dst.0 as usize;
        let mut slab = self.slab.lock().unwrap();
        if idx >= slab.len() || !slab[idx].allocated {
            return Err(MemoryError::InvalidHandle(dst));
        }
        if src.len() > slab[idx].data.len() {
            return Err(MemoryError::Transfer(format!(
                "h2d: src {} > dst {}",
                src.len(),
                slab[idx].data.len()
            )));
        }
        slab[idx].data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn d2h(&self, src: RawHandle, dst: &mut [u8]) -> Result<(), MemoryError> {
        let idx = src.0 as usize;
        let slab = self.slab.lock().unwrap();
        if idx >= slab.len() || !slab[idx].allocated {
            return Err(MemoryError::InvalidHandle(src));
        }
        let copy_len = dst.len().min(slab[idx].data.len());
        dst[..copy_len].copy_from_slice(&slab[idx].data[..copy_len]);
        Ok(())
    }

    fn d2d(&self, src: RawHandle, dst: RawHandle, bytes: usize) -> Result<(), MemoryError> {
        let src_idx = src.0 as usize;
        let dst_idx = dst.0 as usize;
        let mut slab = self.slab.lock().unwrap();
        if src_idx >= slab.len() || !slab[src_idx].allocated {
            return Err(MemoryError::InvalidHandle(src));
        }
        if dst_idx >= slab.len() || !slab[dst_idx].allocated {
            return Err(MemoryError::InvalidHandle(dst));
        }
        let copy_len = bytes
            .min(slab[src_idx].data.len())
            .min(slab[dst_idx].data.len());
        let src_data = slab[src_idx].data[..copy_len].to_vec();
        slab[dst_idx].data[..copy_len].copy_from_slice(&src_data);
        Ok(())
    }

    fn sync(&self) -> Result<(), MemoryError> {
        // CPU is inherently synchronous
        Ok(())
    }
}

/// CUDA-backed memory using cudarc sys calls.
pub struct CudaMemoryBackend {
    stream: Arc<CudaStream>,
    device_info: crate::cuda_runtime::CudaDeviceInfo,
    enabled: bool,
}

impl CudaMemoryBackend {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        // Try to get device info from the stream's context
        let device_info = crate::cuda_runtime::CudaDeviceInfo {
            ordinal: 0,
            name: String::new(),
            compute_capability: (0, 0),
            total_memory: 0,
            free_memory: 0,
        };

        Self {
            stream,
            device_info,
            enabled: true,
        }
    }

    pub fn with_device_info(
        stream: Arc<CudaStream>,
        device_info: crate::cuda_runtime::CudaDeviceInfo,
    ) -> Self {
        Self {
            stream,
            device_info,
            enabled: true,
        }
    }

    pub fn device_info(&self) -> &crate::cuda_runtime::CudaDeviceInfo {
        &self.device_info
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Try to initialize device info from runtime
    pub fn try_init_device_info(&mut self) {
        if !self.enabled {
            return;
        }

        // Get device info from CUDA runtime
        match crate::cuda_runtime::CudaRuntime::for_default_device() {
            Ok(rt) => {
                let info = rt.device_info().clone();
                tracing::info!(
                    device = %info.name,
                    sm = format!("{}.{}", info.compute_capability.0, info.compute_capability.1),
                    free_gib = info.free_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                    total_gib = info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                    "CUDA device info"
                );
                self.device_info = info;
            }
            Err(e) => {
                tracing::warn!(error = %e, "Could not get device info");
            }
        }
    }
}

impl MemoryBackend for CudaMemoryBackend {
    fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError> {
        if !self.enabled {
            tracing::info!("CUDA backend disabled, falling back to host allocation");
            return Err(MemoryError::AllocationFailed {
                requested: bytes,
                max: 0,
            });
        }

        let total_mem = self.device_info.total_memory as usize;
        if total_mem == 0 {
            tracing::info!("Device memory unknown, using fallback");
            return Err(MemoryError::AllocationFailed {
                requested: bytes,
                max: 0,
            });
        }

        if bytes > total_mem {
            return Err(MemoryError::AllocationFailed {
                requested: bytes,
                max: total_mem,
            });
        }

        // Allocate device memory using sys call
        let mut dptr: sys::CUdeviceptr = 0;
        unsafe {
            sys::cuMemAlloc_v2(&mut dptr, bytes).result().map_err(|e| {
                tracing::warn!(error = %e, "cuMemAlloc_v2 failed");
                MemoryError::Cuda(format!("cuMemAlloc_v2: {e:?}"))
            })?;
        }

        Ok(RawHandle(dptr))
    }

    fn free(&self, handle: RawHandle) -> Result<(), MemoryError> {
        let ptr = handle.as_u64();
        if ptr == 0 {
            return Ok(());
        }
        unsafe {
            sys::cuMemFree_v2(ptr)
                .result()
                .map_err(|e| MemoryError::Cuda(format!("cuMemFree_v2 failed: {e:?}")))
        }
    }

    fn h2d(&self, src: &[u8], dst: RawHandle) -> Result<(), MemoryError> {
        let dst_ptr = dst.as_u64();
        unsafe {
            sys::cuMemcpyHtoD_v2(dst_ptr, src.as_ptr() as *const std::ffi::c_void, src.len())
                .result()
                .map_err(|e| MemoryError::Transfer(format!("H2D copy failed: {e:?}")))
        }
    }

    fn d2h(&self, src: RawHandle, dst: &mut [u8]) -> Result<(), MemoryError> {
        let src_ptr = src.as_u64();
        unsafe {
            sys::cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src_ptr,
                dst.len(),
            )
            .result()
            .map_err(|e| MemoryError::Transfer(format!("D2H copy failed: {e:?}")))
        }
    }

    fn d2d(&self, src: RawHandle, dst: RawHandle, bytes: usize) -> Result<(), MemoryError> {
        let src_ptr = src.as_u64();
        let dst_ptr = dst.as_u64();
        unsafe {
            sys::cuMemcpyDtoD_v2(dst_ptr, src_ptr, bytes)
                .result()
                .map_err(|e| MemoryError::Transfer(format!("D2D copy failed: {e:?}")))
        }
    }

    fn sync(&self) -> Result<(), MemoryError> {
        self.stream
            .synchronize()
            .map_err(|e| MemoryError::Cuda(format!("Stream sync failed: {e:?}")))
    }
}

/// Unified memory manager that picks the best available backend.
pub enum MemoryManager {
    /// CPU-only mode (no CUDA available).
    Cpu(CpuMemoryBackend),
    /// CUDA mode with GPU acceleration.
    Cuda(CudaMemoryBackend),
}

impl MemoryManager {
    /// Create a MemoryManager, preferring CUDA if available.
    pub fn new() -> Self {
        if crate::cuda_runtime::is_available() {
            if let Ok(rt) = crate::cuda_runtime::CudaRuntime::for_default_device() {
                if let Ok(stream) = rt.new_stream() {
                    let device_info = rt.device_info().clone();
                    return Self::Cuda(CudaMemoryBackend::with_device_info(stream, device_info));
                }
            }
        }
        Self::Cpu(CpuMemoryBackend::new(usize::MAX))
    }

    /// Try to create a MemoryManager with explicit CUDA.
    pub fn with_cuda(stream: Arc<CudaStream>) -> Self {
        Self::Cuda(CudaMemoryBackend::new(stream))
    }

    pub fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError> {
        match self {
            Self::Cpu(backend) => backend.alloc(bytes),
            Self::Cuda(backend) => backend.alloc(bytes),
        }
    }

    pub fn free(&self, handle: RawHandle) -> Result<(), MemoryError> {
        match self {
            Self::Cpu(backend) => backend.free(handle),
            Self::Cuda(backend) => backend.free(handle),
        }
    }

    pub fn h2d(&self, src: &[u8], dst: RawHandle) -> Result<(), MemoryError> {
        match self {
            Self::Cpu(backend) => backend.h2d(src, dst),
            Self::Cuda(backend) => backend.h2d(src, dst),
        }
    }

    pub fn d2h(&self, src: RawHandle, dst: &mut [u8]) -> Result<(), MemoryError> {
        match self {
            Self::Cpu(backend) => backend.d2h(src, dst),
            Self::Cuda(backend) => backend.d2h(src, dst),
        }
    }

    pub fn d2d(&self, src: RawHandle, dst: RawHandle, bytes: usize) -> Result<(), MemoryError> {
        match self {
            Self::Cpu(backend) => backend.d2d(src, dst, bytes),
            Self::Cuda(backend) => backend.d2d(src, dst, bytes),
        }
    }

    pub fn sync(&self) -> Result<(), MemoryError> {
        match self {
            Self::Cpu(backend) => backend.sync(),
            Self::Cuda(backend) => backend.sync(),
        }
    }

    /// Whether CUDA is available.
    pub fn has_cuda(&self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// Get device info if CUDA is available.
    pub fn device_info(&self) -> Option<&crate::cuda_runtime::CudaDeviceInfo> {
        match self {
            Self::Cuda(backend) => Some(backend.device_info()),
            Self::Cpu(_) => None,
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
