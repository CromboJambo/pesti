//! Device buffer — a thin typed view over a RawHandle.
//!
//! The MemoryBackend owns allocation and lifecycle. DeviceBuffer<T> just
//! holds a RawHandle + element count, providing type info and size computation.
//!
//! ```text
//! MemoryBackend (owns) → RawHandle → DeviceBuffer<T> (typed view)
//! ```
//!
//! Usage pattern:
//! ```text
//! let handle = backend.alloc(N * size_of::<T>())?;
//! let buf = DeviceBuffer::from_backend(handle, N);
//! // ... use buf ...
//! backend.free(handle)?;
//! ```

#[cfg(feature = "cuda")]
use crate::kernel::memory::{MemoryBackend, MemoryError, MemoryManager, RawHandle};

#[cfg(not(feature = "cuda"))]
use crate::kernel::memory_stub::{MemoryBackend, MemoryError, MemoryManager, RawHandle};
use std::marker::PhantomData;

/// Error type for device buffer operations.
#[derive(Debug, thiserror::Error)]
pub enum DeviceBufferError {
    #[error("CUDA context not available")]
    NoContext,

    #[error("CUDA allocation failed: {0}")]
    Allocation(String),

    #[error("CUDA transfer failed: {0}")]
    Transfer(String),

    #[error("buffer size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
}

/// Host-backed buffer for data that lives on the CPU side.
///
/// Used for backward compatibility and CPU-only mode.
#[derive(Clone)]
pub struct HostBuffer<T>(pub Vec<T>);

/// Thin typed view over a RawHandle managed by a MemoryBackend.
///
/// `DeviceBuffer<T>` does NOT own the underlying memory — the backend does.
/// It just provides type information (via PhantomData<T>) and element count.
///
/// For CPU mode, the RawHandle is a slab index. For CUDA mode, it's the
/// device pointer cast to u64. The backend impl knows how to interpret it.
///
/// # CPU-only mode (no backend)
///
/// `DeviceBuffer::zeros()` and `DeviceBuffer::from_host()` allocate
/// host memory directly for use when no backend is available. These
/// are for convenience/testing — they don't go through a MemoryBackend.
#[derive(Clone)]
pub struct DeviceBuffer<T> {
    handle: RawHandle,
    len: usize,
    _marker: PhantomData<T>,
    /// Whether this buffer was allocated via a backend (true) or
    /// via host convenience methods (false).
    backed: bool,
    /// For host convenience buffers (backed=false), stores the data.
    host_data: Option<Vec<T>>,
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        // If this buffer was allocated via a backend, free it.
        // For host-backed convenience buffers, nothing to free.
        if !self.backed {
            return;
        }
        // Note: in the current design, the caller is responsible for
        // calling backend.free() explicitly. We don't free on drop here
        // because the backend may be shared and we don't want to accidentally
        // free memory that's still in use by other DeviceBuffers.
        // The backed flag exists to distinguish backend-allocated from
        // host-allocated buffers for the caller.
    }
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Send> Sync for DeviceBuffer<T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("handle", &self.handle)
            .field("len", &self.len)
            .finish()
    }
}

// --- HostBuffer ---

impl<T> HostBuffer<T> {
    pub fn from_host(data: Vec<T>) -> Self {
        Self(data)
    }

    pub fn zeros(len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self(vec![T::default(); len])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn to_host(self) -> Vec<T> {
        self.0
    }

    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T: Default + Clone> From<Vec<T>> for HostBuffer<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_host(v)
    }
}

impl<T: Default + Clone> From<HostBuffer<T>> for Vec<T> {
    fn from(buf: HostBuffer<T>) -> Self {
        buf.0
    }
}

// --- DeviceBuffer ---

impl<T> DeviceBuffer<T> {
    /// Create a DeviceBuffer from a handle previously allocated by a backend.
    pub fn from_backend(handle: RawHandle, len: usize) -> Self {
        Self {
            handle,
            len,
            _marker: PhantomData,
            backed: true,
            host_data: None,
        }
    }

    /// Get the underlying RawHandle.
    pub fn handle(&self) -> RawHandle {
        self.handle
    }

    /// Get the element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte length of the buffer (len * size_of::<T>).
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Whether this buffer was allocated via a backend.
    pub fn is_backed(&self) -> bool {
        self.backed
    }

    /// Create a device buffer from a raw pointer address and element count.
    ///
    /// # Safety
    ///
    /// `ptr_addr` must point to valid memory with at least
    /// `len * size_of::<T>()` bytes.
    pub unsafe fn from_device(ptr_addr: u64, len: usize) -> Self {
        Self {
            handle: RawHandle(ptr_addr),
            len,
            _marker: PhantomData,
            backed: false,
            host_data: None,
        }
    }

    /// Allocate on the given backend and copy host data to it.
    pub fn from_host_device<B: MemoryBackend>(
        backend: &B,
        data: &[T],
    ) -> Result<Self, DeviceBufferError>
    where
        T: Copy,
    {
        let bytes = data.len() * std::mem::size_of::<T>();
        let handle = backend.alloc(bytes).map_err(|e| {
            DeviceBufferError::Allocation(format!("alloc {} bytes: {e}", bytes))
        })?;

        // Convert data to bytes and copy to device
        let src_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        };
        backend.h2d(src_bytes, handle).map_err(|e| {
            DeviceBufferError::Transfer(format!("H2D: {e}"))
        })?;

        Ok(Self {
            handle,
            len: data.len(),
            _marker: PhantomData,
            backed: true,
            host_data: None,
        })
    }

    /// Convenience: allocate device memory from a CudaStream (no full backend needed).
    #[cfg(feature = "cuda")]
    pub fn from_cuda_stream(
        stream: &std::sync::Arc<cuda_core::CudaStream>,
        data: &[T],
    ) -> Result<Self, DeviceBufferError>
    where
        T: Copy,
    {
        let backend = crate::kernel::memory::CudaMemoryBackend::new(stream.clone());
        Self::from_host_device(&backend, data)
    }

    /// Allocate zero-initialized memory on the given backend.
    pub fn zeros_device<B: MemoryBackend>(backend: &B, len: usize) -> Result<Self, DeviceBufferError>
    where
        T: Default + Copy,
    {
        let bytes = len * std::mem::size_of::<T>();
        let handle = backend.alloc(bytes).map_err(|e| {
            DeviceBufferError::Allocation(format!("alloc {} bytes: {e}", bytes))
        })?;

        Ok(Self {
            handle,
            len,
            _marker: PhantomData,
            backed: true,
            host_data: None,
        })
    }

    /// Copy device data to a host Vec.
    pub fn to_host_vec<B: MemoryBackend>(&self, backend: &B) -> Result<Vec<T>, DeviceBufferError>
    where
        T: Default + Copy,
    {
        let bytes = self.byte_len();
        let mut buf = vec![T::default(); self.len];
        let dst_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, bytes) };
        backend.d2h(self.handle, dst_bytes).map_err(|e| {
            DeviceBufferError::Transfer(format!("D2H: {e}"))
        })?;
        Ok(buf)
    }

    /// Copy device data to a pre-allocated host slice.
    pub fn to_host_slice<B: MemoryBackend>(
        &self,
        backend: &B,
        dst: &mut [T],
    ) -> Result<(), DeviceBufferError> {
        let bytes = self.byte_len();
        let dst_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, bytes) };
        backend.d2h(self.handle, dst_bytes).map_err(|e| {
            DeviceBufferError::Transfer(format!("D2H: {e}"))
        })?;
        Ok(())
    }

    /// Copy host data to this device buffer.
    pub fn from_host_slice<B: MemoryBackend>(
        &self,
        backend: &B,
        src: &[T],
    ) -> Result<(), DeviceBufferError>
    where
        T: Copy,
    {
        if src.len() != self.len {
            return Err(DeviceBufferError::SizeMismatch {
                expected: self.len,
                got: src.len(),
            });
        }
        let bytes = src.len() * std::mem::size_of::<T>();
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, bytes) };
        backend.h2d(src_bytes, self.handle).map_err(|e| {
            DeviceBufferError::Transfer(format!("H2D: {e}"))
        })?;
        Ok(())
    }

    /// Get a reference to the underlying data (host-backed only).
    ///
    /// Returns None for backend-allocated buffers — use `to_host_vec()`
    /// or `to_host_slice()` instead.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.host_data.as_ref().map(|v| v.as_slice())
    }

    /// Get the raw device pointer (if applicable).
    ///
    /// Returns the RawHandle's u64 value. For CPU backend this is the slab
    /// index; for CUDA backend this is the actual device pointer.
    pub fn device_ptr(&self) -> u64 {
        self.handle.0
    }

    /// Allocate zero-initialized host-backed buffer (no backend needed).
    ///
    /// For CPU-only mode and testing. Does not go through MemoryBackend.
    pub fn zeros(len: usize) -> Self
    where
        T: Default + Clone,
    {
        let data = vec![T::default(); len];
        Self {
            handle: RawHandle(0),
            len,
            _marker: PhantomData,
            backed: false,
            host_data: Some(data),
        }
    }

    /// Allocate host-backed buffer from existing data (no backend needed).
    ///
    /// For CPU-only mode and testing. Does not go through MemoryBackend.
    pub fn from_host(data: Vec<T>) -> Self {
        Self {
            handle: RawHandle(0),
            len: data.len(),
            _marker: PhantomData,
            backed: false,
            host_data: Some(data),
        }
    }

    /// Create a CPU-fallback device buffer from a host Vec.
    ///
    /// Data stays on the CPU but is treated as if it were on the device.
    /// The handle is a sentinel value (0xDEAD).
    pub fn from_cpu_device(data: Vec<T>) -> Self {
        Self {
            handle: RawHandle(0xDEAD),
            len: data.len(),
            _marker: PhantomData,
            backed: true,
            host_data: Some(data),
        }
    }

    /// Create a zero-initialized CPU-fallback device buffer.
    pub fn zeros_cpu_device(len: usize) -> Self
    where
        T: Default + Clone,
    {
        let data = vec![T::default(); len];
        Self {
            handle: RawHandle(0xDEAD),
            len,
            _marker: PhantomData,
            backed: true,
            host_data: Some(data),
        }
    }

    /// Get a mutable reference to the underlying data (host-backed only).
    ///
    /// Returns None for backend-allocated buffers — use `to_host_vec()`
    /// or `to_host_slice()` instead.
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        self.host_data.as_mut().map(|v| v.as_mut_slice())
    }
}

impl<T: Default + Clone> DeviceBuffer<T> {
    /// Get a copy of the data as a Vec (host-backed only).
    pub fn to_host(&self) -> Vec<T> {
        self.host_data.clone().unwrap_or_default()
    }
}

/// Allocate on a MemoryManager and return the buffer.
pub fn allocate_on<M: Into<MemoryManager>>(
    manager: M,
    len: usize,
) -> Result<DeviceBuffer<u8>, DeviceBufferError> {
    let mgr = manager.into();
    let bytes = len;
    let handle = mgr.alloc(bytes).map_err(|e| {
        DeviceBufferError::Allocation(format!("alloc {bytes} bytes: {e}"))
    })?;
    Ok(DeviceBuffer {
        handle,
        len,
        _marker: PhantomData,
        backed: true,
        host_data: None,
    })
}

/// Free a handle on a MemoryManager.
pub fn free_on<M: Into<MemoryManager>>(manager: M, handle: RawHandle) -> Result<(), MemoryError> {
    manager.into().free(handle)
}
