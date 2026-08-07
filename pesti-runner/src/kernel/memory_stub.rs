//! Stub memory module for CPU-only builds.


/// Dummy MemoryBackend trait
pub trait MemoryBackend {
    fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError>;
    fn free(&self, handle: RawHandle) -> Result<(), MemoryError>;
    fn h2d(&self, data: &[u8], handle: RawHandle) -> Result<(), MemoryError>;
    fn d2h(&self, handle: RawHandle, dst: &mut [u8]) -> Result<(), MemoryError>;
}

/// Dummy MemoryManager
#[derive(Clone)]
pub enum MemoryManager {
    Cpu(CpuMemoryBackend),
}

impl MemoryManager {
    pub fn new_cpu(capacity: usize) -> Self {
        MemoryManager::Cpu(CpuMemoryBackend::new(capacity))
    }

    pub fn alloc(&self, bytes: usize) -> Result<RawHandle, MemoryError> {
        match self {
            MemoryManager::Cpu(backend) => backend.alloc(bytes),
        }
    }

    pub fn free(&self, handle: RawHandle) -> Result<(), MemoryError> {
        match self {
            MemoryManager::Cpu(backend) => backend.free(handle),
        }
    }

    pub fn h2d(&self, _data: &[u8], _handle: RawHandle) -> Result<(), MemoryError> {
        Ok(())
    }

    pub fn d2h(&self, _handle: RawHandle, _dst: &mut [u8]) -> Result<(), MemoryError> {
        Ok(())
    }

    pub fn sync(&self) -> Result<(), MemoryError> {
        Ok(())
    }
}

/// CPU memory backend (stub)
#[derive(Clone)]
pub struct CpuMemoryBackend {
    capacity: usize,
}

impl CpuMemoryBackend {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}

impl MemoryBackend for CpuMemoryBackend {
    fn alloc(&self, _bytes: usize) -> Result<RawHandle, MemoryError> {
        Ok(RawHandle(0xDEAD)) // Dummy handle
    }

    fn free(&self, _handle: RawHandle) -> Result<(), MemoryError> {
        Ok(())
    }

    fn h2d(&self, _data: &[u8], _handle: RawHandle) -> Result<(), MemoryError> {
        Ok(())
    }

    fn d2h(&self, _handle: RawHandle, _dst: &mut [u8]) -> Result<(), MemoryError> {
        Ok(())
    }
}

/// Memory error type
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("allocation failed: requested {requested}, but only {available} available")]
    AllocationFailed { requested: usize, available: usize },

    #[error("invalid handle: {0:?}")]
    InvalidHandle(RawHandle),

    #[error("transfer error: {0}")]
    Transfer(String),

    #[error("CUDA error: {0}")]
    Cuda(String),
}

/// Raw memory handle (newtype wrapper for consistency with CUDA mode)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawHandle(pub u64);

impl RawHandle {
    pub fn new(handle: u64) -> Self {
        Self(handle)
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}
