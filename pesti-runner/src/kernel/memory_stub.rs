//! Stub memory module for CPU-only builds.

use crate::kernel::device_buf::DeviceBuffer;

/// Dummy MemoryBackend trait
pub trait MemoryBackend {
    fn alloc(&self, bytes: usize) -> Result<u64, MemoryError>;
    fn free(&self, handle: u64) -> Result<(), MemoryError>;
    fn h2d(&self, data: &[u8], handle: u64) -> Result<(), MemoryError>;
    fn d2h(&self, handle: u64, dst: &mut [u8]) -> Result<(), MemoryError>;
}

/// Dummy MemoryManager
#[derive(Clone)]
pub enum MemoryManager {
    Cpu(CpuMemoryBackend),
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
    fn alloc(&self, _bytes: usize) -> Result<u64, MemoryError> {
        Ok(0xDEAD) // Dummy handle
    }

    fn free(&self, _handle: u64) -> Result<(), MemoryError> {
        Ok(())
    }

    fn h2d(&self, _data: &[u8], _handle: u64) -> Result<(), MemoryError> {
        Ok(())
    }

    fn d2h(&self, _handle: u64, _dst: &mut [u8]) -> Result<(), MemoryError> {
        Ok(())
    }
}

/// Memory error type
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("allocation failed: requested {requested}, but only {available} available")]
    AllocationFailed { requested: usize, available: usize },

    #[error("invalid handle: {0}")]
    InvalidHandle(u64),

    #[error("transfer error: {0}")]
    Transfer(String),

    #[error("CUDA error: {0}")]
    Cuda(String),
}

/// Raw memory handle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawHandle(pub u64);

impl RawHandle {
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}
