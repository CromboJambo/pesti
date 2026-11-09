//! Stub kvcache module for CPU-only builds.

use crate::kernel::device_buf::DeviceBuffer;
use half::f16;

/// Dummy KV cache error
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    #[error("cache not available")]
    NotAvailable,
    #[error("invalid slice: {0}")]
    InvalidSlice(String),
}

/// Stub KV cache
pub struct Kvcache {
    buffer: DeviceBuffer<f16>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
}

impl Kvcache {
    pub fn new(
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _max_seq: usize,
        _on_device: bool,
    ) -> Self {
        let total = _num_heads * _head_dim * 2 * _max_seq;
        Self {
            buffer: DeviceBuffer::zeros(total),
            num_heads: _num_heads,
            num_kv_heads: _num_kv_heads,
            head_dim: _head_dim,
            max_seq: _max_seq,
        }
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn seq_len(&self) -> usize {
        0 // Stub - actual sequence length tracking not implemented
    }

    /// Whether this cache is on device memory.
    pub fn is_device(&self) -> bool {
        false // Always false for stub
    }

    pub fn write_kv_at(
        &mut self,
        _pos: usize,
        _key: &[f16],
        _value: &[f16],
    ) -> Result<(), KvError> {
        Err(KvError::NotAvailable)
    }

    /// Stub — see [`crate::kernel::kvcache::Kvcache::write_k_at`].
    pub fn write_k_at(&mut self, _pos: usize, _key: &[f16]) -> Result<(), KvError> {
        Err(KvError::NotAvailable)
    }

    /// Stub — see [`crate::kernel::kvcache::Kvcache::write_v_at`].
    pub fn write_v_at(&mut self, _pos: usize, _value: &[f16]) -> Result<(), KvError> {
        Err(KvError::NotAvailable)
    }

    pub fn clear(&mut self) {
        // Stub - no-op for CPU-only builds
    }

    pub fn append(&mut self, _key: &[f16], _value: &[f16]) -> Result<(), KvError> {
        Err(KvError::NotAvailable)
    }

    pub fn buffer(&self) -> &DeviceBuffer<f16> {
        &self.buffer
    }

    pub fn slice(&self, _start_pos: usize, _seq_len: usize) -> Result<KvcacheSlice<'_>, KvError> {
        Err(KvError::NotAvailable)
    }

    pub fn total_elements(&self) -> usize {
        self.num_heads * self.head_dim * 2 * self.max_seq
    }
}

/// Stub KV cache slice
pub struct KvcacheSlice<'a> {
    pub key_cache: &'a DeviceBuffer<f16>,
    pub value_cache: &'a DeviceBuffer<f16>,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl<'a> KvcacheSlice<'a> {
    pub fn new(
        _key_cache: &'a DeviceBuffer<f16>,
        _value_cache: &'a DeviceBuffer<f16>,
        _seq_len: usize,
        _num_heads: usize,
        _head_dim: usize,
    ) -> Self {
        Self {
            key_cache: _key_cache,
            value_cache: _value_cache,
            seq_len: _seq_len,
            num_heads: _num_heads,
            head_dim: _head_dim,
        }
    }

    pub fn to_tma_descriptor(&self) -> crate::kernel::TmaDescriptor {
        crate::kernel::TmaDescriptor::default()
    }
}

/// Stub TMA descriptor (4x u32 = 16 bytes for Blackwell)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TmaDescriptor(pub [u32; 4]);

impl TmaDescriptor {
    pub fn with_gmem_addr(mut self, addr: u64) -> Self {
        self.0[0] = addr as u32;
        self
    }

    pub fn gmem_addr(&self) -> u64 {
        self.0[0] as u64
    }

    pub fn as_u32_words(&self) -> [u32; 4] {
        self.0
    }
}
