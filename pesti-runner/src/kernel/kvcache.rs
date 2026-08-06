//! KV cache for LLM inference.
//!
//! Stores key and value tensors per layer with dynamic sequence management.
//!
//! Layout: `[num_heads * head_dim, max_seq]` contiguous per layer.
//! The sequence dimension is contiguous for efficient TMA transfers
//! during attention computation.
//!
//! For a model with `num_heads` heads and `head_dim` per head:
//! - Each head's KV slice is `head_dim` elements
//! - All heads are packed: `head_dim * 0, head_dim * 1, ..., head_dim * (heads-1)`
//! - Sequence dimension: positions 0..seq_len

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::tma_descriptor::TmaDescriptor;
use half::f16;

/// Per-layer KV cache allocation.
///
/// Stores K and V tensors in a single contiguous device buffer.
/// Layout: `[num_heads * head_dim * 2, max_seq]` — K and V are interleaved.
/// K occupies `[num_heads * head_dim, :]`, V occupies `[num_heads * head_dim * 2, :]`.
#[derive(Debug)]
pub struct Kvcache {
    /// Device buffer for K and V (interleaved).
    buffer: DeviceBuffer<f16>,
    /// Number of attention heads in the cache.
    num_heads: usize,
    /// Number of KV heads (for GQA).
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Maximum sequence length allocated.
    max_seq: usize,
    /// Current sequence length (used entries).
    seq_len: usize,
    /// Whether the buffer is on device (true) or host (false).
    is_device: bool,
}

impl Kvcache {
    /// Create a new KV cache with the given dimensions.
    ///
    /// `num_heads` — number of attention heads per layer.
    /// `head_dim` — dimension of each head.
    /// `max_seq` — maximum sequence length to allocate.
    /// `on_device` — whether to allocate on device (Device variant) or host (Host variant).
    ///
    /// Total elements: `num_heads * head_dim * 2 * max_seq`.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, max_seq: usize, on_device: bool) -> Self {
        let total = num_heads * head_dim * 2 * max_seq;
        Self {
            buffer: DeviceBuffer::zeros(total),
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
            is_device: on_device,
        }
    }

    /// Create a device-side KV cache from a raw pointer.
    ///
    /// # Safety
    ///
    /// Caller must ensure `ptr_addr` points to a valid buffer of at least
    /// `num_heads * head_dim * 2 * max_seq` f16 elements.
    pub unsafe fn from_device(
        ptr_addr: u64,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Self {
        let total = num_heads * head_dim * 2 * max_seq;
        Self {
            buffer: unsafe { DeviceBuffer::from_device(ptr_addr, total) },
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
            is_device: true,
        }
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Dimension per attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of KV heads (for GQA).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Maximum sequence length.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Current sequence length (number of valid entries).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Whether the cache has been populated.
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Whether this cache is on device memory.
    pub fn is_device(&self) -> bool {
        self.is_device
    }

    /// Total number of elements in the buffer.
    pub fn total_elements(&self) -> usize {
        self.num_heads * self.head_dim * 2 * self.max_seq
    }

    /// Get the current sequence length.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Reset the sequence length to zero (clear cache).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Append a new key vector and value vector at position `seq_len`.
    ///
    /// `key` and `value` must each have `num_heads * head_dim` elements.
    ///
    /// Returns `KvError::SeqLenExceeded` if `seq_len >= max_seq`.
    pub fn append(&mut self, key: &[f16], value: &[f16]) -> Result<(), KvError> {
        if self.seq_len >= self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: self.seq_len,
                max: self.max_seq,
            });
        }

        let head_stride = self.num_heads * self.head_dim;
        let pos = self.seq_len;

        // Write K slice: K occupies [0 .. head_stride * max_seq]
        // K row at pos: [head_stride * pos .. head_stride * (pos+1))
        if let Some(slice) = self.buffer.as_mut_slice() {
            let k_start = head_stride * pos;
            slice[k_start..(head_stride + k_start)].copy_from_slice(key);
            // Write V slice: V occupies [head_stride * max_seq .. head_stride * 2 * max_seq]
            // V row at pos: [head_stride * max_seq + head_stride * pos .. head_stride * max_seq + head_stride * (pos+1))
            let v_start = head_stride * self.max_seq + head_stride * pos;
            slice[v_start..(head_stride + v_start)].copy_from_slice(value);
        }

        self.seq_len += 1;
        Ok(())
    }

    /// Write a KV row for `num_kv_heads` heads at position `pos`.
    ///
    /// `key` and `value` must each have `num_kv_heads * head_dim` elements.
    /// Only the first `num_kv_heads` head slots are written.
    pub fn write_kv_at(&mut self, pos: usize, key: &[f16], value: &[f16]) -> Result<(), KvError> {
        if pos >= self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: pos,
                max: self.max_seq,
            });
        }
        let head_stride = self.num_heads * self.head_dim;
        if let Some(slice) = self.buffer.as_mut_slice() {
            let k_start = head_stride * pos;
            let v_start = head_stride * self.max_seq + head_stride * pos;
            let kv_len = self.num_kv_heads * self.head_dim;
            slice[k_start..(k_start + kv_len)].copy_from_slice(key);
            slice[v_start..(v_start + kv_len)].copy_from_slice(value);
        }
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Ok(())
    }

    /// Append a batch of key/value vectors at positions `start..start+batch`.
    ///
    /// Each entry in `keys` and `values` must have `num_heads * head_dim` elements.
    pub fn append_batch(
        &mut self,
        keys: &[&[f16]],
        values: &[&[f16]],
        start: usize,
    ) -> Result<(), KvError> {
        let batch = keys.len();
        if start + batch > self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: start + batch,
                max: self.max_seq,
            });
        }

        let head_stride = self.num_heads * self.head_dim;

        if let Some(slice) = self.buffer.as_mut_slice() {
            for b in 0..batch {
                let pos = start + b;
                let k_start = head_stride * pos;
                let v_start = head_stride * self.max_seq + head_stride * pos;
                slice[k_start..(head_stride + k_start)].copy_from_slice(keys[b]);
                slice[v_start..(head_stride + v_start)].copy_from_slice(values[b]);
            }
        }

        if start + batch > self.seq_len {
            self.seq_len = start + batch;
        }
        Ok(())
    }

    /// Resize the cache to a larger `max_seq`.
    ///
    /// Copies existing data to the new buffer. Returns `Err` if shrinking.
    pub fn resize(&mut self, new_max_seq: usize) -> Result<(), KvError> {
        if new_max_seq <= self.max_seq {
            return Err(KvError::ResizeFailed {
                reason: "cannot shrink KV cache".to_string(),
            });
        }

        let new_total = self.num_heads * self.head_dim * 2 * new_max_seq;
        let mut new_buf = DeviceBuffer::zeros(new_total);

        if let Some(src) = self.buffer.as_slice() {
            if let Some(dst) = new_buf.as_mut_slice() {
                let copy_len = self.total_elements();
                if copy_len <= dst.len() && copy_len <= src.len() {
                    dst[..copy_len].copy_from_slice(&src[..copy_len]);
                }
            }
        }

        self.buffer = new_buf;
        self.max_seq = new_max_seq;
        Ok(())
    }

    /// Get a TMA descriptor for loading a KV slice into SMEM.
    ///
    /// Returns a descriptor configured for a TMA global cache read
    /// of the K or V tensor for a single attention head at sequence position `pos`.
    ///
    /// `is_key` — true for K tensor, false for V tensor.
    /// `head_idx` — which head to load (0..num_heads).
    /// `box_y` — number of sequence positions to load (1 for decode, >1 for prefill).
    pub fn tma_descriptor(
        &self,
        _gmem_addr: u64,
        is_key: bool,
        head_idx: usize,
        box_y: u16,
    ) -> Result<TmaDescriptor, KvError> {
        if head_idx >= self.num_heads {
            return Err(KvError::HeadIndexOutOfBounds {
                head_idx,
                num_heads: self.num_heads,
            });
        }

        if box_y == 0 || box_y as usize > self.seq_len {
            return Err(KvError::BoxYOutOfBounds {
                box_y: box_y as usize,
                seq_len: self.seq_len,
            });
        }

        let head_stride = self.num_heads * self.head_dim;
        let head_offset = head_idx * self.head_dim;

        // Address offset within the buffer for this head's K or V.
        // K occupies [0 .. head_stride * max_seq], V occupies [head_stride * max_seq .. head_stride * 2 * max_seq].
        // The offset is in element units (f16), converted to byte offset for TMA.
        let byte_offset = if is_key {
            (head_stride * head_offset) as u64 * 2
        } else {
            (head_stride * self.max_seq + head_stride * head_offset) as u64 * 2
        };

        // Box X = head_dim (elements per row), box Y = box_y (rows)
        // GMEM stride = head_stride (skip to next head's data)
        // SMEM stride = head_dim (contiguous in SMEM)
        let desc = TmaDescriptor::new()
            .with_gmem_addr(byte_offset)
            .with_box(
                self.head_dim as u16,            // box X
                (self.head_dim as u16).min(255), // gmem_x_stride (8-bit field, saturates at 255)
                self.head_dim as u16,            // smem_x_stride
                box_y,                           // box Y
                head_stride as u16,              // gmem_y_stride
                self.head_dim as u16,            // smem_y_stride
            )
            .with_element_info(1) // f16 = 2 bytes
            .with_descriptor_type(1) // global cache read
            .with_smem_config(0);

        Ok(desc)
    }

    /// Get the device pointer for this cache.
    ///
    /// Returns `None` if the cache is on host.
    pub fn device_ptr(&self) -> Option<u64> {
        if self.buffer.is_backed() {
            Some(self.buffer.device_ptr())
        } else {
            None
        }
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &DeviceBuffer<f16> {
        &self.buffer
    }

    /// Get a mutable reference to the underlying buffer.
    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<f16> {
        &mut self.buffer
    }
}

/// A slice of the KV cache for a single head at a specific sequence range.
///
/// Used to pass TMA configuration to attention kernels.
#[derive(Debug, Clone, Copy)]
pub struct KvcacheSlice {
    /// Base device pointer for the entire K+V buffer.
    pub gmem_addr: u64,
    /// Number of heads in the cache.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Maximum sequence length (needed for V base calculation).
    pub max_seq: usize,
    /// Head index (0..num_heads).
    pub head_idx: usize,
    /// Sequence start position.
    pub seq_start: usize,
    /// Number of sequence positions (box Y).
    pub seq_len: usize,
    /// Whether this is the K tensor (true) or V tensor (false).
    pub is_key: bool,
}

impl KvcacheSlice {
    /// Create a new KV cache slice.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gmem_addr: u64,
        num_heads: usize,
        head_dim: usize,
        max_seq: usize,
        head_idx: usize,
        seq_start: usize,
        seq_len: usize,
        is_key: bool,
    ) -> Self {
        Self {
            gmem_addr,
            num_heads,
            head_dim,
            max_seq,
            head_idx,
            seq_start,
            seq_len,
            is_key,
        }
    }

    /// Get the base address for this slice's K or V tensor.
    pub fn base_addr(&self) -> u64 {
        let head_stride = self.num_heads * self.head_dim;
        let head_offset = self.head_idx * self.head_dim;
        if self.is_key {
            self.gmem_addr + (head_stride * head_offset) as u64 * 2
        } else {
            self.gmem_addr + (head_stride * self.max_seq + head_stride * head_offset) as u64 * 2
        }
    }

    /// Build a TMA descriptor for this slice.
    pub fn to_tma_descriptor(&self) -> TmaDescriptor {
        let head_stride = self.num_heads * self.head_dim;
        let head_offset = self.head_idx * self.head_dim;
        // Byte offset within the buffer for this head's K or V
        let byte_offset = if self.is_key {
            (head_stride * head_offset) as u64 * 2
        } else {
            (head_stride * self.max_seq + head_stride * head_offset) as u64 * 2
        };
        TmaDescriptor::new()
            .with_gmem_addr(byte_offset)
            .with_box(
                self.head_dim as u16,
                (self.head_dim as u16).min(255),
                self.head_dim as u16,
                self.seq_len as u16,
                head_stride as u16,
                self.head_dim as u16,
            )
            .with_element_info(1)
            .with_descriptor_type(1)
            .with_smem_config(0)
    }
}

/// KV cache errors.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    #[error("sequence length exceeded: current={current}, max={max}")]
    SeqLenExceeded { current: usize, max: usize },

    #[error("head index out of bounds: head_idx={head_idx}, num_heads={num_heads}")]
    HeadIndexOutOfBounds { head_idx: usize, num_heads: usize },

    #[error("box_y out of bounds: box_y={box_y}, seq_len={seq_len}")]
    BoxYOutOfBounds { box_y: usize, seq_len: usize },

    #[error("resize failed: {reason}")]
    ResizeFailed { reason: String },
}

