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
    #[cfg(feature = "cuda")]
    pub fn new_with_backend(
        backend: std::sync::Arc<crate::kernel::memory::CudaMemoryBackend>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        on_device: bool,
    ) -> Self {
        let total = num_heads * head_dim * 2 * max_seq;
        let buffer = if on_device {
            DeviceBuffer::zeros_device(&*backend, total).unwrap()
        } else {
            DeviceBuffer::zeros(total)
        };
        Self {
            buffer,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
            is_device: on_device,
        }
    }

    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        on_device: bool,
    ) -> Self {
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

    /// Set the current sequence length directly (for pre-populated device buffers).
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len.min(self.max_seq);
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
    ///
    /// # Cross-contamination warning
    ///
    /// This writes **both** the K region and the V region of *this one* cache.
    /// It is correct when a single `Kvcache` holds both K and V (the intended
    /// design). But if the caller stores K and V in **separate** caches (one
    /// `Kvcache` each) and calls this on both, each cache ends up holding a
    /// copy of the *other* tensor in its unused region: the key cache's V
    /// region is filled with V and the value cache's K region is filled with K.
    /// A reader that only touches the correct region (e.g. the CPU fallback)
    /// is unaffected, but a reader that consumes a whole buffer (e.g. the GPU
    /// path) silently ingests the contamination. In that two-cache setup use
    /// [`write_k_at`] for the key cache and [`write_v_at`] for the value cache
    /// instead.
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

    /// Write only the **K** row for `num_kv_heads` heads at position `pos`.
    ///
    /// `key` must have `num_kv_heads * head_dim` elements. Only the K region is
    /// written; the V region is left untouched. Use this (rather than
    /// [`write_kv_at`]) when K and V live in separate caches, so V is not
    /// cross-contaminated into the key cache's V region. See
    /// `kv_write_no_cross_contamination` for the invariant this guards.
    pub fn write_k_at(&mut self, pos: usize, key: &[f16]) -> Result<(), KvError> {
        if pos >= self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: pos,
                max: self.max_seq,
            });
        }
        let head_stride = self.num_heads * self.head_dim;
        if let Some(slice) = self.buffer.as_mut_slice() {
            let k_start = head_stride * pos;
            let kv_len = self.num_kv_heads * self.head_dim;
            slice[k_start..(k_start + kv_len)].copy_from_slice(key);
        }
        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }
        Ok(())
    }

    /// Write only the **V** row for `num_kv_heads` heads at position `pos`.
    ///
    /// `value` must have `num_kv_heads * head_dim` elements. Only the V region
    /// is written; the K region is left untouched. Use this (rather than
    /// [`write_kv_at`]) when K and V live in separate caches, so K is not
    /// cross-contaminated into the value cache's K region. See
    /// `kv_write_no_cross_contamination` for the invariant this guards.
    pub fn write_v_at(&mut self, pos: usize, value: &[f16]) -> Result<(), KvError> {
        if pos >= self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: pos,
                max: self.max_seq,
            });
        }
        let head_stride = self.num_heads * self.head_dim;
        if let Some(slice) = self.buffer.as_mut_slice() {
            let v_start = head_stride * self.max_seq + head_stride * pos;
            let kv_len = self.num_kv_heads * self.head_dim;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn halfs(values: &[f32]) -> Vec<f16> {
        values.iter().map(|&v| f16::from_f32(v)).collect()
    }

    /// Regression test: the dispatch path stores K and V in **separate**
    /// `Kvcache`s (one key cache + one value cache per layer) and must write
    /// each tensor only into its own cache's own region.
    ///
    /// The bug this guards against: calling `write_kv_at(pos, &k_row, &v_row)`
    /// on *both* caches (which writes K and V into *each* buffer) cross-
    /// contaminates the key cache's V region with V and the value cache's K
    /// region with K. Region-selective readers (the CPU fallback) never notice,
    /// but whole-buffer readers (the GPU path) silently ingest the garbage.
    ///
    /// Invariant under test: after the dispatch-style write pattern,
    ///   - key cache:   K region == K,  V region all zeros
    ///   - value cache: K region all zeros, V region == V
    #[test]
    fn kv_write_no_cross_contamination() {
        // Tiny GQA-shaped cache: 2 heads, 2 KV heads, head_dim 4, max_seq 8.
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let max_seq = 8;
        let kv_len = num_kv_heads * head_dim; // 8

        let mut key_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, false);
        let mut value_cache = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, false);

        // Distinct, non-zero K and V rows so any bleed is detectable.
        let k_row = halfs(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let v_row = halfs(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

        // The dispatch write pattern: K into the key cache, V into the value
        // cache, each into its own region only.
        key_cache.write_k_at(0, &k_row).expect("write_k_at(key)");
        value_cache
            .write_v_at(0, &v_row)
            .expect("write_v_at(value)");

        let head_stride = num_heads * head_dim; // 8
        let v_base = head_stride * max_seq; // 64

        let kbuf = key_cache.buffer().as_slice().expect("host key buffer");
        let vbuf = value_cache.buffer().as_slice().expect("host value buffer");

        // Key cache: K region holds K.
        assert!(
            kbuf[..kv_len] == k_row[..],
            "key cache K region must hold K"
        );
        // Key cache: V region must remain pristine (all zeros).
        assert!(
            (v_base..v_base + kv_len).all(|i| kbuf[i] == f16::from_f32(0.0)),
            "key cache V region must stay zero — V leaked into the key cache"
        );

        // Value cache: V region holds V.
        assert!(
            vbuf[v_base..v_base + kv_len] == v_row[..],
            "value cache V region must hold V"
        );
        // Value cache: K region must remain pristine (all zeros).
        assert!(
            (0..kv_len).all(|i| vbuf[i] == f16::from_f32(0.0)),
            "value cache K region must stay zero — K leaked into the value cache"
        );

        // Sanity: the contaminated pattern (write_kv_at on both caches) must
        // NOT be what the dispatch path does — document the failure mode.
        let mut bad_key = Kvcache::new(num_heads, num_kv_heads, head_dim, max_seq, false);
        bad_key.write_kv_at(0, &k_row, &v_row).expect("write_kv_at");
        let bad_kbuf = bad_key.buffer().as_slice().expect("host buffer");
        assert!(
            bad_kbuf[v_base..v_base + kv_len] == v_row[..],
            "write_kv_at on the key cache DOES contaminate its V region (the bug)"
        );
    }
}
