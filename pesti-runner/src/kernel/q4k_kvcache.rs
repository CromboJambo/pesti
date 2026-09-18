//! Q4_K quantized KV cache for reduced memory bandwidth during autoregressive decode.
//!
//! Stores K/V tensors in ggml block_q4_K format (256 elements per 144-byte block)
//! to achieve ~4x reduction in KV cache memory footprint vs FP16. Dequantization
//! happens on-the-fly during attention computation via GPU kernels.
//!
//! Layout: Each Q4_K block stores 256 values using:
//! - 2 bytes: scale factor (f16)
//! - 2 bytes: min offset (f16)
//! - 12 bytes: per-group scales/mins (packed 6-bit pairs)
//! - 128 bytes: quantized data (nibbles, 2 values per byte)
//! Total: 144 bytes per 256 f32-equivalent values

use crate::kernel::device_buf::DeviceBuffer;

/// Q4_K block size constants (ggml canonical format).
const Q4K_BLOCK_SIZE: usize = 144; // bytes per block
const Q4K_BLOCK_ELEMS: usize = 256; // elements per block

/// Per-layer Q4_K KV cache.
pub struct Q4KVCache {
    /// Device buffer for quantized K data (Q4_K blocks).
    k_buffer: DeviceBuffer<u8>,
    /// Device buffer for quantized V data (Q4_K blocks).
    v_buffer: DeviceBuffer<u8>,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Maximum sequence length.
    max_seq: usize,
    /// Current sequence length (valid entries).
    seq_len: usize,
}

impl Q4KVCache {
    /// Create a new Q4_K quantized KV cache.
    pub fn new(num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        // Calculate total bytes needed for K and V in Q4_K format
        let total_elems = num_kv_heads * head_dim * max_seq;
        let blocks_needed = total_elems.div_ceil(Q4K_BLOCK_ELEMS);
        let buffer_bytes = blocks_needed * Q4K_BLOCK_SIZE;

        Self {
            k_buffer: DeviceBuffer::zeros(buffer_bytes),
            v_buffer: DeviceBuffer::zeros(buffer_bytes),
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
        }
    }

    /// Create from existing device buffer (for pre-allocated memory).
    pub fn from_device(
        k_ptr: u64,
        v_ptr: u64,
        _k_bytes: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Self {
        let total_elems = num_kv_heads * head_dim * max_seq;
        let blocks_needed = total_elems.div_ceil(Q4K_BLOCK_ELEMS);
        let buffer_bytes = blocks_needed * Q4K_BLOCK_SIZE;

        Self {
            k_buffer: unsafe { DeviceBuffer::from_device(k_ptr, buffer_bytes) },
            v_buffer: unsafe { DeviceBuffer::from_device(v_ptr, buffer_bytes) },
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
        }
    }

    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Dimension per head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum sequence length.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Current sequence length (valid entries).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Set the current sequence length directly (for pre-populated device buffers).
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len.min(self.max_seq);
    }

    /// Reset cache.
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Write a quantized KV row at position `pos`.
    ///
    /// Expects pre-quantized Q4_K block data for K and V.
    pub fn write_kv_at(
        &mut self,
        pos: usize,
        k_block: &[u8],
        v_block: &[u8],
    ) -> Result<(), KvError> {
        if pos >= self.max_seq {
            return Err(KvError::SeqLenExceeded {
                current: pos,
                max: self.max_seq,
            });
        }

        let row_bytes = self.num_kv_heads * self.head_dim;
        let blocks_per_row = row_bytes.div_ceil(Q4K_BLOCK_ELEMS);
        let block_offset = pos * blocks_per_row * Q4K_BLOCK_SIZE;

        // Write K block
        if let Some(slice) = self.k_buffer.as_mut_slice() {
            slice[block_offset..block_offset + k_block.len()].copy_from_slice(k_block);
        }

        // Write V block
        if let Some(slice) = self.v_buffer.as_mut_slice() {
            slice[block_offset..block_offset + v_block.len()].copy_from_slice(v_block);
        }

        if pos + 1 > self.seq_len {
            self.seq_len = pos + 1;
        }

        Ok(())
    }

    /// Get device pointer for K data.
    pub fn k_device_ptr(&self) -> Option<u64> {
        if self.k_buffer.is_backed() {
            Some(self.k_buffer.device_ptr())
        } else {
            None
        }
    }

    /// Get device pointer for V data.
    pub fn v_device_ptr(&self) -> Option<u64> {
        if self.v_buffer.is_backed() {
            Some(self.v_buffer.device_ptr())
        } else {
            None
        }
    }

    /// Calculate memory savings vs FP16 cache.
    pub fn memory_savings_percentage(&self) -> f32 {
        let fp16_bytes = self.num_kv_heads * self.head_dim * self.max_seq * 2; // K + V in FP16
        let q4k_bytes = self.k_buffer.len() + self.v_buffer.len();
        ((fp16_bytes as f32 - q4k_bytes as f32) / fp16_bytes as f32) * 100.0
    }

    /// Total memory used by this cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.k_buffer.len() + self.v_buffer.len()
    }
}

/// KV cache errors for Q4_K implementation.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    #[error("sequence length exceeded: current={current}, max={max}")]
    SeqLenExceeded { current: usize, max: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_memory_savings() {
        // 8 KV heads, head_dim=64, max_seq=2048
        let cache = Q4KVCache::new(8, 64, 2048);

        // FP16 would use: 8 * 64 * 2048 * 2 bytes (K+V) = 2,097,152 bytes
        let fp16_bytes = 8 * 64 * 2048 * 2;

        // Q4_K should use significantly less (~4x reduction)
        let q4k_bytes = cache.memory_bytes();
        assert!(
            q4k_bytes < fp16_bytes,
            "Q4_K cache should be smaller than FP16"
        );

        // Expect ~75% savings (4:1 compression ratio)
        let savings = cache.memory_savings_percentage();
        assert!(
            savings > 70.0,
            "Expected >70% memory savings, got {:.1}%",
            savings
        );
    }

    #[test]
    fn test_q4k_write_kv() {
        let mut cache = Q4KVCache::new(8, 64, 2048);

        // Create dummy quantized block data (Q4_K block is 144 bytes)
        let k_block = vec![0u8; Q4K_BLOCK_SIZE];
        let v_block = vec![0u8; Q4K_BLOCK_SIZE];

        cache.write_kv_at(0, &k_block, &v_block).unwrap();
        assert_eq!(cache.seq_len(), 1);

        cache.write_kv_at(1, &k_block, &v_block).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }
}
