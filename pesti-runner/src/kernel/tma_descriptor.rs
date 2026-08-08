//! Blackwell TMA global cache read descriptor.
//!
//! **SPECULATIVE — unverified bit layout.**
//!
//! The 128-byte CUtensorMap structure is opaque per the CUDA driver API.
//! `cuTensorMapEncodeTiled()` must be called on the host to create valid
//! descriptors. The raw bit positions below are educated guesses from
//! reverse-engineering cuda-oxide examples and the PTX `tensormap.replace`
//! instruction fields. They have NOT been verified against actual hardware.
//!
//! For production use, see `tma_bridge.rs` which uses `cuTensorMapEncodeTiled`.
//!
//! The descriptor encodes the source of an asynchronous GMEM-to-SMEM copy:
//! - GMEM address offset
//! - Box dimensions (X and Y)
//! - GMEM and SMEM strides
//! - Element info, descriptor type, SMEM config, cache hint
//!
//! For the KV cache use case, the address offset is a byte offset from
//! the buffer base (passed separately to the kernel). The box defines
//! the region to copy, and strides define how to stride through GMEM/SMEM.

/// 128-bit TMA descriptor packed as u128 for correct alignment and zero-copy casting.
///
/// **SPECULATIVE:** Bit positions are unverified guesses. See module-level docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct TmaDescriptor(pub u128);

impl TmaDescriptor {
    /// Create a new zeroed descriptor.
    pub const fn new() -> Self {
        Self(0u128)
    }

    /// Get the underlying u128 value.
    pub const fn as_u128(&self) -> u128 {
        self.0
    }

    /// Get the descriptor as [u32; 4] for hardware register access.
    pub const fn as_u32_words(&self) -> [u32; 4] {
        [
            (self.0) as u32,
            (self.0 >> 32) as u32,
            (self.0 >> 64) as u32,
            (self.0 >> 96) as u32,
        ]
    }

    /// Set the GMEM address offset (within the buffer base).
    ///
    /// The offset must be 256-byte aligned for TMA.
    /// Stored in word 0 [31:0].
    pub const fn with_gmem_addr(mut self, addr: u64) -> Self {
        self.0 |= (addr as u128 & 0xFFFFFFFF) << 0;
        self
    }

    /// Get the GMEM address offset from the descriptor.
    pub const fn gmem_addr(&self) -> u64 {
        (self.0 & 0xFFFFFFFF) as u64
    }

    /// Set the box (region) dimensions and strides.
    ///
    /// `box_x` — number of elements along X axis (16-bit).
    /// `gmem_x_stride` — GMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `smem_x_stride` — SMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `box_y` — number of elements along Y axis (16-bit).
    /// `gmem_y_stride` — GMEM stride in elements between consecutive columns (16-bit).
    /// `smem_y_stride` — SMEM stride in elements between consecutive columns (16-bit).
    #[allow(clippy::identity_op, clippy::erasing_op)]
    pub const fn with_box(
        mut self,
        box_x: u16,
        gmem_x_stride: u16,
        smem_x_stride: u16,
        box_y: u16,
        gmem_y_stride: u16,
        smem_y_stride: u16,
    ) -> Self {
        self.0 |= (box_x as u128) << 32;
        let gmem_x = if gmem_x_stride > 255 {
            255u16
        } else {
            gmem_x_stride
        };
        let smem_x = if smem_x_stride > 255 {
            255u16
        } else {
            smem_x_stride
        };
        self.0 |= (gmem_x as u128) << 48;
        self.0 |= (smem_x as u128) << 56;
        self.0 |= (box_y as u128) << 64;
        self.0 |= (gmem_y_stride as u128) << 80;
        self.0 |= (smem_y_stride as u128) << 96;
        self
    }

    /// Set the element info field.
    pub const fn with_element_info(mut self, element_size: u8) -> Self {
        self.0 |= (element_size as u128 & 0xF) << 120;
        self
    }

    /// Set the descriptor type.
    pub const fn with_descriptor_type(mut self, dtype: u8) -> Self {
        self.0 |= (dtype as u128) << 112;
        self
    }

    /// Set the SMEM config field (bits 104-107, word[3] bits 8-11).
    pub const fn with_smem_config(mut self, config: u8) -> Self {
        self.0 |= (config as u128) << 104;
        self
    }

    /// Unpack from [u32; 4] words received from a kernel.
    pub const fn from_u32_words(words: [u32; 4]) -> Self {
        Self(
            (words[0] as u128)
                | ((words[1] as u128) << 32)
                | ((words[2] as u128) << 64)
                | ((words[3] as u128) << 96),
        )
    }
}

impl Default for TmaDescriptor {
    fn default() -> Self {
        Self::new()
    }
}
