//! Bridge between PESTI TmaDescriptor and cuda-oxide TMA infrastructure.
//!
//! PESTI's `TmaDescriptor` (128-bit u128) is a **speculative** hand-written bit layout
//! — the actual CUtensorMap encoding is opaque and not publicly documented.
//! This module provides the bridge to cuda-oxide's 128-byte `TmaDescriptor` / `CUtensorMap` type used
//! by `cp_async_bulk_tensor_2d_g2s` and the `#[kernel]` approach.
//!
//! **Production descriptors should be created via `cuTensorMapEncodeTiled` on the host**,
//! as cuda-oxide does. Our host-side `HostTmaDescriptor` wraps that approach.
//!
//! Host-side TMA descriptors are created via `cuTensorMapEncodeTiled` and
//! passed to kernels as `*const TmaDescriptor`.

use cuda_core::sys::{
    CUtensorMap, CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE, cuTensorMapEncodeTiled,
};
use std::mem::MaybeUninit;

/// Host-side TMA descriptor for f16 tensors.
///
/// Wraps `CUtensorMap` created via `cuTensorMapEncodeTiled`.
/// This is the 128-byte opaque descriptor passed to `#[kernel]` functions.
#[derive(Debug, Clone)]
pub struct HostTmaDescriptor {
    /// Raw descriptor data (128 bytes).
    pub opaque: [u64; 16],
}

impl HostTmaDescriptor {
    /// Create a new TMA descriptor for a 2D f16 tensor.
    ///
    /// `global_address` — base device pointer to the tensor.
    /// `global_width` — number of elements along X (head_dim for K/V).
    /// `global_height` — number of elements along Y (max_seq or box_y).
    /// `tile_width` — TMA box dimension X (elements per row per copy).
    /// `tile_height` — TMA box dimension Y (rows per copy).
    ///
    /// # Safety
    ///
    /// `global_address` must point to valid device memory with at least
    /// `global_width * global_height` f16 elements.
    pub unsafe fn create_f16(
        global_address: *mut std::ffi::c_void,
        global_width: u64,
        global_height: u64,
        tile_width: u32,
        tile_height: u32,
    ) -> Result<Self, String> {
        let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
        let global_dim: [u64; 2] = [global_width, global_height];
        // Byte stride between consecutive rows in global memory
        let global_strides: [u64; 1] = [global_width * 2]; // f16 = 2 bytes
        let box_dim: [u32; 2] = [tile_width, tile_height];
        let element_strides: [u32; 2] = [1, 1];

        let result = unsafe {
            cuTensorMapEncodeTiled(
                tensor_map.as_mut_ptr(),
                CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                2,
                global_address,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };

        if result != 0 {
            return Err(format!(
                "cuTensorMapEncodeTiled failed: error code {}",
                result
            ));
        }

        let descriptor = unsafe { tensor_map.assume_init() };
        Ok(Self {
            opaque: descriptor.opaque,
        })
    }

    /// Create a TMA descriptor with SWIZZLE_128B for tensor memory compatibility.
    ///
    /// tcgen05 TMEM requires SWIZZLE_128B layout to match the core matrix
    /// tiling pattern used by tcgen05_mma instructions.
    ///
    /// # Safety
    ///
    /// `global_address` must point to valid device memory with at least
    /// `global_width * global_height` f16 elements.
    pub unsafe fn create_f16_swizzled(
        global_address: *mut std::ffi::c_void,
        global_width: u64,
        global_height: u64,
        tile_width: u32,
        tile_height: u32,
    ) -> Result<Self, String> {
        let mut tensor_map = MaybeUninit::<CUtensorMap>::uninit();
        let global_dim: [u64; 2] = [global_width, global_height];
        let global_strides: [u64; 1] = [global_width * 2];
        let box_dim: [u32; 2] = [tile_width, tile_height];
        let element_strides: [u32; 2] = [1, 1];

        let result = unsafe {
            cuTensorMapEncodeTiled(
                tensor_map.as_mut_ptr(),
                CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                2,
                global_address,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
                CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };

        if result != 0 {
            return Err(format!(
                "cuTensorMapEncodeTiled (SWIZZLE_128B) failed: error code {}",
                result
            ));
        }

        let descriptor = unsafe { tensor_map.assume_init() };
        Ok(Self {
            opaque: descriptor.opaque,
        })
    }
}

/// Get the raw pointer to the descriptor data for passing to a kernel.
pub fn to_descriptor_ptr(desc: &HostTmaDescriptor) -> *const cuda_device::tma::TmaDescriptor {
    // The descriptor data lives in the opaque field.
    // We cast the pointer to the first u64 element to match the TmaDescriptor layout.
    desc.opaque.as_ptr() as *const cuda_device::tma::TmaDescriptor
}
