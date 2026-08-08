//! Bridge between PESTI TMA descriptor and cudarc TMA infrastructure.
//!
//! PESTI's `TmaDescriptor` (128-bit u128) is a **speculative** hand-written bit layout
//! — the actual CUtensorMap encoding is opaque and not publicly documented.
//! This module provides the bridge to cudarc's sys bindings for `cuTensorMapEncodeTiled`.
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

use cudarc::driver::sys;
use std::mem::MaybeUninit;

/// Host-side TMA descriptor for f16 tensors.
///
/// Wraps `CUtensorMap` created via `cuTensorMapEncodeTiled`.
/// This is the 128-byte opaque descriptor passed to kernels.
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
        let mut tensor_map = MaybeUninit::<sys::CUtensorMap>::uninit();
        let global_dim: [u64; 2] = [global_width, global_height];
        // Byte stride between consecutive rows in global memory
        let global_strides: [u64; 1] = [global_width * 2]; // f16 = 2 bytes
        let box_dim: [u32; 2] = [tile_width, tile_height];
        let element_strides: [u32; 2] = [1, 1];

        let result = sys::cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        );

        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuTensorMapEncodeTiled failed: error code {result:?}"
            ));
        }

        let descriptor = tensor_map.assume_init();
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
        let mut tensor_map = MaybeUninit::<sys::CUtensorMap>::uninit();
        let global_dim: [u64; 2] = [global_width, global_height];
        let global_strides: [u64; 1] = [global_width * 2]; // f16 = 2 bytes
        let box_dim: [u32; 2] = [tile_width, tile_height];
        let element_strides: [u32; 2] = [1, 1];

        let result = sys::cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        );

        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!(
                "cuTensorMapEncodeTiled (SWIZZLE_128B) failed: error code {result:?}"
            ));
        }

        let descriptor = tensor_map.assume_init();
        Ok(Self {
            opaque: descriptor.opaque,
        })
    }
}
