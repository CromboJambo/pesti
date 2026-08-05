//! Integration tests for TMA descriptor creation using cuTensorMapEncodeTiled

#[cfg(test)]
mod tma_integration_tests {
    use half::f16;

    // Import the bridge module
    #[cfg(feature = "cuda")]
    use pesti_runner::kernel::tma_bridge::HostTmaDescriptor;

    #[test]
    fn test_tma_descriptor_size() {
        // Verify the descriptor is 128 bytes as expected
        let desc = HostTmaDescriptor { opaque: [0u64; 16] };
        assert_eq!(std::mem::size_of_val(&desc.opaque), 128);
        assert_eq!(desc.opaque.len(), 16); // 16 x u64 = 128 bytes
    }

    #[test]
    fn test_tma_descriptor_zeroed() {
        let desc = HostTmaDescriptor { opaque: [0u64; 16] };
        for &word in &desc.opaque {
            assert_eq!(word, 0u64);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_tma_descriptor_create_f16() {
        use cuda_core::{CudaContext, CudaStream};
        use std::sync::Arc;

        // Initialize CUDA
        unsafe {
            cuda_core::init(0).expect("CUDA init should succeed");
        }

        let ctx = Arc::new(CudaContext::new(0).expect("Context creation should succeed"));
        let stream = Arc::new(ctx.new_stream().expect("Stream creation should succeed"));

        // Allocate device memory for test (128 f16 elements = 256 bytes)
        let num_elements = 128;
        let bytes = num_elements * 2; // f16 = 2 bytes
        let ptr = unsafe {
            cuda_core::memory::malloc_async(stream.cu_stream(), bytes)
                .expect("Device allocation should succeed")
        };

        // Write some test data to device
        let host_data: Vec<f16> = (0..num_elements).map(|i| f16::from_f32(i as f32)).collect();
        unsafe {
            cuda_core::memory::memcpy_htod_async::<u8>(
                ptr,
                host_data.as_ptr() as *const u8,
                bytes,
                stream.cu_stream(),
            )
            .expect("H2D copy should succeed");
        }

        // Create TMA descriptor for 128x1 tile (single row)
        let result = unsafe {
            HostTmaDescriptor::create_f16(
                ptr as *mut std::ffi::c_void,
                128, // width
                1,   // height
                64,  // tile_width
                1,   // tile_height
            )
        };

        assert!(result.is_ok(), "TMA descriptor creation should succeed: {:?}", result.err());
        let desc = result.unwrap();

        // Verify descriptor is non-zero (not all zeros means it was properly encoded)
        let has_non_zero = desc.opaque.iter().any(|&word| word != 0);
        assert!(has_non_zero, "TMA descriptor should have non-zero encoding");

        // Clean up
        unsafe {
            cuda_core::memory::free_async(ptr, stream.cu_stream())
                .expect("Free should succeed");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_tma_descriptor_create_swizzled() {
        use cuda_core::{CudaContext, CudaStream};
        use std::sync::Arc;

        unsafe {
            cuda_core::init(0).expect("CUDA init should succeed");
        }

        let ctx = Arc::new(CudaContext::new(0).expect("Context creation should succeed"));
        let stream = Arc::new(ctx.new_stream().expect("Stream creation should succeed"));

        let num_elements = 256;
        let bytes = num_elements * 2;
        let ptr = unsafe {
            cuda_core::memory::malloc_async(stream.cu_stream(), bytes)
                .expect("Device allocation should succeed")
        };

        // Create TMA descriptor with SWIZZLE_128B for tcgen05 compatibility
        let result = unsafe {
            HostTmaDescriptor::create_f16_swizzled(
                ptr as *mut std::ffi::c_void,
                128, // width
                2,   // height (for batched access)
                64,  // tile_width
                2,   // tile_height
            )
        };

        assert!(result.is_ok(), "Swizzled TMA descriptor creation should succeed: {:?}", result.err());
        let desc = result.unwrap();

        // Verify descriptor is properly sized
        assert_eq!(desc.opaque.len(), 16);

        unsafe {
            cuda_core::memory::free_async(ptr, stream.cu_stream())
                .expect("Free should succeed");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_tma_descriptor_vs_speculative() {
        use cuda_core::{CudaContext, CudaStream};
        use std::sync::Arc;
        use pesti_runner::kernel::tma_descriptor::TmaDescriptor;

        unsafe {
            cuda_core::init(0).expect("CUDA init should succeed");
        }

        let ctx = Arc::new(CudaContext::new(0).expect("Context creation should succeed"));
        let stream = Arc::new(ctx.new_stream().expect("Stream creation should succeed"));

        let num_elements = 64;
        let bytes = num_elements * 2;
        let ptr = unsafe {
            cuda_core::memory::malloc_async(stream.cu_stream(), bytes)
                .expect("Device allocation should succeed")
        };

        // Create real descriptor via cuTensorMapEncodeTiled
        let real_desc = unsafe {
            HostTmaDescriptor::create_f16(
                ptr as *mut std::ffi::c_void,
                64,
                1,
                32,
                1,
            )
        }.expect("Real descriptor creation should succeed");

        // Compare with speculative bit-packing approach
        let spec_desc = TmaDescriptor::new()
            .with_gmem_addr(0) // offset within buffer
            .with_box(64, 64, 64, 1, 64, 64)
            .with_element_info(1)
            .with_descriptor_type(1)
            .with_smem_config(0);

        // They should be different (one is real CUDA encoding, one is speculative)
        assert_ne!(real_desc.opaque[0], spec_desc.as_u32_words()[0]);

        unsafe {
            cuda_core::memory::free_async(ptr, stream.cu_stream())
                .expect("Free should succeed");
        }
    }
}
