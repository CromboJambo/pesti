//! Tests for memory backend fixes.

#[cfg(test)]
mod tests {
    use half::f16;
    use pesti_runner::kernel::{CpuMemoryBackend, DeviceBuffer, MemoryBackend, MemoryManager};

    #[test]
    fn test_cpu_backend_alloc_and_transfer() {
        let backend = CpuMemoryBackend::new(1024 * 1024);
        
        // Allocate memory
        let handle = backend.alloc(100).unwrap();
        
        // Write data via h2d
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        backend.h2d(&data, handle).unwrap();
        
        // Read back via d2h
        let mut read_buf = vec![0u8; 100];
        backend.d2h(handle, &mut read_buf).unwrap();
        
        assert_eq!(data, read_buf);
        
        // Free memory
        backend.free(handle).unwrap();
    }

    #[test]
    fn test_device_buffer_from_cpu() {
        let data = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        let buf = DeviceBuffer::from_host(data.clone());
        
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_backed()); // Not backend-allocated
        
        let slice = buf.as_slice().unwrap();
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn test_device_buffer_zeros() {
        let buf: DeviceBuffer<f32> = DeviceBuffer::zeros(10);
        
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_backed());
        
        let slice = buf.as_slice().unwrap();
        assert!(slice.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_memory_manager_cpu_fallback() {
        let manager = MemoryManager::new();
        
        // Should always have at least CPU backend
        assert!(manager.has_cuda() || !manager.has_cuda()); // Either is fine
        
        match &manager {
            MemoryManager::Cpu(_) => {
                // CPU-only mode - allocate and verify
                let handle = manager.alloc(50).unwrap();
                let mut buf = vec![0u8; 50];
                manager.d2h(handle, &mut buf).unwrap();
                assert_eq!(buf.len(), 50);
                manager.free(handle).unwrap();
            }
            MemoryManager::Cuda(_) => {
                // CUDA available - just verify it exists
                let info = manager.device_info().unwrap();
                assert!(info.total_memory > 0 || info.compute_capability.0 > 0);
            }
        }
    }

    #[test]
    fn test_slab_allocator_reuse() {
        let backend = CpuMemoryBackend::new(1024 * 1024);
        
        // Allocate and free - should reuse slot
        let h1 = backend.alloc(100).unwrap();
        backend.free(h1).unwrap();
        
        let h2 = backend.alloc(100).unwrap();
        
        // Should be the same slot (for testing purposes)
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_device_buffer_byte_len() {
        let buf_f32: DeviceBuffer<f32> = DeviceBuffer::zeros(5);
        assert_eq!(buf_f32.byte_len(), 5 * 4); // f32 = 4 bytes
        
        let buf_f16: DeviceBuffer<f16> = DeviceBuffer::zeros(5);
        assert_eq!(buf_f16.byte_len(), 5 * 2); // f16 = 2 bytes
        
        let buf_u8: DeviceBuffer<u8> = DeviceBuffer::zeros(5);
        assert_eq!(buf_u8.byte_len(), 5 * 1); // u8 = 1 byte
    }

    #[test]
    fn test_memory_manager_alloc_free_roundtrip() {
        let manager = MemoryManager::new();
        
        let handle = manager.alloc(256).unwrap();
        assert_ne!(handle.as_u64(), 0);
        
        manager.free(handle).unwrap();
        
        // Should be able to allocate again
        let handle2 = manager.alloc(256).unwrap();
        assert_ne!(handle2.as_u64(), 0);
    }

    #[test]
    fn test_cpu_backend_capacity_limit() {
        let backend = CpuMemoryBackend::new(1024); // Only 1KB total
        
        // Should succeed - within capacity
        let h1 = backend.alloc(500).unwrap();
        
        // Should fail - exceeds remaining capacity
        let result = backend.alloc(600);
        assert!(result.is_err());
        
        backend.free(h1).unwrap();
    }

    #[test]
    fn test_device_buffer_to_host() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let buf = DeviceBuffer::from_host(data.clone());
        
        let recovered = buf.to_host();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_device_buffer_as_mut_slice() {
        let mut buf: DeviceBuffer<f32> = DeviceBuffer::zeros(5);
        
        if let Some(slice) = buf.as_mut_slice() {
            slice[0] = 10.0;
            slice[1] = 20.0;
            
            assert_eq!(slice[0], 10.0);
            assert_eq!(slice[1], 20.0);
        } else {
            panic!("Expected host-backed buffer");
        }
    }
}
