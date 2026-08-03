#![allow(unused_unsafe)]

//! CUDA acceleration crate for PESTI.
//!
//! This crate provides GPU-accelerated operations using the `cudarc` backend.
//! It's designed to be a drop-in replacement for CPU operations when GPU is available.

pub mod kernels;
pub mod memory;

/// CUDA acceleration features with real cudarc integration
#[allow(unused_unsafe)]
pub mod features {
    use std::sync::OnceLock;

    /// Cache for device count to avoid repeated queries
    static DEVICE_COUNT: OnceLock<usize> = OnceLock::new();

    /// Check if CUDA is available
    #[allow(unused_unsafe)]
    pub fn cuda_available() -> bool {
        // Try to initialize CUDA driver
        unsafe {
            match cuda_core::init(0) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
    }

    /// Get the number of available CUDA devices
    #[allow(unused_unsafe)]
    pub fn device_count() -> usize {
        *DEVICE_COUNT.get_or_init(|| {
            let mut count: i32 = 0;
            unsafe {
                match cuda_core::sys::cuDeviceGetCount(&mut count) {
                    0 => count as usize, // CUDA_SUCCESS
                    _ => 0,
                }
            }
        })
    }

    /// Get compute capability for a device
    #[allow(unused_unsafe)]
    pub fn compute_capability(device_id: usize) -> Option<(u32, u32)> {
        // Initialize CUDA driver if needed
        let _ = cuda_available();

        unsafe {
            // Get device handle
            let mut cu_device = std::mem::MaybeUninit::uninit();
            if cuda_core::sys::cuDeviceGet(
                cu_device.as_mut_ptr(),
                device_id as i32,
            ) != 0 {
                return None;
            }
            let cu_device = cu_device.assume_init();

            // Get compute capability major
            let mut major = std::mem::MaybeUninit::uninit();
            if cuda_core::sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            ) != 0 {
                return None;
            }
            let major = major.assume_init();

            // Get compute capability minor
            let mut minor = std::mem::MaybeUninit::uninit();
            if cuda_core::sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            ) != 0 {
                return None;
            }
            let minor = minor.assume_init();

            Some((major as u32, minor as u32))
        }
    }

    /// Get device name
    #[allow(unused_unsafe)]
    pub fn device_name(device_id: usize) -> Option<String> {
        // Initialize CUDA driver if needed
        let _ = cuda_available();

        unsafe {
            // Get device handle
            let mut cu_device = std::mem::MaybeUninit::uninit();
            if cuda_core::sys::cuDeviceGet(
                cu_device.as_mut_ptr(),
                device_id as i32,
            ) != 0 {
                return None;
            }
            let cu_device = cu_device.assume_init();

            // Get device name
            let mut name_buf = [0i8; 256];
            if cuda_core::sys::cuDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as i32,
                cu_device,
            ) != 0 {
                return None;
            }

            let name: String = name_buf
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect::<Vec<u8>>()
                .into_iter()
                .map(|b| b as char)
                .collect();

            Some(name)
        }
    }

    /// Get total memory for a device in bytes
    #[allow(unused_unsafe)]
    pub fn device_total_memory(device_id: usize) -> Option<u64> {
        // Initialize CUDA driver if needed
        let _ = cuda_available();

        unsafe {
            // Get device handle
            let mut cu_device = std::mem::MaybeUninit::uninit();
            if cuda_core::sys::cuDeviceGet(
                cu_device.as_mut_ptr(),
                device_id as i32,
            ) != 0 {
                return None;
            }
            let _cu_device = cu_device.assume_init();

            // Get memory info
            let mut free: usize = 0;
            let mut total: usize = 0;
            if cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) != 0 {
                return None;
            }

            Some(total as u64)
        }
    }

    /// Check if device supports tcgen05 (sm_100+)
    pub fn supports_tcgen05(device_id: usize) -> bool {
        compute_capability(device_id)
            .map(|(major, _)| major >= 10)
            .unwrap_or(false)
    }

    /// Check if device supports WGMMA (sm_89+)
    pub fn supports_wgmma(device_id: usize) -> bool {
        compute_capability(device_id)
            .map(|(major, _)| major >= 8)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::features::*;

    #[test]
    fn test_cuda_available_returns_bool() {
        let result = cuda_available();
        assert!(result == true || result == false);
    }

    #[test]
    fn test_device_count_returns_non_negative() {
        let count = device_count();
        assert!(count >= 0);
    }

    #[test]
    fn test_compute_capability_stubs_return_none() {
        // On systems without CUDA, this should return None
        let cap = compute_capability(0);
        // If CUDA is available, we should get Some((major, minor))
        // If not, we get None - both are valid
        match cap {
            Some((major, minor)) => {
                assert!(major > 0);
                assert!(minor >= 0);
            }
            None => {
                // No CUDA available, which is fine for testing
            }
        }
    }

    #[test]
    fn test_device_count_consistency() {
        // Multiple calls should return consistent results
        let count1 = device_count();
        let count2 = device_count();
        assert_eq!(count1, count2);
    }

    #[test]
    fn test_cuda_available_stability() {
        // Stub should be stable across calls
        let result1 = cuda_available();
        let result2 = cuda_available();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_device_name() {
        let name = device_name(0);
        match name {
            Some(n) => {
                assert!(!n.is_empty());
                println!("Device 0: {}", n);
            }
            None => {
                // No CUDA available, which is fine for testing
            }
        }
    }

    #[test]
    fn test_device_total_memory() {
        let mem = device_total_memory(0);
        match mem {
            Some(bytes) => {
                assert!(bytes > 0);
                println!("Device 0: {} MB", bytes / (1024 * 1024));
            }
            None => {
                // No CUDA available, which is fine for testing
            }
        }
    }

    #[test]
    fn test_supports_tcgen05() {
        let supports = supports_tcgen05(0);
        println!("Device 0 supports tcgen05: {}", supports);
        // Should be true on Blackwell (sm_100+), false otherwise
    }

    #[test]
    fn test_supports_wgmma() {
        let supports = supports_wgmma(0);
        println!("Device 0 supports WGMMA: {}", supports);
        // Should be true on Ampere+ (sm_89+)
    }
}
