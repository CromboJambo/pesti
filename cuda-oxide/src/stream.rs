//! CUDA stream management for async operations.

use cudarc::driver::{CudaDevice, CudaStream};

/// Get the default stream for a device
pub fn get_default_stream(device: &CudaDevice) -> CudaStream {
    device.default_stream().unwrap()
}

/// Create a new pinned stream
pub fn create_pinned_stream(device: &CudaDevice) -> Result<CudaStream, String> {
    device
        .create_stream()
        .map_err(|e| format!("Failed to create stream: {}", e))
}
