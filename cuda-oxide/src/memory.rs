//! CUDA memory management utilities.

use cudarc::driver::{CudaDevice, CudaSlice};

/// Allocate GPU memory for f32 data
pub fn alloc_f32(device: &CudaDevice, size: usize) -> Result<CudaSlice<f32>, String> {
    device
        .alloc_zeros::<f32>(size)
        .map_err(|e| format!("Failed to allocate GPU memory: {}", e))
}

/// Copy data from CPU to GPU
pub fn upload_f32(
    device: &CudaDevice,
    data: &[f32],
) -> Result<CudaSlice<f32>, String> {
    device
        .alloc_from_vec(data)
        .map_err(|e| format!("Failed to upload data to GPU: {}", e))
}

/// Copy data from GPU to CPU
pub fn download_f32(
    device: &CudaDevice,
    gpu_data: &CudaSlice<f32>,
) -> Result<Vec<f32>, String> {
    device
        .copy_slice(gpu_data)
        .map_err(|e| format!("Failed to download data from GPU: {}", e))
}
