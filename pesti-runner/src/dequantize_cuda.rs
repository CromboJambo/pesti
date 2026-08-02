//! CUDA-accelerated dequantization kernels.
//!
//! Uses `cuda-oxide` crate for GPU-based dequantization of GGUF tensors.
//! This provides significant speedups over CPU-based implementations,
//! especially for large tensor loads.

use cudarc::driver::CudaDevice;

use crate::dequantize::DequantizeError;

/// CUDA-accelerated Q4_0 dequantization.
pub fn dequantize_q4_0_cuda(
    device: &CudaDevice,
    data: &[u8],
    element_count: usize,
) -> Result<Vec<f32>, DequantizeError> {
    // Delegate to cuda-oxide crate for actual implementation
    use cuda_oxide::kernels::dequantize_q4_0_kernel;

    dequantize_q4_0_kernel(device, data, element_count)
        .map_err(|e| DequantizeError::Cuda(e))?;

    // TODO: Actually convert back to Vec<f32> after kernel returns CudaSlice
    Err(DequantizeError::NotImplemented("Q4_0 CUDA".to_string()))
}

/// CUDA-accelerated Q4_1 dequantization.
pub fn dequantize_q4_1_cuda(
    device: &CudaDevice,
    data: &[u8],
    element_count: usize,
) -> Result<Vec<f32>, DequantizeError> {
    // Delegate to cuda-oxide crate for actual implementation
    use cuda_oxide::kernels::dequantize_q4_1_kernel;

    dequantize_q4_1_kernel(device, data, element_count)
        .map_err(|e| DequantizeError::Cuda(e))?;

    Err(DequantizeError::NotImplemented("Q4_1 CUDA".to_string()))
}

/// CUDA-accelerated Q8_0 dequantization.
pub fn dequantize_q8_0_cuda(
    device: &CudaDevice,
    data: &[u8],
    element_count: usize,
) -> Result<Vec<f32>, DequantizeError> {
    // Delegate to cuda-oxide crate for actual implementation
    use cuda_oxide::kernels::dequantize_q8_0_kernel;

    dequantize_q8_0_kernel(device, data, element_count)
        .map_err(|e| DequantizeError::Cuda(e))?;

    Err(DequantizeError::NotImplemented("Q8_0 CUDA".to_string()))
}
