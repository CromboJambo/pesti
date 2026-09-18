//! Device-resident tensor abstraction for LLM inference.
//!
//! A `DeviceTensor` wraps a weight matrix and manages its lifecycle across CPU
//! and GPU memory. The key insight from Phase 2b: upload weights ONCE at model
//! load time, not per-forward-pass. This eliminates the dominant transfer
//! bottleneck in transformer inference.
//!
//! # Usage
//! ```rust
//! // Create from CPU data (model loading)
//! let mut t = DeviceTensor::from_cpu(vec![1.0, 2.0, 3.0], vec![3]);
//!
//! // Upload to GPU once
//! if let Some(gpu_buf) = &t.gpu_buffer {
//!     // Use gpu_buf directly in kernel launches
//! }
//! ```

use crate::kernel::device_buf::DeviceBuffer;
use half::f16;
use std::sync::Arc;

/// A tensor that can reside on CPU, GPU, or both.
///
/// The tensor owns its CPU data (`cpu_data`) and optionally a GPU buffer
/// (`gpu_buffer`). Transfers are explicit: `upload_to_gpu()` and
/// `download_from_gpu()`. This makes the memory lifecycle visible and
/// debuggable.
#[derive(Clone)]
pub struct DeviceTensor {
    /// Host-side f32 data. Always present for compatibility with CPU-only paths.
    pub cpu_data: Vec<f32>,

    /// GPU device buffer (f16 for GEMM efficiency). None if not uploaded.
    gpu_buffer: Option<Arc<DeviceBuffer<f16>>>,

    /// Tensor shape in row-major order (e.g., [out_features, in_features]).
    pub shape: Vec<usize>,
}

impl DeviceTensor {
    /// Create a tensor from CPU data with explicit shape.
    pub fn from_cpu(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape product {}",
            data.len(),
            expected
        );
        Self {
            cpu_data: data,
            gpu_buffer: None,
            shape,
        }
    }

    /// Upload the tensor to GPU memory as f16 (for GEMM efficiency).
    /// Idempotent: calling twice only uploads once.
    pub fn upload_to_gpu(&mut self) {
        if self.gpu_buffer.is_some() {
            return;
        }
        let f16_data: Vec<f16> = self.cpu_data.iter().map(|v| f16::from_f32(*v)).collect();
        let buf = DeviceBuffer::from_host(f16_data);
        self.gpu_buffer = Some(Arc::new(buf));
    }

    /// Check if the tensor is currently resident on GPU.
    pub fn is_on_gpu(&self) -> bool {
        self.gpu_buffer.is_some()
    }

    /// Get a reference to the GPU buffer, if uploaded.
    pub fn gpu_buffer(&self) -> Option<&Arc<DeviceBuffer<f16>>> {
        self.gpu_buffer.as_ref()
    }

    /// Download tensor from GPU back to CPU (f32).
    pub fn download_from_gpu(&mut self) {
        if let Some(gpu_buf) = &self.gpu_buffer {
            // Note: f16 -> f32 conversion happens in DeviceBuffer::to_host()
            let f16_data = gpu_buf.to_host();
            self.cpu_data = f16_data.iter().map(|v| v.to_f32()).collect();
        }
    }

    /// Total elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return shape as string for logging.
    pub fn shape_str(&self) -> String {
        self.shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_tensor_cpu_only() {
        let t = DeviceTensor::from_cpu(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.cpu_data, vec![1.0, 2.0, 3.0]);
        assert!(!t.is_on_gpu());
    }

    #[test]
    fn test_device_tensor_upload_download() {
        let mut t = DeviceTensor::from_cpu(vec![1.5, 2.5], vec![2]);
        t.upload_to_gpu();
        assert!(t.is_on_gpu());

        // Download back (f16 roundtrip)
        t.download_from_gpu();
        assert_eq!(t.cpu_data.len(), 2);
    }
}