//! Stub inference engine for CPU-only builds.

use candle_core::{Device, DType};

/// Dummy inference engine (CPU-only mode)
pub struct InferenceEngine {
    pub device: Device,
    pub dtype: DType,
}

impl InferenceEngine {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }

    pub fn full_device_info(&self) -> Result<String, crate::RunnerError> {
        Ok(format!("CPU ({} {})", self.device, self.dtype))
    }
}
