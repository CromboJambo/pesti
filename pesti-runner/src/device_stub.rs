//! Stub device module for CPU-only builds.

use std::sync::Arc;

/// Dummy device info
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
}

/// Device backend preference
#[derive(Debug, Clone)]
pub struct DeviceBackend {
    pub preference: String,
    pub device: candle_core::Device,
}

impl DeviceBackend {
    pub fn new(preference: &str) -> Self {
        Self {
            preference: preference.to_string(),
            device: candle_core::Device::Cpu,
        }
    }

    pub fn select(&mut self) -> Result<(), crate::RunnerError> {
        Ok(())
    }
}

/// Device type enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Cpu,
    Gpu,
}

/// Device selection result
#[derive(Debug)]
pub struct DeviceSelection {
    pub device_type: DeviceType,
    pub ordinal: usize,
}

/// Selector for choosing devices
pub struct DeviceSelector {
    _private: (),
}

impl DeviceSelector {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn refresh(&mut self) {}

    pub fn select_for_model(&mut self, _model_bytes: u64) -> DeviceSelection {
        DeviceSelection {
            device_type: DeviceType::Cpu,
            ordinal: 0,
        }
    }

    pub fn quick_select(&self, _model_bytes: u64) -> DeviceSelection {
        DeviceSelection {
            device_type: DeviceType::Cpu,
            ordinal: 0,
        }
    }

    pub fn list_available(&self) -> Vec<DeviceInfo> {
        vec![DeviceInfo {
            name: "CPU".to_string(),
        }]
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new()
    }
}
