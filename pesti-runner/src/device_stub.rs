//! Stub device module for CPU-only builds.
//!
//! Provides LocalDevice and DeviceSelector stubs to match the real API.

/// Dummy device info (mirrors real DeviceInfo)
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
}

/// Dummy device backend (mirrors real DeviceBackend)
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

/// Device type enum (mirrors real DeviceType)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Cpu,
    Gpu,
}

/// Device selection result (mirrors real DeviceSelection)
#[derive(Debug)]
pub struct DeviceSelection {
    pub device_type: DeviceType,
    pub ordinal: usize,
}

/// LocalDevice stub (mirrors real LocalDevice from device_discovery.rs)
#[derive(Debug, Clone)]
pub struct LocalDevice {
    pub name: String,
    pub memory_mb: u64,
}

impl LocalDevice {
    pub fn cpu_fallback() -> Self {
        Self {
            name: "CPU".to_string(),
            memory_mb: 0,
        }
    }
}

/// Selector for choosing devices (mirrors real DeviceSelector)
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
