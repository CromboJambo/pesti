use crate::cuda_runtime::is_available;
use crate::device_discovery::LocalDevice;
use crate::error::RunnerError;
use crate::remote_discovery::RemoteDevice;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Device backend for tensor computation.
///
/// CUDA/CPU/MKL backend selection.
pub struct DeviceBackend {
    pub preference: String,
    pub device: Device,
}

impl DeviceBackend {
    pub fn new(preference: impl Into<String>) -> Self {
        Self {
            preference: preference.into(),
            device: Device::Cpu,
        }
    }

    /// Select device based on preference.
    pub fn select(&mut self) -> Result<(), RunnerError> {
        match self.preference.as_str() {
            "cuda" if is_available() => {
                // CUDA available, use it
                self.device = Device::cuda_if_available(0)
                    .map_err(|e: candle_core::Error| RunnerError::Device(e.to_string()))?;
            }
            "cuda" => {
                // CUDA requested but not available, warn and fall back to CPU
                tracing::warn!("CUDA requested but unavailable, falling back to CPU");
                self.device = Device::Cpu;
            }
            "cpu" => {
                self.device = Device::Cpu;
            }
            "mkl" => {
                self.device = Device::Cpu;
            }
            "accelerate" => {
                self.device = Device::Cpu;
            }
            _ => {
                // Unknown preference, fall back to CPU
                tracing::warn!(
                    "Unknown device backend '{}', falling back to CPU",
                    self.preference
                );
                self.device = Device::Cpu;
            }
        }
        let info = self.info().unwrap_or_default();
        debug!(preference = %self.preference, device = %info, "Device backend: selected");
        Ok(())
    }

    /// Get device info.
    pub fn info(&self) -> Result<String, RunnerError> {
        Ok(match &self.device {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(ordinal) => format!("cuda:{ordinal:?}"),
            Device::Metal(_) => "metal".to_string(),
        })
    }

    pub fn is_available(&self) -> Result<bool, RunnerError> {
        Ok(match &self.device {
            Device::Cpu => true,
            Device::Cuda(_) => true,
            Device::Metal(_) => false,
        })
    }
}

/// Hybrid device selector with priority routing.
///
/// Routes inference requests based on:
/// 1. Model size vs available VRAM
/// 2. Priority list from config
/// 3. Health check of remote endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSelector {
    /// Priority list of device types.
    pub priority: Vec<DeviceType>,
    /// Local devices discovered at last scan.
    local_devices: Vec<LocalDevice>,
    /// Remote devices discovered at last scan.
    remote_devices: Vec<RemoteDevice>,
}

/// Type of compute device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    /// Local CUDA GPU by ordinal.
    LocalGpu(u32),
    /// Remote LM Studio instance.
    Remote(String),
    /// CPU fallback.
    Cpu,
}

impl DeviceSelector {
    /// Create a new device selector with default priority: GPU0 → GPU1 → Remote → CPU.
    pub fn new() -> Self {
        let local = crate::device_discovery::discover_local_devices();
        let priority = local
            .iter()
            .filter(|d| d.ordinal != u32::MAX)
            .map(|d| DeviceType::LocalGpu(d.ordinal))
            .collect();

        Self {
            priority,
            local_devices: local,
            remote_devices: Vec::new(),
        }
    }

    /// Create selector with explicit priority list.
    pub fn with_priority(priority: Vec<DeviceType>) -> Self {
        Self {
            priority,
            local_devices: crate::device_discovery::discover_local_devices(),
            remote_devices: Vec::new(),
        }
    }

    /// Refresh discovered devices.
    pub async fn refresh(&mut self) {
        // Force initialization and device enumeration upfront to resolve state issues
        if let Err(e) = crate::cuda_runtime::enumerate_devices() {
            tracing::warn!("CUDA runtime pre-check failed during refresh: {}", e);
        }
        self.local_devices = crate::device_discovery::discover_local_devices();
        self.remote_devices = crate::remote_discovery::get_healthy_remote_devices().await;
    }
    /// Select device for a model of the given size (in bytes).
    pub async fn select_for_model(&mut self, model_bytes: u64) -> DeviceSelection {
        self.refresh().await;

        for device_type in &self.priority {
            match device_type {
                DeviceType::LocalGpu(ordinal) => {
                    if let Some(device) = self
                        .local_devices
                        .iter()
                        .find(|d| d.ordinal == *ordinal && d.can_hold_model(model_bytes))
                    {
                        return DeviceSelection {
                            device_type: device_type.clone(),
                            selected: LocalDevice::clone(device),
                            remote: None,
                            reason: format!(
                                "Priority: {} ({} GiB free)",
                                device.name,
                                device.free_vram / 1024 / 1024 / 1024
                            ),
                        };
                    }
                }
                DeviceType::Remote(_) => {
                    if let Some(remote) = self
                        .remote_devices
                        .iter()
                        .find(|d| d.can_hold_model(model_bytes))
                    {
                        return DeviceSelection {
                            device_type: device_type.clone(),
                            selected: LocalDevice::cpu_fallback(),
                            remote: Some(remote.clone()),
                            reason: format!(
                                "Priority: {} (latency: {}ms)",
                                remote.name, remote.latency_ms
                            ),
                        };
                    }
                }
                DeviceType::Cpu => {
                    return DeviceSelection {
                        device_type: DeviceType::Cpu,
                        selected: LocalDevice::cpu_fallback(),
                        remote: None,
                        reason: "CPU fallback".to_string(),
                    };
                }
            }
        }

        if let Some(best) = crate::device_discovery::select_best_gpu(model_bytes) {
            return DeviceSelection {
                device_type: DeviceType::LocalGpu(best.ordinal),
                selected: best,
                remote: None,
                reason: "Best available GPU".to_string(),
            };
        }

        DeviceSelection {
            device_type: DeviceType::Cpu,
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "CPU fallback (no GPU available)".to_string(),
        }
    }

    /// List all available devices.
    pub fn list_available(&self) -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        for local in &self.local_devices {
            devices.push(DeviceInfo {
                name: local.name.clone(),
                device_type: "local_gpu".to_string(),
                vram_total: Some(local.total_vram),
                vram_free: Some(local.free_vram),
                available: local.available,
                ordinal: Some(local.ordinal),
                endpoint: None,
            });
        }

        for remote in &self.remote_devices {
            devices.push(DeviceInfo {
                name: remote.name.clone(),
                device_type: "remote".to_string(),
                vram_total: remote.vram_total,
                vram_free: remote.vram_free,
                available: remote.healthy,
                ordinal: None,
                endpoint: Some(remote.endpoint.clone()),
            });
        }

        devices.push(DeviceInfo {
            name: "CPU".to_string(),
            device_type: "cpu".to_string(),
            vram_total: None,
            vram_free: None,
            available: true,
            ordinal: None,
            endpoint: None,
        });

        devices
    }

    /// Get device selection for a model (without refreshing).
    pub async fn quick_select(&self, model_bytes: u64) -> DeviceSelection {
        for device_type in &self.priority {
            match device_type {
                DeviceType::LocalGpu(ordinal) => {
                    if let Some(device) = self
                        .local_devices
                        .iter()
                        .find(|d| d.ordinal == *ordinal && d.can_hold_model(model_bytes))
                    {
                        return DeviceSelection {
                            device_type: device_type.clone(),
                            selected: LocalDevice::clone(device),
                            remote: None,
                            reason: format!(
                                "Priority: {} ({} GiB free)",
                                device.name,
                                device.free_vram / 1024 / 1024 / 1024
                            ),
                        };
                    }
                }
                DeviceType::Remote(_) => {
                    if let Some(remote) = self
                        .remote_devices
                        .iter()
                        .find(|d| d.healthy && d.can_hold_model(model_bytes))
                    {
                        return DeviceSelection {
                            device_type: device_type.clone(),
                            selected: LocalDevice::cpu_fallback(),
                            remote: Some(remote.clone()),
                            reason: format!(
                                "Priority: {} (latency: {}ms)",
                                remote.name, remote.latency_ms
                            ),
                        };
                    }
                }
                DeviceType::Cpu => {
                    return DeviceSelection {
                        device_type: DeviceType::Cpu,
                        selected: LocalDevice::cpu_fallback(),
                        remote: None,
                        reason: "CPU fallback".to_string(),
                    };
                }
            }
        }

        if let Some(best) = crate::device_discovery::select_best_gpu(model_bytes) {
            return DeviceSelection {
                device_type: DeviceType::LocalGpu(best.ordinal),
                selected: best,
                remote: None,
                reason: "Best available GPU".to_string(),
            };
        }

        DeviceSelection {
            device_type: DeviceType::Cpu,
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "CPU fallback (no GPU available)".to_string(),
        }
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a discovered device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    pub device_type: String,
    pub vram_total: Option<u64>,
    pub vram_free: Option<u64>,
    pub available: bool,
    pub ordinal: Option<u32>,
    pub endpoint: Option<String>,
}

/// Result of device selection.
#[derive(Debug, Clone)]
pub struct DeviceSelection {
    pub device_type: DeviceType,
    pub selected: LocalDevice,
    pub remote: Option<RemoteDevice>,
    pub reason: String,
}

impl DeviceSelection {
    /// Whether inference should be routed to a remote device.
    pub fn is_remote(&self) -> bool {
        self.remote.is_some()
    }

    /// Get the remote endpoint if available.
    pub fn remote_endpoint(&self) -> Option<&str> {
        self.remote.as_ref().map(|r| r.endpoint.as_str())
    }
}

impl LocalDevice {
    /// Create CPU fallback device.
    pub(crate) fn cpu_fallback() -> Self {
        Self {
            ordinal: u32::MAX,
            name: "CPU (fallback)".to_string(),
            total_vram: 0,
            free_vram: 0,
            compute_capability: "N/A".to_string(),
            available: true,
            used_vram: 0,
        }
    }
}
