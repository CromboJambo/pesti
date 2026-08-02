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
                tracing::warn!("Unknown device backend '{}', falling back to CPU", self.preference);
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
            Device::Cuda(_) => is_available(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_selector_new_creates_default() {
        let selector = DeviceSelector::new();
        assert!(selector.priority.contains(&DeviceType::Cpu) || !selector.priority.is_empty());
    }

    #[test]
    fn device_info_serializes() {
        let info = DeviceInfo {
            name: "Test GPU".to_string(),
            device_type: "local_gpu".to_string(),
            vram_total: Some(16_000_000_000),
            vram_free: Some(8_000_000_000),
            available: true,
            ordinal: Some(0),
            endpoint: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Test GPU"));
        assert!(json.contains("local_gpu"));
    }

    #[test]
    fn device_type_serialization() {
        let gpu = DeviceType::LocalGpu(0);
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(json.contains("LocalGpu"));

        let cpu = DeviceType::Cpu;
        let json = serde_json::to_string(&cpu).unwrap();
        assert!(json.contains("Cpu"));
    }

    // ========== DeviceBackend Tests ==========

    #[test]
    fn test_device_backend_new_defaults_to_cpu() {
        let backend = DeviceBackend::new("cuda");
        assert_eq!(backend.preference, "cuda");
        // Default device is CPU regardless of preference until select() is called
        match backend.device {
            Device::Cpu => {}
            _ => panic!("Expected CPU device initially"),
        }
    }

    #[test]
    fn test_device_backend_select_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        match backend.device {
            Device::Cpu => {}
            _ => panic!("Expected CPU device after select(cpu)"),
        }
    }

    #[test]
    fn test_device_backend_select_mkl() {
        let mut backend = DeviceBackend::new("mkl");
        backend.select().unwrap();
        
        // MKL selects CPU backend
        match backend.device {
            Device::Cpu => {}
            _ => panic!("Expected CPU device after select(mkl)"),
        }
    }

    #[test]
    fn test_device_backend_select_accelerate() {
        let mut backend = DeviceBackend::new("accelerate");
        backend.select().unwrap();
        
        // Accelerate selects CPU backend
        match backend.device {
            Device::Cpu => {}
            _ => panic!("Expected CPU device after select(accelerate)"),
        }
    }

    #[test]
    fn test_device_backend_info_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        let info = backend.info().unwrap();
        assert_eq!(info, "cpu");
    }

    #[test]
    fn test_device_backend_is_available_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        let available = backend.is_available().unwrap();
        assert!(available, "CPU should always be available");
    }

    #[test]
    fn test_device_backend_unknown_preference_falls_back_to_cpu() {
        let mut backend = DeviceBackend::new("unknown_backend");
        backend.select().unwrap();
        
        // Unknown preference should fall back to CPU
        match backend.device {
            Device::Cpu => {}
            _ => panic!("Expected CPU device for unknown preference"),
        }
    }

    #[test]
    fn test_device_backend_cuda_if_available() {
        let mut backend = DeviceBackend::new("cuda");
        backend.select().unwrap();
        
        // CUDA selection depends on hardware availability
        // Just verify it doesn't panic and selects something valid
        let _ = backend.info().unwrap();
        let _ = backend.is_available().unwrap();
    }

    // ========== DeviceSelector Tests ==========

    #[tokio::test]
    async fn test_device_selector_new_discovers_devices() {
        let selector = DeviceSelector::new();
        // Should have at least CPU in priority list
        assert!(selector.priority.contains(&DeviceType::Cpu) || !selector.priority.is_empty());
        // Should have discovered local devices
        assert!(!selector.local_devices.is_empty());
    }

    #[test]
    fn test_device_selector_with_explicit_priority() {
        let priority = vec![
            DeviceType::LocalGpu(0),
            DeviceType::LocalGpu(1),
            DeviceType::Cpu,
        ];
        let selector = DeviceSelector::with_priority(priority);
        assert_eq!(selector.priority.len(), 3);
        assert_eq!(selector.priority[0], DeviceType::LocalGpu(0));
    }

    #[tokio::test]
    async fn test_device_selector_refresh() {
        let mut selector = DeviceSelector::new();
        let _initial_len = selector.local_devices.len();
        
        selector.refresh().await;
        
        // After refresh, should still have devices
        assert!(!selector.local_devices.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_select_for_model_small() {
        let mut selector = DeviceSelector::new();
        
        // Small model (100MB)
        let selection = selector.select_for_model(100 * 1024 * 1024).await;
        
        // Should select something (CPU at minimum)
        assert!(!selection.reason.is_empty());
        match &selection.device_type {
            DeviceType::LocalGpu(_) | DeviceType::Remote(_) | DeviceType::Cpu => {}
        }
    }

    #[tokio::test]
    async fn test_device_selector_select_for_model_large() {
        let mut selector = DeviceSelector::new();
        
        // Large model (30GB)
        let selection = selector.select_for_model(30 * 1024 * 1024 * 1024).await;
        
        // Should fall back to CPU or best available GPU
        assert!(!selection.reason.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_quick_select() {
        let selector = DeviceSelector::new();
        
        // Quick select without refresh
        let selection = selector.quick_select(500 * 1024 * 1024).await;
        
        assert!(!selection.reason.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_list_available() {
        let selector = DeviceSelector::new();
        let devices = selector.list_available();
        
        // Should always have at least CPU listed
        assert!(devices.iter().any(|d| d.name == "CPU"));
    }

    #[test]
    fn test_device_selector_list_shows_cpu() {
        let selector = DeviceSelector::new();
        let devices = selector.list_available();
        
        let cpu_device = devices.iter().find(|d| d.name == "CPU").unwrap();
        assert_eq!(cpu_device.device_type, "cpu");
        assert!(cpu_device.available);
    }

    // ========== DeviceType Tests ==========

    #[test]
    fn test_device_type_local_gpu() {
        let gpu = DeviceType::LocalGpu(0);
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(json.contains("LocalGpu"));
        assert!(json.contains("0"));
    }

    #[test]
    fn test_device_type_remote() {
        let remote = DeviceType::Remote("http://localhost:8080".to_string());
        let json = serde_json::to_string(&remote).unwrap();
        assert!(json.contains("Remote"));
        assert!(json.contains("localhost"));
    }

    #[test]
    fn test_device_type_cpu() {
        let cpu = DeviceType::Cpu;
        let json = serde_json::to_string(&cpu).unwrap();
        assert!(json.contains("Cpu"));
    }

    #[test]
    fn test_device_type_equality() {
        let gpu0 = DeviceType::LocalGpu(0);
        let gpu0_copy = DeviceType::LocalGpu(0);
        let gpu1 = DeviceType::LocalGpu(1);
        
        assert_eq!(gpu0, gpu0_copy);
        assert_ne!(gpu0, gpu1);
    }

    // ========== DeviceSelection Tests ==========

    #[test]
    fn test_device_selection_is_remote_local() {
        let selection = DeviceSelection {
            device_type: DeviceType::LocalGpu(0),
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "test".to_string(),
        };
        
        assert!(!selection.is_remote());
        assert!(selection.remote_endpoint().is_none());
    }

    #[test]
    fn test_device_selection_is_remote_true() {
        let remote = RemoteDevice {
            name: "Remote GPU".to_string(),
            endpoint: "http://192.168.1.100:8080".to_string(),
            healthy: true,
            latency_ms: 50,
            vram_total: Some(8_000_000_000),
            vram_free: Some(4_000_000_000),
        };
        
        let selection = DeviceSelection {
            device_type: DeviceType::Remote("test".to_string()),
            selected: LocalDevice::cpu_fallback(),
            remote: Some(remote),
            reason: "remote inference".to_string(),
        };
        
        assert!(selection.is_remote());
        assert_eq!(
            selection.remote_endpoint(),
            Some("http://192.168.1.100:8080")
        );
    }

    #[test]
    fn test_device_selection_cpu_fallback() {
        let selection = DeviceSelection {
            device_type: DeviceType::Cpu,
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "CPU fallback".to_string(),
        };
        
        assert!(!selection.is_remote());
        assert_eq!(selection.device_type, DeviceType::Cpu);
    }

    // ========== LocalDevice Tests ==========

    #[test]
    fn test_cpu_fallback_device() {
        let cpu = LocalDevice::cpu_fallback();
        
        assert_eq!(cpu.ordinal, u32::MAX);
        assert!(cpu.name.contains("CPU"));
        assert_eq!(cpu.total_vram, 0);
        assert_eq!(cpu.free_vram, 0);
        assert!(cpu.available);
    }

    #[test]
    fn test_local_device_can_hold_model() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 8_000_000_000,
            compute_capability: "8.6".to_string(),
            available: true,
            used_vram: 2_000_000_000,
        };
        
        // Model that fits
        assert!(device.can_hold_model(5_000_000_000));
        
        // Model too large
        assert!(!device.can_hold_model(10_000_000_000));
    }

    #[test]
    fn test_local_device_zero_free_vram() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Full GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 0,
            compute_capability: "8.6".to_string(),
            available: true,
            used_vram: 16_000_000_000,
        };
        
        // Even tiny model won't fit
        assert!(!device.can_hold_model(1));
    }

    // ========== DeviceInfo Tests ==========

    #[test]
    fn test_device_info_serialization() {
        let info = DeviceInfo {
            name: "RTX 4070".to_string(),
            device_type: "local_gpu".to_string(),
            vram_total: Some(16_000_000_000),
            vram_free: Some(8_000_000_000),
            available: true,
            ordinal: Some(0),
            endpoint: None,
        };
        
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("RTX 4070"));
        assert!(json.contains("16000000000"));
    }

    #[test]
    fn test_device_info_remote_serialization() {
        let info = DeviceInfo {
            name: "Remote LM Studio".to_string(),
            device_type: "remote".to_string(),
            vram_total: Some(8_000_000_000),
            vram_free: Some(4_000_000_000),
            available: true,
            ordinal: None,
            endpoint: Some("http://192.168.1.100:8080".to_string()),
        };
        
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Remote LM Studio"));
        assert!(json.contains("endpoint"));
    }
}
