//! Remote device discovery for LM Studio instances on network.
//!
//! Discovers LM Studio inference servers via health checks on known ports.
//! Supports Tailscale network addresses and configurable endpoints.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A discovered remote LM Studio instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDevice {
    /// Display name for the device.
    pub name: String,
    /// HTTP endpoint (e.g., "http://100.123.45.67:1234").
    pub endpoint: String,
    /// Whether the device passed health check.
    pub healthy: bool,
    /// Latency in milliseconds (0 if not checked).
    pub latency_ms: u64,
    /// Estimated VRAM from LM Studio API (None if unavailable).
    pub vram_total: Option<u64>,
    /// Estimated free VRAM from LM Studio API (None if unavailable).
    pub vram_free: Option<u64>,
}

impl RemoteDevice {
    /// Check if this device can hold a model of the given size.
    pub fn can_hold_model(&self, model_bytes: u64) -> bool {
        match self.vram_free {
            Some(free) => free >= model_bytes,
            None => true, // Unknown VRAM, assume available
        }
    }

    /// Health check the remote LM Studio instance.
    pub async fn health_check(&mut self) {
        let start = std::time::Instant::now();

        match tokio::time::timeout(
            std::time::Duration::from_secs(3),
            reqwest::get(format!("{}/health", self.endpoint)),
        )
        .await
        {
            Ok(Ok(response)) => {
                self.healthy = response.status().is_success();
                self.latency_ms = start.elapsed().as_millis() as u64;
                debug!(
                    endpoint = %self.endpoint,
                    healthy = self.healthy,
                    latency_ms = self.latency_ms,
                    "Remote device health check"
                );
            }
            Ok(Err(e)) => {
                self.healthy = false;
                self.latency_ms = start.elapsed().as_millis() as u64;
                debug!(error = %e, endpoint = %self.endpoint, "Remote device health check failed");
            }
            Err(_) => {
                self.healthy = false;
                self.latency_ms = start.elapsed().as_millis() as u64;
                debug!(endpoint = %self.endpoint, "Remote device health check timed out");
            }
        }
    }

    /// Fetch VRAM info from LM Studio API if available.
    pub async fn fetch_vram_info(&mut self) {
        // LM Studio exposes GPU info via /api/gpu endpoint
        let url = format!("{}/api/gpu", self.endpoint);
        match tokio::time::timeout(std::time::Duration::from_secs(2), reqwest::get(&url)).await {
            Ok(Ok(response)) => {
                if response.status().is_success()
                    && let Ok(json) = response.json::<serde_json::Value>().await
                {
                    if let Some(total) = json.get("totalMemory").and_then(|v| v.as_u64()) {
                        self.vram_total = Some(total);
                    }
                    if let Some(free) = json.get("freeMemory").and_then(|v| v.as_u64()) {
                        self.vram_free = Some(free);
                    }
                }
            }
            Ok(Err(e)) => {
                debug!(error = %e, "Failed to fetch VRAM info from remote device");
            }
            Err(_) => {
                debug!("Timeout fetching VRAM info from remote device");
            }
        }
    }
}

/// Remote device discovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteDiscoveryConfig {
    /// LM Studio default port.
    pub lm_studio_port: u16,
    /// Custom endpoints to probe (e.g., Tailscale IPs).
    pub endpoints: Vec<String>,
    /// Whether to probe local network via UDP multicast.
    pub probe_local_network: bool,
}

impl Default for RemoteDiscoveryConfig {
    fn default() -> Self {
        Self {
            lm_studio_port: 1234,
            endpoints: Vec::new(),
            probe_local_network: false,
        }
    }
}

impl RemoteDiscoveryConfig {
    /// Create config from environment variables.
    ///
    /// Supports:
    /// - `LLM_REMOTE_ENDPOINTS` — semicolon-separated list of endpoints
    /// - `LLM_LM_STUDIO_PORT` — custom LM Studio port
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(ports) = std::env::var("LLM_LM_STUDIO_PORT")
            && let Ok(port) = ports.trim().parse::<u16>()
        {
            config.lm_studio_port = port;
        }

        if let Ok(endpoints) = std::env::var("LLM_REMOTE_ENDPOINTS") {
            config.endpoints = endpoints
                .split(';')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let s = s.trim();
                    if s.contains("://") {
                        s.to_string()
                    } else {
                        format!("http://{s}:{}", config.lm_studio_port)
                    }
                })
                .collect();
        }

        config
    }

    /// Get all endpoints to probe.
    pub fn all_endpoints(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut endpoints = Vec::new();

        // Add default LM Studio endpoints
        let defaults = vec![
            format!("http://localhost:{}", self.lm_studio_port),
            format!("http://127.0.0.1:{}", self.lm_studio_port),
        ];

        for ep in defaults {
            if seen.insert(ep.clone()) {
                endpoints.push(ep);
            }
        }

        // Add custom endpoints
        for ep in &self.endpoints {
            if seen.insert(ep.clone()) {
                endpoints.push(ep.clone());
            }
        }

        endpoints
    }
}

/// Discover remote LM Studio instances.
///
/// Probes configured endpoints and performs health checks.
pub async fn discover_remote_devices(config: &RemoteDiscoveryConfig) -> Vec<RemoteDevice> {
    let mut devices = Vec::new();

    for endpoint in config.all_endpoints() {
        let name = if endpoint.contains("127.0.0.1") || endpoint.contains("localhost") {
            "Local LM Studio".to_string()
        } else {
            format!("Remote LM Studio ({})", endpoint)
        };

        let mut device = RemoteDevice {
            name,
            endpoint: endpoint.clone(),
            healthy: false,
            latency_ms: 0,
            vram_total: None,
            vram_free: None,
        };

        device.health_check().await;

        if device.healthy {
            device.fetch_vram_info().await;
        }

        devices.push(device);
    }

    debug!(count = devices.len(), "Remote device discovery complete");
    devices
}

/// Find the best available remote device for a model.
pub async fn select_best_remote(model_bytes: u64) -> Option<RemoteDevice> {
    let config = RemoteDiscoveryConfig::from_env();
    let devices = discover_remote_devices(&config).await;

    devices
        .into_iter()
        .filter(|d| d.healthy)
        .filter(|d| d.can_hold_model(model_bytes))
        .min_by_key(|d| d.latency_ms) // Prefer lowest latency
}

/// Get all healthy remote devices.
pub async fn get_healthy_remote_devices() -> Vec<RemoteDevice> {
    let config = RemoteDiscoveryConfig::from_env();
    let devices = discover_remote_devices(&config).await;
    devices.into_iter().filter(|d| d.healthy).collect()
}

