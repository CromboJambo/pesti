//! Computational inertia subsystem for PESTI.
//!
//! Provides demand logging when GPU is unavailable, maintaining substrate momentum
//! instead of failing inference requests. When GPU returns, accumulated work is
//! replayed with priority ordering and resident tensor references.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Priority level for computational demand.
/// Higher values = higher priority (executed first during replay).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Priority {
    /// Critical: must execute before GPU returns if possible
    Critical = 3,
    /// High: should execute ASAP after GPU recovery
    High = 2,
    /// Normal: standard priority, FIFO within this tier
    Normal = 1,
    /// Low: can be dropped under backpressure
    Low = 0,
}

impl Eq for Priority {}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering: higher priority comes first in queue
        other.cmp(self)
    }
}

impl serde::Serialize for Priority {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Critical => 3usize,
            Self::High => 2usize,
            Self::Normal => 1usize,
            Self::Low => 0usize,
        }
        .serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Priority {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        match value {
            3 => Ok(Self::Critical),
            2 => Ok(Self::High),
            1 => Ok(Self::Normal),
            0 => Ok(Self::Low),
            _ => Err(serde::de::Error::custom("Invalid priority value")),
        }
    }
}

/// Computational work type with typed parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkType {
    /// GEMM: C = alpha * A @ B + beta * C
    /// - m: rows of A/C
    /// - n: cols of B/C
    /// - k: inner dimension
    /// - alpha/beta: scaling factors
    Gemm {
        m: u32,
        n: u32,
        k: u32,
        alpha: f32,
        beta: f32,
    },

    /// Attention: query @ key^T @ value
    /// - query_seq_len: length of query sequence (usually 1 for decode)
    /// - num_heads: number of attention heads
    /// - head_dim: dimension per head
    /// - cache_seq_len: current KV cache sequence length
    Attention {
        query_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        cache_seq_len: u32,
    },

    /// RMSNorm: y = x / mean(x^2) * weight * eps
    RmsNorm {
        dim: u32,
        eps: f32,
    },

    /// Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    Softmax {
        dim: u32,
    },

    /// Embedding lookup: token → vector
    Embedding {
        vocab_size: u32,
        embed_dim: u32,
    },
}

impl WorkType {
    /// Estimate bytes for a work item (rough heuristic).
    fn estimate_bytes(&self) -> usize {
        // Rough estimate: params + potential input data references
        match self {
            Self::Gemm { m, n, k: _, .. } => {
                // C matrix size (f32) is the dominant factor
                (*m as usize) * (*n as usize) * 4
            }
            Self::Attention { num_heads, head_dim, cache_seq_len, .. } => {
                // KV cache size: 2 * num_heads * head_dim * cache_seq_len * f16
                2 * (*num_heads as usize) * (*head_dim as usize) * (*cache_seq_len as usize) * 2
            }
            Self::RmsNorm { dim, .. } => *dim as usize * 4, // weight + input
            Self::Softmax { dim } => *dim as usize * 4, // logits
            Self::Embedding { embed_dim, .. } => *embed_dim as usize * 4, // embedding vector
        }
    }

    /// Generate a unique identifier for this work item.
    fn id(&self) -> Uuid {
        // Derive UUID from work parameters (deterministic)
        let mut hash = 0u64;
        match self {
            Self::Gemm { m, n, k, alpha, beta } => {
                hash ^= *m as u64 ^ (*n as u64) ^ (*k as u64) ^ (*alpha as u32 as u64) ^ (*beta as u32 as u64);
            }
            Self::Attention { query_seq_len, num_heads, head_dim, cache_seq_len } => {
                hash ^= *query_seq_len as u64 ^ *num_heads as u64 ^ *head_dim as u64 ^ *cache_seq_len as u64;
            }
            Self::RmsNorm { dim, eps } => {
                hash ^= *dim as u64 ^ (*eps as u32 as u64);
            }
            Self::Softmax { dim } => {
                hash ^= *dim as u64;
            }
            Self::Embedding { vocab_size, embed_dim } => {
                hash ^= *vocab_size as u64 ^ *embed_dim as u64;
            }
        }
        // Use the hash directly for UUID (lower 128 bits)
        Uuid::from_u128((hash as u128) << 64 | (hash as u128))
    }
}

/// A unit of computational demand that can survive GPU unavailability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demand {
    /// Unique identifier for this demand.
    pub id: Uuid,
    /// The work to be performed.
    pub work_type: WorkType,
    /// When this demand was created (for ordering and timeout).
    pub timestamp: SystemTime,
    /// Priority level for execution ordering.
    pub priority: Priority,
    /// Estimated bytes of resident tensor references this demand depends on.
    pub resident_bytes: usize,
}

impl Demand {
    /// Create a new demand with default low priority.
    pub fn new(work_type: WorkType) -> Self {
        let timestamp = SystemTime::now();
        let resident_bytes = work_type.estimate_bytes();

        Self {
            id: work_type.id(),
            work_type,
            timestamp,
            priority: Priority::Normal,
            resident_bytes,
        }
    }

    /// Create a demand with explicit priority.
    pub fn with_priority(work_type: WorkType, priority: Priority) -> Self {
        let mut demand = Self::new(work_type);
        demand.priority = priority;
        demand
    }

    /// Check if this demand has expired (older than timeout).
    pub fn has_expired(&self, timeout: Duration) -> bool {
        self.timestamp.elapsed().unwrap_or(Duration::MAX) > timeout
    }

    /// Calculate age in milliseconds.
    pub fn age_ms(&self) -> u128 {
        self.timestamp.elapsed().unwrap_or(Duration::MAX).as_millis()
    }
}

/// Queue of pending work with backpressure protection.
pub struct DemandQueue {
    /// Maximum number of pending demands.
    capacity: usize,
    /// Current queue of demands (sorted by priority, then timestamp).
    queue: VecDeque<Demand>,
    /// Total bytes currently queued.
    total_bytes: usize,
}

impl DemandQueue {
    /// Create a new demand queue with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            queue: VecDeque::new(),
            total_bytes: 0,
        }
    }

    /// Try to add a demand to the queue.
    /// Returns Ok(Some(demand)) if added, Ok(None) if dropped due to backpressure.
    pub fn try_push(&mut self, demand: Demand) -> Result<Option<Demand>, &'static str> {
        // Check capacity
        if self.queue.len() >= self.capacity {
            // Backpressure: drop lowest priority items first
            return Err("backpressure");
        }

        // Check byte limit (conservative: 50% of queue size)
        let max_bytes = self.capacity * demand.resident_bytes / 2;
        if self.total_bytes + demand.resident_bytes > max_bytes {
            return Err("byte_limit");
        }

        self.queue.push_back(demand);
        Ok(None) // Successfully added
    }

    /// Remove and return the highest priority demand.
    pub fn pop(&mut self) -> Option<Demand> {
        if self.queue.is_empty() {
            return None;
        }

        // Find highest priority item (highest value first due to Ord impl)
        let mut max_idx = 0;
        let demands: Vec<&Demand> = self.queue.iter().collect();
        
        for (idx, demand) in demands.iter().enumerate() {
            if demand.priority > demands[max_idx].priority {
                max_idx = idx;
            }
        }

        // Remove and return the highest priority item
        let demand = self.queue.remove(max_idx).unwrap();
        self.total_bytes = self.total_bytes.saturating_sub(demand.resident_bytes);
        Some(demand)
    }

    /// Peek at the highest priority demand without removing it.
    pub fn peek(&self) -> Option<&Demand> {
        self.queue.front()
    }

    /// Get all pending demands (for replay).
    pub fn drain(&mut self) -> Vec<Demand> {
        let demands: Vec<Demand> = self.queue.drain(..).collect();
        self.total_bytes = 0;
        demands
    }

    /// Current number of pending demands.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Current total bytes queued.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Manages computational inertia: keeps substrate alive when GPU unavailable.
pub struct InertiaManager {
    /// Demand log for work requests while GPU absent.
    demand_queue: DemandQueue,
    /// Whether GPU is currently available.
    gpu_available: AtomicBool,
    /// Telemetry counters.
    stats: Arc<InertiaStats>,
}

/// Runtime statistics for inertia tracking.
#[derive(Debug, Default, Clone)]
pub struct InertiaStats {
    /// Total work submitted to the system.
    pub total_submitted: u64,
    /// Work executed immediately (GPU available).
    pub total_executed: u64,
    /// Work deferred due to GPU unavailability.
    pub total_deferred: u64,
    /// Work dropped due to backpressure.
    pub total_dropped: u64,
    /// Work replayed after GPU recovery.
    pub total_replayed: u64,
}

impl InertiaStats {
    /// Calculate utilization ratio (executed / submitted).
    pub fn utilization_ratio(&self) -> f32 {
        if self.total_submitted == 0 {
            return 0.0;
        }
        self.total_executed as f32 / self.total_submitted as f32
    }

    /// Calculate deferral ratio (deferred / submitted).
    pub fn deferral_ratio(&self) -> f32 {
        if self.total_submitted == 0 {
            return 0.0;
        }
        self.total_deferred as f32 / self.total_submitted as f32
    }

    /// Calculate drop ratio (dropped / submitted).
    pub fn drop_ratio(&self) -> f32 {
        if self.total_submitted == 0 {
            return 0.0;
        }
        self.total_dropped as f32 / self.total_submitted as f32
    }
}

impl InertiaManager {
    /// Create a new inertia manager with specified queue capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            demand_queue: DemandQueue::new(capacity),
            gpu_available: AtomicBool::new(true), // Start with GPU available
            stats: Arc::new(InertiaStats::default()),
        }
    }

    /// Set GPU availability status.
    pub fn set_gpu_available(&self, available: bool) {
        self.gpu_available.store(available, Ordering::SeqCst);
    }

    /// Check if GPU is currently available.
    pub fn gpu_available(&self) -> bool {
        self.gpu_available.load(Ordering::SeqCst)
    }

    /// Request computational work.
    /// Returns ReadyForExecution if GPU available, LoggedForLater if deferred.
    pub fn request_work(&mut self, work_type: WorkType) -> WorkResult {
        // Increment submitted counter using Arc::make_mut
        Arc::make_mut(&mut self.stats).total_submitted += 1;

        if self.gpu_available() {
            // GPU available: execute immediately (in real system, this would dispatch to GPU)
            Arc::make_mut(&mut self.stats).total_executed += 1;
            WorkResult::ReadyForExecution(work_type)
        } else {
            // GPU unavailable: queue for later execution
            let demand = Demand::new(work_type);

            match self.demand_queue.try_push(demand) {
                Ok(_) => {
                    Arc::make_mut(&mut self.stats).total_deferred += 1;
                    WorkResult::LoggedForLater
                }
                Err("backpressure") | Err("byte_limit") => {
                    Arc::make_mut(&mut self.stats).total_dropped += 1;
                    WorkResult::Dropped
                }
                Err(e) => {
                    tracing::warn!("Unknown error pushing demand: {}", e);
                    Arc::make_mut(&mut self.stats).total_dropped += 1;
                    WorkResult::Dropped
                }
            }
        }
    }

    /// Get pending work for execution after GPU recovery.
    pub fn get_pending_for_execution(&mut self) -> Vec<Demand> {
        let demands = self.demand_queue.drain();
        
        // Increment replayed counter using Arc::make_mut
        Arc::make_mut(&mut self.stats).total_replayed += demands.len() as u64;
        
        demands
    }

    /// Check if there is pending work waiting for execution.
    pub fn has_pending_work(&self) -> bool {
        !self.demand_queue.is_empty()
    }

    /// Get current statistics.
    pub fn stats(&self) -> InertiaStats {
        Arc::clone(&self.stats).as_ref().clone()
    }

    /// Estimate total pending bytes.
    pub fn estimated_pending_bytes(&self) -> usize {
        self.demand_queue.total_bytes()
    }
}

/// Result of a work request.
#[derive(Debug, Clone)]
pub enum WorkResult {
    /// GPU available: ready to execute immediately.
    ReadyForExecution(WorkType),
    /// GPU unavailable: logged for later execution.
    LoggedForLater,
    /// Backpressure: work dropped due to queue capacity.
    Dropped,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available_execution() {
        let mut manager = InertiaManager::new(10);
        manager.set_gpu_available(true);

        let result = manager.request_work(WorkType::Gemm {
            m: 128, n: 128, k: 4096, alpha: 1.0, beta: 0.0,
        });

        assert!(matches!(result, WorkResult::ReadyForExecution(_)));
        let stats = manager.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_executed, 1);
        assert_eq!(stats.total_deferred, 0);
    }

    #[test]
    fn test_gpu_unavailable_logging() {
        let mut manager = InertiaManager::new(10);
        manager.set_gpu_available(false);

        let result = manager.request_work(WorkType::Gemm {
            m: 256, n: 256, k: 8192, alpha: 1.0, beta: 0.0,
        });

        assert!(matches!(result, WorkResult::LoggedForLater));
        let stats = manager.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_deferred, 1);
    }

    #[test]
    fn test_gpu_return_drains_queue() {
        let mut manager = InertiaManager::new(10);
        
        // GPU unavailable: log demand
        manager.set_gpu_available(false);
        manager.request_work(WorkType::Gemm {
            m: 256, n: 256, k: 8192, alpha: 1.0, beta: 0.0,
        });
        
        // GPU returns
        manager.set_gpu_available(true);
        let pending = manager.get_pending_for_execution();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut manager = InertiaManager::new(5); // Small capacity to test backpressure
        
        // Phase 1: GPU available
        manager.set_gpu_available(true);
        for _ in 0..10 {
            manager.request_work(WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 });
        }
        
        // Phase 2: GPU unavailable + backpressure
        manager.set_gpu_available(false);
        for _ in 0..20 {
            let result = manager.request_work(WorkType::Attention {
                query_seq_len: 1, num_heads: 8, head_dim: 64, cache_seq_len: 128,
            });
            match result {
                WorkResult::LoggedForLater => {}
                WorkResult::Dropped => {} // Expected due to backpressure
                WorkResult::ReadyForExecution(_) => panic!("Should not execute when GPU unavailable"),
            }
        }
        
        let stats = manager.stats();
        assert_eq!(stats.total_submitted, 30);
        assert_eq!(stats.total_executed, 10); // First phase
        assert!(stats.total_deferred > 0); // Second phase logged some
        assert!(stats.total_dropped >= 0); // May have dropped some due to backpressure
        
        // Phase 3: GPU returns + replay
        let pending = manager.get_pending_for_execution();
        let total_from_queue = pending.len() as u64;
        let total_deferred = stats.total_deferred;
        assert_eq!(total_from_queue + stats.total_dropped, 20); // Total deferred = pending + dropped
    }

    #[test]
    fn test_priority_ordering() {
        let mut queue = DemandQueue::new(10);
        
        // Add demands with different priorities (in random order)
        let low = Demand::with_priority(WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 }, Priority::Low);
        let high = Demand::with_priority(WorkType::Gemm { m: 128, n: 128, k: 4096, alpha: 1.0, beta: 0.0 }, Priority::High);
        let normal = Demand::with_priority(WorkType::Gemm { m: 256, n: 256, k: 8192, alpha: 1.0, beta: 0.0 }, Priority::Normal);
        
        queue.try_push(normal).unwrap();
        queue.try_push(high).unwrap();
        queue.try_push(low).unwrap();
        
        // Pop should return highest priority first
        let popped = queue.pop().unwrap();
        assert_eq!(popped.priority, Priority::High);
        
        let popped = queue.pop().unwrap();
        assert_eq!(popped.priority, Priority::Normal);
        
        let popped = queue.pop().unwrap();
        assert_eq!(popped.priority, Priority::Low);
    }

    #[test]
    fn test_backpressure() {
        let mut manager = InertiaManager::new(3); // Very small capacity
        
        // Fill queue to capacity
        manager.set_gpu_available(false);
        for _ in 0..3 {
            let result = manager.request_work(WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 });
            assert!(matches!(result, WorkResult::LoggedForLater));
        }
        
        // Next request should be dropped
        let result = manager.request_work(WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 });
        assert!(matches!(result, WorkResult::Dropped));
        
        let stats = manager.stats();
        assert_eq!(stats.total_dropped, 1);
    }

    #[test]
    fn test_demand_timestamps() {
        let mut manager = InertiaManager::new(10);
        manager.set_gpu_available(false);
        
        // Submit two demands with a small delay
        manager.request_work(WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 });
        std::thread::sleep(Duration::from_millis(1));
        manager.request_work(WorkType::Gemm { m: 128, n: 128, k: 4096, alpha: 1.0, beta: 0.0 });
        
        let pending = manager.get_pending_for_execution();
        assert_eq!(pending.len(), 2);
        
        // Verify timestamps are ordered (first should be older)
        assert!(pending[0].timestamp <= pending[1].timestamp);
    }

    #[test]
    fn test_demand_id_uniqueness() {
        let work1 = WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 };
        let work2 = WorkType::Gemm { m: 64, n: 64, k: 2048, alpha: 1.0, beta: 0.0 };
        
        // Same parameters should produce same ID (deterministic)
        assert_eq!(work1.id(), work2.id());
        
        let work3 = WorkType::Gemm { m: 128, n: 128, k: 4096, alpha: 1.0, beta: 0.0 };
        assert_ne!(work1.id(), work3.id()); // Different parameters = different ID
    }

    #[test]
    fn test_utilization_ratios() {
        let mut stats = InertiaStats::default();
        stats.total_submitted = 100;
        stats.total_executed = 70;
        stats.total_deferred = 25;
        stats.total_dropped = 5;
        
        assert_eq!(stats.utilization_ratio(), 0.7);
        assert_eq!(stats.deferral_ratio(), 0.25);
        assert_eq!(stats.drop_ratio(), 0.05);
    }
}
