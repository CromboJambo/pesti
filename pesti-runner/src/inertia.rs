//! Computational inertia subsystem for PESTI.
//!
//! Provides demand logging when GPU is unavailable, maintaining substrate momentum
//! instead of failing inference requests. When GPU returns, accumulated work is
//! replayed efficiently.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of computational work that can be logged for later execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkType {
    /// GEMM: C = alpha * A @ B + beta * C
    Gemm { m: usize, n: usize, k: usize, alpha: f32, beta: f32 },
    /// Attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
    Attention {
        query_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        cache_seq_len: usize,
    },
    /// Sampling: token selection from logits
    Sampling {
        logits_size: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    },
    /// Tokenization: text → tokens
    Tokenization { input_len: usize },
}

/// Pending work item logged while GPU was unavailable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingWork {
    /// Unique identifier for this work item.
    pub request_id: Uuid,
    /// Type of work to perform when GPU returns.
    pub work_type: WorkType,
    /// When the work was requested (for timeout/aging).
    pub timestamp: SystemTime,
    /// Priority: higher = execute first when GPU returns.
    pub priority: u8,
    /// Whether this work has been executed after GPU returned.
    #[serde(default)]
    pub executed: bool,
}

impl PendingWork {
    pub fn new(work_type: WorkType) -> Self {
        Self {
            request_id: Uuid::new_v4(),
            work_type,
            timestamp: SystemTime::now(),
            priority: 0,
            executed: false,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Demand log that accumulates work requests while GPU is unavailable.
pub struct DemandLog {
    /// Queue of pending work items (FIFO by timestamp).
    queue: VecDeque<PendingWork>,
    /// Total bytes of pending work (for backpressure).
    estimated_bytes: usize,
    /// Maximum number of items to keep in queue.
    max_items: usize,
}

impl Default for DemandLog {
    fn default() -> Self {
        Self::new(1024) // reasonable default
    }
}

impl DemandLog {
    pub fn new(max_items: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(max_items),
            estimated_bytes: 0,
            max_items,
        }
    }

    /// Estimate bytes for a work item (rough heuristic).
    fn estimate_bytes(work_type: &WorkType) -> usize {
        match work_type {
            WorkType::Gemm { m, n, .. } => *m * *n * 4, // f32 output buffer size
            WorkType::Attention {
                query_seq_len,
                num_heads,
                head_dim,
                cache_seq_len,
            } => {
                *query_seq_len * num_heads * *head_dim * 4 + *cache_seq_len * 2 * 1024
            }
            WorkType::Sampling { logits_size, .. } => logits_size * 4,
            WorkType::Tokenization { input_len } => input_len * 2,
        }
    }

    /// Log a work request. Returns true if logged, false if queue full.
    pub fn log(&mut self, work: PendingWork) -> bool {
        let bytes = Self::estimate_bytes(&work.work_type);

        // Backpressure: check size limit
        if self.estimated_bytes + bytes > self.max_items * 1024 * 1024 {
            while self.estimated_bytes > self.max_items * 512 * 1024 {
                if let Some(evicted) = self.queue.pop_front() {
                    self.estimated_bytes -= Self::estimate_bytes(&evicted.work_type);
                } else {
                    return false;
                }
            }
        }

        // Check item count limit
        if self.queue.len() >= self.max_items {
            if let Some(evicted) = self.queue.pop_front() {
                self.estimated_bytes -= Self::estimate_bytes(&evicted.work_type);
            } else {
                return false;
            }
        }

        self.queue.push_back(work);
        self.estimated_bytes += bytes;
        true
    }

    /// Get all pending work, sorted by priority (descending) then timestamp.
    pub fn drain(&mut self) -> Vec<PendingWork> {
        let mut work: Vec<PendingWork> = self.queue.drain(..).collect();
        work.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.timestamp.cmp(&b.timestamp)));
        self.estimated_bytes = 0;
        work
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }
}

/// Manages computational inertia: keeps substrate alive when GPU unavailable.
pub struct InertiaManager {
    demand_log: DemandLog,
    gpu_available: Arc<AtomicBool>,
    total_work_logged: usize,
    total_work_executed: usize,
}

impl InertiaManager {
    pub fn new(max_demand_log_items: usize) -> Self {
        Self {
            demand_log: DemandLog::new(max_demand_log_items),
            gpu_available: Arc::new(AtomicBool::new(true)),
            total_work_logged: 0,
            total_work_executed: 0,
        }
    }

    pub fn gpu_available(&self) -> bool {
        self.gpu_available.load(Ordering::SeqCst)
    }

    pub fn set_gpu_available(&self, available: bool) {
        self.gpu_available.store(available, Ordering::SeqCst);
    }

    /// Request computational work.
    pub fn request_work(&mut self, work_type: WorkType) -> WorkResult {
        if self.gpu_available() {
            WorkResult::ReadyForExecution(work_type)
        } else {
            let pending = PendingWork::new(work_type);
            if self.demand_log.log(pending) {
                self.total_work_logged += 1;
                WorkResult::LoggedForLater
            } else {
                WorkResult::Dropped
            }
        }
    }

    /// Get pending work for execution when GPU becomes available.
    pub fn get_pending_for_execution(&mut self) -> Vec<PendingWork> {
        let work = self.demand_log.drain();
        self.total_work_executed += work.len();
        work
    }

    /// Statistics about inertia management.
    pub fn stats(&self) -> InertiaStats {
        InertiaStats {
            gpu_available: self.gpu_available(),
            pending_items: self.demand_log.len(),
            estimated_pending_bytes: self.demand_log.estimated_bytes(),
            total_work_logged: self.total_work_logged,
            total_work_executed: self.total_work_executed,
        }
    }

    pub fn has_accumulated_demand(&self) -> bool {
        self.demand_log.len() > 0 || self.total_work_logged > self.total_work_executed
    }
}

/// Result of a work request to the inertia manager.
#[derive(Debug)]
pub enum WorkResult {
    ReadyForExecution(WorkType),
    LoggedForLater,
    Dropped,
}

/// Statistics about inertia management.
#[derive(Debug)]
pub struct InertiaStats {
    pub gpu_available: bool,
    pub pending_items: usize,
    pub estimated_pending_bytes: usize,
    pub total_work_logged: usize,
    pub total_work_executed: usize,
}

impl InertiaStats {
    pub fn utilization_ratio(&self) -> f32 {
        if self.total_work_logged == 0 {
            1.0
        } else {
            (self.total_work_executed as f32 / self.total_work_logged as f32).min(1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available_execution() {
        let mut manager = InertiaManager::new(100);
        manager.set_gpu_available(true);

        let result = manager.request_work(WorkType::Gemm {
            m: 128, n: 128, k: 4096, alpha: 1.0, beta: 0.0,
        });
        assert!(matches!(result, WorkResult::ReadyForExecution(_)));
    }

    #[test]
    fn test_gpu_unavailable_logging() {
        let mut manager = InertiaManager::new(100);
        manager.set_gpu_available(false);

        let result = manager.request_work(WorkType::Gemm {
            m: 256, n: 256, k: 8192, alpha: 1.0, beta: 0.0,
        });
        assert!(matches!(result, WorkResult::LoggedForLater));
        assert_eq!(manager.demand_log.len(), 1);
    }

    #[test]
    fn test_gpu_return_drains_queue() {
        let mut manager = InertiaManager::new(100);
        manager.set_gpu_available(false);
        manager.request_work(WorkType::Gemm { m: 1, n: 1, k: 1, alpha: 1.0, beta: 0.0 });
        manager.request_work(WorkType::Attention {
            query_seq_len: 1, num_heads: 8, head_dim: 64, cache_seq_len: 128,
        });

        assert_eq!(manager.demand_log.len(), 2);

        manager.set_gpu_available(true);
        let pending = manager.get_pending_for_execution();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut manager = InertiaManager::new(100);
        manager.set_gpu_available(false);

        for _ in 0..10 {
            manager.request_work(WorkType::Gemm { m: 1, n: 1, k: 1, alpha: 1.0, beta: 0.0 });
        }

        let stats = manager.stats();
        assert_eq!(stats.total_work_logged, 10);
        assert_eq!(stats.total_work_executed, 0);

        let _pending = manager.get_pending_for_execution();
        let stats = manager.stats();
        assert_eq!(stats.total_work_executed, 10);
        assert_eq!(stats.utilization_ratio(), 1.0);
    }
}
