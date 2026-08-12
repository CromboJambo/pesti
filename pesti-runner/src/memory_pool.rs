//! Device memory pool for efficient GPU memory management
//!
//! Pre-allocates device memory in configurable chunks to amortize allocation overhead.
//! Provides reusable buffers for benchmark runs and inference.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use crate::cuda_runtime::{allocate_device_memory, free_device_memory, CudaError};

/// A reusable GPU memory buffer within the pool.
pub struct PooledBuffer {
    /// Raw device pointer.
    pub ptr: *mut u8,
    /// Size in bytes.
    pub size: usize,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Note: The pool owns the memory, so we don't free here
        // This prevents double-free when buffers are returned to pool
    }
}

/// Configuration for the memory pool.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
/// Initial number of allocations per size class (default: 3 to save memory).
pub initial_allocations: usize,
/// Maximum total allocations across all size classes (default: 50).
pub max_allocations: usize,
    /// Size classes to pre-allocate (in bytes).
    pub size_classes: Vec<usize>,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_allocations: 10,
            max_allocations: 100,
            /// Common sizes from benchmark analysis
            size_classes: vec![
                32 * 1024,       // 32 KB (small tensors)
                128 * 1024,      // 128 KB (medium tensors)
                512 * 1024,      // 512 KB (large tensors)
            ],
        }
    }
}

/// Pool usage statistics.
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// Total allocations made.
    pub total_allocations: usize,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes.
    pub current_memory_bytes: usize,
}

/// Device memory pool that pre-allocates buffers for reuse.
pub struct MemoryPool {
    /// Pre-allocated buffers organized by size class.
    pools: Vec<Mutex<VecDeque<PooledBuffer>>>,
    /// Size classes corresponding to each pool.
    size_classes: Vec<usize>,
    /// Configuration.
    config: MemoryPoolConfig,
    /// Statistics.
    stats: Arc<Mutex<PoolStats>>,
}

#[cfg(feature = "cuda")]
impl MemoryPool {
    /// Create a new memory pool with default configuration.
    pub fn new() -> Result<Self, CudaError> {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create a new memory pool with custom configuration.
    pub fn with_config(config: MemoryPoolConfig) -> Result<Self, CudaError> {
        let mut pools = Vec::new();
        let mut size_classes = Vec::new();

        for &size in &config.size_classes {
            let pool: Mutex<VecDeque<PooledBuffer>> = Mutex::new(VecDeque::new());
            
            // Pre-allocate initial buffers
            for _ in 0..config.initial_allocations {
                let ptr = allocate_device_memory(size)?;
                (*pool.lock().unwrap()).push_back(PooledBuffer { ptr, size });
            }
            
            pools.push(pool);
            size_classes.push(size);
        }

        Ok(Self {
            pools,
            size_classes,
            config,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        })
    }

    /// Allocate a buffer of at least `size` bytes from the pool.
    pub fn allocate(&self, size: usize) -> Result<PooledBuffer, CudaError> {
        // Find appropriate size class (smallest that fits)
        let class_idx = self
            .size_classes
            .iter()
            .position(|&s| s >= size)
            .ok_or_else(|| {
                CudaError::DriverError(format!("No suitable size class for {}", size))
            })?;

        // Try to get from pool
        {
            let mut pool = self.pools[class_idx].lock().unwrap();
            if let Some(buffer) = pool.pop_front() {
                return Ok(buffer);
            }
        }

        // Pool exhausted, allocate new buffer (if under limit)
        let mut stats = self.stats.lock().unwrap();
        if stats.total_allocations < self.config.max_allocations {
            drop(stats); // Release lock before allocation
            
            let ptr = allocate_device_memory(self.size_classes[class_idx])?;
            let buffer = PooledBuffer { 
                ptr, 
                size: self.size_classes[class_idx] 
            };

            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations += 1;
            stats.current_memory_bytes += self.size_classes[class_idx];
            if stats.current_memory_bytes > stats.peak_memory_bytes {
                stats.peak_memory_bytes = stats.current_memory_bytes;
            }

            Ok(buffer)
        } else {
            // Under memory pressure, fall back to direct allocation
            let ptr = allocate_device_memory(size)?;
            Ok(PooledBuffer { ptr, size })
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn deallocate(&self, buffer: PooledBuffer) {
        // Find which pool this belongs to
        if let Some(class_idx) = self.size_classes.iter().position(|&s| s == buffer.size) {
            let mut pool = self.pools[class_idx].lock().unwrap();
            pool.push_back(buffer);

            // Update stats
            let mut stats = self.stats.lock().unwrap();
            stats.current_memory_bytes -= self.size_classes[class_idx];
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> PoolStats {
        let guard = self.stats.lock().unwrap();
        (*guard).clone()
    }

    /// Clear all buffers from the pool (free memory).
    pub fn clear(&self) -> Result<(), CudaError> {
        for (class_idx, pool) in self.pools.iter().enumerate() {
            let mut pool = pool.lock().unwrap();
            while let Some(buffer) = pool.pop_front() {
                unsafe {
                    free_device_memory(buffer.ptr)?;
                }
            }
            
            // Replenish to initial count
            for _ in 0..self.config.initial_allocations {
                let ptr = allocate_device_memory(self.size_classes[class_idx])?;
                pool.push_back(PooledBuffer { 
                    ptr, 
                    size: self.size_classes[class_idx] 
                });
            }
        }

        // Reset stats
        *self.stats.lock().unwrap() = PoolStats::default();
        Ok(())
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Free all buffers when pool is destroyed (only if CUDA is available)
        #[cfg(feature = "cuda")]
        for (class_idx, pool) in self.pools.iter().enumerate() {
            if let Ok(mut pool) = pool.lock() {
                while let Some(buffer) = pool.pop_front() {
                    let _ = unsafe { free_device_memory(buffer.ptr) };
                }
            }
        }
    }
}

/// Batched execution context for amortizing kernel launch overhead.
pub struct ExecutionBatch {
    /// Number of sequences in batch.
    pub batch_size: usize,
    /// Minimum sequence length to batch together.
    min_seq_len: usize,
    /// Current accumulated sequences.
    current_sequences: Vec<SequenceConfig>,
}

#[derive(Debug, Clone)]
struct SequenceConfig {
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
}

impl ExecutionBatch {
    /// Create a new execution batch.
    pub fn new(min_seq_len: usize) -> Self {
        Self {
            batch_size: 0,
            min_seq_len,
            current_sequences: Vec::new(),
        }
    }

    /// Add a sequence to the batch.
    pub fn add_sequence(&mut self, seq_q: usize, seq_k: usize, num_heads: usize, head_dim: usize) {
        self.current_sequences.push(SequenceConfig {
            seq_q,
            seq_k,
            num_heads,
            head_dim,
        });
        self.batch_size += 1;
    }

    /// Check if batch is ready to execute (has enough sequences).
    pub fn is_ready(&self) -> bool {
        // Ready when we have at least 2 sequences or all are small (< min_seq_len)
        self.batch_size >= 2
            || self
                .current_sequences
                .iter()
                .all(|s| s.seq_q < self.min_seq_len && s.seq_k < self.min_seq_len)
    }

    /// Get total sequence length for batch.
    pub fn total_length(&self) -> usize {
        self.current_sequences
            .iter()
            .map(|s| s.seq_q + s.seq_k)
            .sum()
    }

    /// Clear the batch (call after execution).
    pub fn clear(&mut self) {
        self.batch_size = 0;
        self.current_sequences.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_batch() {
        let mut batch = ExecutionBatch::new(128);
        
        // Small sequences should batch immediately
        batch.add_sequence(64, 64, 4, 64);
        assert!(batch.is_ready());
        
        // Add more
        batch.add_sequence(128, 128, 8, 64);
        assert_eq!(batch.batch_size, 2);
    }
}
