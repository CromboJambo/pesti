//! KV cache for autoregressive generation.
//!
//! Stores key and value tensors across positions for efficient
//! incremental decoding. During generation, only the new position's
//! K/V is appended — attention reads from the full cache.
//!
//! Layout per layer: `[num_kv_heads * max_seq_len * head_dim]`
//! Index: `head * max_seq * head_dim + pos * head_dim + d`

/// Key-value cache for a single layer.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// Key cache: `[num_kv_heads * max_seq * head_dim]`
    k: Vec<f32>,
    /// Value cache: `[num_kv_heads * max_seq * head_dim]`
    v: Vec<f32>,
    /// Number of filled positions
    seq_len: usize,
    /// Maximum sequence length
    max_seq: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl LayerKvCache {
    /// Create an empty cache.
    pub fn new(num_kv_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        let total = num_kv_heads * max_seq * head_dim;
        Self {
            k: vec![0.0; total],
            v: vec![0.0; total],
            seq_len: 0,
            max_seq,
            num_kv_heads,
            head_dim,
        }
    }

    /// Number of positions currently cached.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Maximum capacity.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Append a single position's K and V.
    ///
    /// `k`: `[num_kv_heads * head_dim]` — RoPE-rotated key for this position
    /// `v`: `[num_kv_heads * head_dim]` — value for this position
    ///
    /// Returns the position index where data was written.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> usize {
        assert_eq!(k.len(), self.num_kv_heads * self.head_dim);
        assert_eq!(v.len(), self.num_kv_heads * self.head_dim);
        assert!(
            self.seq_len < self.max_seq,
            "KV cache full: seq_len={}, max_seq={}",
            self.seq_len,
            self.max_seq
        );

        let pos = self.seq_len;
        let stride = self.max_seq * self.head_dim;

        for h in 0..self.num_kv_heads {
            let k_dst = h * stride + pos * self.head_dim;
            let k_src = h * self.head_dim;
            self.k[k_dst..k_dst + self.head_dim].copy_from_slice(&k[k_src..k_src + self.head_dim]);

            let v_dst = h * stride + pos * self.head_dim;
            let v_src = h * self.head_dim;
            self.v[v_dst..v_dst + self.head_dim].copy_from_slice(&v[v_src..v_src + self.head_dim]);
        }

        self.seq_len += 1;
        pos
    }

    /// Get a slice of cached keys for a specific head.
    ///
    /// Returns `[seq_len * head_dim]` — all cached positions for this head.
    pub fn k_head(&self, head: usize) -> &[f32] {
        let stride = self.max_seq * self.head_dim;
        let start = head * stride;
        &self.k[start..start + self.seq_len * self.head_dim]
    }

    /// Get a slice of cached values for a specific head.
    ///
    /// Returns `[seq_len * head_dim]` — all cached positions for this head.
    pub fn v_head(&self, head: usize) -> &[f32] {
        let stride = self.max_seq * self.head_dim;
        let start = head * stride;
        &self.v[start..start + self.seq_len * self.head_dim]
    }

    /// Reset the cache (clear all positions).
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }
}
