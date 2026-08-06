//! RoPE (Rotary Positional Embeddings).
//!
//! Applies rotary embeddings to query and key vectors for attention.
//! For each head, position m:
//!   q_m' = q_m * cos(m * theta) - q_{m+head_dim/2} * sin(m * theta)
//!   k_m' = k_m * cos(m * theta) - k_{m+head_dim/2} * sin(m * theta)

use half::f16;

/// Rotary positional embeddings configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RopeConfig {
    /// Dimension per head (must be even).
    pub head_dim: usize,
    /// Base for frequency computation.
    pub base: f32,
    /// Maximum context length for precomputing frequencies.
    pub max_position_embeddings: usize,
    /// Rope scaling factor (1.0 = no scaling).
    pub scaling_factor: Option<f32>,
    /// Rope scaling type ("linear", "yarn", etc.).
    pub scaling_type: Option<String>,
}

impl RopeConfig {
    pub fn new(head_dim: usize, base: f32, max_position_embeddings: usize) -> Self {
        Self {
            head_dim,
            base,
            max_position_embeddings,
            scaling_factor: None,
            scaling_type: None,
        }
    }

    /// Compute frequencies: theta_i = base^(-2i/dim) for i in 0..head_dim/2.
    fn compute_theta(&self) -> Vec<f32> {
        let dim_half = self.head_dim / 2;
        (0..dim_half)
            .map(|i| self.base.powf(-(i as f32) / dim_half as f32))
            .collect()
    }

    /// Apply RoPE to query and key vectors in-place.
    ///
    /// q: [batch, seq_len, num_heads, head_dim] flattened row-major
    /// k: [batch, seq_len, num_heads, head_dim] flattened row-major
    /// seq_len: sequence length
    /// start_pos: starting position in the context (for scaling)
    pub fn apply(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        num_heads: usize,
        seq_len: usize,
        start_pos: usize,
    ) {
        let theta = self.compute_theta();
        let dim_half = self.head_dim / 2;

        for pos in 0..seq_len {
            let actual_pos = start_pos + pos;
            for head in 0..num_heads {
                for (i, &freq) in theta.iter().enumerate() {
                    let angle = actual_pos as f32 * freq;
                    let cos = angle.cos();
                    let sin = angle.sin();

                    let q_idx = pos * num_heads * self.head_dim + head * self.head_dim + i;
                    let k_idx = pos * num_heads * self.head_dim + head * self.head_dim + i;

                    let q_next = q_idx + dim_half;
                    let k_next = k_idx + dim_half;

                    let q_orig = q[q_idx];
                    let k_orig = k[k_idx];

                    q[q_idx] = q_orig * cos - q[q_next] * sin;
                    q[q_next] = q_orig * sin + q[q_next] * cos;

                    k[k_idx] = k_orig * cos - k[k_next] * sin;
                    k[k_next] = k_orig * sin + k[k_next] * cos;
                }
            }
        }
    }
}

/// Apply RoPE to f16 query and key tensors.
pub fn apply_rope_f16(
    q: &mut [f16],
    k: &mut [f16],
    head_dim: usize,
    num_heads: usize,
    seq_len: usize,
    start_pos: usize,
    base: f32,
) {
    let dim_half = head_dim / 2;
    let theta: Vec<f32> = (0..dim_half)
        .map(|i| base.powf(-(i as f32) / dim_half as f32))
        .collect();

    for pos in 0..seq_len {
        let actual_pos = start_pos + pos;
        for head in 0..num_heads {
            for (i, &freq) in theta.iter().enumerate() {
                let angle = actual_pos as f32 * freq;
                let cos = angle.cos();
                let sin = angle.sin();

                let q_idx = pos * num_heads * head_dim + head * head_dim + i;
                let q_next = q_idx + dim_half;
                let k_idx = q_idx;
                let k_next = k_idx + dim_half;

                let q_orig = q[q_idx].to_f32();
                let q_next_val = q[q_next].to_f32();
                let k_orig = k[k_idx].to_f32();
                let k_next_val = k[k_next].to_f32();

                q[q_idx] = f16::from_f32(q_orig * cos - q_next_val * sin);
                q[q_next] = f16::from_f32(q_orig * sin + q_next_val * cos);

                k[k_idx] = f16::from_f32(k_orig * cos - k_next_val * sin);
                k[k_next] = f16::from_f32(k_orig * sin + k_next_val * cos);
            }
        }
    }
}

