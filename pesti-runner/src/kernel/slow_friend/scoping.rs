//! Expert scoping prior from slow-friend state.
//!
//! Projects the stable EMA summary to a bounded relevance map over synthetic
//! "expert" slots (hidden-state regions). Used to scope activation and measure
//! whether stable memory can keep long-context behavior closer to low-context reference.

/// Bounded expert-relevance prior computed from slow-friend state.
pub struct ExpertPrior {
    /// Number of synthetic expert slots.
    num_experts: usize,
    /// Dimensionality of the input hidden state.
    dim: usize,
    /// Projection matrix: [num_experts][dim] — learned or fixed random.
    projection: Vec<Vec<f32>>,
}

impl ExpertPrior {
    /// Create a new expert prior with random projection (seeded for reproducibility).
    pub fn new(dim: usize, num_experts: usize) -> Self {
        // Simple deterministic "random" projection using index-based hashing.
        let mut rng_state = 42u64;
        let mut projection = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            let row: Vec<f32> = (0..dim)
                .map(|_| {
                    // Simple xorshift RNG → f32 in [-1, 1]
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 5;
                    let val = (rng_state as i64) % 1000;
                    ((val as f32) / 500.0) - 1.0
                })
                .collect();
            projection.push(row);
        }
        Self {
            num_experts,
            dim,
            projection,
        }
    }

    /// Compute relevance scores for each expert slot from the stable state.
    /// Returns [num_experts] scores in [0, 1] (softmax-normalized).
    pub fn compute(&self, state: &[f32]) -> Vec<f32> {
        debug_assert_eq!(state.len(), self.dim);

        // Compute raw scores via projection.
        let mut scores = Vec::with_capacity(self.num_experts);
        for row in &self.projection {
            let mut dot = 0.0_f64;
            for i in 0..self.dim {
                dot += state[i] as f64 * row[i] as f64;
            }
            scores.push(dot as f32);
        }

        // Softmax to get bounded [0,1] relevance.
        let max_score = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        for s in &mut exp_scores {
            *s /= sum_exp;
        }

        exp_scores
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

/// Apply soft scoping: modulate hidden state regions by expert relevance.
/// Regions with low prior relevance are attenuated (scaled toward zero).
pub fn apply_scoping(hidden: &[f32], prior: &[f32]) -> Vec<f32> {
    let num_experts = prior.len();
    let region_size = hidden.len() / num_experts;
    let mut result = hidden.to_vec();

    for (expert_idx, &relevance) in prior.iter().enumerate() {
        let start = expert_idx * region_size;
        let end = start + region_size;
        // Soft attenuation: scale by relevance (0 = off, 1 = full).
        // Use sqrt to be less aggressive at low relevance.
        let scale = relevance.sqrt();
        for i in start..end {
            result[i] *= scale;
        }
    }

    result
}

/// Compute activation pattern: which expert regions are "active" (above threshold).
pub fn activation_pattern(hidden: &[f32], num_experts: usize) -> Vec<bool> {
    let region_size = hidden.len() / num_experts;
    let mut pattern = vec![false; num_experts];

    for expert_idx in 0..num_experts {
        let start = expert_idx * region_size;
        let end = start + region_size;
        // Compute mean absolute value in this region.
        let mut sum = 0.0_f64;
        for i in start..end {
            sum += hidden[i].abs() as f64;
        }
        let mean_mag = (sum / region_size as f64) as f32;
        // Threshold: active if mean magnitude > 0.1 (arbitrary but consistent).
        pattern[expert_idx] = mean_mag > 0.1;
    }

    pattern
}

/// Jaccard similarity between two activation patterns.
pub fn jaccard_similarity(a: &[bool], b: &[bool]) -> f32 {
    let mut intersection = 0;
    let mut union = 0;
    for i in 0..a.len() {
        if a[i] && b[i] {
            intersection += 1;
            union += 1;
        } else if a[i] || b[i] {
            union += 1;
        }
    }
    if union == 0 {
        return 1.0; // Both empty = identical.
    }
    intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_scores_sum_to_one() {
        let prior = ExpertPrior::new(896, 8);
        let state = vec![0.5_f32; 896];
        let scores = prior.compute(&state);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "scores must sum to 1");
    }

    #[test]
    fn activation_pattern_length() {
        let hidden = vec![0.5_f32; 896];
        let pattern = activation_pattern(&hidden, 8);
        assert_eq!(pattern.len(), 8);
    }

    #[test]
    fn jaccard_identical_patterns() {
        let a = vec![true, false, true, false];
        let b = vec![true, false, true, false];
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint_patterns() {
        let a = vec![true, false, false, false];
        let b = vec![false, true, false, false];
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn apply_scoping_attenuates_low_relevance() {
        let hidden = vec![1.0_f32; 8];
        let prior = vec![1.0, 0.0]; // First expert fully relevant, second not.
        let scoped = apply_scoping(&hidden, &prior);
        // First region should be unchanged, second attenuated to zero.
        assert!((scoped[0] - 1.0).abs() < 1e-6);
        assert!(scoped[4].abs() < 1e-6);
    }
}
