//! Type-safe masking for attention and softmax operations.
//!
//! Replaces ad-hoc negative-infinity / large-negative float masking with an
//! explicit structural representation. A masked position is *absent*, not
//! "very small" — the type system encodes that distinction.
//!
//! # Design Principles
//!
//! 1. **Masked is not negative.** The type reflects absence, not magnitude.
//! 2. **No silent underflow reliance.** Correctness doesn't depend on
//!    `exp(-1e9) ≈ 0` being numerically true.
//! 3. **Zero-cost when possible.** Segmented softmax skips work on masked
//!    positions; all-active masks fast-path to plain softmax.
//! 4. **Composable.** Masks combine via `and()`, not float arithmetic.
//!
//! # Usage
//!
//! ```rust
//! use pesti_runner::softmask::{Softmask, segmented_softmax};
//!
//! // Causal mask: query at position 2 over 5 key positions
//! let mask = Softmask::causal(2, 5);
//! let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let weights = segmented_softmax(&scores, &mask);
//! // weights[3] and weights[4] are exactly 0.0
//! ```

/// A single position's state in a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// This position participates in softmax normalization.
    Active,
    /// This position is excluded from softmax normalization.
    Masked,
}

impl Slot {
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self, Slot::Active)
    }
}

/// A sequence of slot states over attention positions.
///
/// Composable and zero-cost when dense. Storage is 1 byte per position
/// (fieldless enum niche optimization).
#[derive(Debug, Clone)]
pub struct Softmask {
    slots: Vec<Slot>,
}

impl Softmask {
    /// All positions active (no masking).
    #[inline]
    pub fn all_active(len: usize) -> Self {
        Self {
            slots: vec![Slot::Active; len],
        }
    }

    /// Causal mask for a query at position `q_pos` over `seq_k` key positions.
    /// Positions 0..=q_pos are Active, positions > q_pos are Masked.
    #[inline]
    pub fn causal(q_pos: usize, seq_k: usize) -> Self {
        let slots = (0..seq_k)
            .map(|i| {
                if i <= q_pos {
                    Slot::Active
                } else {
                    Slot::Masked
                }
            })
            .collect();
        Self { slots }
    }

    /// From an attention mask where 1 = active, 0 = masked.
    /// Matches the convention in `trl/dataset.rs` (`attention_mask: Vec<u8>`).
    #[inline]
    pub fn from_bytes(mask: &[u8]) -> Self {
        let slots = mask
            .iter()
            .map(|&b| if b != 0 { Slot::Active } else { Slot::Masked })
            .collect();
        Self { slots }
    }

    /// Compose two masks: a position is Active only if BOTH are Active.
    /// Lengths must match; panics in debug, truncates to min in release.
    #[inline]
    pub fn and(self, other: Softmask) -> Self {
        assert_eq!(
            self.slots.len(),
            other.slots.len(),
            "Softmask::and requires equal lengths ({} != {})",
            self.slots.len(),
            other.slots.len()
        );
        let slots = self
            .slots
            .into_iter()
            .zip(other.slots)
            .map(|(a, b)| {
                if a.is_active() && b.is_active() {
                    Slot::Active
                } else {
                    Slot::Masked
                }
            })
            .collect();
        Self { slots }
    }

    /// Number of active positions (the effective softmax denominator).
    #[inline]
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_active()).count()
    }

    /// Check if a position is active.
    #[inline]
    pub fn is_active(&self, i: usize) -> bool {
        self.slots.get(i).is_some_and(Slot::is_active)
    }

    /// Iterate over indices of active positions only.
    /// Enables the segmented softmax path.
    #[inline]
    pub fn active_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.is_active().then_some(i))
    }

    /// Total number of positions in this mask.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the mask covers zero positions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Convert to a float buffer for GPU kernel consumption.
    /// Active positions get `active_value`, masked positions get `masked_value`.
    /// This is the type-safety boundary: host-side is typed, device transport is float.
    #[inline]
    pub fn to_float_buffer(&self, active_value: f32, masked_value: f32) -> Vec<f32> {
        self.slots
            .iter()
            .map(|s| {
                if s.is_active() {
                    active_value
                } else {
                    masked_value
                }
            })
            .collect()
    }

    /// Direct access to the underlying slots (for advanced composition).
    #[inline]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }
}

impl PartialEq for Softmask {
    fn eq(&self, other: &Self) -> bool {
        self.slots == other.slots
    }
}

impl Eq for Softmask {}

/// Plain numerically-stable softmax over all positions.
///
/// This is the standard max-subtraction pattern (fold identity `-inf` is
/// algorithmic, not a masking convention). Used as the fast path when
/// no masking is needed.
#[inline]
pub fn plain_softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    if sum > 0.0 && !sum.is_nan() {
        exps.into_iter().map(|e| e / sum).collect()
    } else {
        // Degenerate: all scores are -inf or NaN. Uniform distribution.
        let inv = 1.0 / scores.len() as f32;
        vec![inv; scores.len()]
    }
}

/// Segmented softmax: compute softmax only over active positions, write
/// exact zeros at masked positions.
///
/// Key properties:
/// - Masked positions get **exactly 0.0**, not `exp(-1e9 - max) ≈ 1e-434`
/// - Normalization sum includes only active positions
/// - No magic negative number in the hot path
/// - Fast-paths to `plain_softmax` when all positions are active
///
/// # Panics
///
/// In debug builds, panics if the mask length doesn't match scores length.
/// If ALL positions are masked (empty active set), panics in debug and
/// returns uniform zeros in release — a fully-masked attention head is
/// almost certainly a bug.
#[inline]
pub fn segmented_softmax(scores: &[f32], mask: &Softmask) -> Vec<f32> {
    assert_eq!(
        scores.len(),
        mask.len(),
        "segmented_softmax: scores length ({}) != mask length ({})",
        scores.len(),
        mask.len()
    );

    if scores.is_empty() {
        return Vec::new();
    }

    // Fast path: no masking needed
    if mask.active_count() == scores.len() {
        return plain_softmax(scores);
    }

    let active: Vec<usize> = mask.active_indices().collect();

    if active.is_empty() {
        // All positions masked — degenerate case.
        debug_assert!(
            false,
            "segmented_softmax: all positions masked (empty active set)"
        );
        return vec![0.0f32; scores.len()];
    }

    // Max over active only (fold identity is algorithmic, not semantic)
    let max_val = active
        .iter()
        .map(|&i| scores[i])
        .fold(f32::NEG_INFINITY, f32::max);

    // Exp and sum over active only
    let exps: Vec<f32> = active
        .iter()
        .map(|&i| (scores[i] - max_val).exp())
        .collect();
    let sum: f32 = exps.iter().sum();

    let mut weights = vec![0.0f32; scores.len()];

    if sum > 0.0 && !sum.is_nan() {
        for (&idx, &e) in active.iter().zip(exps.iter()) {
            weights[idx] = e / sum;
        }
    } else {
        // Degenerate: all active scores are -inf (shouldn't happen with real data)
        let inv = 1.0 / active.len() as f32;
        for &idx in &active {
            weights[idx] = inv;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Softmask construction ──

    #[test]
    fn test_all_active() {
        let mask = Softmask::all_active(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.active_count(), 5);
        for i in 0..5 {
            assert!(mask.is_active(i));
        }
    }

    #[test]
    fn test_causal_mask() {
        // Query at position 2 over 5 key positions: [A, A, A, M, M]
        let mask = Softmask::causal(2, 5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.active_count(), 3);
        assert!(mask.is_active(0));
        assert!(mask.is_active(1));
        assert!(mask.is_active(2));
        assert!(!mask.is_active(3));
        assert!(!mask.is_active(4));
    }

    #[test]
    fn test_causal_mask_first_position() {
        // Query at position 0: only position 0 is active
        let mask = Softmask::causal(0, 4);
        assert_eq!(mask.active_count(), 1);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
    }

    #[test]
    fn test_causal_mask_last_position() {
        // Query at last position: all active
        let mask = Softmask::causal(4, 5);
        assert_eq!(mask.active_count(), 5);
    }

    #[test]
    fn test_from_bytes() {
        let bytes = vec![1, 0, 1, 1, 0];
        let mask = Softmask::from_bytes(&bytes);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.active_count(), 3);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert!(mask.is_active(2));
        assert!(mask.is_active(3));
        assert!(!mask.is_active(4));
    }

    #[test]
    fn test_from_bytes_nonzero_is_active() {
        // Any non-zero byte should be active (robustness)
        let bytes = vec![1, 255, 42, 0];
        let mask = Softmask::from_bytes(&bytes);
        assert!(mask.is_active(0));
        assert!(mask.is_active(1));
        assert!(mask.is_active(2));
        assert!(!mask.is_active(3));
    }

    // ── Composition ──

    #[test]
    fn test_and_composition() {
        let causal = Softmask::causal(3, 5); // [A, A, A, A, M]
        let padding = Softmask::from_bytes(&[1, 0, 1, 1, 0]); // [A, M, A, A, M]
        let combined = causal.and(padding);

        assert_eq!(combined.len(), 5);
        assert_eq!(combined.active_count(), 3);
        assert!(combined.is_active(0));
        assert!(!combined.is_active(1)); // masked by padding
        assert!(combined.is_active(2));
        assert!(combined.is_active(3));
        assert!(!combined.is_active(4)); // masked by both
    }

    #[test]
    fn test_and_commutative() {
        let a = Softmask::causal(2, 4);
        let b = Softmask::from_bytes(&[1, 1, 0, 1]);
        assert_eq!(a.clone().and(b.clone()), b.and(a));
    }

    #[test]
    #[should_panic(expected = "equal lengths")]
    fn test_and_mismatched_lengths_panics() {
        let a = Softmask::all_active(3);
        let b = Softmask::all_active(5);
        let _ = a.and(b);
    }

    // ── active_indices ──

    #[test]
    fn test_active_indices() {
        let mask = Softmask::from_bytes(&[1, 0, 1, 0, 1]);
        let indices: Vec<usize> = mask.active_indices().collect();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_active_indices_empty() {
        let mask = Softmask::from_bytes(&[0, 0, 0]);
        let indices: Vec<usize> = mask.active_indices().collect();
        assert!(indices.is_empty());
    }

    // ── to_float_buffer (GPU boundary) ──

    #[test]
    fn test_to_float_buffer() {
        let mask = Softmask::causal(1, 3); // [A, A, M]
        let buf = mask.to_float_buffer(0.0, -1e9);
        assert_eq!(buf, vec![0.0, 0.0, -1e9]);
    }

    // ── plain_softmax ──

    #[test]
    fn test_plain_softmax_basic() {
        let scores = vec![1.0, 2.0, 3.0];
        let weights = plain_softmax(&scores);
        assert_eq!(weights.len(), 3);
        // Sum should be 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Monotonically increasing
        assert!(weights[0] < weights[1]);
        assert!(weights[1] < weights[2]);
    }

    #[test]
    fn test_plain_softmax_uniform() {
        let scores = vec![0.5, 0.5, 0.5];
        let weights = plain_softmax(&scores);
        for &w in &weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_plain_softmax_empty() {
        let weights = plain_softmax(&[]);
        assert!(weights.is_empty());
    }

    // ── segmented_softmax ──

    #[test]
    fn test_segmented_softmax_exact_zeros() {
        // THE key property: masked positions get EXACTLY 0.0
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = Softmask::causal(2, 5); // [A, A, A, M, M]
        let weights = segmented_softmax(&scores, &mask);

        assert_eq!(weights.len(), 5);
        // Masked positions are exactly zero (not ≈ 0)
        assert_eq!(weights[3], 0.0f32);
        assert_eq!(weights[4], 0.0f32);
        // Active positions sum to 1.0
        let active_sum: f32 = weights.iter().sum();
        assert!((active_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_segmented_softmax_matches_plain_when_all_active() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = Softmask::all_active(4);
        let segmented = segmented_softmax(&scores, &mask);
        let plain = plain_softmax(&scores);

        for (s, p) in segmented.iter().zip(plain.iter()) {
            assert!((s - p).abs() < 1e-7);
        }
    }

    #[test]
    fn test_segmented_softmax_causal_correctness() {
        // Causal softmax at position 2 should equal plain softmax over first 3 elements
        let scores = vec![1.0, 2.0, 3.0, 100.0]; // 100.0 would dominate if not masked
        let mask = Softmask::causal(2, 4);
        let weights = segmented_softmax(&scores, &mask);

        // The unmasked 100.0 at position 3 must have zero weight
        assert_eq!(weights[3], 0.0f32);

        // Weights over [1, 2, 3] should match plain softmax of those three
        let expected = plain_softmax(&scores[..3]);
        for i in 0..3 {
            assert!((weights[i] - expected[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_segmented_softmax_single_active() {
        let scores = vec![1.0, 2.0, 3.0];
        let mask = Softmask::from_bytes(&[0, 1, 0]); // Only position 1 active
        let weights = segmented_softmax(&scores, &mask);

        assert_eq!(weights[0], 0.0f32);
        assert!((weights[1] - 1.0).abs() < 1e-7); // Single active gets weight 1.0
        assert_eq!(weights[2], 0.0f32);
    }

    #[test]
    fn test_segmented_softmax_composed_masks() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let causal = Softmask::causal(3, 5); // [A, A, A, A, M]
        let padding = Softmask::from_bytes(&[1, 0, 1, 1, 0]); // [A, M, A, A, M]
        let combined = causal.and(padding); // [A, M, A, A, M]

        let weights = segmented_softmax(&scores, &combined);

        // combined = [A, M, A, A, M] → positions 0, 2, 3 active
        assert!(weights[0] > 0.0);
        assert_eq!(weights[1], 0.0f32);
        assert!(weights[2] > 0.0);
        assert!(weights[3] > 0.0);
        assert_eq!(weights[4], 0.0f32);

        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_segmented_softmax_empty_scores() {
        let mask = Softmask::all_active(0);
        let weights = segmented_softmax(&[], &mask);
        assert!(weights.is_empty());
    }

    #[test]
    #[should_panic(expected = "all positions masked")]
    fn test_segmented_softmax_all_masked_panics_in_debug() {
        let scores = vec![1.0, 2.0, 3.0];
        let mask = Softmask::from_bytes(&[0, 0, 0]);
        let _weights = segmented_softmax(&scores, &mask);
    }

    #[test]
    fn test_segmented_softmax_numerical_stability() {
        // Large scores should still work (max-subtraction handles it)
        let scores = vec![1000.0, 1001.0, 999.0];
        let mask = Softmask::all_active(3);
        let weights = segmented_softmax(&scores, &mask);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(weights[1] > weights[0]);
        assert!(weights[0] > weights[2]);
    }

    // ── Edge cases ──

    #[test]
    fn test_is_active_out_of_bounds() {
        let mask = Softmask::all_active(3);
        assert!(!mask.is_active(100)); // Out of bounds → not active (safe)
    }

    #[test]
    fn test_softmask_clone_and_eq() {
        let a = Softmask::causal(2, 5);
        let b = a.clone();
        assert_eq!(a, b);
    }
}
