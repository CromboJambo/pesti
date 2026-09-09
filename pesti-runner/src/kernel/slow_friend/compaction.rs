//! Compaction trigger and re-anchor logic for the slow-friend substrate.
//!
//! When divergence exceeds a threshold, re-anchor the EMA summary toward
//! the precise path to bound long-context drift.

use super::divergence::DivergenceScore;
use super::state::SlowFriendState;

/// Triggers re-anchoring when divergence exceeds a threshold.
pub struct CompactionTrigger {
    pub threshold: f32,
}

impl CompactionTrigger {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Should we re-anchor at this step?
    pub fn should_compact(&self, score: &DivergenceScore) -> bool {
        score.value > self.threshold
    }
}

/// Re-anchor the slow-friend summary toward the precise path.
/// Blends the current summary with the precise hidden state.
/// weight in [0,1]: 0 = no change, 1 = full reset to precise state.
pub fn reanchor(state: &mut SlowFriendState, precise: &[f32], weight: f32) {
    debug_assert_eq!(precise.len(), state.summary().len(), "dimension mismatch");
    let summary = state.summary_mut();
    for i in 0..summary.len() {
        summary[i] = (1.0 - weight) * summary[i] + weight * precise[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::slow_friend::divergence::{DivergenceMetric, DivergenceScore};
    use crate::kernel::slow_friend::state::{SlowFriendConfig, SlowFriendState};

    fn dummy_score(value: f32) -> DivergenceScore {
        DivergenceScore {
            metric: DivergenceMetric::Cosine,
            value,
            ref_norm: None,
        }
    }

    #[test]
    fn trigger_fires_above_threshold() {
        let t = CompactionTrigger::new(0.1);
        assert!(!t.should_compact(&dummy_score(0.05)));
        assert!(t.should_compact(&dummy_score(0.15)));
    }

    #[test]
    fn reanchor_weight_zero_leaves_unchanged() {
        let cfg = SlowFriendConfig { dim: 4, alpha: 0.9 };
        let mut state = SlowFriendState::new(&cfg);
        state.update(&[1.0, 2.0, 3.0, 4.0]);

        let before = state.summary().to_vec();
        reanchor(&mut state, &[5.0, 6.0, 7.0, 8.0], 0.0);

        for i in 0..4 {
            assert!((state.summary()[i] - before[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn reanchor_weight_one_full_reset() {
        let cfg = SlowFriendConfig { dim: 4, alpha: 0.9 };
        let mut state = SlowFriendState::new(&cfg);
        state.update(&[1.0, 2.0, 3.0, 4.0]);

        reanchor(&mut state, &[5.0, 6.0, 7.0, 8.0], 1.0);

        assert!((state.summary()[0] - 5.0).abs() < 1e-6);
        assert!((state.summary()[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn reanchor_weight_half_blends() {
        let cfg = SlowFriendConfig { dim: 4, alpha: 0.9 };
        let mut state = SlowFriendState::new(&cfg);
        // Converge to a known value first
        for _ in 0..100 {
            state.update(&[2.0, 2.0, 2.0, 2.0]);
        }

        reanchor(&mut state, &[4.0, 4.0, 4.0, 4.0], 0.5);

        // Should be halfway: (1-0.5)*2 + 0.5*4 = 3
        assert!((state.summary()[0] - 3.0).abs() < 0.1);
    }
}