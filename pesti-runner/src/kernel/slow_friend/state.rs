//! Bounded low-pass (EMA) over hidden states — the "slow friend".
//!
//! Fixed size, O(1) per update, cannot accumulate unbounded error.
//! alpha < 1.0 is the bounded forget; (1 - alpha) is the bounded write.

#[derive(Debug)]
pub struct SlowFriendConfig {
    pub dim: usize,
    pub alpha: f32,
}

impl Default for SlowFriendConfig {
    fn default() -> Self {
        Self { dim: 896, alpha: 0.95 }
    }
}

#[derive(Debug)]
pub struct SlowFriendState {
    summary: Vec<f32>,
    alpha: f32,
    step: u64,
}

impl SlowFriendState {
    pub fn new(cfg: &SlowFriendConfig) -> Self {
        debug_assert!((0.0..1.0).contains(&cfg.alpha), "alpha must be in [0, 1)");
        Self {
            summary: vec![0.0; cfg.dim],
            alpha: cfg.alpha,
            step: 0,
        }
    }

    pub fn update(&mut self, h: &[f32]) {
        debug_assert_eq!(h.len(), self.summary.len(), "dimension mismatch");
        let one_minus_alpha = 1.0 - self.alpha;
        for i in 0..self.summary.len() {
            self.summary[i] = self.alpha * self.summary[i] + one_minus_alpha * h[i];
        }
        self.step += 1;
    }

    pub fn summary(&self) -> &[f32] {
        &self.summary
    }

    /// Mutable access to the summary for in-place modification (e.g., re-anchoring).
    pub fn summary_mut(&mut self) -> &mut [f32] {
        &mut self.summary
    }

    pub fn norm(&self) -> f32 {
        let sum: f64 = self.summary.iter().map(|x| (*x as f64) * (*x as f64)).sum();
        (sum.sqrt()) as f32
    }

    pub fn step(&self) -> u64 {
        self.step
    }

    pub fn reset(&mut self) {
        for x in &mut self.summary {
            *x = 0.0;
        }
        self.step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_converges_to_constant() {
        let cfg = SlowFriendConfig { dim: 4, alpha: 0.9 };
        let mut sf = SlowFriendState::new(&cfg);
        let constant = vec![1.0, 2.0, 3.0, 4.0];
        for _ in 0..1000 {
            sf.update(&constant);
        }
        let s = sf.summary();
        assert!((s[0] - 1.0).abs() < 0.01);
        assert!((s[3] - 4.0).abs() < 0.01);
    }

    #[test]
    fn boundedness_over_many_steps() {
        let cfg = SlowFriendConfig { dim: 8, alpha: 0.99 };
        let mut sf = SlowFriendState::new(&cfg);
        let bounded_input = vec![1.0; 8];
        for _ in 0..10_000 {
            sf.update(&bounded_input);
        }
        // EMA of all-ones converges to all-ones, norm = sqrt(8) ≈ 2.83
        assert!(sf.norm() < 3.0, "norm blew up: {}", sf.norm());
    }

    #[test]
    fn reset_clears_state() {
        let cfg = SlowFriendConfig { dim: 4, alpha: 0.9 };
        let mut sf = SlowFriendState::new(&cfg);
        sf.update(&[1.0, 2.0, 3.0, 4.0]);
        assert!(sf.norm() > 0.0);
        sf.reset();
        assert_eq!(sf.step(), 0);
        assert!((sf.norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn step_counting() {
        let cfg = SlowFriendConfig { dim: 2, alpha: 0.5 };
        let mut sf = SlowFriendState::new(&cfg);
        assert_eq!(sf.step(), 0);
        sf.update(&[1.0, 2.0]);
        assert_eq!(sf.step(), 1);
        sf.update(&[3.0, 4.0]);
        assert_eq!(sf.step(), 2);
    }
}
