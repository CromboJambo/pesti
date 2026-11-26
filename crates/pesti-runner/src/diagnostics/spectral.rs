//! RMT-based attention spectral analysis for degeneracy detection.
//!
//! Uses Random Matrix Theory (Marchenko-Pastur law) to establish baselines
//! for "healthy" attention matrices, then detects structural anomalies:
//! - Attention collapse (rank-1 behavior)
//! - Dead heads (near-zero spectra)
//! - Long-context drift (gradual spectral deviation)
//!
//! Core principle: healthy transformer attention has eigenvalue distributions
//! close to RMT predictions. Significant deviation indicates structural problems.
//!
//! Implementation uses randomized SVD (Halko et al. 2011) for O(n·k·d) complexity
//! instead of O(n³) full eigendecomposition.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};



/// Spectral statistics for a single attention matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSpectrum {
    /// Layer index
    pub layer: usize,
    /// Head index
    pub head: usize,
    /// Sequence length n
    pub seq_len: usize,

    /// Top-k eigenvalues of A·Aᵀ (k=10 by default)
    pub top_eigenvalues: Vec<f64>,

    /// Spectral norm ||A||₂ = sqrt(λ_max(A·Aᵀ))
    pub spectral_norm: f64,

    /// Numerical rank (eigenvalues > 1e-8 * λ_max)
    pub numerical_rank: usize,

    /// Effective rank: exp(H(p)) where p_i = λ_i / Σλ_j
    /// Measures "how many dimensions" the attention actually uses
    pub effective_rank: f64,

    /// Marchenko-Pastur distance: KL divergence between empirical
    /// eigenvalue distribution and MP law prediction
    pub mp_distance: f64,

    /// Degeneracy score: 0 = healthy (RMT-like), 1 = fully collapsed
    pub degeneracy_score: f64,
}

/// Summary of a full spectral profiling run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralProfileReport {
    /// Unique identifier for this inference run
    pub run_id: String,
    /// Model name/identifier
    pub model: String,
    /// Sequence length analyzed
    pub seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads per layer
    pub num_heads: usize,

    /// Per-head spectral statistics
    pub heads: Vec<AttentionSpectrum>,

    /// Aggregate summary
    pub summary: ProfileSummary,
}

/// Aggregate statistics for a profiling run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    pub total_heads: usize,
    pub healthy_heads: usize,      // degeneracy < 0.3
    pub suspect_heads: usize,      // 0.3 ≤ degeneracy < 0.7
    pub degenerate_heads: usize,   // degeneracy ≥ 0.7
    pub avg_effective_rank: f64,
    pub min_effective_rank: f64,
    pub max_degeneracy: f64,
}

/// Computes spectral statistics using randomized SVD (Halko et al. 2011).
///
/// Complexity: O(n·k·d) instead of O(n³) for full eigendecomposition.
/// Where n = sequence length, k = number of eigenvalues to compute, d = oversampling.
pub struct SpectralProfiler {
    /// Number of eigenvalues to compute (default 10)
    k: usize,
    /// Oversampling parameter p (default 5)
    oversample: usize,
    /// Power iterations for accuracy (default 2)
    power_iter: usize,
}

impl SpectralProfiler {
    /// Create a new profiler with default parameters.
    pub fn new() -> Self {
        Self { k: 10, oversample: 5, power_iter: 2 }
    }

    /// Create a profiler with custom parameters.
    pub fn with_params(k: usize, oversample: usize, power_iter: usize) -> Self {
        Self { k, oversample, power_iter }
    }

    /// Compute spectral statistics for an attention matrix.
    ///
    /// `attention` is the softmax output of Q·Kᵀ/√d, shape (seq_len × seq_len).
    /// Each row sums to 1 (probability distribution over keys).
    pub fn analyze_attention(
        &self,
        layer: usize,
        head: usize,
        attention: &Array2<f32>,
    ) -> AttentionSpectrum {
        let n = attention.nrows();
        assert_eq!(n, attention.ncols(), "Attention matrix must be square");

        // Convert to f64 for numerical stability in eigenvalue computation
        let a_f64: Array2<f64> = attention.mapv(|x| x as f64);

        // Compute A·Aᵀ (symmetric, positive semi-definite)
        // For softmax attention rows sum to 1, so A is row-stochastic.
        // A·Aᵀ eigenvalues tell us about the "spread" of attention patterns.
        let a_at = &a_f64 * &a_f64.t();

        // Compute top-k eigenvalues using randomized SVD
        let top_eigenvalues = self.randomized_top_k_eigenvalues(&a_at);

        // Spectral norm is sqrt of largest eigenvalue of A·Aᵀ
        let spectral_norm = top_eigenvalues.first().copied().unwrap_or(0.0).sqrt();

        // Numerical rank: count eigenvalues > 1e-8 * λ_max
        let lambda_max = top_eigenvalues.first().copied().unwrap_or(1e-12);
        let numerical_rank = top_eigenvalues
            .iter()
            .filter(|&&λ| λ > 1e-8 * lambda_max)
            .count();

        // Effective rank: exp(entropy of normalized eigenvalue distribution)
        let total_eig: f64 = top_eigenvalues.iter().sum();
        let effective_rank = if total_eig > 1e-12 {
            let probs: Vec<f64> = top_eigenvalues.iter().map(|&λ| λ / total_eig).collect();
            let entropy: f64 = probs
                .iter()
                .filter(|&&p| p > 1e-15)
                .map(|&p| -p * p.ln())
                .sum();
            entropy.exp()
        } else {
            0.0
        };

        // Marchenko-Pastur distance
        let mp_distance = self.marchenko_pastur_distance(&top_eigenvalues, n, n);

        // Degeneracy score
        let degeneracy_score = self.compute_degeneracy(
            effective_rank,
            n as f64,
            &top_eigenvalues,
            mp_distance,
        );

        AttentionSpectrum {
            layer,
            head,
            seq_len: n,
            top_eigenvalues,
            spectral_norm,
            numerical_rank,
            effective_rank,
            mp_distance,
            degeneracy_score,
        }
    }

    /// Compute top-k eigenvalues of a symmetric PSD matrix using randomized SVD.
    ///
    /// Algorithm (Halko et al. 2011):
    /// 1. Form sketch Y = A·Ω where Ω is n×(k+p) Gaussian random
    /// 2. QR decompose Y = Q·R (Q has orthonormal columns)
    /// 3. Form B = Qᵀ·A·Q (small (k+p)×(k+p) matrix)
    /// 4. Power iterate: B = (B·Bᵀ)^s · B for accuracy
    /// 5. Full eigendecomposition of B (cheap since k+p << n)
    /// 6. Top-k eigenvalues of A ≈ top-k eigenvalues of B
    fn randomized_top_k_eigenvalues(&self, a: &Array2<f64>) -> Vec<f64> {
        let n = a.nrows();
        let l = (self.k + self.oversample).min(n);  // Target dimension

        if n <= 32 {
            // For small matrices, just do full eigendecomposition
            return self.full_top_k_eigenvalues(a);
        }

        // Generate random Gaussian matrix Ω (n × l)
        let mut rng = StdRng::seed_from_u64(42);  // Deterministic for reproducibility
        let omega: Array2<f64> = Array2::from_shape_fn((n, l), |(_, _)| {
            // Box-Muller transform for standard normal distribution
            let u1: f64 = rng.gen_range(0.0_f64..1.0_f64).max(1e-10);
            let u2: f64 = rng.gen_range(0.0_f64..1.0_f64);
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        });

        // Y = A · Ω
        let mut y = a * &omega;

        // Power iterations for better approximation of dominant subspace
        for _ in 0..self.power_iter {
            // Y = A · (Aᵀ · Y)  [for symmetric A, this amplifies large eigenvalues]
            let at_y = &a.t() * &y;
            y = a * &at_y;
        }

        // QR decomposition of Y to get orthonormal basis Q
        let q = self.qr_orthonormalize(&y);

        // B = Qᵀ · A · Q  (small l×l matrix)
        let b = &q.t() * a * &q;

        // Full eigendecomposition of small matrix B
        let eigenvalues = self.symmetric_eigenvalues(&b);

        // Return top-k
        let mut top_k: Vec<f64> = eigenvalues.into_iter().take(self.k).collect();
        top_k.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        top_k
    }

    /// Full eigendecomposition for small matrices (n ≤ 32).
    fn full_top_k_eigenvalues(&self, a: &Array2<f64>) -> Vec<f64> {
        let eigenvalues = self.symmetric_eigenvalues(a);
        let mut top_k: Vec<f64> = eigenvalues.into_iter().take(self.k).collect();
        top_k.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        top_k
    }

    /// QR decomposition via Gram-Schmidt (simple, no external deps).
    fn qr_orthonormalize(&self, y: &Array2<f64>) -> Array2<f64> {
        let n = y.nrows();
        let l = y.ncols();
        let mut q = Array2::<f64>::zeros((n, l));

        for j in 0..l {
            // Start with column j of Y
            let mut v = y.column(j).to_vec();

            // Orthogonalize against previous Q columns
            for i in 0..j {
                let qi = q.column(i).to_vec();
                let dot: f64 = v.iter().zip(qi.iter()).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(qi.iter()).for_each(|(v_i, q_i)| *v_i -= dot * q_i);
            }

            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for (i, v_i) in v.iter_mut().enumerate() {
                    *v_i /= norm;
                    q[[i, j]] = *v_i;
                }
            } else {
                // Column is linearly dependent; set to unit vector
                if j < n {
                    q[[j, j]] = 1.0;
                }
            }
        }

        q
    }

    /// Compute eigenvalues of a small symmetric matrix using Jacobi rotation.
    fn symmetric_eigenvalues(&self, a: &Array2<f64>) -> Vec<f64> {
        let n = a.nrows();
        if n == 0 {
            return vec![];
        }

        // Work on a copy
        let mut matrix = a.to_owned();
        let max_sweeps = 50;

        for _sweep in 0..max_sweeps {
            // Find largest off-diagonal element
            let mut max_val = 0.0f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if matrix[[i, j]].abs() > max_val {
                        max_val = matrix[[i, j]].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Converged if off-diagonal is negligible
            if max_val < 1e-12 * matrix[[0, 0]].abs().max(1.0) {
                break;
            }

            // Jacobi rotation to zero out element (p, q)
            let app = matrix[[p, p]];
            let aqq = matrix[[q, q]];
            let apq = matrix[[p, q]];

            if apq.abs() < 1e-15 {
                continue;
            }

            // Standard Jacobi rotation formula
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };

            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation to rows/columns p and q
            for i in 0..n {
                if i != p && i != q {
                    let aip = matrix[[i, p]];
                    let aiq = matrix[[i, q]];
                    matrix[[i, p]] = c * aip - s * aiq;
                    matrix[[i, q]] = s * aip + c * aiq;
                    matrix[[p, i]] = matrix[[i, p]];
                    matrix[[q, i]] = matrix[[i, q]];
                }
            }

            let app_new = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            let aqq_new = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            matrix[[p, p]] = app_new;
            matrix[[q, q]] = aqq_new;
            matrix[[p, q]] = 0.0;
            matrix[[q, p]] = 0.0;
        }

        // Extract diagonal (eigenvalues)
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| matrix[[i, i]]).collect();
        // Keep only non-negative (PSD matrix should have these)
        eigenvalues.retain(|&λ| λ > 1e-15);
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Compute KL divergence between empirical eigenvalue distribution
    /// and Marchenko-Pastur law prediction.
    ///
    /// For β=1 (square matrices), MP density is:
    /// ρ(x) = (1/(2πx)) · √((4-x)·x) for x ∈ [0, 4]
    ///
    /// We normalize eigenvalues by their trace to compare distributions.
    pub fn marchenko_pastur_distance(
        &self,
        eigenvalues: &[f64],
        n: usize,
        m: usize,
    ) -> f64 {
        if eigenvalues.is_empty() || n == 0 {
            return 0.0;
        }

        // Normalize eigenvalues to sum to 1 (probability distribution)
        let total: f64 = eigenvalues.iter().sum();
        if total < 1e-15 {
            return 0.0;
        }
        let probs: Vec<f64> = eigenvalues.iter().map(|&λ| λ / total).collect();

        // For β = n/m = 1 (square attention), MP support is [0, 4]
        // Scale eigenvalues to compare with MP on [0, 4]
        let beta = n as f64 / m as f64;
        let lambda_max_mp = 4.0 * beta;  // Upper edge of MP support

        // Compute KL divergence: Σ p_i · log(p_i / q_i)
        // where q_i is the MP density at the corresponding point
        let mut kl = 0.0f64;
        for (&p, &λ) in probs.iter().zip(eigenvalues.iter()) {
            if p < 1e-15 {
                continue;
            }
            // Scale λ to MP support [0, 4β]
            let x_scaled = if total > 1e-15 {
                (λ / total) * n as f64 * lambda_max_mp  // Map to MP scale
            } else {
                0.0
            };

            // MP density at x (for β=1): ρ(x) = √((4-x)x) / (2πx)
            let q = if x_scaled > 1e-10 && x_scaled < lambda_max_mp - 1e-10 {
                let x = x_scaled;
                ((lambda_max_mp - x) * x).sqrt() / (std::f64::consts::PI * 2.0 * x)
            } else if x_scaled <= 1e-10 || x_scaled >= lambda_max_mp - 1e-10 {
                1e-10  // Small value outside support
            } else {
                1e-10
            };

            kl += p * (p / q).ln();
        }

        kl.max(0.0)  // KL divergence is non-negative
    }

    /// Compute degeneracy score from spectral statistics.
    ///
    /// Formula:
    /// degeneracy = 0.4 · (1 - effective_rank / seq_len)    // rank collapse
    ///            + 0.3 · min(λ₁/Σλᵢ, 1.0)                  // top eigenvalue dominance
    ///            + 0.3 · min(mp_distance / 1.0, 1.0)       // deviation from RMT
    fn compute_degeneracy(
        &self,
        effective_rank: f64,
        seq_len: f64,
        eigenvalues: &[f64],
        mp_distance: f64,
    ) -> f64 {
        // Rank collapse component
        let rank_ratio = (effective_rank / seq_len).clamp(0.0, 1.0);
        let rank_collapse = 1.0 - rank_ratio;

        // Top eigenvalue dominance
        let total: f64 = eigenvalues.iter().sum();
        let top_dominance = if total > 1e-15 {
            (eigenvalues.first().copied().unwrap_or(0.0) / total).min(1.0)
        } else {
            0.0
        };

        // RMT deviation
        let rmt_deviation = (mp_distance / 1.0).min(1.0);

        // Weighted combination
        0.4 * rank_collapse + 0.3 * top_dominance + 0.3 * rmt_deviation
    }
}

impl Default for SpectralProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks spectral health of a single attention head over time.
///
/// Used during decode to detect gradual attention collapse (e.g., model
/// starting to repeat itself). Maintains exponential moving averages of
/// effective rank and flags heads that show sustained degeneration.
#[derive(Debug, Clone)]
pub struct HeadHealthTracker {
    /// Exponential moving average of effective rank
    pub ema_effective_rank: f64,
    /// Standard deviation of recent effective ranks
    pub rank_stddev: f64,
    /// Count of consecutive steps where degeneracy > threshold
    pub consecutive_degenerate: usize,
    /// Total number of updates
    pub update_count: usize,
    /// EMA decay factor (0.95 = slow, 0.5 = fast)
    alpha: f64,
}

impl HeadHealthTracker {
    /// Create a new tracker with the given EMA decay.
    pub fn new(alpha: f64) -> Self {
        Self {
            ema_effective_rank: 0.0,
            rank_stddev: 0.0,
            consecutive_degenerate: 0,
            update_count: 0,
            alpha,
        }
    }

    /// Update with a new effective rank observation.
    pub fn update(&mut self, effective_rank: f64, degeneracy: f64) {
        self.update_count += 1;
        if self.update_count == 1 {
            self.ema_effective_rank = effective_rank;
            self.rank_stddev = 0.0;
        } else {
            // EMA update
            let prev_ema = self.ema_effective_rank;
            self.ema_effective_rank =
                self.alpha * self.ema_effective_rank + (1.0 - self.alpha) * effective_rank;
            // Update stddev approximation
            let diff = (effective_rank - prev_ema).abs();
            self.rank_stddev = self.alpha * self.rank_stddev + (1.0 - self.alpha) * diff;
        }

        if degeneracy > 0.5 {
            self.consecutive_degenerate += 1;
        } else {
            self.consecutive_degenerate = 0;
        }
    }

    /// Returns true if this head shows signs of collapse.
    /// Collapse = sustained high degeneracy + low rank variance (stuck in one mode).
    pub fn is_collapsing(&self) -> bool {
        self.consecutive_degenerate >= 5
            && self.ema_effective_rank > 0.0
            && (self.rank_stddev / self.ema_effective_rank) < 0.1
    }

    /// Returns true if this head appears dead (near-zero effective rank).
    pub fn is_dead(&self) -> bool {
        self.update_count >= 3 && self.ema_effective_rank < 1.5
    }
}

impl Default for HeadHealthTracker {
    fn default() -> Self {
        Self::new(0.95)
    }
}

/// Detects spectral drift during generation — when attention patterns
/// change significantly from their initial state.
///
/// Drift can indicate:
/// - Repetition loops (effective rank collapsing toward 1)
/// - Topic shifts (rank increasing as new context is attended to)
/// - Degradation (rank becoming unstable/noisy)
#[derive(Debug, Clone)]
pub struct SpectralDriftDetector {
    /// Baseline effective ranks captured during initial generation steps
    baseline_ranks: Vec<Vec<f64>>,  // [layer][head]
    /// Current EMA of effective ranks
    current_ranks: Vec<Vec<f64>>,   // [layer][head]
    /// Number of steps in baseline window
    baseline_window: usize,
    /// Steps since last drift check
    steps_since_check: usize,
    /// Drift check interval (steps)
    check_interval: usize,
    /// Maximum relative drift before alerting
    drift_threshold: f64,
    /// Last computed drift score (0 = no drift, 1 = maximum)
    pub last_drift_score: f64,
    /// Heads with significant drift since baseline
    pub drifted_heads: Vec<(usize, usize, f64)>,  // (layer, head, relative_drift)
}

impl SpectralDriftDetector {
    /// Create a new drift detector.
    ///
    /// `baseline_window`: number of initial steps to average for baseline
    /// `check_interval`: how often (in steps) to check for drift
    pub fn new(baseline_window: usize, check_interval: usize) -> Self {
        Self {
            baseline_ranks: Vec::new(),
            current_ranks: Vec::new(),
            baseline_window,
            steps_since_check: 0,
            check_interval,
            drift_threshold: 0.5,  // Alert when rank changes by >50%
            last_drift_score: 0.0,
            drifted_heads: Vec::new(),
        }
    }

    /// Initialize with the model's layer/head structure.
    pub fn init_structure(&mut self, num_layers: usize, num_heads: usize) {
        self.baseline_ranks = vec![vec![0.0; num_heads]; num_layers];
        self.current_ranks = vec![vec![0.0; num_heads]; num_layers];
    }

    /// Update with current effective ranks from all heads.
    /// Returns true if drift was detected this step.
    pub fn update(&mut self, current_ranks: &[[[f64]]]) -> bool {
        // Update current ranks (EMA with fast decay for responsiveness)
        for layer in 0..self.current_ranks.len() {
            for head in 0..self.current_ranks[layer].len() {
                if let Some(&rank) = current_ranks.get(layer).and_then(|h| h.get(head)) {
                    self.current_ranks[layer][head] =
                        0.7 * self.current_ranks[layer][head] + 0.3 * rank;
                }
            }
        }

        // Build baseline from first N steps (average of current ranks)
        if self.baseline_ranks.iter().all(|l| l.iter().all(|&r| r == 0.0)) {
            for layer in 0..self.current_ranks.len() {
                for head in 0..self.current_ranks[layer].len() {
                    self.baseline_ranks[layer][head] = self.current_ranks[layer][head];
                }
            }
        }

        // Check drift at intervals
        self.steps_since_check += 1;
        if self.steps_since_check < self.check_interval {
            return false;
        }
        self.steps_since_check = 0;

        // Compute relative drift for each head
        self.drifted_heads.clear();
        let mut total_drift = 0.0f64;
        let mut count = 0;

        for layer in 0..self.current_ranks.len() {
            for head in 0..self.current_ranks[layer].len() {
                let baseline = self.baseline_ranks[layer][head];
                let current = self.current_ranks[layer][head];
                if baseline > 1e-6 {
                    let relative_drift = ((current - baseline) / baseline).abs();
                    total_drift += relative_drift;
                    count += 1;
                    if relative_drift > self.drift_threshold {
                        self.drifted_heads.push((layer, head, relative_drift));
                    }
                }
            }
        }

        self.last_drift_score = if count > 0 {
            total_drift / count as f64
        } else {
            0.0
        };

        // Sort by drift magnitude (most drifted first)
        self.drifted_heads.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        self.drifted_heads.truncate(10);  // Keep top 10

        !self.drifted_heads.is_empty()
    }

    /// Returns true if significant drift was detected.
    pub fn has_drift(&self) -> bool {
        !self.drifted_heads.is_empty()
    }

    /// Reset the baseline to current state (use after topic change or new prompt).
    pub fn reset_baseline(&mut self) {
        for layer in 0..self.current_ranks.len() {
            for head in 0..self.current_ranks[layer].len() {
                self.baseline_ranks[layer][head] = self.current_ranks[layer][head];
            }
        }
        self.drifted_heads.clear();
        self.last_drift_score = 0.0;
    }
}

impl Default for SpectralDriftDetector {
    fn default() -> Self {
        Self::new(5, 10)  // 5-step baseline, check every 10 steps
    }
}

/// Collects spectral profiles across layers and heads during inference.
///
/// Enabled via `PESTI_SPECTRAL_PROFILE=1` environment variable.
/// Zero overhead when disabled (checked once at startup).
pub struct SpectralProfilerCollector {
    profiler: SpectralProfiler,
    run_id: String,
    model: String,
    seq_len: usize,
    num_layers: usize,
    num_heads: usize,
    spectra: Vec<AttentionSpectrum>,
    enabled: bool,
    current_layer_idx: usize,
    /// Per-head health trackers for online monitoring (decode-time)
    head_trackers: Vec<Vec<HeadHealthTracker>>,  // [layer][head]
    /// Spectral drift detector for long-generation monitoring
    drift_detector: SpectralDriftDetector,
}

impl SpectralProfilerCollector {
    /// Check if profiling is enabled (without creating a collector).
    pub fn is_profiling_enabled() -> bool {
        std::env::var("PESTI_SPECTRAL_PROFILE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Create a new collector. Checks `PESTI_SPECTRAL_PROFILE` env var.
    pub fn new(model: String, seq_len: usize, num_layers: usize, num_heads: usize) -> Self {
        let enabled = std::env::var("PESTI_SPECTRAL_PROFILE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        let run_id = format!(
            "spectral-{}-{}",
            model.replace(['/', ':', ' '], "-"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        );

        Self {
            profiler: SpectralProfiler::new(),
            run_id,
            model,
            seq_len,
            num_layers,
            num_heads,
            spectra: Vec::with_capacity(num_layers * num_heads),
            enabled,
            current_layer_idx: 0,
            head_trackers: vec![vec![HeadHealthTracker::default(); num_heads]; num_layers],
            drift_detector: SpectralDriftDetector::new(5, 10),
        }

        // Initialize drift detector structure
        let mut collector = Self {
            profiler: SpectralProfiler::new(),
            run_id,
            model,
            seq_len,
            num_layers,
            num_heads,
            spectra: Vec::with_capacity(num_layers * num_heads),
            enabled,
            current_layer_idx: 0,
            head_trackers: vec![vec![HeadHealthTracker::default(); num_heads]; num_layers],
            drift_detector: SpectralDriftDetector::new(5, 10),
        };
        collector.drift_detector.init_structure(num_layers, num_heads);
        collector
    }

    /// Check if this collector is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current layer being profiled (for integration with attention dispatch).
    /// Returns 0 by default; use set_current_layer() to track progress.
    pub fn current_layer(&self) -> usize {
        self.current_layer_idx
    }

    /// Set the current layer index for profiling.
    pub fn set_current_layer(&mut self, layer: usize) {
        self.current_layer_idx = layer;
    }

    /// Record spectral statistics for one attention head.
    pub fn record(&mut self, spectrum: AttentionSpectrum) {
        if self.enabled {
            self.spectra.push(spectrum);
        }
    }

    /// Analyze and record an attention matrix directly.
    pub fn analyze_head(
        &mut self,
        layer: usize,
        head: usize,
        attention: &Array2<f32>,
    ) {
        if !self.enabled {
            return;
        }
        let spectrum = self.profiler.analyze_attention(layer, head, attention);
        self.spectra.push(spectrum);
    }

    /// Generate the final profiling report.
    pub fn finish(self) -> Option<SpectralProfileReport> {
        if !self.enabled || self.spectra.is_empty() {
            return None;
        }

        let total_heads = self.spectra.len();
        let healthy = self.spectra.iter().filter(|s| s.degeneracy_score < 0.3).count();
        let suspect = self
            .spectra
            .iter()
            .filter(|s| (0.3..0.7).contains(&s.degeneracy_score))
            .count();
        let degenerate = self.spectra.iter().filter(|s| s.degeneracy_score >= 0.7).count();

        let avg_effective_rank: f64 = self
            .spectra
            .iter()
            .map(|s| s.effective_rank)
            .sum::<f64>()
            / total_heads as f64;
        let min_effective_rank = self
            .spectra
            .iter()
            .map(|s| s.effective_rank)
            .fold(f64::MAX, f64::min);
        let max_degeneracy = self
            .spectra
            .iter()
            .map(|s| s.degeneracy_score)
            .fold(0.0f64, f64::max);

        Some(SpectralProfileReport {
            run_id: self.run_id,
            model: self.model,
            seq_len: self.seq_len,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            heads: self.spectra,
            summary: ProfileSummary {
                total_heads,
                healthy_heads: healthy,
                suspect_heads: suspect,
                degenerate_heads: degenerate,
                avg_effective_rank,
                min_effective_rank,
                max_degeneracy,
            },
        })
    }

    /// Update a head's health tracker with a decode-time attention vector.
    ///
    /// During decode, attention is a single row (1 × seq_len) — the new query
    /// attending to all cached keys. We compute effective rank from this vector
    /// and update the EMA-based health tracker.
    pub fn update_decode_head(
        &mut self,
        layer: usize,
        head: usize,
        attention_vector: &[f32],
    ) {
        if !self.enabled || attention_vector.is_empty() {
            return;
        }

        let n = attention_vector.len();
        // Compute effective rank from the attention vector (entropy-based)
        let total: f64 = attention_vector.iter().map(|&x| x as f64).sum();
        if total < 1e-15 {
            return;
        }

        let probs: Vec<f64> = attention_vector.iter().map(|&x| (x as f64) / total).collect();
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum();
        let effective_rank = entropy.exp();

        // Degeneracy for a single vector: how concentrated is it?
        // A one-hot vector (rank 1) is maximally degenerate.
        let degeneracy = 1.0 - (effective_rank / n as f64).clamp(0.0, 1.0);

        if layer < self.head_trackers.len() && head < self.head_trackers[layer].len() {
            self.head_trackers[layer][head].update(effective_rank, degeneracy);
        }
    }

    /// Get health status for all heads.
    pub fn head_health(&self) -> Vec<Vec<&HeadHealthTracker>> {
        self.head_trackers.iter().map(|l| l.iter().collect()).collect()
    }

    /// Update spectral drift detection with current effective ranks from all heads.
    /// Call this periodically during generation (e.g., every 10 tokens).
    ///
    /// `ranks` should be shaped [num_layers][num_heads] with current effective ranks.
    /// Returns true if significant drift was detected.
    pub fn update_drift(&mut self, ranks: &[[[f64]]]) -> bool {
        if !self.enabled {
            return false;
        }
        // Build 3D array from head trackers
        let current_ranks: Vec<Vec<f64>> = self
            .head_trackers
            .iter()
            .map(|layer| layer.iter().map(|t| t.ema_effective_rank).collect())
            .collect();
        self.drift_detector.update(&[&current_ranks[..]])
    }

    /// Get the drift detector for direct access.
    pub fn drift_detector(&self) -> &SpectralDriftDetector {
        &self.drift_detector
    }

    /// Reset drift baseline (use after topic change or new prompt).
    pub fn reset_drift_baseline(&mut self) {
        self.drift_detector.reset_baseline();
    }

    /// Find heads that are collapsing or dead.
    pub fn unhealthy_heads(&self) -> Vec<(usize, usize, &'static str)> {
        let mut result = Vec::new();
        for (layer, heads) in self.head_trackers.iter().enumerate() {
            for (head, tracker) in heads.iter().enumerate() {
                if tracker.is_collapsing() {
                    result.push((layer, head, "collapsing"));
                } else if tracker.is_dead() {
                    result.push((layer, head, "dead"));
                }
            }
        }
        result
    }

    /// Export report as JSON string.
    pub fn to_json(&self) -> Option<String> {
        if self.spectra.is_empty() {
            return None;
        }
        let total_heads = self.spectra.len();
        let healthy = self.spectra.iter().filter(|s| s.degeneracy_score < 0.3).count();
        let suspect = self
            .spectra
            .iter()
            .filter(|s| (0.3..0.7).contains(&s.degeneracy_score))
            .count();
        let degenerate = self.spectra.iter().filter(|s| s.degeneracy_score >= 0.7).count();

        let report = SpectralProfileReport {
            run_id: self.run_id.clone(),
            model: self.model.clone(),
            seq_len: self.seq_len,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            heads: self.spectra.clone(),
            summary: ProfileSummary {
                total_heads,
                healthy_heads: healthy,
                suspect_heads: suspect,
                degenerate_heads: degenerate,
                avg_effective_rank: 0.0,
                min_effective_rank: 0.0,
                max_degeneracy: 0.0,
            },
        };
        serde_json::to_string_pretty(&report).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_detector_no_drift() {
        let mut detector = SpectralDriftDetector::new(2, 5);
        detector.init_structure(1, 2);  // 1 layer, 2 heads

        // Simulate stable generation (same ranks every step)
        for _ in 0..10 {
            let drifted = detector.update(&[[[5.0, 3.0]]]);
            assert!(!drifted);
        }
        assert!(!detector.has_drift());
    }

    #[test]
    fn test_drift_detector_detects_collapse() {
        let mut detector = SpectralDriftDetector::new(2, 5);
        detector.init_structure(1, 2);  // 1 layer, 2 heads

        // Establish baseline with healthy ranks
        for _ in 0..5 {
            detector.update(&[[[8.0, 6.0]]]);
        }

        // Simulate collapse (rank drops significantly)
        let mut detected = false;
        for _ in 0..10 {
            if detector.update(&[[[2.0, 1.5]]]) {
                detected = true;
                break;
            }
        }

        assert!(detected, "Should detect drift when rank collapses");
        assert!(detector.has_drift());
        assert!(!detector.drifted_heads.is_empty());
    }

    #[test]
    fn test_drift_detector_reset_baseline() {
        let mut detector = SpectralDriftDetector::new(2, 5);
        detector.init_structure(1, 1);

        // Establish baseline
        for _ in 0..5 {
            detector.update(&[[[8.0]]]);
        }

        // Shift to new stable state
        for _ in 0..5 {
            detector.update(&[[[4.0]]]);
        }

        // Reset baseline to current (lower) state
        detector.reset_baseline();

        // Should no longer detect drift at the new level
        for _ in 0..10 {
            assert!(!detector.update(&[[[4.0]]]));
        }
    }

    #[test]
    fn test_randomized_svd_small_matrix() {
        let profiler = SpectralProfiler::new();
        // Identity matrix: all eigenvalues = 1
        let n = 8;
        let a = Array2::eye(n);
        let eigs = profiler.randomized_top_k_eigenvalues(&a.mapv(|x: f32| x as f64));
        assert_eq!(eigs.len(), 8.min(10));
        // All eigenvalues should be close to 1
        for &λ in &eigs {
            assert!((λ - 1.0).abs() < 0.1, "Eigenvalue {} not close to 1", λ);
        }
    }

    #[test]
    fn test_rank_one_matrix_detection() {
        let profiler = SpectralProfiler::new();
        // Rank-1 matrix: all rows identical
        let n = 16;
        let row = vec![1.0f32 / n as f32; n];
        let a = Array2::from_shape_vec((n, n), row.repeat(n)).unwrap();
        let spectrum = profiler.analyze_attention(0, 0, &a);

        // Rank-1 should have high degeneracy
        assert!(spectrum.degeneracy_score > 0.5,
            "Rank-1 matrix should be degenerate, got {}", spectrum.degeneracy_score);
        assert_eq!(spectrum.numerical_rank, 1,
            "Rank-1 matrix should have numerical rank 1, got {}", spectrum.numerical_rank);
    }

    #[test]
    fn test_local_attention_is_healthy() {
        let profiler = SpectralProfiler::new();
        // Local/banded attention: each query attends mainly to nearby keys
        // This is a realistic "healthy" pattern in transformers
        let n = 32;
        let bandwidth = 5;
        let mut a = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let dist = (i as i32 - j as i32).abs() as usize;
                if dist <= bandwidth {
                    a[[i, j]] = 1.0f32 / (2 * bandwidth + 1) as f32;
                }
            }
        }
        let spectrum = profiler.analyze_attention(0, 0, &a);

        // Local attention should be relatively healthy (not fully degenerate)
        // It has higher rank than uniform attention because rows differ
        assert!(spectrum.degeneracy_score < 0.8,
            "Local attention should not be fully degenerate, got {}", spectrum.degeneracy_score);
    }

    #[test]
    fn test_uniform_attention_is_rank_one() {
        let profiler = SpectralProfiler::new();
        // Uniform attention: all rows identical -> rank 1 matrix
        let n = 32;
        let a = Array2::from_elem((n, n), 1.0f32 / n as f32);
        let spectrum = profiler.analyze_attention(0, 0, &a);

        // All rows identical means rank-1, which is degenerate
        assert_eq!(spectrum.numerical_rank, 1,
            "Uniform attention should have numerical rank 1, got {}", spectrum.numerical_rank);
        assert!(spectrum.degeneracy_score > 0.5,
            "Rank-1 uniform attention should be degenerate, got {}", spectrum.degeneracy_score);
    }

    #[test]
    fn test_effective_rank_calculation() {
        let profiler = SpectralProfiler::new();
        // Matrix with known eigenvalue structure
        let n = 10;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = (n - i) as f64;  // Eigenvalues: 10, 9, 8, ..., 1
        }
        let eigs = profiler.full_top_k_eigenvalues(&a);
        assert_eq!(eigs.len(), 10);
        assert!((eigs[0] - 10.0).abs() < 0.1);
        assert!((eigs[9] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_mp_distance_zero_for_empty() {
        let profiler = SpectralProfiler::new();
        let dist = profiler.marchenko_pastur_distance(&[], 10, 10);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_collector_enabled_flag() {
        // Collector should be disabled by default (no env var)
        unsafe { std::env::remove_var("PESTI_SPECTRAL_PROFILE") };
        let collector = SpectralProfilerCollector::new(
            "test-model".to_string(), 10, 2, 4
        );
        assert!(!collector.enabled);

        // Should be enabled with env var
        unsafe { std::env::set_var("PESTI_SPECTRAL_PROFILE", "1") };
        let collector = SpectralProfilerCollector::new(
            "test-model".to_string(), 10, 2, 4
        );
        assert!(collector.enabled);
        unsafe { std::env::remove_var("PESTI_SPECTRAL_PROFILE") };
    }
}
