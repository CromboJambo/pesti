//! Divergence metrics: how far the precise path has drifted from the stable summary.

#[derive(Debug, Clone, Copy)]
pub enum DivergenceMetric {
    /// cosine distance = 1 - cos(a, b). In [0,2]. Scale-invariant.
    Cosine,
    /// normalized L2 = ||a-b|| / (||a|| + ||b|| + eps). In [0,1]. Magnitude-sensitive.
    RelL2,
}

#[derive(Debug, Clone, Copy)]
pub struct DivergenceScore {
    pub metric: DivergenceMetric,
    pub value: f32,
    pub ref_norm: Option<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..a.len() {
        sum += a[i] as f64 * b[i] as f64;
    }
    sum
}

fn norm(a: &[f32]) -> f64 {
    let sum: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    sum.sqrt()
}

pub fn divergence(metric: DivergenceMetric, stable: &[f32], precise: &[f32]) -> DivergenceScore {
    let n = norm(stable);
    let p = norm(precise);
    let eps = 1e-12_f64;

    match metric {
        DivergenceMetric::Cosine => {
            if (n < eps) || (p < eps) {
                return DivergenceScore {
                    metric: DivergenceMetric::Cosine,
                    value: 0.0,
                    ref_norm: Some(n as f32),
                };
            }
            let d = dot(stable, precise);
            let cos_val = (d / (n * p)).clamp(-1.0, 1.0);
            DivergenceScore {
                metric: DivergenceMetric::Cosine,
                value: (1.0 - cos_val) as f32,
                ref_norm: Some(n as f32),
            }
        }
        DivergenceMetric::RelL2 => {
            let mut sum_sq = 0.0_f64;
            for i in 0..stable.len() {
                let diff = stable[i] as f64 - precise[i] as f64;
                sum_sq += diff * diff;
            }
            let dist = sum_sq.sqrt();
            let denom = (n + p).max(eps);
            DivergenceScore {
                metric: DivergenceMetric::RelL2,
                value: (dist / denom) as f32,
                ref_norm: Some(n as f32),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_zero_divergence() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let score = divergence(DivergenceMetric::Cosine, &a, &b);
        assert!(
            (score.value - 0.0).abs() < 1e-6,
            "cosine should be 0 for identical vectors"
        );

        let score2 = divergence(DivergenceMetric::RelL2, &a, &b);
        assert!(
            (score2.value - 0.0).abs() < 1e-6,
            "relL2 should be 0 for identical vectors"
        );
    }

    #[test]
    fn orthogonal_vectors_cosine_one() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = divergence(DivergenceMetric::Cosine, &a, &b);
        assert!(
            (score.value - 1.0).abs() < 1e-6,
            "cosine distance for orthogonal should be 1"
        );
    }

    #[test]
    fn zero_vector_no_nan() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        let score = divergence(DivergenceMetric::Cosine, &a, &b);
        assert!(score.value.is_finite(), "should not be NaN for zero vector");
    }

    #[test]
    fn rel_l2_bounded() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let score = divergence(DivergenceMetric::RelL2, &a, &b);
        assert!(
            score.value >= 0.0 && score.value <= 1.0,
            "relL2 must be in [0,1]"
        );
    }
}
