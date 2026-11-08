//! Weight statistics operations for analysis.

use serde::{Deserialize, Serialize};

/// A weight statistics operation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStatOp {
    /// Tensor name to compute statistics on.
    pub param: String,
    /// Statistic type to compute.
    pub stat_type: StatType,
}

/// Types of statistics that can be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatType {
    /// Mean value of all elements.
    Mean,
    /// Standard deviation of all elements.
    StdDev,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Number of non-zero elements.
    NonZeroCount,
    /// L2 norm (Frobenius norm for matrices).
    L2Norm,
}

/// Result of a weight statistics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStatResult {
    /// Tensor name.
    pub param: String,
    /// Statistic type computed.
    pub stat_type: StatType,
    /// Computed value.
    pub value: f64,
    /// Total number of elements in the tensor.
    pub element_count: usize,
}
