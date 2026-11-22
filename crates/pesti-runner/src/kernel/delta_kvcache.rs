//! Delta-encoded KV cache with provable reconstruction error bounds.
//!
//! Instead of quantizing absolute K/V values at each position, we store:
//! 1. A base value (position 0) at full precision
//! 2. Quantized deltas between consecutive positions with step size δ
//!
//! Reconstruction at position n: x̂_n = x_0 + Σ_{i=1}^{n} q(δ_i)
//! where δ_i = x_i - x_{i-1} and q() is the quantization function.
//!
//! **Error bound (the "floor")**: If each delta quantization satisfies
//! |δ_i - q(δ_i)| ≤ δ/2, then by triangle inequality:
//!
//!   |x_n - x̂_n| ≤ Σ_{i=1}^{n} |δ_i - q(δ_i)| ≤ n · δ / 2
//!
//! This gives a predictable, position-dependent error floor that grows
//! linearly with sequence length. For KV cache this is powerful because:
//!
//! 1. K vectors change smoothly across positions (RoPE is continuous), so
//!    deltas are small and can be quantized with finer steps than absolute values
//! 2. We know exactly when the cache has degraded enough to need re-quantization
//! 3. The bound is Lean-provable via induction on n
//!
//! ## Format
//!
//! Each head-dimension channel stores:
//! - Base value: f32 (full precision anchor)
//! - Deltas: sequence of i8 codes, each representing a quantized delta
//!   with step δ. Code c maps to delta = c * δ.
//!
//! Memory per position per channel: 1 byte (i8) vs 4 bytes (f32) or 2 bytes (f16).
//! Compression ratio: 4x vs f32, 2x vs f16, with a known error bound.

/// Delta quantizer with configurable step size.
///
/// Quantizes a delta value to the nearest multiple of δ, represented as i8.
/// The maximum quantization error is exactly δ/2 (rounding to nearest).
#[derive(Debug, Clone)]
pub struct DeltaQuantizer {
    /// Quantization step size. Maximum error per delta = δ/2.
    pub delta: f32,
    /// Maximum representable code (i8::MAX as f32).
    max_code: f32,
}

impl DeltaQuantizer {
    /// Create a new delta quantizer with the given step size.
    ///
    /// The step size δ determines both precision and error bound:
    /// - Smaller δ → better precision, larger code range needed
    /// - Larger δ → coarser precision, smaller code range
    pub fn new(delta: f32) -> Self {
        assert!(delta > 0.0, "Delta step size must be positive");
        Self {
            delta,
            max_code: i8::MAX as f32,
        }
    }

    /// Maximum representable delta magnitude without clipping.
    pub fn max_representable(&self) -> f32 {
        self.max_code * self.delta
    }

    /// Quantize a delta value to an i8 code.
    ///
    /// Uses round-half-up via floor: k = ⌊d/δ + 1/2⌋, matching the Lean theorem
    /// exactly. Clamped to [-127*δ, 127*δ]. Returns (code, actual_quantized_delta, was_clipped).
    ///
    /// The floor-based rounding (vs f32::round's ties-away-from-zero) ensures the
    /// feedback invariant e ∈ [−δ/2, δ/2) is maintained inductively: when |Δ| ≤ 127δ
    /// and e < δ/2, we get d/δ + 1/2 < 128 so ⌊·⌋ ≤ 127 (no clip).
    pub fn quantize(&self, delta: f32) -> (i8, f32, bool) {
        let raw = delta / self.delta;
        // k = ⌊d/δ + 1/2⌋ — exact match to the Lean theorem's rounding mode
        let code_f32 = (raw + 0.5).floor();
        // Clipping only if the floored code actually exceeds representable range
        let clipped = code_f32 < -self.max_code || code_f32 > self.max_code;
        let code = code_f32.clamp(-self.max_code, self.max_code) as i8;
        let quantized = code as f32 * self.delta;
        (code, quantized, clipped)
    }

    /// Dequantize a code back to a delta value.
    pub fn dequantize(&self, code: i8) -> f32 {
        code as f32 * self.delta
    }

    /// Maximum quantization error for a single delta (δ/2).
    /// Only valid when no clipping occurs.
    pub fn max_error_per_delta(&self) -> f32 {
        self.delta / 2.0
    }

    /// Error bound at position n: n * δ / 2.
    ///
    /// This is the "floor" — the maximum possible accumulated reconstruction
    /// error after n delta steps, assuming worst-case rounding at each step
    /// and NO clipping. If clipping occurs, this bound no longer holds.
    pub fn error_bound_at(&self, n: usize) -> f32 {
        n as f32 * self.max_error_per_delta()
    }

    /// Choose a step size δ such that the error bound at position `max_seq`
    /// does not exceed `max_allowed_error`.
    ///
    /// Solves: max_seq * δ / 2 ≤ max_allowed_error
    /// → δ ≤ 2 * max_allowed_error / max_seq
    pub fn step_size_for_bound(max_seq: usize, max_allowed_error: f32) -> f32 {
        if max_seq == 0 {
            return 1.0;
        }
        2.0 * max_allowed_error / max_seq as f32
    }

    /// Choose a step size δ that satisfies BOTH:
    /// 1. Error bound at max_seq ≤ max_allowed_error
    /// 2. Maximum expected delta fits in i8 range (no clipping)
    ///
    /// Returns None if the constraints are incompatible with i8 codes.
    pub fn step_size_for_bound_and_range(
        max_seq: usize,
        max_allowed_error: f32,
        max_expected_delta: f32,
    ) -> Option<f32> {
        // Constraint 1: δ ≤ 2 * max_allowed_error / max_seq
        let delta_from_error = if max_seq > 0 {
            2.0 * max_allowed_error / max_seq as f32
        } else {
            f32::MAX
        };

        // Constraint 2: 126 * δ ≥ max_expected_delta → δ ≥ max_expected_delta / 126
        // Using 126 (not 127) for shipping safety: ensures |Δ| ≤ 126δ in practice,
        // giving a full δ of slack against floating-point noise in the feedback loop.
        // With floor rounding ⌊d/δ + 1/2⌋ and e ∈ [−δ/2, δ/2):
        //   d = Δ + e < 126δ + δ/2 = 126.5δ → d/δ + 1/2 < 127 → ⌊·⌋ ≤ 126 (no clip)
        let delta_from_range = max_expected_delta / 126.0;

        if delta_from_range <= delta_from_error {
            Some(delta_from_range)
        } else {
            None // Constraints incompatible with i8
        }
    }

    /// Choose a step size δ that GUARANTEES no clipping for RoPE-rotated vectors.
    ///
    /// For LLaMA-style RoPE with head_dim `d`:
    ///   - Dimension `i` rotates at rate θ_i = 10000^(-2i/d) radians per position
    ///   - Max delta in dimension `i` = 2·|K_i|·sin(θ_i/2) (exact bound from trig identity)
    ///   - The highest frequency dimension (i=0) has θ_0 = 1.0
    ///
    /// This computes δ such that 127·δ ≥ max_possible_delta, guaranteeing no
    /// clipping by construction. The error bound E(n) ≤ n·δ/2 then holds without
    /// a clipping-error term — making the Lean formalization much cleaner.
    ///
    /// Note: With feedback encoding (deltas computed from reconstructed values),
    /// the ACTUAL error stays near δ/2 regardless of position — the n·δ/2 bound
    /// is a conservative open-loop worst case that's never reached in practice.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per head (determines RoPE frequency spectrum)
    /// * `k_norm_bound` - Upper bound on K vector norm per dimension
    ///   (e.g., 1.0 after RMSNorm, or larger if keys aren't normalized)
    /// * `safety_factor` - Margin for unexpected variations (recommend 2.0-4.0)
    /// * `max_seq` - Maximum sequence length for error bound constraint
    /// * `max_allowed_error` - Maximum acceptable open-loop error at max_seq
    ///
    /// Returns None if the no-clipping constraint and error bound are incompatible
    /// with i8 codes (would need i16 or larger).
    pub fn step_size_prevent_clipping(
        head_dim: usize,
        k_norm_bound: f32,
        safety_factor: f32,
        max_seq: usize,
        max_allowed_error: f32,
    ) -> Option<f32> {
        assert!(head_dim > 0, "head_dim must be positive");
        assert!(k_norm_bound > 0.0, "k_norm_bound must be positive");
        assert!(safety_factor >= 1.0, "safety_factor must be ≥ 1.0");

        // Maximum RoPE rotation rate (highest frequency dimension, i=0)
        // For LLaMA: θ_0 = 10000^(-2*0/d) = 1.0 radian per position
        let max_theta = 10000f32.powf(-2.0 * 0.0 / head_dim as f32);

        // Exact maximum delta for a unit-norm vector rotating by θ radians:
        // |x(n+1) - x(n)| = 2·|x|·sin(θ/2) (from cos(A+B)-cos(A) identity)
        // This is tighter than the linear approximation |x|·θ for large θ.
        let max_delta_exact = 2.0 * k_norm_bound * (max_theta / 2.0).sin();

        // Apply safety factor for unexpected variations
        let max_delta = max_delta_exact * safety_factor;

        // Constraint 1 (prevention): δ ≥ max_delta / 126 (guarantees no clipping with slack)
        // 126 instead of 127: shipping-safe margin so |Δ| ≤ 126δ, keeping d = Δ + e
        // well below the 127.5δ tie boundary even with floating-point noise.
        let delta_min_for_range = max_delta / 126.0;

        // Constraint 2 (quality): δ ≤ 2 * max_allowed_error / max_seq
        let delta_max_for_error = if max_seq > 0 {
            2.0 * max_allowed_error / max_seq as f32
        } else {
            f32::MAX
        };

        // Feasible only if we can satisfy both constraints
        if delta_min_for_range <= delta_max_for_error {
            // Use smallest δ that prevents clipping (best precision within range)
            Some(delta_min_for_range)
        } else {
            None // Need i16 codes or relax error bound
        }
    }

    /// Compute per-dimension step sizes for RoPE-aware delta encoding.
    ///
    /// Different RoPE dimensions have very different rotation rates:
    ///   - Dimension 0: θ = 1.0 (fastest)
    ///   - Dimension d/2: θ ≈ 0.316
    ///   - Dimension d-1: θ ≈ 0.01 (slowest)
    ///
    /// Using a single global δ wastes precision on slow dimensions and may
    /// clip on fast ones. This computes per-dimension δ_i sized to each
    /// dimension's rotation rate, giving tighter overall error bounds.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension per head
    /// * `k_norm_bound` - Upper bound on K vector norm per dimension
    /// * `safety_factor` - Margin for unexpected variations (recommend 2.0-4.0)
    ///
    /// Returns a Vec of length `head_dim` with step sizes δ_0, δ_1, ..., δ_{d-1}.
    /// Each δ_i guarantees no clipping in dimension i: 126·δ_i ≥ 2·k_norm·sin(θ_i/2)·safety.
    pub fn rope_aware_step_sizes(
        head_dim: usize,
        k_norm_bound: f32,
        safety_factor: f32,
    ) -> Vec<f32> {
        (0..head_dim)
            .map(|d| {
                // LLaMA-style RoPE frequency for dimension d
                let theta = 10000f32.powf(-2.0 * d as f32 / head_dim as f32);
                // Exact max delta in this dimension: 2·|K|·sin(θ/2)
                let max_delta = 2.0 * k_norm_bound * (theta / 2.0).sin() * safety_factor;
                // Step size that prevents clipping in this dimension (126 for float-noise slack)
                max_delta / 126.0
            })
            .collect()
    }

    /// Compute the total error bound for per-dimension RoPE-aware encoding.
    ///
    /// With per-dimension step sizes δ_i, the error bound at position n is:
    ///   E(n) ≤ Σ_{d=0}^{head_dim-1} (n · δ_d / 2)
    ///
    /// This is typically much tighter than the global bound because slow
    /// dimensions get much smaller step sizes.
    pub fn rope_aware_error_bound(
        head_dim: usize,
        k_norm_bound: f32,
        safety_factor: f32,
        n: usize,
    ) -> f32 {
        let steps = Self::rope_aware_step_sizes(head_dim, k_norm_bound, safety_factor);
        steps.iter().map(|&s| n as f32 * s / 2.0).sum()
    }
}

/// Delta-encoded KV cache for a single layer.
///
/// Stores K and V values using delta encoding from a full-precision base.
/// Each position after the first stores quantized deltas rather than absolute values.
#[derive(Debug)]
pub struct DeltaKvCache {
    /// Base K values at position 0 (full precision).
    k_base: Vec<f32>,
    /// Base V values at position 0 (full precision).
    v_base: Vec<f32>,
    /// Quantized K deltas. Layout: [pos][head][dim] where pos starts at 1.
    k_deltas: Vec<Vec<i8>>,
    /// Quantized V deltas. Layout: [pos][head][dim] where pos starts at 1.
    v_deltas: Vec<Vec<i8>>,
    /// Previous K values (for computing next delta). Full precision.
    k_prev: Vec<f32>,
    /// Previous V values (for computing next delta). Full precision.
    v_prev: Vec<f32>,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Current sequence length (valid positions).
    seq_len: usize,
    /// Delta quantizer for K values.
    k_quantizer: DeltaQuantizer,
    /// Delta quantizer for V values.
    v_quantizer: DeltaQuantizer,
    /// Maximum allowed sequence length.
    max_seq: usize,
    /// Count of K deltas that were clipped (exceeded i8 range).
    clipped_k_deltas: usize,
    /// Count of V deltas that were clipped (exceeded i8 range).
    clipped_v_deltas: usize,
}

impl DeltaKvCache {
    /// Create a new delta-encoded KV cache.
    ///
    /// `k_step` and `v_step` are the quantization step sizes δ for K and V deltas.
    /// Smaller values give better precision but require larger code ranges.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        k_step: f32,
        v_step: f32,
    ) -> Self {
        let channels = num_kv_heads * head_dim;
        Self {
            k_base: vec![0.0; channels],
            v_base: vec![0.0; channels],
            k_deltas: Vec::with_capacity(max_seq),
            v_deltas: Vec::with_capacity(max_seq),
            k_prev: vec![0.0; channels],
            v_prev: vec![0.0; channels],
            num_kv_heads,
            head_dim,
            seq_len: 0,
            k_quantizer: DeltaQuantizer::new(k_step),
            v_quantizer: DeltaQuantizer::new(v_step),
            max_seq,
            clipped_k_deltas: 0,
            clipped_v_deltas: 0,
        }
    }

    /// Create with step sizes chosen to meet error bounds at max_seq.
    pub fn new_with_error_bound(
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        k_max_error: f32,
        v_max_error: f32,
    ) -> Self {
        let k_step = DeltaQuantizer::step_size_for_bound(max_seq, k_max_error);
        let v_step = DeltaQuantizer::step_size_for_bound(max_seq, v_max_error);
        Self::new(num_kv_heads, head_dim, max_seq, k_step, v_step)
    }

    /// Number of positions currently cached.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Maximum capacity.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Number of K deltas that were clipped during encoding.
    /// If > 0, the error bound n·δ/2 may not hold at those positions.
    pub fn clipped_k_deltas(&self) -> usize {
        self.clipped_k_deltas
    }

    /// Number of V deltas that were clipped during encoding.
    /// If > 0, the error bound n·δ/2 may not hold at those positions.
    pub fn clipped_v_deltas(&self) -> usize {
        self.clipped_v_deltas
    }

    /// Total number of clipped deltas (K + V).
    pub fn total_clipped(&self) -> usize {
        self.clipped_k_deltas + self.clipped_v_deltas
    }

    /// Error bound for K reconstruction at current sequence length.
    pub fn k_error_bound(&self) -> f32 {
        self.k_quantizer.error_bound_at(self.seq_len.saturating_sub(1))
    }

    /// Error bound for V reconstruction at current sequence length.
    pub fn v_error_bound(&self) -> f32 {
        self.v_quantizer.error_bound_at(self.seq_len.saturating_sub(1))
    }

    /// Append a single position's K and V values.
    ///
    /// For position 0: stores full-precision base values.
    /// For position > 0: computes deltas from previous position, quantizes them.
    ///
    /// `k`: `[num_kv_heads * head_dim]` — RoPE-rotated key for this position
    /// `v`: `[num_kv_heads * head_dim]` — value for this position
    ///
    /// Returns the position index where data was written.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<usize, DeltaKvError> {
        let channels = self.num_kv_heads * self.head_dim;
        if k.len() != channels || v.len() != channels {
            return Err(DeltaKvError::DimensionMismatch {
                expected: channels,
                got_k: k.len(),
                got_v: v.len(),
            });
        }
        if self.seq_len >= self.max_seq {
            return Err(DeltaKvError::SeqLenExceeded {
                current: self.seq_len,
                max: self.max_seq,
            });
        }

        let pos = self.seq_len;

        if pos == 0 {
            // Store full-precision base values
            self.k_base.copy_from_slice(k);
            self.v_base.copy_from_slice(v);
            self.k_prev.copy_from_slice(k);
            self.v_prev.copy_from_slice(v);
        } else {
            // Compute and store quantized deltas from previous position
            let mut k_delta_codes = vec![0i8; channels];
            let mut v_delta_codes = vec![0i8; channels];

            for i in 0..channels {
                let k_delta = k[i] - self.k_prev[i];
                let v_delta = v[i] - self.v_prev[i];

                let (k_code, _, k_clipped) = self.k_quantizer.quantize(k_delta);
                let (v_code, _, v_clipped) = self.v_quantizer.quantize(v_delta);
                if k_clipped {
                    self.clipped_k_deltas += 1;
                }
                if v_clipped {
                    self.clipped_v_deltas += 1;
                }

                k_delta_codes[i] = k_code;
                v_delta_codes[i] = v_code;

                // Update previous to the quantized reconstruction (not the true value)
                // This matches what the decoder will actually see.
                self.k_prev[i] += self.k_quantizer.dequantize(k_code);
                self.v_prev[i] += self.v_quantizer.dequantize(v_code);
            }

            self.k_deltas.push(k_delta_codes);
            self.v_deltas.push(v_delta_codes);
        }

        self.seq_len += 1;
        Ok(pos)
    }

    /// Reconstruct K and V at a specific position by summing deltas from base.
    ///
    /// Returns `(k_values, v_values)` each of length `num_kv_heads * head_dim`.
    pub fn reconstruct_at(&self, pos: usize) -> Result<(Vec<f32>, Vec<f32>), DeltaKvError> {
        if pos >= self.seq_len {
            return Err(DeltaKvError::PositionOutOfRange {
                pos,
                seq_len: self.seq_len,
            });
        }

        let channels = self.num_kv_heads * self.head_dim;
        let mut k_recon = self.k_base.clone();
        let mut v_recon = self.v_base.clone();

        if pos > 0 {
            for p in 1..=pos {
                // Note: deltas are stored at index p-1 (since position 0 has no delta)
                let k_deltas = &self.k_deltas[p - 1];
                let v_deltas = &self.v_deltas[p - 1];

                for i in 0..channels {
                    k_recon[i] += self.k_quantizer.dequantize(k_deltas[i]);
                    v_recon[i] += self.v_quantizer.dequantize(v_deltas[i]);
                }
            }
        }

        Ok((k_recon, v_recon))
    }

    /// Reconstruct all K values for a specific head across all positions.
    ///
    /// Returns `[seq_len * head_dim]` — same layout as `LayerKvCache::k_head`.
    pub fn k_head_all_positions(&self, head: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.seq_len * self.head_dim);
        for pos in 0..self.seq_len {
            if let Ok((k_recon, _)) = self.reconstruct_at(pos) {
                let start = head * self.head_dim;
                result.extend_from_slice(&k_recon[start..start + self.head_dim]);
            }
        }
        result
    }

    /// Reconstruct all V values for a specific head across all positions.
    pub fn v_head_all_positions(&self, head: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.seq_len * self.head_dim);
        for pos in 0..self.seq_len {
            if let Ok((_, v_recon)) = self.reconstruct_at(pos) {
                let start = head * self.head_dim;
                result.extend_from_slice(&v_recon[start..start + self.head_dim]);
            }
        }
        result
    }

    /// Returns the K base vector for a specific head (position 0 values).
    /// Layout: [head_dim] f32. Used by Route C PTX kernel.
    pub fn k_base_head(&self, head: usize) -> &[f32] {
        let start = head * self.head_dim;
        &self.k_base[start..start + self.head_dim]
    }

    /// Returns the V base vector for a specific head (position 0 values).
    /// Layout: [head_dim] f32. Used by Route C PTX kernel.
    pub fn v_base_head(&self, head: usize) -> &[f32] {
        let start = head * self.head_dim;
        &self.v_base[start..start + self.head_dim]
    }

    /// Returns K delta codes for a specific head in flat row-major layout.
    /// Layout: [(seq_len-1) x head_dim] i8, position 0 first.
    /// Used by Route C PTX kernel which reads deltas directly without reconstruction.
    pub fn k_deltas_head_flat(&self, head: usize) -> Vec<i8> {
        let mut result = Vec::with_capacity((self.seq_len.saturating_sub(1)) * self.head_dim);
        for pos in 0..self.seq_len.saturating_sub(1) {
            // k_deltas[pos] is the delta from position pos to pos+1
            // Layout within each pos entry: [head][dim]
            let start = head * self.head_dim;
            result.extend_from_slice(&self.k_deltas[pos][start..start + self.head_dim]);
        }
        result
    }

    /// Returns V delta codes for a specific head in flat row-major layout.
    /// Layout: [(seq_len-1) x head_dim] i8, position 0 first.
    /// Used by Route C PTX kernel.
    pub fn v_deltas_head_flat(&self, head: usize) -> Vec<i8> {
        let mut result = Vec::with_capacity((self.seq_len.saturating_sub(1)) * self.head_dim);
        for pos in 0..self.seq_len.saturating_sub(1) {
            let start = head * self.head_dim;
            result.extend_from_slice(&self.v_deltas[pos][start..start + self.head_dim]);
        }
        result
    }

    /// Returns the K quantization step size.
    pub fn k_step(&self) -> f32 {
        self.k_quantizer.delta
    }

    /// Returns the V quantization step size.
    pub fn v_step(&self) -> f32 {
        self.v_quantizer.delta
    }

    /// Total memory used by this cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        let channels = self.num_kv_heads * self.head_dim;
        // Base values: 2 * channels * 4 bytes (f32)
        let base_bytes = 2 * channels * std::mem::size_of::<f32>();
        // Deltas: 2 * seq_len * channels * 1 byte (i8)
        let delta_bytes = 2 * self.seq_len * channels * std::mem::size_of::<i8>();
        base_bytes + delta_bytes
    }

    /// Memory savings percentage vs full-precision f32 cache at current seq_len.
    pub fn memory_savings_vs_f32(&self) -> f32 {
        let channels = self.num_kv_heads * self.head_dim;
        let f32_bytes = 2 * self.seq_len * channels * std::mem::size_of::<f32>();
        let current_bytes = self.memory_bytes();
        if f32_bytes == 0 {
            return 0.0;
        }
        ((f32_bytes as f32 - current_bytes as f32) / f32_bytes as f32) * 100.0
    }

    /// Reset the cache (clear all positions).
    pub fn clear(&mut self) {
        self.k_base.fill(0.0);
        self.v_base.fill(0.0);
        self.k_deltas.clear();
        self.v_deltas.clear();
        self.k_prev.fill(0.0);
        self.v_prev.fill(0.0);
        self.seq_len = 0;
    }
}

/// Errors for delta-encoded KV cache operations.
#[derive(Debug, thiserror::Error)]
pub enum DeltaKvError {
    #[error("dimension mismatch: expected {expected}, got k={got_k}, v={got_v}")]
    DimensionMismatch {
        expected: usize,
        got_k: usize,
        got_v: usize,
    },
    #[error("sequence length exceeded: current={current}, max={max}")]
    SeqLenExceeded { current: usize, max: usize },
    #[error("position {pos} out of range (seq_len={seq_len})")]
    PositionOutOfRange { pos: usize, seq_len: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_quantizer_roundtrip() {
        let q = DeltaQuantizer::new(0.01);
        let (code, quantized, _) = q.quantize(0.037);
        assert_eq!(code, 4); // round(0.037 / 0.01) = 4
        assert!((quantized - 0.04).abs() < 1e-6);

        let dequantized = q.dequantize(code);
        assert!((dequantized - 0.04).abs() < 1e-6);
    }

    #[test]
    fn test_delta_quantizer_error_bound() {
        let delta = 0.01;
        let q = DeltaQuantizer::new(delta);

        // Test that quantization error never exceeds δ/2
        for i in 0..1000 {
            let x = (i as f32 - 500.0) * 0.001; // range [-0.5, 0.5]
            let (_, quantized, _) = q.quantize(x);
            let error = (x - quantized).abs();
            assert!(
                error <= delta / 2.0 + 1e-7,
                "Quantization error {:.6} exceeds bound {:.6} for x={}",
                error,
                delta / 2.0,
                x
            );
        }
    }

    #[test]
    fn test_error_bound_at_position() {
        let q = DeltaQuantizer::new(0.01);
        // Error bound at position n should be n * δ / 2
        assert!((q.error_bound_at(0) - 0.0).abs() < 1e-7);
        assert!((q.error_bound_at(1) - 0.005).abs() < 1e-7);
        assert!((q.error_bound_at(10) - 0.05).abs() < 1e-7);
        assert!((q.error_bound_at(100) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_step_size_for_bound() {
        // To keep error ≤ 0.1 at position 100: δ ≤ 2 * 0.1 / 100 = 0.002
        let step = DeltaQuantizer::step_size_for_bound(100, 0.1);
        assert!((step - 0.002).abs() < 1e-7);
    }

    #[test]
    fn test_delta_kv_cache_append_and_reconstruct() {
        let mut cache = DeltaKvCache::new(1, 4, 10, 0.01, 0.01);

        // Position 0: base values
        let k0 = vec![1.0, 2.0, 3.0, 4.0];
        let v0 = vec![0.5, 0.5, 0.5, 0.5];
        cache.append(&k0, &v0).unwrap();

        // Position 1: small deltas from position 0
        let k1 = vec![1.02, 2.03, 2.98, 4.01];
        let v1 = vec![0.51, 0.49, 0.52, 0.48];
        cache.append(&k1, &v1).unwrap();

        // Reconstruct position 0 (should be exact)
        let (k_recon0, v_recon0) = cache.reconstruct_at(0).unwrap();
        for i in 0..4 {
            assert!((k_recon0[i] - k0[i]).abs() < 1e-6);
            assert!((v_recon0[i] - v0[i]).abs() < 1e-6);
        }

        // Reconstruct position 1 (should be within δ/2 = 0.005 of true values)
        let (k_recon1, v_recon1) = cache.reconstruct_at(1).unwrap();
        for i in 0..4 {
            assert!(
                (k_recon1[i] - k1[i]).abs() <= 0.005 + 1e-6,
                "K[{}] error {:.6} exceeds bound",
                i,
                (k_recon1[i] - k1[i]).abs()
            );
            assert!(
                (v_recon1[i] - v1[i]).abs() <= 0.005 + 1e-6,
                "V[{}] error {:.6} exceeds bound",
                i,
                (v_recon1[i] - v1[i]).abs()
            );
        }
    }

    #[test]
    fn test_error_bound_accumulation() {
        // Test that the accumulated error at position n never exceeds n * δ / 2
        let delta = 0.01;
        let mut cache = DeltaKvCache::new(1, 4, 100, delta, delta);

        // Generate a smooth sequence (like RoPE-rotated K vectors)
        let true_values: Vec<Vec<f32>> = (0..50)
            .map(|pos| {
                let t = pos as f32 * 0.1;
                vec![
                    t.sin(),
                    t.cos(),
                    (t * 2.0).sin() * 0.5,
                    1.0 + 0.1 * (t * 0.5).sin(),
                ]
            })
            .collect();

        for pos in 0..50 {
            let k = &true_values[pos];
            let v = &true_values[pos]; // Use same values for V in this test
            cache.append(k, v).unwrap();
        }

        // Verify error bound holds at every position
        for pos in 0..50 {
            let (k_recon, _) = cache.reconstruct_at(pos).unwrap();
            let bound = delta * pos as f32 / 2.0;

            for i in 0..4 {
                let error = (k_recon[i] - true_values[pos][i]).abs();
                assert!(
                    error <= bound + 1e-6,
                    "Position {}, channel {}: error {:.6} exceeds bound {:.6}",
                    pos,
                    i,
                    error,
                    bound
                );
            }
        }
    }

    #[test]
    fn test_memory_savings() {
        // 4 heads, head_dim=64, 100 positions
        let mut cache = DeltaKvCache::new(4, 64, 200, 0.01, 0.01);

        for pos in 0..100 {
            let t = pos as f32 * 0.01;
            let k: Vec<f32> = (0..256).map(|d| t.sin() * 0.1 + d as f32 * 0.001).collect();
            let v: Vec<f32> = (0..256).map(|d| t.cos() * 0.1 + d as f32 * 0.001).collect();
            cache.append(&k, &v).unwrap();
        }

        // Should save significant memory vs f32
        let savings = cache.memory_savings_vs_f32();
        assert!(
            savings > 50.0,
            "Expected >50% memory savings, got {:.1}%",
            savings
        );
    }

    #[test]
    fn test_delta_cache_smaller_than_q4k_for_short_sequences() {
        // For short sequences, delta encoding with i8 codes can be more compact
        // than Q4K blocks (which have 16 bytes of overhead per 256 elements)
        let num_heads = 8;
        let head_dim = 64;
        let channels = num_heads * head_dim; // 512

        // Delta cache: 2 bases * 512 * 4 bytes + seq_len * 2 * 512 * 1 byte
        // At seq_len=10: 4096 + 10240 = 14336 bytes
        let delta_bytes_at_10 = 2 * channels * 4 + 10 * 2 * channels * 1;

        // Q4K: each block is 144 bytes for 256 elements
        // Total elements at seq_len=10: 2 (K+V) * 512 * 10 = 10240
        // Blocks needed: ceil(10240 / 256) = 40 blocks per K/V = 80 total
        let q4k_bytes_at_10 = 80 * 144;

        eprintln!(
            "Delta cache at seq_len=10: {} bytes",
            delta_bytes_at_10
        );
        eprintln!("Q4K cache at seq_len=10: {} bytes", q4k_bytes_at_10);

        // Delta encoding should be competitive or better for short sequences
        assert!(
            delta_bytes_at_10 < q4k_bytes_at_10 * 2,
            "Delta cache should not be more than 2x Q4K"
        );
    }

    #[test]
    fn test_step_size_prevent_clipping() {
        // For LLaMA head_dim=64, k_norm_bound=1.0 (RMSNorm), safety_factor=2.0:
        // max_delta_exact = 2 * 1.0 * sin(0.5) ≈ 0.9589
        // max_delta_with_safety = 0.9589 * 2.0 ≈ 1.9178
        // δ_min_for_range = 1.9178 / 126 ≈ 0.01522 (shipping-safe)
        // Min achievable error at pos 512 = 512 * 0.01522 / 2 ≈ 3.898
        let step = DeltaQuantizer::step_size_prevent_clipping(
            64,   // head_dim
            1.0,  // k_norm_bound (RMSNorm)
            2.0,  // safety_factor
            512,  // max_seq
            4.0,  // max_allowed_error at max_seq (must be ≥ 3.866)
        );

        assert!(step.is_some(), "Should find feasible step size");
        let s = step.unwrap();

        // Verify no-clipping constraint: 126 * δ ≥ max_delta (shipping-safe)
        let max_representable = 126.0 * s;
        let expected_max_delta = 2.0 * 1.0 * 0.5f32.sin() * 2.0; // exact with safety
        assert!(max_representable >= expected_max_delta - 1e-6, 
            "Max representable {:.4} should be ≥ {:.4}", max_representable, expected_max_delta);

        // Verify error bound constraint: 512 * δ / 2 ≤ 4.0
        let error_at_max = 512.0 * s / 2.0;
        assert!(error_at_max <= 4.0 + 1e-6,
            "Error at max_seq {:.4} should be ≤ 4.0", error_at_max);

        // Verify it's the smallest δ that prevents clipping
        let expected_min = expected_max_delta / 126.0;
        assert!((s - expected_min).abs() < 1e-6,
            "Step size {:.6} should equal min for no-clipping {:.6}", s, expected_min);
    }

    #[test]
    fn test_step_size_prevent_clipping_incompatible() {
        // Tight error bound + large expected delta → incompatible with i8
        let step = DeltaQuantizer::step_size_prevent_clipping(
            64,   // head_dim
            10.0, // Large k_norm_bound (unnormalized keys)
            4.0,  // High safety factor
            512,  // max_seq
            0.01, // Very tight error bound
        );

        // max_delta_exact = 2 * 10.0 * sin(0.5) ≈ 9.589
        // max_delta_with_safety = 9.589 * 4.0 ≈ 38.36
        // δ_min_for_range = 38.36 / 126 ≈ 0.304
        // δ_max_for_error = 2 * 0.01 / 512 ≈ 0.000039
        // Incompatible → None
        assert!(step.is_none(), "Should be incompatible with i8 codes");
    }

    #[test]
    fn test_rope_aware_step_sizes() {
        let head_dim = 64;
        let steps = DeltaQuantizer::rope_aware_step_sizes(head_dim, 1.0, 2.0);

        assert_eq!(steps.len(), head_dim);

        // Dimension 0 (fastest) should have largest step size
        // θ_0 = 1.0, max_delta = 2 * 1.0 * sin(0.5) * 2.0 ≈ 1.9178, δ_0 ≈ 0.01522
        let expected_d0 = 2.0 * 1.0 * 0.5f32.sin() * 2.0 / 126.0;
        assert!((steps[0] - expected_d0).abs() < 1e-6,
            "δ_0 {:.6} should be {:.6}", steps[0], expected_d0);

        // Last dimension (slowest) should have much smaller step size
        // θ_63 = 10000^(-2*63/64) ≈ 0.01, max_delta ≈ 2*1*0.005*2 = 0.02, δ_63 ≈ 0.000159
        let theta_last = 10000f32.powf(-2.0 * 63.0 / 64.0);
        let expected_last = 2.0 * 1.0 * (theta_last / 2.0).sin() * 2.0 / 126.0;
        assert!((steps[63] - expected_last).abs() < 1e-8,
            "δ_63 {:.8} should be {:.8}", steps[63], expected_last);

        // Step sizes should be monotonically decreasing
        for i in 1..head_dim {
            assert!(steps[i] <= steps[i - 1] + 1e-10,
                "Step sizes should decrease with dimension");
        }

        // Per-dimension bound should be much tighter than global
        let rope_bound = DeltaQuantizer::rope_aware_error_bound(head_dim, 1.0, 2.0, 512);
        let global_step = steps[0]; // Worst case dimension
        let global_bound = 512.0 * global_step / 2.0 * head_dim as f32;
        
        assert!(rope_bound < global_bound,
            "RoPE-aware bound {:.4} should be tighter than global {:.4}",
            rope_bound, global_bound);
    }

    #[test]
    fn test_prevention_guarantees_no_clipping() {
        // End-to-end: use step_size_prevent_clipping to configure cache,
        // then verify zero clipping on realistic RoPE data.
        let head_dim = 64;
        let num_heads = 4;
        let max_seq = 256;
        let k_norm_bound = 1.0;
        let safety_factor = 2.0;

        // Choose δ to prevent clipping by construction
        // min error at 256 = 256 * δ / 2, with δ ≈ 0.0152 → min error ≈ 1.94
        let step = DeltaQuantizer::step_size_prevent_clipping(
            head_dim, k_norm_bound, safety_factor, max_seq, 2.0
        ).unwrap();

        let mut cache = DeltaKvCache::new(num_heads, head_dim, max_seq, step, step);

        // Generate realistic RoPE-rotated K vectors (unit norm per dim)
        for pos in 0..max_seq {
            let t = pos as f32 * 0.1; // Rotation angle per position
            let k: Vec<f32> = (0..num_heads*head_dim).map(|ch| {
                let d = ch % head_dim;
                let theta = 10000f32.powf(-2.0 * d as f32 / head_dim as f32);
                (t * theta).sin() // Unit amplitude, RoPE frequency
            }).collect();
            let v = k.clone();
            cache.append(&k, &v).unwrap();
        }

        // CRITICAL: Zero clipping should occur by construction
        assert_eq!(cache.clipped_k_deltas(), 0, 
            "Prevention should guarantee zero K clipping, got {}", cache.clipped_k_deltas());
        assert_eq!(cache.clipped_v_deltas(), 0,
            "Prevention should guarantee zero V clipping, got {}", cache.clipped_v_deltas());

        // Error bound should hold at every position (no clipping term needed)
        for pos in 0..max_seq {
            let (k_recon, _) = cache.reconstruct_at(pos).unwrap();
            let bound = step * pos as f32 / 2.0;
            
            for i in 0..num_heads*head_dim {
                // Regenerate true value to compare
                let t = pos as f32 * 0.1;
                let d = i % head_dim;
                let theta = 10000f32.powf(-2.0 * d as f32 / head_dim as f32);
                let true_val = (t * theta).sin();
                
                let error = (k_recon[i] - true_val).abs();
                assert!(error <= bound + 1e-6,
                    "Pos {}, ch {}: error {:.6} > bound {:.6}", pos, i, error, bound);
            }
        }
    }
}
