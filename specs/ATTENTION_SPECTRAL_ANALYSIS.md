# PESTI Attention Spectral Analysis: RMT-Based Degeneracy Detection

## §0 Goal + Scope

**Goal**: Add lightweight spectral analysis of attention matrices during inference to detect degenerate attention patterns (heads that collapse into low-rank behavior) using Random Matrix Theory (RMT) baselines. This turns "is the model paying attention?" from a qualitative question into a measurable diagnostic.

**Core principle**: *Attention matrices in healthy transformer layers have eigenvalue distributions close to those predicted by RMT for random Gaussian matrices (GUE/GOE ensembles). Significant deviation from the Marchenko-Pastur or Wigner semicircle law indicates structural problems — dead heads, attention collapse, or pathological token interactions.*

**NOT in scope (yet)**:
- ❌ Real-time per-token spectral analysis during generation (too expensive for decode step)
- ❌ Full eigendecomposition of large attention matrices (use randomized SVD instead)
- ❌ Training-time interventions based on spectral analysis (inference diagnostics only)
- ❌ Replacing existing attention mechanisms (this is observability, not a new kernel)

**Scope**: Three phases in `pesti-runner`:
1. **Phase 1**: Offline spectral profiler — analyze attention matrices from a completed inference run, generate RMT comparison report
2. **Phase 2**: Online head health monitoring — track per-head spectral statistics during generation with O(n) randomized methods
3. **Phase 3**: Degeneracy alerts and adaptive behavior — trigger warnings or fallback strategies when heads collapse

## §1 Context & Why Now

**Principle**: We just verified that Riemann zeta zeros follow GUE statistics (commit `c41ec9a`). The same mathematical framework — Random Matrix Theory — governs eigenvalue distributions of large random matrices, including attention matrices in transformers. This gives us a principled baseline for "what does healthy attention look like?" without needing to hand-tune thresholds.

**Why now**:
- Pesti is the primary LLM inference engine for the fleet. As models get larger and contexts longer, attention degradation becomes a real operational concern (long-context drift, repeated token generation, loss of instruction following).
- Current debugging of "the model stopped making sense" requires manual inspection of logits or attention visualizations — slow and subjective.
- RMT gives us objective, mathematically-grounded thresholds: if the eigenvalue distribution of an attention matrix deviates from the Marchenko-Pastur law by more than X standard deviations (where X is calibrated against random matrices), that head is structurally different from what you'd expect from a well-functioning transformer.
- The GUE verification work we just did (`examples/rh_prime_counting.rs`, `c41ec9a`) established the computational infrastructure for RMT analysis in Rust. This spec applies that to a practical problem.

**The mathematical connection**:
- Attention matrix A ∈ R^{n×n} (where n = sequence length) has eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ
- For a random Gaussian matrix with entries ~ N(0, 1/n), the empirical spectral distribution converges to the **Marchenko-Pastur law** (for A·Aᵀ) or **Wigner semicircle law** (for symmetric A)
- Healthy transformer attention should be "close to random" in the sense that it's not dominated by a few eigenvalues (which would indicate collapse onto a low-dimensional subspace)
- Degenerate heads show: one dominant eigenvalue (λ₁ >> λ₂), or a gap in the spectrum, or negative eigenvalues (for non-symmetric attention)

**Failure modes this detects**:
1. **Attention collapse**: Head attends only to 1-2 tokens → rank-1 matrix → single dominant eigenvalue
2. **Dead head**: Attention weights all ~0 → zero eigenvalues → spectral radius near 0
3. **Oscillatory attention**: Alternating positive/negative patterns → complex eigenvalues for non-symmetric A
4. **Long-context drift**: As context grows, healthy heads should maintain RMT-like spectra; deviation signals the model is losing track

**RMT baselines we'll use**:

| Ensemble | Matrix type | Eigenvalue distribution | Application |
|----------|-------------|------------------------|-------------|
| GUE (Gaussian Unitary) | Complex Hermitian | Wigner semicircle + level repulsion | Baseline for "random" complex attention |
| GOE (Gaussian Orthogonal) | Real symmetric | Wigner semicircle | Baseline for symmetrized attention (A+Aᵀ)/2 |
| Marchenko-Pastur | A·Aᵀ for A ∈ R^{n×m} | Support [α⁻¹(1-√β)², α⁻¹(1+√β)²] where β=n/m | Baseline for attention covariance structure |

## §2 Architecture

### Phase 1: Offline Spectral Profiler

**Problem**: After a long inference run (e.g., document summarization), we want to know which attention heads were healthy and which degenerated. Currently this requires dumping full attention matrices (memory-intensive) and analyzing them externally.

**Solution**: A `SpectralProfiler` that hooks into the attention dispatch and computes spectral statistics on-the-fly without storing full matrices.

```rust
// crates/pesti-runner/src/diagnostics/spectral.rs (new file)

/// Spectral statistics for a single attention matrix.
#[derive(Debug, Clone, Serialize)]
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

/// Computes spectral statistics using randomized SVD (Halko et al. 2011).
/// O(n·k·d) instead of O(n³) for full eigendecomposition.
pub struct SpectralProfiler {
    /// Number of eigenvalues to compute
    k: usize,
    /// Power iterations for accuracy
    power_iter: usize,
}

impl SpectralProfiler {
    pub fn new(k: usize) -> Self { ... }
    
    /// Compute top-k singular values of attention matrix A (n×n).
    /// Uses randomized SVD: sketch with Gaussian random matrix Ω (n×(k+p)),
    /// then QR decompose and solve small eigenvalue problem.
    pub fn analyze_attention(&self, a: &Tensor2D) -> AttentionSpectrum { ... }
    
    /// Compare empirical spectrum to Marchenko-Pastur prediction.
    /// Returns KL divergence (0 = perfect match).
    pub fn marchenko_pastur_distance(
        &self, 
        eigenvalues: &[f64], 
        n: usize, 
        m: usize  // for attention, m=n so β=1
    ) -> f64 { ... }
    
    /// Compute degeneracy score from spectral statistics.
    /// Combines: effective rank ratio, top eigenvalue dominance, MP distance.
    pub fn degeneracy_score(&self, spectrum: &AttentionSpectrum) -> f64 { ... }
}
```

**Integration point**: Hook into `AttentionDispatch::forward` after computing attention weights but before applying to values. Only active when `PESTI_SPECTRAL_PROFILE=1` env var is set (zero overhead in production).

**Output**: JSON report with per-layer, per-head spectral statistics:
```json
{
  "run_id": "...",
  "model": "qwen2.5-3b",
  "seq_len": 2048,
  "heads": [
    {
      "layer": 5,
      "head": 3,
      "spectral_norm": 1.23,
      "effective_rank": 45.2,
      "numerical_rank": 128,
      "mp_distance": 0.034,
      "degeneracy_score": 0.12
    },
    ...
  ],
  "summary": {
    "total_heads": 96,
    "healthy_heads": 89,       // degeneracy < 0.3
    "suspect_heads": 5,        // 0.3 ≤ degeneracy < 0.7
    "degenerate_heads": 2      // degeneracy ≥ 0.7
  }
}
```

### Phase 2: Online Head Health Monitoring

**Problem**: Offline profiling is useful for debugging but doesn't help during long generations where degradation happens gradually. We want to detect "the model is starting to repeat itself" before it becomes obvious in the output.

**Solution**: Track rolling spectral statistics per head during generation using O(n) streaming methods.

```rust
// crates/pesti-runner/src/diagnostics/head_health.rs (new file)

/// Tracks spectral health of a single attention head over time.
pub struct HeadHealthTracker {
    /// Exponential moving average of effective rank
    ema_effective_rank: f64,
    /// Standard deviation of recent effective ranks
    rank_stddev: f64,
    /// Count of consecutive steps where degeneracy > threshold
    consecutive_degenerate: usize,
}

impl HeadHealthTracker {
    pub fn update(&mut self, spectrum: &AttentionSpectrum) { ... }
    
    /// Returns true if this head shows signs of collapse.
    pub fn is_collapsing(&self) -> bool {
        self.consecutive_degenerate >= 5 
            && (self.ema_effective_rank / self.rank_stddev) < 2.0
    }
}

/// Fleet-level monitor: tracks all heads, raises alerts.
pub struct AttentionHealthMonitor {
    trackers: HashMap<(usize, usize), HeadHealthTracker>,  // (layer, head) -> tracker
    alert_threshold: f64,  // fraction of degenerate heads that triggers warning
}
```

**Performance**: Randomized SVD with k=5 and 1 power iteration costs ~O(5n) per head per step. For n=2048 and 32 heads, that's ~327K operations — negligible compared to the GEMM in attention (~O(n²·d)).

**Trigger conditions**:
- Warning: >10% of heads have degeneracy_score > 0.5 for 3+ consecutive steps
- Critical: >30% of heads degenerate, or any head has effective_rank < 2 for 5+ steps

### Phase 3: Adaptive Response

**Problem**: Detecting degeneration is useful, but can we do something about it?

**Solution**: 
1. **Log and continue** (default): Record degeneracy events in the inference log for post-hoc analysis
2. **Temperature boost**: If attention is collapsing (model too confident), slightly increase sampling temperature to encourage exploration
3. **Context truncation hint**: If long-context drift detected (degeneration correlates with position), suggest truncating old context

```rust
pub enum DegeneracyResponse {
    LogOnly,
    AdjustTemperature { delta: f64 },
    SuggestTruncate { keep_last_n: usize },
}
```

## §3 Implementation Notes

**Randomized SVD (Halko et al. 2011)**:
- Given A ∈ R^{n×n}, form sketch Y = A·Ω where Ω ∈ R^{n×(k+p)} is Gaussian random
- QR decompose Y = Q·R where Q ∈ R^{n×(k+p)} has orthonormal columns
- Form B = Qᵀ·A·Q (small (k+p)×(k+p) matrix)
- Compute full SVD of B (cheap since k+p << n)
- Top-k singular values of A ≈ top-k singular values of B

**Marchenko-Pastur law** for β=1 (square matrices):
- Density: ρ(x) = (1/(2πx)) · √((4-x)·x) for x ∈ [0, 4]
- For attention A with entries scaled by 1/√n, A·Aᵀ eigenvalues should follow this
- KL divergence between empirical histogram and MP density gives "randomness score"

**Degeneracy score formula**:
```
degeneracy = 0.4 · (1 - effective_rank / seq_len)    // rank collapse
           + 0.3 · min(λ₁/Σλᵢ, 1.0)                  // top eigenvalue dominance  
           + 0.3 · min(mp_distance / 1.0, 1.0)       // deviation from RMT
```

**Env vars**:
- `PESTI_SPECTRAL_PROFILE=1` — enable offline profiling (Phase 1)
- `PESTI_HEAD_HEALTH=1` — enable online monitoring (Phase 2)
- `PESTI_DEGENERACY_RESPONSE=log|temp|truncate` — Phase 3 behavior

## §4 Validation Plan

1. **Baseline calibration**: Run spectral profiler on known-good generations (short prompts, clear answers). Verify that healthy heads have degeneracy_score < 0.3 and MP distance < 0.1.
2. **Degeneracy injection**: Manually create degenerate attention patterns (e.g., force a head to attend only to token 1) and verify the detector catches it with score > 0.7.
3. **Long-context test**: Generate 8K+ token outputs and track when/if heads start degrading. Compare against manual inspection of output quality.
4. **Performance overhead**: Measure that Phase 2 adds <2% latency to decode step (target: <1ms per 32-head layer).

## §5 Connection to RH Work

This spec applies the same RMT machinery we used to verify GUE statistics for zeta zeros (`c41ec9a`) to a practical ML problem. The mathematical connection:
- Zeta zeros → eigenvalues of random unitary matrices (GUE)
- Attention matrices → approximately random when healthy
- Deviation from RMT prediction → structural anomaly

The RH proof itself doesn't directly help here, but the *verification work* we did (computing GUE statistics in Rust, comparing to theoretical predictions) is exactly the pattern this spec implements for attention matrices.

## §6 Implementation Status

### Phase 1: Offline Spectral Profiling ✅
- `SpectralProfiler` with randomized SVD eigenvalue estimation
- Marchenko-Pastur bounds and KL divergence scoring
- Degeneracy score formula (rank collapse + eigenvalue dominance + RMT deviation)
- Severity classification (healthy/suspect/degenerate)
- CLI: `pesti --diagnose` for end-to-end analysis

### Phase 2: Online Head Health Monitoring ✅
- `HeadHealthTracker` with EMA-based effective rank tracking
- Decode-time attention capture via profiler hooks in transformer layers
- Per-head health scores updated every decode step
- Configurable sampling rate to limit overhead

### Spectral Drift Detection ✅
- `SpectralDriftDetector` tracks baseline ranks during initial steps
- Periodic comparison of current vs baseline effective ranks
- Detects repetition loops (rank → 1), topic shifts (rank ↑), degradation (rank unstable)
- Integrated with collector via `update_drift()` method

### MCP Server Integration ✅
- `spectral_analyze` tool accepts square attention matrices remotely
- Returns full RMT analysis as JSON (effective rank, PR, MP distance, severity)
- Enables agent-driven diagnostics without local model loading

### Remaining Work
- **Quantization validation**: Compare spectral properties pre/post Q4 quantization to detect when compression degrades attention quality. Would use the existing `--diagnose` infrastructure with paired model runs.
- **CLI drift monitoring**: Add `--monitor-drift N` flag to generate N tokens while tracking spectral drift (builds on existing drift detector).
- **Performance overhead measurement**: Validate Phase 2 adds <2% decode latency per validation plan.
