//! Kernel selection: memoized decision-making for which kernel to execute.
//!
//! This module implements the "query system" pattern from rustc's architecture:
//! instead of re-evaluating hardware/dtype/shape constraints on every dispatch
//! call, we compute the optimal kernel once per unique constraint combination
//! and cache the result in a HashMap.
//!
//! ## Design Principles
//!
//! - **Selection vs execution**: This module decides *which* kernel to use.
//!   The actual kernel execution code stays where it is (in dispatch.rs).
//! - **Data-driven**: Adding a new kernel = adding one entry to the registry,
//!   not modifying N dispatch methods.
//! - **Guaranteed fallback**: Every operation has a CPU baseline kernel that
//!   always works. Selection never fails — worst case returns the baseline.
//!
//! ## Usage
//!
//! ```ignore
//! let registry = KernelRegistry::detect(); // one-time at init
//! let selector = KernelSelector::new(registry);
//!
//! // Per dispatch call:
//! let req = KernelRequest {
//!     op: OpKind::GEMM,
//!     arch: ComputeArch::Cuda { sm: 86 },
//!     dtype: DType::F16,
//!     m: 4096, n: 4096, k: 4096,
//! };
//! let kernel = selector.select(&req); // HashMap lookup after first call
//! ```

use std::collections::HashMap;
use std::fmt;

/// The class of operation being dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Matrix-matrix multiplication: C = alpha * A @ B + beta * C
    GEMM,
    /// Linear layer: y = x @ W^T + bias (specialized GEMM with transpose)
    LinearWithBias,
    /// Scaled dot-product attention: softmax(QK^T / sqrt(d)) @ V
    Attention,
    /// RMS normalization
    RMSNorm,
    /// SwiGLU activation: swiglu(gate, up) = gate * sigmoid(gate) * up
    SwiGLU,
    /// Rotary position embedding
    RoPE,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::GEMM => write!(f, "GEMM"),
            OpKind::LinearWithBias => write!(f, "Linear"),
            OpKind::Attention => write!(f, "Attention"),
            OpKind::RMSNorm => write!(f, "RMSNorm"),
            OpKind::SwiGLU => write!(f, "SwiGLU"),
            OpKind::RoPE => write!(f, "RoPE"),
        }
    }
}

/// Target compute architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeArch {
    /// CPU (x86_64 or aarch64 — no distinction needed for kernel selection)
    Cpu,
    /// NVIDIA CUDA with given compute capability (e.g., 75 = Turing, 86 = Ampere, 90 = Hopper)
    Cuda { sm: u16 },
}

impl fmt::Display for ComputeArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeArch::Cpu => write!(f, "CPU"),
            ComputeArch::Cuda { sm } => write!(f, "CUDA sm_{sm}"),
        }
    }
}

impl ComputeArch {
    /// Check if this architecture satisfies a minimum CUDA requirement.
    pub fn is_cuda_at_least(&self, min_sm: u16) -> bool {
        match self {
            ComputeArch::Cuda { sm } => *sm >= min_sm,
            ComputeArch::Cpu => false,
        }
    }
}

/// Data type of the computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelDType {
    F32,
    F16,
    BF16,
    Q4_K,
    Q8_0,
}

impl fmt::Display for KernelDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelDType::F32 => write!(f, "f32"),
            KernelDType::F16 => write!(f, "f16"),
            KernelDType::BF16 => write!(f, "bf16"),
            KernelDType::Q4_K => write!(f, "q4_k"),
            KernelDType::Q8_0 => write!(f, "q8_0"),
        }
    }
}

/// A specific kernel implementation that can execute an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelId {
    // ── GEMM kernels ──
    /// CPU f16→f32 GEMM (always available, baseline)
    CpuGemmF16To32,
    /// CUDA cuBLAS HGEMM via candle_bridge (F16 inputs, F32 accumulate)
    CudaCublasHgemm,
    /// Direct CUDA cuBLAS bridge (true F16 inference path)
    CudaDirectHgemm,

    // ── Attention kernels ──
    /// CPU manual attention (always available, baseline)
    CpuAttentionManual,
    /// CUDA fused attention (cache_len >= 2, prefill + decode with cache)
    CudaAttentionFused,
    /// CUDA fused decode attention (batch=1, seq=1, no cache — first token)
    CudaAttentionDecodeFused,

    // ── Non-matmul kernels (Phase 1: selection only, execution stays in dispatch) ──
    CpuRmsNorm,
    CudaRmsNorm,
    CpuSwiGLU,
    CudaSwiGLU,
    CpuRoPE,
    CudaRoPE,
}

impl fmt::Display for KernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelId::CpuGemmF16To32 => write!(f, "cpu_gemm_f16"),
            KernelId::CudaCublasHgemm => write!(f, "cuda_cublas_hgemm"),
            KernelId::CudaDirectHgemm => write!(f, "cuda_direct_hgemm"),
            KernelId::CpuAttentionManual => write!(f, "cpu_attention_manual"),
            KernelId::CudaAttentionFused => write!(f, "cuda_attention_fused"),
            KernelId::CudaAttentionDecodeFused => write!(f, "cuda_attention_decode_fused"),
            KernelId::CpuRmsNorm => write!(f, "cpu_rms_norm"),
            KernelId::CudaRmsNorm => write!(f, "cuda_rms_norm"),
            KernelId::CpuSwiGLU => write!(f, "cpu_swiglu"),
            KernelId::CudaSwiGLU => write!(f, "cuda_swiglu"),
            KernelId::CpuRoPE => write!(f, "cpu_rope"),
            KernelId::CudaRoPE => write!(f, "cuda_rope"),
        }
    }
}

/// The constraints that determine which kernel to use for an operation.
///
/// This is the "query key" — two requests with identical fields will always
/// get the same kernel selection (and subsequent calls are cache hits).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelRequest {
    /// Operation type
    pub op: OpKind,
    /// Target architecture
    pub arch: ComputeArch,
    /// Data type of the computation
    pub dtype: KernelDType,
    /// Matrix dimensions (for GEMM/Linear) or sequence length (for attention).
    /// Stored as a tuple to keep the Hash simple; use 0 for unused dimensions.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// For attention: current KV cache length (0 = no cache yet)
    pub cache_len: usize,
}

impl KernelRequest {
    /// Create a GEMM request.
    pub fn gemm(arch: ComputeArch, dtype: KernelDType, m: usize, n: usize, k: usize) -> Self {
        Self { op: OpKind::GEMM, arch, dtype, m, n, k, cache_len: 0 }
    }

    /// Create a linear layer request.
    pub fn linear(arch: ComputeArch, dtype: KernelDType, batch: usize, in_features: usize, out_features: usize) -> Self {
        Self {
            op: OpKind::LinearWithBias,
            arch,
            dtype,
            m: batch,
            n: out_features,
            k: in_features,
            cache_len: 0,
        }
    }

    /// Create an attention request.
    pub fn attention(arch: ComputeArch, dtype: KernelDType, seq_len: usize, cache_len: usize) -> Self {
        Self { op: OpKind::Attention, arch, dtype, m: seq_len, n: 0, k: 0, cache_len }
    }
}

/// Performance class for ranking kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerfClass {
    /// CPU fallback — always works, slowest. Guaranteed to exist for every OpKind.
    Baseline,
    /// GPU kernel with good performance (e.g., cuBLAS via candle_bridge)
    Fast,
    /// Best known kernel for this operation class (e.g., direct cuBLAS, fused attention)
    Optimal,
}

impl PerfClass {
    pub fn rank(&self) -> u8 {
        match self {
            PerfClass::Baseline => 0,
            PerfClass::Fast => 1,
            PerfClass::Optimal => 2,
        }
    }
}

/// Architecture constraint for a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchConstraint {
    /// Works on any architecture (CPU or GPU)
    Any,
    /// CPU only
    CpuOnly,
    /// CUDA with minimum compute capability
    CudaMin(u16),
}

impl ArchConstraint {
    pub fn satisfies(&self, arch: &ComputeArch) -> bool {
        match self {
            ArchConstraint::Any => true,
            ArchConstraint::CpuOnly => matches!(arch, ComputeArch::Cpu),
            ArchConstraint::CudaMin(min_sm) => arch.is_cuda_at_least(*min_sm),
        }
    }
}

/// Data type constraint for a kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeConstraint {
    /// Works with any data type
    Any,
    /// Specific data type required
    Exact(KernelDType),
}

impl DTypeConstraint {
    pub fn satisfies(&self, dtype: &KernelDType) -> bool {
        match self {
            DTypeConstraint::Any => true,
            DTypeConstraint::Exact(d) => d == dtype,
        }
    }
}

/// Shape constraint for a kernel (e.g., cache length requirements).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeConstraint {
    /// No shape requirements — works with any dimensions
    Any,
    /// Requires KV cache length to be exactly 0 (first token / no cache)
    NoCache,
    /// Requires KV cache length >= N (has existing cache)
    MinCacheLen(usize),
}

impl ShapeConstraint {
    pub fn satisfies(&self, req: &KernelRequest) -> bool {
        match self {
            ShapeConstraint::Any => true,
            ShapeConstraint::NoCache => req.cache_len == 0,
            ShapeConstraint::MinCacheLen(n) => req.cache_len >= *n,
        }
    }
}

/// A specification for a kernel in the registry.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// The kernel identifier
    pub id: KernelId,
    /// Operation this kernel implements
    pub op: OpKind,
    /// Architecture constraint
    pub arch: ArchConstraint,
    /// Data type constraint
    pub dtype: DTypeConstraint,
    /// Shape constraint (e.g., cache length requirements for attention)
    pub shape: ShapeConstraint,
    /// Performance class (used for ranking when multiple kernels satisfy constraints)
    pub perf: PerfClass,
}

impl KernelSpec {
    /// Check if this kernel can handle the given request.
    pub fn satisfies(&self, req: &KernelRequest) -> bool {
        self.op == req.op
            && self.arch.satisfies(&req.arch)
            && self.dtype.satisfies(&req.dtype)
            && self.shape.satisfies(req)
    }
}

/// Static registry of available kernels and their constraints.
///
/// Built once at initialization based on what's compiled in and what hardware
/// is present. This is where "adding a kernel = adding an entry" happens.
pub struct KernelRegistry {
    kernels: Vec<KernelSpec>,
}

impl KernelRegistry {
    /// Build the registry from runtime detection.
    ///
    /// CPU kernels are always included (guaranteed fallback).
    /// CUDA kernels are included only when the `cuda` feature is enabled
    /// and a GPU with sufficient compute capability is detected.
    pub fn detect(cuda_sm: Option<u16>) -> Self {
        let mut kernels = vec![
            // ── CPU baseline kernels (always available) ──
            KernelSpec {
                id: KernelId::CpuGemmF16To32,
                op: OpKind::GEMM,
                arch: ArchConstraint::Any,
                dtype: DTypeConstraint::Exact(KernelDType::F16),
                shape: ShapeConstraint::Any,
                perf: PerfClass::Baseline,
            },
            KernelSpec {
                id: KernelId::CpuAttentionManual,
                op: OpKind::Attention,
                arch: ArchConstraint::Any,
                dtype: DTypeConstraint::Any,
                shape: ShapeConstraint::Any,
                perf: PerfClass::Baseline,
            },
            KernelSpec {
                id: KernelId::CpuRmsNorm,
                op: OpKind::RMSNorm,
                arch: ArchConstraint::Any,
                dtype: DTypeConstraint::Any,
                shape: ShapeConstraint::Any,
                perf: PerfClass::Baseline,
            },
            KernelSpec {
                id: KernelId::CpuSwiGLU,
                op: OpKind::SwiGLU,
                arch: ArchConstraint::Any,
                dtype: DTypeConstraint::Any,
                shape: ShapeConstraint::Any,
                perf: PerfClass::Baseline,
            },
            KernelSpec {
                id: KernelId::CpuRoPE,
                op: OpKind::RoPE,
                arch: ArchConstraint::Any,
                dtype: DTypeConstraint::Any,
                shape: ShapeConstraint::Any,
                perf: PerfClass::Baseline,
            },
        ];

        // ── CUDA kernels (only when GPU detected) ──
        if let Some(sm) = cuda_sm {
            // cuBLAS via candle_bridge — requires sm_75+ (Volta+)
            if sm >= 75 {
                kernels.push(KernelSpec {
                    id: KernelId::CudaCublasHgemm,
                    op: OpKind::GEMM,
                    arch: ArchConstraint::CudaMin(75),
                    dtype: DTypeConstraint::Exact(KernelDType::F16),
                    shape: ShapeConstraint::Any,
                    perf: PerfClass::Fast,
                });
            }

            // Direct cuBLAS bridge — requires sm_80+ (Ampere+) for best performance
            if sm >= 80 {
                kernels.push(KernelSpec {
                    id: KernelId::CudaDirectHgemm,
                    op: OpKind::GEMM,
                    arch: ArchConstraint::CudaMin(80),
                    dtype: DTypeConstraint::Exact(KernelDType::F16),
                    shape: ShapeConstraint::Any,
                    perf: PerfClass::Optimal,
                });
            }

            // Fused attention kernels — require sm_75+
            if sm >= 75 {
                // General fused attention: used when KV cache exists (cache_len >= 1)
                kernels.push(KernelSpec {
                    id: KernelId::CudaAttentionFused,
                    op: OpKind::Attention,
                    arch: ArchConstraint::CudaMin(75),
                    dtype: DTypeConstraint::Exact(KernelDType::F16),
                    shape: ShapeConstraint::MinCacheLen(1),
                    perf: PerfClass::Optimal,
                });
                // Specialized decode kernel: used for first token (no cache)
                kernels.push(KernelSpec {
                    id: KernelId::CudaAttentionDecodeFused,
                    op: OpKind::Attention,
                    arch: ArchConstraint::CudaMin(75),
                    dtype: DTypeConstraint::Exact(KernelDType::F16),
                    shape: ShapeConstraint::NoCache,
                    perf: PerfClass::Optimal,
                });
            }

            // Non-matmul CUDA kernels
            if sm >= 70 {
                kernels.push(KernelSpec {
                    id: KernelId::CudaRmsNorm,
                    op: OpKind::RMSNorm,
                    arch: ArchConstraint::CudaMin(70),
                    dtype: DTypeConstraint::Any,
                    shape: ShapeConstraint::Any,
                    perf: PerfClass::Fast,
                });
                kernels.push(KernelSpec {
                    id: KernelId::CudaSwiGLU,
                    op: OpKind::SwiGLU,
                    arch: ArchConstraint::CudaMin(70),
                    dtype: DTypeConstraint::Any,
                    shape: ShapeConstraint::Any,
                    perf: PerfClass::Fast,
                });
                kernels.push(KernelSpec {
                    id: KernelId::CudaRoPE,
                    op: OpKind::RoPE,
                    arch: ArchConstraint::CudaMin(70),
                    dtype: DTypeConstraint::Any,
                    shape: ShapeConstraint::Any,
                    perf: PerfClass::Fast,
                });
            }
        }

        Self { kernels }
    }

    /// Find the best kernel satisfying all constraints in the request.
    ///
    /// Returns the highest-performing kernel that matches. If no GPU kernel
    /// matches, falls back to the CPU baseline (which always exists).
    pub fn best_for(&self, req: &KernelRequest) -> KernelId {
        self.kernels.iter()
            .filter(|k| k.satisfies(req))
            .max_by_key(|k| k.perf.rank())
            .map(|k| k.id)
            .unwrap_or_else(|| self.fallback_for(req.op))
    }

    /// Get the guaranteed CPU fallback for an operation.
    fn fallback_for(&self, op: OpKind) -> KernelId {
        match op {
            OpKind::GEMM | OpKind::LinearWithBias => KernelId::CpuGemmF16To32,
            OpKind::Attention => KernelId::CpuAttentionManual,
            OpKind::RMSNorm => KernelId::CpuRmsNorm,
            OpKind::SwiGLU => KernelId::CpuSwiGLU,
            OpKind::RoPE => KernelId::CpuRoPE,
        }
    }

    /// Number of kernels in the registry.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the registry is empty (should never happen — CPU baselines always exist).
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Selects the optimal kernel for a given execution request.
///
/// Memoizes selections so repeated requests with identical constraints
/// become HashMap lookups instead of constraint evaluation. During a single
/// inference run, the same (arch, dtype, shape) combinations appear thousands
/// of times — after the first call, selection is O(1).
pub struct KernelSelector {
    cache: HashMap<KernelRequest, KernelId>,
    registry: KernelRegistry,
}

impl KernelSelector {
    /// Create a new selector with the given kernel registry.
    pub fn new(registry: KernelRegistry) -> Self {
        Self { cache: HashMap::new(), registry }
    }

    /// Select the best kernel for the request.
    ///
    /// First call evaluates constraints against the registry; subsequent calls
    /// with identical requests are direct HashMap lookups.
    pub fn select(&mut self, req: &KernelRequest) -> KernelId {
        if let Some(&id) = self.cache.get(req) {
            return id;
        }
        let id = self.registry.best_for(req);
        self.cache.insert(req.clone(), id);
        id
    }

    /// Select the best kernel for the request (immutable borrow).
    ///
    /// Does not update the cache — use `select` when you want memoization.
    /// This variant exists for cases where the selector is behind a shared reference.
    pub fn select_ref(&self, req: &KernelRequest) -> KernelId {
        if let Some(&id) = self.cache.get(req) {
            return id;
        }
        self.registry.best_for(req)
    }

    /// Clear the selection cache (e.g., after a hardware state change).
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    /// Number of unique selections made so far.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the kernel registry (for inspection/debugging).
    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_registry_selects_cpu_kernels() {
        let registry = KernelRegistry::detect(None); // no GPU
        let mut selector = KernelSelector::new(registry);

        let req = KernelRequest::gemm(ComputeArch::Cpu, KernelDType::F16, 4096, 4096, 4096);
        let kernel = selector.select(&req);
        assert_eq!(kernel, KernelId::CpuGemmF16To32);

        let attn_req = KernelRequest::attention(ComputeArch::Cpu, KernelDType::F16, 128, 0);
        let attn_kernel = selector.select(&attn_req);
        assert_eq!(attn_kernel, KernelId::CpuAttentionManual);
    }

    #[test]
    fn cuda_registry_selects_gpu_kernels() {
        let registry = KernelRegistry::detect(Some(86)); // RTX 3070 Ti (Ampere)
        let mut selector = KernelSelector::new(registry);

        // GEMM on CUDA should select cuBLAS kernel
        let req = KernelRequest::gemm(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 4096, 4096, 4096);
        let kernel = selector.select(&req);
        assert_eq!(kernel, KernelId::CudaDirectHgemm);

        // Attention with no cache → specialized decode kernel
        let attn_no_cache =
            KernelRequest::attention(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 128, 0);
        assert_eq!(
            selector.select(&attn_no_cache),
            KernelId::CudaAttentionDecodeFused
        );

        // Attention with cache → general fused kernel
        let attn_cached =
            KernelRequest::attention(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 128, 5);
        assert_eq!(selector.select(&attn_cached), KernelId::CudaAttentionFused);
    }

    #[test]
    fn cuda_registry_falls_back_for_cpu_arch() {
        let registry = KernelRegistry::detect(Some(86)); // GPU present
        let mut selector = KernelSelector::new(registry);

        // CPU request on a system with GPU should still get CPU kernel
        let req = KernelRequest::gemm(ComputeArch::Cpu, KernelDType::F16, 4096, 4096, 4096);
        let kernel = selector.select(&req);
        assert_eq!(kernel, KernelId::CpuGemmF16To32);
    }

    #[test]
    fn selection_is_memoized() {
        let registry = KernelRegistry::detect(Some(86));
        let mut selector = KernelSelector::new(registry);

        let req = KernelRequest::gemm(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 4096, 4096, 4096);

        // First call: computes and caches
        let k1 = selector.select(&req);
        assert_eq!(selector.cache_size(), 1);

        // Second call: cache hit
        let k2 = selector.select(&req);
        assert_eq!(k1, k2);
        assert_eq!(selector.cache_size(), 1); // still 1 entry
    }

    #[test]
    fn different_requests_get_different_cache_entries() {
        let registry = KernelRegistry::detect(Some(86));
        let mut selector = KernelSelector::new(registry);

        let req1 = KernelRequest::gemm(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 4096, 4096, 4096);
        let req2 = KernelRequest::gemm(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 2048, 2048, 2048);

        selector.select(&req1);
        selector.select(&req2);
        assert_eq!(selector.cache_size(), 2);
    }

    #[test]
    fn arch_constraint_cuda_min() {
        let constraint = ArchConstraint::CudaMin(75);
        assert!(constraint.satisfies(&ComputeArch::Cuda { sm: 75 }));
        assert!(constraint.satisfies(&ComputeArch::Cuda { sm: 86 }));
        assert!(!constraint.satisfies(&ComputeArch::Cuda { sm: 70 }));
        assert!(!constraint.satisfies(&ComputeArch::Cpu));
    }

    #[test]
    fn perf_class_ranking() {
        assert!(PerfClass::Baseline < PerfClass::Fast);
        assert!(PerfClass::Fast < PerfClass::Optimal);
        assert_eq!(PerfClass::Baseline.rank(), 0);
        assert_eq!(PerfClass::Fast.rank(), 1);
        assert_eq!(PerfClass::Optimal.rank(), 2);
    }

    #[test]
    fn registry_len_includes_cpu_and_cuda() {
        let cpu_only = KernelRegistry::detect(None);
        let with_gpu = KernelRegistry::detect(Some(86));
        assert!(with_gpu.len() > cpu_only.len());
    }

    #[test]
    fn invalidate_clears_cache() {
        let registry = KernelRegistry::detect(Some(86));
        let mut selector = KernelSelector::new(registry);

        let req = KernelRequest::gemm(ComputeArch::Cuda { sm: 86 }, KernelDType::F16, 4096, 4096, 4096);
        selector.select(&req);
        assert_eq!(selector.cache_size(), 1);

        selector.invalidate();
        assert_eq!(selector.cache_size(), 0);
    }
}
