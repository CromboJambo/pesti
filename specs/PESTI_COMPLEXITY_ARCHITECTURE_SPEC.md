# PESTI Complexity Architecture: Kernel Selection & Device Abstraction

## §0 Goal + Scope

**Goal**: Restructure pesti-runner's kernel dispatch and device selection to eliminate scattered procedural branching, following the architectural patterns rustc uses to manage combinatorial complexity at 4M LOC. The result: adding a new kernel or hardware target becomes a data change (one entry in a registry) instead of a code change (modifying N dispatch methods).

**Core principle**: *"Complexity in a computation system doesn't come from the number of branches — it comes from where the branches live."* Branches scattered across every function are unmanageable. Branches concentrated in one decision point with memoized results are manageable.

**NOT in scope (yet)**:
- ❌ Full query engine with incremental invalidation (rustc's dep graph is overkill for a fixed model architecture)
- ❌ Multi-backend support beyond CUDA + CPU (don't abstract for backends we don't have)
- ❌ Speculative decoding or beam search (the activation caching pattern enables them, but implementing them is separate work)
- ❌ Replacing cuBLAS with custom GEMM kernels (that's the GPU optimization track, not complexity architecture)

**Scope**: Three phases in `pesti-runner`:
1. **Phase 1**: Extract `KernelSelector` — memoized kernel selection replacing duplicated if/else chains in dispatch methods
2. **Phase 2**: Unify device decision at init time — replace per-call GPU availability checks with a single `ExecutionMode` enum
3. **Phase 3**: Device-agnostic model graph — separate "what to compute" from "where to compute it" (the SSA boundary)

## §1 Context & Why Now

**Principle**: pesti is "pretty much the whole point" of applying Rust complexity-solving logic to LLM execution. The model architecture is fixed (like the language grammar), but the execution strategy (hardware × dtype × seq_len × batch_size) varies combinatorially. That's the same shape as "compile this code for this target with these flags" — exactly the problem rustc solved.

**Why now**:
- `dispatch.rs` is 2,272 lines with the same GPU/CPU decision tree duplicated across `dispatch_gemm`, `dispatch_linear`, and `dispatch_attention`. Each method independently checks `self.prefer_gpu && self.gpu_available() && bridge_is_cuda()` — conditions that never change between decode steps but are re-evaluated thousands of times per inference run.
- Adding a new attention kernel (e.g., FlashAttention-3) currently requires modifying 3+ call sites with identical branching logic. This is the exact failure mode rustc's trait solver migration fixed: procedural branching that scales as O(kernels × dtypes × archs) instead of O(1) data entries.
- Week 30 optimization work (KV cache quantization, speculative decoding) will add *more* branching dimensions (quantized vs unquantized KV, proposal vs verification paths). Without restructuring dispatch first, these additions compound the existing complexity.
- The rustc deep-dive analysis identified four patterns that map directly: query memoization, dep graph caching, SSA boundary, and procedural→logical reframing. This spec implements the first three in pesti's context.

**The combinatorial space pesti currently handles with procedural branching**:

| Dimension | Values | Current handling |
|-----------|--------|-----------------|
| Hardware | CPU / CUDA (sm_75/80/86/90) | `if self.prefer_gpu && self.gpu_available() && bridge_is_cuda()` in every method |
| Dtype | f32 / f16 / bf16 / Q4_K / Q8_0 | Separate code paths per dtype, duplicated across methods |
| Sequence length | 1 (decode) vs N (prefill) | `if seq_len == 1 { fused_decode } else if cache_len >= 2 { fused } else { manual }` |
| Batch size | 1 vs N | Implicit in shape checks, no explicit selection logic |
| Kernel availability | fused / fused_decode / manual | `self.has_fused_kernel()` checks scattered through attention dispatch |

That's a 5-dimensional decision space. Currently: nested if/else in each method. Cyclomatic complexity of `AttentionDispatch::forward` alone is >20.

**rustc patterns being applied**:

| rustc Pattern | What it solved | pesti Application |
|---------------|----------------|-------------------|
| Query system (232 memoized computations) | "Which type does this expression have?" computed once, cached forever | Kernel selection: "Which kernel for (arch, dtype, seq_len, batch)?" computed once, cached forever |
| Dep graph (incremental recompilation) | Change one file → only dirty queries recompute | Shared prefix caching: same system prompt → same early layer outputs, reused across conversations |
| SSA boundary (MIR → LLVM) | Device-agnostic IR lowered to hardware backend | Model graph (device-agnostic ops) lowered to CUDA/CPU execution |
| Trait solver migration (51k → 12k lines) | Procedural branching → logical proof search | Hand-coded if/else chains → constraint-based kernel selection |

**Failure modes this prevents**:
1. A new kernel is added but one of the three dispatch methods isn't updated — it silently falls back to a slower path with no error.
2. GPU becomes unavailable mid-run (driver crash, OOM) — each method handles fallback differently, leading to inconsistent state.
3. Adding ROCm support requires touching every dispatch method instead of implementing one backend trait.
4. The optimization track adds KV quantization and speculative decoding on top of the existing branching, pushing cyclomatic complexity past the point where a single engineer can reason about it.

## §2 Architecture

### Phase 1: KernelSelector (Immediate)

**Problem**: Every dispatch method re-derives "which kernel should I use?" from scratch on every call. The conditions are identical but the code is duplicated three times.

**Solution**: A `KernelSelector` struct that takes execution constraints and returns a memoized kernel selection.

```rust
// crates/pesti-runner/src/kernel/selector.rs (new file)

/// Constraints that determine which kernel to use for an operation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct KernelRequest {
    /// Operation type (GEMM, Attention, Norm, etc.)
    pub op: OpKind,
    /// Target architecture
    pub arch: ComputeArch,       // Cpu, Cuda { sm: u16 }, ...
    /// Data type of the computation
    pub dtype: DType,            // F32, F16, BF16, Q4_K, ...
    /// Sequence length (1 = decode, N = prefill)
    pub seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// For attention: current KV cache length
    pub cache_len: Option<usize>,
}

/// The result of kernel selection — which concrete kernel to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelId {
    // GEMM kernels
    CpuGemmF32,
    CudaCublasHgemm,     // F16 via cuBLAS
    CudaBridgeGemm,      // candle_bridge GPU GEMM

    // Attention kernels
    CpuAttentionManual,
    CudaAttentionFused,        // cache_len >= 2
    CudaAttentionDecodeFused,  // batch=1, seq=1, cache_len == 0
    CudaAttentionManual,       // fallback when fused not available

    // Non-matmul kernels
    CpuRmsNorm,
    CudaRmsNorm,
    CpuSwiGLU,
    CudaSwiGLU,
    CpuRoPE,
    CudaRoPE,
    // ... one variant per kernel
}

/// OpKind identifies the class of operation being dispatched.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum OpKind {
    GEMM,
    LinearWithBias,
    Attention,
    RMSNorm,
    SwiGLU,
    RoPE,
}

/// Selects the optimal kernel for a given execution request.
/// Memoizes selections so repeated requests (same arch/dtype/shape)
/// become HashMap lookups instead of constraint evaluation.
pub struct KernelSelector {
    cache: HashMap<KernelRequest, KernelId>,
    /// Static registry of available kernels and their requirements.
    /// Populated at init time based on what's actually compiled in.
    registry: KernelRegistry,
}

impl KernelSelector {
    /// Create a selector with the given set of available kernels.
    pub fn new(registry: KernelRegistry) -> Self;

    /// Select the best kernel for the request.
    /// First call evaluates constraints; subsequent calls are cache hits.
    pub fn select(&mut self, req: &KernelRequest) -> KernelId {
        if let Some(&id) = self.cache.get(req) {
            return id;
        }
        let id = self.registry.best_for(req);
        self.cache.insert(req.clone(), id);
        id
    }

    /// Clear the cache (e.g., after a hardware state change).
    pub fn invalidate(&mut self);
}

/// Static registry of available kernels and their constraints.
/// Built once at init — this is where "adding a kernel = adding an entry" happens.
pub struct KernelRegistry {
    kernels: Vec<KernelSpec>,
}

impl KernelRegistry {
    /// Build the registry from what's actually available at runtime.
    pub fn detect() -> Self {
        let mut kernels = vec![
            // Always available (CPU fallback)
            KernelSpec { id: KernelId::CpuGemmF32, arch: ArchConstraint::Any, dtype: DTypeConstraint::F32, perf: PerfClass::Baseline },
            KernelSpec { id: KernelId::CpuRmsNorm, arch: ArchConstraint::Any, dtype: DTypeConstraint::Any, perf: PerfClass::Baseline },
            // ... all CPU kernels
        ];

        #[cfg(feature = "cuda")]
        if let Some(arch) = detect_cuda_arch() {
            kernels.push(KernelSpec { id: KernelId::CudaCublasHgemm, arch: ArchConstraint::CudaMin(75), dtype: DTypeConstraint::F16, perf: PerfClass::Optimal });
            kernels.push(KernelSpec { id: KernelId::CudaRmsNorm, arch: ArchConstraint::CudaMin(75), dtype: DTypeConstraint::Any, perf: PerfClass::Fast });
            // ... all CUDA kernels
        }

        Self { kernels }
    }

    /// Find the best kernel satisfying all constraints in the request.
    fn best_for(&self, req: &KernelRequest) -> KernelId {
        self.kernels.iter()
            .filter(|k| k.satisfies(req))
            .max_by_key(|k| k.perf.rank())
            .map(|k| k.id)
            .unwrap_or(KernelId::CpuGemmF32) // guaranteed fallback
    }
}

/// Capability constraints for a kernel.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub id: KernelId,
    pub arch: ArchConstraint,
    pub dtype: DTypeConstraint,
    pub perf: PerfClass,
    /// Optional shape requirements (e.g., fused decode requires seq_len == 1)
    pub shape_requirements: Option<ShapeRequirements>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfClass {
    Baseline,   // CPU fallback — always works, slowest
    Fast,       // GPU kernel, good performance
    Optimal,    // Best known kernel for this operation class
}
```

**Integration with existing dispatch methods**:

```rust
// crates/pesti-runner/src/kernel/dispatch.rs (modified)

pub struct DispatchContext {
    selector: KernelSelector,      // replaces scattered if/else chains
    gpu_bridge: Option<CudaBridge>, // still needed for actual execution
    // ... other state unchanged
}

impl DispatchContext {
    pub fn new() -> Self {
        let registry = KernelRegistry::detect();  // one-time hardware detection
        let selector = KernelSelector::new(registry);
        // ...
    }

    /// dispatch_linear after refactor:
    pub fn dispatch_linear(
        &mut self, input: &[f32], weight: &[f32], bias: Option<&[f32]>, out: &mut [f32],
    ) {
        let req = KernelRequest {
            op: OpKind::LinearWithBias,
            arch: self.current_arch(),
            dtype: DType::F32,
            seq_len: 1, // linear layers are per-token
            batch_size: 1,
            cache_len: None,
        };
        let kernel = self.selector.select(&req);

        match kernel {
            KernelId::CudaCublasHgemm => self.gemm_gpu_cublas(input, weight, bias, out),
            KernelId::CudaBridgeGemm => self.gemm_gpu_bridge(input, weight, bias, out),
            _ => self.gemm_cpu(input, weight, bias, out), // fallback
        }
    }
}
```

**What changes in dispatch.rs**:
- Remove all `if self.prefer_gpu && self.gpu_available() && bridge_is_cuda()` chains
- Replace with: build a `KernelRequest`, call `self.selector.select(&req)`, match on the `KernelId`
- The actual kernel execution functions (`gemm_cpu`, `gemm_gpu_cublas`, etc.) stay exactly where they are — only the *selection* logic moves
- Expected: ~200 lines of duplicated branching removed, replaced by ~30 lines of request construction + match

### Phase 2: ExecutionMode Unification (Immediate)

**Problem**: GPU availability is checked at every dispatch call. If the GPU state changes mid-run (driver crash, OOM), different methods handle it differently.

**Solution**: Resolve execution mode once at init. Store as an enum. Dispatch methods pattern-match on it.

```rust
// crates/pesti-runner/src/kernel/mod.rs (modified)

/// Resolved execution mode — determined once at init, not per-call.
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    Cpu,
    Gpu {
        backend: GpuBackend,
        arch: ComputeArch,   // e.g., Cuda { sm: 86 }
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    CandleBridge,  // candle_bridge with CUDA feature
    DirectCuda,    // direct CUDA calls (future)
}

impl DispatchContext {
    pub fn new() -> Self {
        let mode = Self::detect_execution_mode();
        // KernelRegistry uses the same detection result
        let registry = KernelRegistry::detect_for(&mode);
        Self {
            mode,
            selector: KernelSelector::new(registry),
            // ...
        }
    }

    fn detect_execution_mode() -> ExecutionMode {
        #[cfg(feature = "cuda")]
        if let Some(arch) = detect_cuda_arch() {
            return ExecutionMode::Gpu { backend: GpuBackend::CandleBridge, arch };
        }
        ExecutionMode::Cpu
    }

    /// Current architecture for kernel requests — no per-call detection.
    fn current_arch(&self) -> ComputeArch {
        match &self.mode {
            ExecutionMode::Cpu => ComputeArch::Cpu,
            ExecutionMode::Gpu { arch, .. } => *arch,
        }
    }
}
```

**What changes**: 
- `self.prefer_gpu`, `self.gpu_available()`, `bridge_is_cuda()` checks disappear from dispatch methods
- They're replaced by `match &self.mode` in the KernelRegistry (at init) and `self.current_arch()` (per-call, but it's an enum match, not a system query)
- If GPU state *does* change mid-run (rare), add `DispatchContext::re_detect_mode()` that rebuilds the selector — explicit recovery path instead of implicit per-call checking

### Phase 3: Device-Agnostic Model Graph (Medium-term)

**Problem**: CPU and GPU execution paths are interleaved in the same functions. Adding ROCm means touching every dispatch method again.

**Solution**: Two-layer architecture mirroring rustc's MIR→LLVM boundary.

```rust
// crates/pesti-runner/src/graph/mod.rs (new module)

/// Layer 1: Device-agnostic computation graph ("the MIR").
/// Describes WHAT to compute, not WHERE or HOW.
pub struct ModelGraph {
    pub layers: Vec<LayerOp>,
}

#[derive(Debug, Clone)]
pub enum LayerOp {
    Embedding { input: TensorRef },
    Attention { q: TensorRef, k: TensorRef, v: TensorRef, mask: Option<TensorRef> },
    Linear { input: TensorRef, weight: TensorRef, bias: Option<TensorRef> },
    RMSNorm { input: TensorRef, weight: TensorRef },
    SwiGLU { gate: TensorRef, up: TensorRef },
    RoPE { q: TensorRef, k: TensorRef, positions: TensorRef },
    // ... one variant per transformer operation
}

/// A reference to a tensor in the computation graph.
/// Resolved to actual device memory at execution time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorRef(pub usize);

/// Layer 2: Hardware backend ("the LLVM").
/// Describes HOW to execute a ModelGraph on specific hardware.
pub trait ComputeBackend: Send + Sync {
    /// Execute the full model graph. Returns the final output tensor.
    fn execute(&self, graph: &ModelGraph) -> Result<TensorHandle, BackendError>;

    /// Allocate a tensor on this backend's device.
    fn allocate(&self, shape: &[usize], dtype: DType) -> Result<TensorHandle, BackendError>;

    /// Copy data from host to device.
    fn upload(&self, data: &[f32]) -> Result<TensorHandle, BackendError>;

    /// Copy data from device to host.
    fn download(&self, handle: &TensorHandle) -> Result<Vec<f32>, BackendError>;
}

/// CUDA backend — implements ComputeBackend using candle_bridge / cuBLAS.
pub struct CudaBackend { /* ... */ }

/// CPU backend — implements ComputeBackend using native loops / BLAS.
pub struct CpuBackend { /* ... */ }
```

**Execution flow after Phase 3**:
1. Model is loaded → `ModelGraph` is built (device-agnostic, describes the transformer)
2. Backend is selected at init (`CudaBackend` or `CpuBackend`)
3. Each inference step: `backend.execute(&model_graph)` — the backend decides which kernels to use internally
4. Adding ROCm = implementing `ComputeBackend` for ROCm. Zero changes to model logic.

**Migration path**: Phase 3 doesn't happen all at once. Start with attention (the most complex dispatch), then GEMM, then non-matmul ops. Each migration is: move the operation from `dispatch.rs` into a `ModelGraph` node + backend implementation.

### What NOT to Build (and Why)

**No full query engine**: rustc needs incremental invalidation because it compiles arbitrary user code with arbitrary dependencies. pesti runs one model with a fixed architecture. A HashMap cache in KernelSelector is sufficient — no dep graph, no green/red classification, no 50k-line infrastructure.

**No premature backend abstraction**: Phase 3's `ComputeBackend` trait is justified because we already have two backends (CPU + CUDA). Don't design for ROCm/Metal/WebGPU until one of those is actually needed. The trait exists to make adding them *easy*, not to pretend they're coming next week.

**No per-operation memoization beyond kernel selection**: rustc caches query results because the same type-checking question comes up repeatedly across a large codebase. pesti's "same question" pattern is simpler: the same (arch, dtype, shape) combination appears thousands of times during a single inference run. One HashMap handles that. Don't build a general-purpose computation cache.

## §3 Acceptance Criteria

### Phase 1: KernelSelector
- [ ] `crates/pesti-runner/src/kernel/selector.rs` exists with `KernelRequest`, `KernelId`, `OpKind`, `KernelSelector`, `KernelRegistry`, `KernelSpec`, `PerfClass`
- [ ] `KernelRegistry::detect()` correctly identifies available kernels based on compiled features and runtime hardware
- [ ] `KernelSelector::select()` returns correct kernel for all (arch, dtype, seq_len, batch) combinations — unit tests for each dimension
- [ ] `KernelSelector` cache works: second call with same request is a HashMap lookup (verify with benchmark or test)
- [ ] `dispatch_gemm`, `dispatch_linear`, `dispatch_attention` use `KernelSelector` instead of inline if/else chains
- [ ] No behavioral change: all existing tests pass, tok/s unchanged within measurement noise
- [ ] Adding a new kernel to the registry requires zero changes to dispatch methods (verified by adding a test kernel)

### Phase 2: ExecutionMode
- [ ] `ExecutionMode` enum exists in `kernel/mod.rs`
- [ ] `DispatchContext::new()` resolves mode once; no per-call GPU availability checks remain in dispatch methods
- [ ] `self.prefer_gpu`, `self.gpu_available()`, `bridge_is_cuda()` removed from all dispatch methods (search for these strings returns 0 matches in kernel/)
- [ ] CPU-only builds work identically to before (no CUDA feature)
- [ ] GPU builds select the same kernels as before (regression test: compare kernel selection before/after for known hardware)

### Phase 3: Model Graph (when started)
- [ ] `ModelGraph` and `LayerOp` types defined in `graph/mod.rs`
- [ ] `ComputeBackend` trait defined with `execute`, `allocate`, `upload`, `download`
- [ ] At least one operation (attention) migrated to graph + backend pattern
- [ ] Existing dispatch path still works alongside graph path (dual-mode during migration)
- [ ] Adding a mock backend that logs operations (no actual computation) compiles and runs — proves the abstraction is clean

### Build Gates (all phases)
- [ ] `cargo build -p pesti-runner` succeeds
- [ ] `cargo test -p pesti-runner` all tests pass
- [ ] No new clippy warnings in pesti-runner
- [ ] tok/s benchmark: within 2% of baseline (no performance regression from indirection)

## §4 Done-When / Handoff

When Phase 1 + 2 complete:
1. Update `ROADMAP.md` — add "Complexity Architecture" section under Week 30/31
2. Add EDR to `CHANGELOG.md`: why KernelSelector over other approaches (trait objects, macros, etc.)
3. Update `pesti-runner/ROADMAP.md` with the new dispatch architecture
4. Document `KernelRegistry::detect()` behavior in `docs/README.md` (how to add a new kernel)

Stop at next gate: Phase 3 (Model Graph) starts when either (a) a third hardware target is needed, or (b) the optimization track adds enough branching dimensions that dispatch.rs exceeds ~1500 lines again. Don't do Phase 3 preemptively.

## §5 Constraints & Cautions

- **No performance regression**: The KernelSelector adds one HashMap lookup per dispatch call. This must be negligible (< 1µs). If profiling shows otherwise, use a smaller key type or direct indexing instead of HashMap.
- **Backward compatibility**: `DispatchContext::new()` signature may change (to accept optional config), but all public inference APIs (`pesti::run()`, server endpoints) must remain unchanged. This is an internal refactor.
- **Don't over-abstract the registry**: `KernelSpec` should be a simple struct with fields, not a trait object or dynamic plugin system. We have ~20 kernels total, not 2000. A `Vec<KernelSpec>` with linear scan (or small HashMap by OpKind) is fine.
- **CUDA feature gating stays**: `#[cfg(feature = "cuda")]` blocks are still needed around CUDA-specific code. The refactor moves *selection* logic but doesn't eliminate conditional compilation — it just concentrates the decision in one place.
- **Phase 3 is optional and deferred**: If pesti never needs a third backend, Phase 3 may never be justified. The Model Graph abstraction has value for testability (mock backend) and code clarity, but don't do it "because rustc does." Do it when the problem demands it.
- **Kernel execution functions don't move in Phase 1/2**: `gemm_cpu`, `gemm_gpu_cublas`, `attention_fused`, etc. stay where they are. Only the *selection* logic changes. Moving execution code is Phase 3's job.

## §6 Suggested TDD Order

### Phase 1 (1-2 days)
1. Create `selector.rs` with type definitions: `OpKind`, `ComputeArch`, `DType`, `KernelRequest`, `KernelId`, `PerfClass`, `KernelSpec`
2. Implement `KernelRegistry::detect()` — start with CPU-only, add CUDA behind feature flag
3. Write unit tests for `KernelRegistry::best_for()`: verify correct kernel selected for each (arch, dtype, seq_len) combination
4. Implement `KernelSelector` with HashMap cache + tests for memoization
5. Refactor `dispatch_gemm` to use selector — run existing tests, verify no behavior change
6. Refactor `dispatch_linear` to use selector
7. Refactor `dispatch_attention` to use selector (most complex — three kernel variants)
8. Remove all inline GPU availability checks from dispatch methods
9. Add test: "adding a kernel to registry requires no dispatch changes" (add a fake kernel, verify it's selectable without touching dispatch code)

### Phase 2 (1 day)
1. Define `ExecutionMode` and `GpuBackend` enums in `kernel/mod.rs`
2. Implement `DispatchContext::detect_execution_mode()` — consolidate existing detection logic
3. Replace all `self.prefer_gpu && self.gpu_available() && bridge_is_cuda()` with `match &self.mode`
4. Add `DispatchContext::re_detect_mode()` for explicit GPU state recovery
5. Verify CPU-only build still works (`cargo build -p pesti-runner --no-default-features`)
6. Regression test: kernel selection on known hardware matches pre-refactor behavior

### Phase 3 (when justified, 3-5 days)
1. Define `ModelGraph`, `LayerOp`, `TensorRef` in new `graph/mod.rs`
2. Define `ComputeBackend` trait + `BackendError`
3. Implement `CpuBackend` wrapping existing CPU dispatch functions
4. Migrate attention to graph pattern: build `LayerOp::Attention` nodes, execute via backend
5. Add mock backend test: verify graph executes in correct order without actual computation
6. Migrate remaining operations one at a time (GEMM → non-matmul)
7. Remove old dispatch path when all operations are migrated

## §7 Open Questions

1. **Should KernelSelector be per-DispatchContext or global?**
   - Default: Per-context. Each inference session may have different available kernels (e.g., one process has GPU, another doesn't). A global selector would need to be keyed by hardware state, which is more complex than just building one per context.

2. **How should kernel performance ranking work when multiple kernels satisfy constraints?**
   - Default: Static `PerfClass` enum (Baseline < Fast < Optimal) assigned at registry construction. This is a simplification — the "best" kernel can depend on exact shapes, not just dtype/arch. If this becomes a problem, replace with a shape-aware scoring function. Don't build an autotuner as part of this spec (that's the existing auto-tuning track).

3. **Should KernelRequest include the full tensor shapes or just seq_len/batch_size?**
   - Default: Just seq_len and batch_size for Phase 1. Full shapes would make the HashMap key larger and reduce cache hit rate. If we find cases where the same (seq_len, batch) maps to different optimal kernels based on hidden dimension, add shape fields then.

4. **Phase 3 timing: what's the actual trigger?**
   - Suggested triggers (any one justifies starting):
     - A third hardware backend is actually needed (ROCm for AMD lab machines?)
     - dispatch.rs exceeds 1500 lines after optimization track additions
     - The weight-space track needs to execute subgraphs of the model independently
   - If none of these happen, Phase 3 stays deferred indefinitely. That's fine.

5. **Should this spec cover pesti-state integration (logging kernel selections)?**
   - Default: No. `pesti-state` logs weight-space operations, not inference dispatch decisions. If kernel selection telemetry is needed for debugging, that's a separate concern (probably just `log::debug!` statements, not persistent state).
