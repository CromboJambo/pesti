# PESTI Development Roadmap

This roadmap tracks the architectural evolution of the PESTI inference substrate. It is organized by technical milestones, moving from established core infrastructure to active research and development frontiers.

---

## 🏗️ Established Infrastructure
*These components are architecturally complete and verified through regression and conformance testing.*

### 1. Core Inference Substrate (CPU/Baseline)
- **GGUF/GGML Parsing:** Full support for GGUF v3, including all 2/8/4-bit quantization families (Q2_K through Q8_0).
- **Transformer Primitive Layer:** Pure-Rust implementation of RMSNorm, RoPE, SwiGLU, and multi-head attention.
- **Inference Loop:** Autoregressive generation with robust sampling (Top-P, Top-K).
- **Weight Loading:** Optimized loading of GGUF/GGML weights with byte-exact verification.

### 2. Backend Abstraction & Dispatch
- **Execution Trait Layer:** Unified interface for swapping between CPU, CUDA, and third-party backends (Mistral.rs, Candle).
- **Device Selection Engine:** Intelligent routing logic based on hardware availability and model requirements.
- **Hybrid Routing:** Capability to route workloads between local GPU, remote services (LM Studio), and CPU fallback.

### 3. Data & Serialization
- **GGUF Writer:** Support for GGUF v3 serialization and tensor metadata alignment.
- **SafeTensors Bridge:** Integration for loading and converting SafeTensors weights into the PESTI execution graph.

---

## 🚀 Active Research & Development (The Frontier)
*These areas represent the current engineering frontier where active implementation and verification are ongoing.*

### 1. Hardware-Accelerated Execution (High Priority)
- **[IN PROGRESS] WGMMA Kernel Launch:** Moving from PTX loading to functional `function.launch()` calls for sm_89/sm_120.
- **[IN PROGRESS] CUTLASS Integration:** Refinement of the CUTLASS GEMM wrapper for high-throughput FP16 operations.
- **[PLANNING] Kernel Fusion:** Implementation of fused RoPE + Softmax kernels to reduce global memory round-trips.
- **[PLANNING] TMA/Async Prefetching:** Utilizing TMA descriptors for asynchronous data movement in the attention path.

### 2. Advanced Quantization & Optimization
- **[PLANNING] Ternary/Low-bit Research:** Exploring architectural support for 1.58-bit and 3-bit quantization formats.
- **[PLANNING] Continuous Batching:** Developing the infrastructure for multi-sequence throughput optimization.
- **[PLANNING] KV Cache Management:** Implementing advanced strategies for GPU-backed KV cache eviction and reuse.

### 3. Performance Benchmarking & Profiling
- **[PLANNING] End-to-End Throughput Analysis:** Systematic measurement of tokens/sec across different hardware generations.
- **[PLANNING] Memory Bandwidth Profiling:** Identifying bottlenecks in H2D/D2H transfers and kernel execution.

---

## 🛠 Technical Constraints & Constraints
- **Target Architectures:** sm_89 (Ada Lovelace), sm_100/120 (Blackwell).
- **Toolchain:** Pinned Rust Nightly (required for advanced CUDA integration and experimental features).
- **Primary Backend:** CUDA via `cuda-oxide` and `cudarc`.

---
*This roadmap is a living document. Progress is measured by the movement of tasks from 'Planning' to 'Active' to 'Established'.*
