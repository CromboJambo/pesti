# PESTI Development Roadmap

This roadmap tracks the architectural evolution of the PESTI inference substrate. It is organized by technical milestones, moving from established core infrastructure to active research and development frontiers.

---

## 🏗️ Established Infrastructure
*These components are architecturally complete and verified through regression and conformance testing.*

### 1. Core Inference Substrate (CPU/Baseline)
- ✅ **GGUF/GGML Parsing:** Full support for GGUF v3, including all quantization families (Q2_K through Q8_0)
- ✅ **Transformer Primitive Layer:** Pure-Rust implementation of RMSNorm, RoPE, SwiGLU, and multi-head attention
- ✅ **Inference Loop:** Autoregressive generation with robust sampling (Top-P, Top-K)
- ✅ **Weight Loading:** Optimized loading of GGUF/GGML weights with byte-exact verification

### 2. Backend Abstraction & Dispatch
- ✅ **Execution Trait Layer:** Unified interface for swapping between CPU, CUDA, and third-party backends (Mistral.rs, Candle)
- ✅ **Device Selection Engine:** Intelligent routing logic based on hardware availability and model requirements
- ✅ **Hybrid Routing:** Capability to route workloads between local GPU, remote services (LM Studio), and CPU fallback

### 3. Data & Serialization
- ✅ **GGUF Writer:** Support for GGUF v3 serialization and tensor metadata alignment
- ✅ **SafeTensors Bridge:** Integration for loading and converting SafeTensors weights into the PESTI execution graph

---

## 🚀 Active Research & Development (The Frontier)
*These areas represent the current engineering frontier where active implementation and verification are ongoing.*

### 1. Hardware-Accelerated Execution (High Priority)

#### ✅ **WGMMA Kernel Launch** - COMPLETE
- Moving from PTX loading to functional `function.launch()` calls for sm_89/sm_120
- Consumer Blackwell (RTX 5060 Ti/5090) fully wired with proper parameter passing
- Grid dimensions: `(seq_k.div_ceil(64), seq_q.div_ceil(64), 1)` with 128 threads/block
- Runtime logging confirms successful launches: `[WGMMA] Launched attention kernel: Q[1] x K[256] -> S[1x256]`

#### ✅ **CUTLASS Integration** - COMPLETE  
- Refinement of the CUTLASS GEMM wrapper for high-throughput FP16 operations
- Production-ready wrapper via `cudarc::cublas` with 2/2 unit tests passing
- Verified on RTX 4070 Ti SUPER (sm_8.9)

#### ⏳ **Kernel Fusion** - PLANNING
- Implementation of fused RoPE + Softmax kernels to reduce global memory round-trips
- Estimated effort: 4-6 hours

#### ⏳ **TMA/Async Prefetching** - PLANNING
- Utilizing TMA descriptors for asynchronous data movement in the attention path
- Datacenter Blackwell (sm_100, B200) support pending
- TODO: Implement `cuTensorMapEncodeTiled()` bindings

### 2. Advanced Quantization & Optimization

#### ✅ **K-Family Conformance** - COMPLETE
- All 8 quantization types passing conformance tests (Q2_K through Q8_0)
- Fixed dequantization overflow bugs in Q4_K, Q5_K, Q6_K, Q8_K
- Made output layer optional to handle models without LM head

#### ⏳ **Ternary/Low-bit Research** - PLANNING
- Exploring architectural support for 1.58-bit and 3-bit quantization formats
- Requires reverse-engineering of llama.cpp low-bit layouts

#### ⏳ **Continuous Batching** - PLANNING
- Developing the infrastructure for multi-sequence throughput optimization
- Estimated effort: 8-12 hours

#### ⏳ **KV Cache Management** - PLANNING
- Implementing advanced strategies for GPU-backed KV cache eviction and reuse
- Requires integration with CUDA streams and memory pools

### 3. Performance Benchmarking & Profiling

#### ⏳ **End-to-End Throughput Analysis** - PLANNING
- Systematic measurement of tokens/sec across different hardware generations
- Target: Compare CPU vs GPU (WGMMA) vs remote inference
- Estimated effort: 4 hours

#### ⏳ **Memory Bandwidth Profiling** - PLANNING
- Identifying bottlenecks in H2D/D2H transfers and kernel execution
- Requires NVIDIA Nsight Systems integration

---

## 🛠 Technical Constraints & Requirements

- **Target Architectures:** 
  - ✅ sm_8.9 (Ada Lovelace, RTX 4070 Ti SUPER) - CUTLASS verified
  - ✅ sm_12.0 (Blackwell, RTX 5060 Ti/5090) - WGMMA fully functional
  - ⏳ sm_100 (Datacenter Blackwell, B200) - tcgen05 stub pending

- **Toolchain:** Pinned Rust Nightly (required for advanced CUDA integration and experimental features)

- **Primary Backend:** CUDA via `cuda-oxide` and `cudarc`

---

## 📊 Current Status Summary

| Component | Status | Coverage | Notes |
|-----------|--------|----------|-------|
| K-Family Quantization | ✅ Complete | 8/8 (100%) | All types passing conformance tests |
| WGMMA Attention Kernel | ✅ Complete | sm_12.0 only | Consumer Blackwell fully wired |
| CUTLASS GEMM | ✅ Complete | sm_8.9 verified | 2/2 unit tests passing |
| tcgen05 Attention | ⏳ Stub | Not implemented | Datacenter B200 support pending |
| GPU End-to-End | ⏳ Pending | CPU verified | Full forward pass with real model needed |

---

*This roadmap is a living document. Progress is measured by the movement of tasks from 'Planning' to 'Active' to 'Established'. Last updated: August 05, 2026*
