# PESTI Architecture: Full-Rust vs Hybrid Python/Rust Approach

**Status:** Proposed  
**Date:** 2026-10-10  
**Related:** litellm (BerriAI) architecture analysis, SPEC_LLM_RUNNER_TRAINING.md

## Context

Litellm demonstrates a proven hybrid architecture: Python SDK layer with Rust hotpath optimizations. Their Rust workspace (~80 crates, ~10MB) handles performance-critical paths while the Python layer (~77MB) maintains 100+ provider integrations and user-facing interfaces.

We analyzed litellm's architecture to determine whether pesti should follow the same hybrid approach or go full-Rust.

## Decision: Full-Rust Architecture

Pesti will be a **pure Rust LLM inference stack** with optional language bindings, not a Python project with Rust optimizations. This differs fundamentally from litellm's "Python first, Rust for speed" model.

### Rationale

1. **Smaller provider surface**: Pesti targets OpenAI-compatible APIs and local GGUF/safetensors models, not 100+ cloud providers. No need for Python's ecosystem breadth.
2. **Local-first deployment**: Home-lab GPU inference doesn't benefit from Python's cloud service integrations (GCP Vertex, Azure OpenAI, AWS Bedrock).
3. **Performance as primary goal**: Every layer in Rust eliminates FFI overhead, serialization costs, and language boundary friction.
4. **Cleaner dependency graph**: Single Cargo workspace vs dual Python/Rust build systems with pyo3 bridge complexity.
5. **Target audience alignment**: Rust users who want local LLM inference get native tooling without Python installation.

### What We Adopt from Litellm

Despite the architectural difference, litellm's Rust core demonstrates patterns worth adopting:

#### 1. Cache Layer Architecture (High Priority)
Litellm has multiple cache implementations: `cache-redis`, `cache-s3`, `cache-gcs`, `cache-disk`, `cache-memory`. Pesti should implement a pluggable response caching layer for repeated prompts.

**Pesti implementation:** New `pesti-cache` crate with disk-backed LRU cache as default (SQLite or file-based), optional Redis backend for multi-instance deployments.

#### 2. Router Architecture (Medium Priority)
Litellm's `litellm-router` provides model routing with fallback, load balancing, and circuit breaking. Pesti-server could adopt similar patterns for multi-model deployments.

**Pesti implementation:** Extend pesti-server to support multiple loaded models with routing logic based on request parameters or configuration rules.

#### 3. Observability/Tracing (Medium Priority)
Litellm has a dedicated `tracing` crate with OpenTelemetry integration and structured logging across all crates.

**Pesti implementation:** Add tracing instrumentation to pesti-runner and pesti-server with metrics for tokens/sec, latency, cache hit rates, and GPU utilization.

#### 4. Token Counter Abstraction (Low Priority)
Litellm provides multiple token counting implementations: tiktoken, HuggingFace tokenizer, fast vs accurate paths.

**Pesti implementation:** Already addressed by qwen2-bpe crate; may need additional tokenizers for other model families in the future.

#### 5. Secrets Management Pattern (Future)
Litellm has dedicated crates for each secrets provider (AWS, Azure, GCP, HashiCorp). Pesti could learn from this pattern if multi-tenant deployment becomes a requirement.

**Pesti implementation:** Not needed for single-user home-lab deployments; revisit if pesti-server becomes production multi-tenant.

### What We Don't Adopt

#### Python SDK via pyo3
Litellm exposes its Rust core as a Python module (`litellm-python-compat` via pyo3). Pesti will NOT do this initially because:
- Python users have llama.cpp, vLLM, and other established options
- Adding pyo3 bridge adds build complexity for marginal benefit
- Focus on clean HTTP API that any language can call

**Revisit if:** User demand emerges from Python users who specifically want pesti's performance characteristics.

#### 100+ Provider Integrations
Litellm supports dozens of cloud providers with custom auth, rate limiting, and error handling per provider. Pesti focuses on:
- OpenAI-compatible API (covers most cloud providers)
- Local model inference (GGUF, safetensors)
- GPU-accelerated CUDA kernels

### Architecture Comparison

```
Litellm Architecture:
┌─────────────────────────────────────┐
│ Python SDK (litellm package)        │
│ - 100+ provider integrations        │
│ - API endpoints                     │
│ - User-facing interfaces            │
└──────────────┬──────────────────────┘
               │ pyo3 bridge
┌──────────────▼──────────────────────┐
│ Rust Core (litellm-rust workspace)  │
│ - Routing, caching, token counting  │
│ - Performance-critical paths        │
└─────────────────────────────────────┘

Pesti Architecture:
┌─────────────────────────────────────┐
│ pesti CLI / pesti-server (Rust)     │
│ - OpenAI-compatible API             │
│ - Model management                  │
│ - Training pipeline                 │
└──────────────┬──────────────────────┘
               │ Direct Rust calls
┌──────────────▼──────────────────────┐
│ pesti-runner (Rust)                 │
│ - GPU kernels, inference loop       │
│ - CUDA dispatch, weight loading     │
└──────────────┬──────────────────────┘
               │ Direct Rust calls
┌──────────────▼──────────────────────┐
│ pesti-safetensors / qwen2-bpe (Rust)│
│ - Weight I/O, tokenization          │
└─────────────────────────────────────┘
```

### Implementation Plan

**Phase 1: Cache Layer (Weeks 1-2)**
- Create `pesti-cache` crate with disk-backed LRU cache
- Integrate into pesti-server for response caching
- Configure via TOML (cache size, TTL, backend)

**Phase 2: Observability (Weeks 3-4)**
- Add tracing instrumentation to pesti-runner and pesti-server
- Metrics: tokens/sec, latency percentiles, cache hit rates, GPU utilization
- OpenTelemetry-compatible structured logging

**Phase 3: Multi-Model Router (Weeks 5-6)**
- Extend pesti-server to support multiple loaded models
- Routing logic based on request parameters
- Model hot-swapping with minimal downtime

### Success Metrics

- **Cache hit rate:** >30% for repeated prompts in production deployments
- **Latency improvement:** <10ms overhead from caching layer vs uncached requests
- **Observability completeness:** All inference operations traced with latency and token metrics
- **Multi-model support:** Serve 2+ models simultaneously with <5% performance degradation per model

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Cache layer adds complexity | Start with simple disk-backed LRU; add Redis later if needed |
| Multi-model memory pressure | Implement VRAM-aware model loading/unloading |
| Observability overhead | Use sampling for high-throughput deployments |
| Scope creep toward litellm feature set | Maintain strict focus on local inference + OpenAI-compatible API |

### Conclusion

Litellm's hybrid architecture makes sense for their use case (drop-in replacement for existing Python code with 100+ provider integrations). Pesti's full-Rust approach is better suited for local GPU inference where performance, simplicity, and clean architecture matter more than ecosystem breadth.

We adopt litellm's proven patterns (caching, routing, observability) while avoiding their architectural complexity (Python bridge, massive provider surface). This gives us the best of both: battle-tested design patterns with a leaner, faster implementation.
