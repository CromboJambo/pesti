# PESTI — Portable Execution Substrate for Transformer Inference

## Project Scope

**In scope:** GPU-first LLM inference engine written in Rust. CUDA dispatch, cuBLAS integration, fused attention kernels, weight caching, and optimized decode performance. Target: production-grade local inference with measurable tok/s benchmarks.

**Not in scope:**
- Multi-agent orchestration framework (that's crabjar)
- Infrastructure-as-code / machine provisioning (that's home-lab/mirror-lab)
- Fleet monitoring/orchestration (that's herdr)
- Educational content for users (that's user_coarses)

## Development Priorities

GPU decode optimization is the primary focus. All changes should be measurable in tok/s improvement or numerical conformance. See `docs/ROADMAP.md` for weekly sprint tracking and performance targets.
