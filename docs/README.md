# PESTI Documentation

> Portable Execution Substrate for Transformer Inference

This directory holds all PESTI engineering documentation. The root of the repo
keeps only the three always-current files: [`README.md`](../README.md),
[`ROADMAP.md`](../ROADMAP.md), and [`CHANGELOG.md`](../CHANGELOG.md).

## Start here

| Document | What it is |
|----------|-----------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Single source of truth: current version, phase status, verified capabilities, known gaps |
| [QUICKSTART-OPTION-B.md](QUICKSTART-OPTION-B.md) | Build & run guide (learning mode vs. production mistral.rs mode) |
| [RELEASE.md](RELEASE.md) | Release process |

## By topic

- **[concepts/](concepts/)** — Architecture strategy, the Computational Inertia concept, Slow-Friend Substrate (bounded memory / scoped MoE / drift-gated compaction)
- **[specs/](specs/)** — Self-contained implementation specs for fresh coding sessions (e.g. Slow-Friend G1)
- **[gpu/](gpu/)** — GPU attention kernels, WGMMA/tcgen05, flash attention, CUDA integration, RoPE caching
- **[cpu/](cpu/)** — CPU forward pass, SIMD, optimization, CPU↔GPU mapping
- **[conformance/](conformance/)** — Numerical conformance vs. llama.cpp, regression-testing strategy
- **[benchmarks/](benchmarks/)** — Measured benchmark results and profiling
- **[tokenizer/](tokenizer/)** — Qwen2 BPE tokenizer integration (mistral.rs backend)
- **[notes/](notes/)** — Scratch notes, dead-code analysis, misc
- **[history/](history/)** — Weekly progress logs (Week 1–17) and superseded session reports

## Conventions

- **Current state** lives in `PROJECT_STATUS.md` and the root `ROADMAP.md`.
- **Historical** week-by-week logs live in `history/` and are not updated after the fact.
- Performance claims must cite a measured baseline (see `benchmarks/`).
- Phase status uses: ✅ Complete · ⚠️ Partial · ❌ Blocked · 🔄 In Progress.
