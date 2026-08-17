# Week 14 Plan: Real End-to-End Inference with 30GB VRAM

**Date**: August 17, 2026
**Status**: Updated for current hardware
**Goal**: Measure real-world autoregressive throughput on a real model with a real decoder, and get a credible comparison vs llama.cpp.

---

## 🎯 Executive Summary

**Week 13 status**: CUDA GEMM is numerically correct, but the only “end-to-end” numbers were synthetic
micro-benchmarks that do not represent transformer decode cost. Generation time was effectively stubbed.

**Week 14 reality**: With **32GB VRAM** across RTX 4070 Ti SUPER + RTX 5060 Ti, the constraint is no
longer model size. The right Week 14 work is:
- run real token-in → token-out decode on a real GGUF model,
- measure per-token latency and tok/s,
- compare against llama.cpp on the same model/prompt,
- profile the hot path to decide whether attention/FFN/RoPE/copy is the bottleneck.

**This is no longer a “0.5B/72 tok/s baseline” plan. It is a real throughput-measurement sprint.**

---

## 🎯 Model Choice

**Primary target**: **Bonsai-27B-Q1_0.gguf**
- Path: `/mnt/data/state/ai/lmstudio/models/lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf`
- Size: 3.6GB
- Why: fits easily in VRAM, gives a realistic workload, uses GGUF which PESTI already loads.
- Expected performance: 10-40 tok/s on a single RTX 4070 Ti SUPER depending on optimization.

**Fallback options** (in order):
- `gemma-4-26B-A4B-it-Q4_K_M.gguf` (2.5GB)
- `Qwen3.6-27B-Q4_K_M.gguf` (~14GB)

If Bonsai is unavailable or decoding fails, fall through to the next available model.

---

## 📋 Deliverables for Week 14

### Code Changes
- [ ] `pesti-runner/examples/week14_e2e_decode.rs` — real decode benchmark:
  - tokenize prompt
  - run real transformer layers
  - sample next token
  - time generation loop
- [ ] profiling notes documenting where time is spent

### Documentation
- [ ] `WEEK_14_E2E_RESULTS.md` — measured tok/s, prompt, hardware, model
- [ ] update `ROADMAP.md` with measured numbers, not projections

### Artifacts
- [ ] verified tok/s measurement
- [ ] llama.cpp baseline on same model/prompt/hardware
- [ ] bottleneck profile for decode path

---

## 🎯 Week 14 Priorities

### Priority 1: Real decode benchmark
**Goal**: get one real tok/s number.

**Tasks**
- [ ] build a working example using existing loader/tokenizer/sample path
- [ ] generate 64–128 tokens and report throughput
- [ ] fix any model-loading errors encountered

**Success metric**: example completes and prints tokens/sec.

### Priority 2: llama.cpp baseline
**Goal**: comparable reference.

**Tasks**
- [ ] run same prompt through llama.cpp on same GPU
- [ ] record prompt-processing + decode tok/s

**Success metric**: side-by-side PESTI vs llama.cpp numbers.

### Priority 3: Profile hot path
**Goal**: know where time goes.

**Tasks**
- [ ] instrument layer timing in the benchmark
- [ ] identify whether bottleneck is attention, FFN, RoPE, or host/device copy
- [ ] write findings in `WEEK_14_E2E_RESULTS.md`

**Success metric**: clear statement of top 3 bottlenecks.

---

## 🚦 Success Criteria

### Must-Have
- [ ] real decode benchmark completes successfully
- [ ] real measured tok/s reported
- [ ] llama.cpp baseline on same model/prompt

### Nice-to-Have
- [ ] layer-level profiling
- [ ] first meaningful optimization target identified

---

## 📝 Notes for Next Session

**Starting point**
```bash
cd /home/crombo/projects/pesti
cargo run --package pesti-runner --example week14_e2e_decode
```

**Context**
- CPU decode path already exists and loads GGUF correctly.
- Old projections are obsolete; real measurement is the only useful artifact now.
- 32GB VRAM means model-size arguments are secondary to decoder efficiency.
