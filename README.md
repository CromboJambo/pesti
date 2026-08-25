# PESTI - Portable Execution Substrate for Transformer Inference

```
▄▄▄▄▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄         ▄▄▄▄▄     ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ▄▄▄▄▄
█████ ▀█████▄  █████ ▀████▄   ▄▓▓▓▓▀▓▓▓▓▄  ▐▓█▓█▀▓███▓▀▓▓█▓▌ █▓█▓█
▓███▓  ▐█▓██▓▌ ▓█▓█▓  ▐▓█▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▓▌ ▐▓▓▓▌ ▓▓▓▓▓ ▐▓▓▓▌ ▓▓▓▓▓
▓▓▓▓▓   ▓▓▓▓▓▓ ▓▓▓▓▓   ▒▓▒▒░ ▓▒▓▒▒   ▒▒▓▒▓ ▀     ▒▓▒▓▓     ▀ ▓▓▓▒▓
▓▓▓▒▓   ▒▒▒▒▒▒ ▒▓▓▒▓         ▓▒▓▓▒▄              ▒▒▒▒▓       ▓▒▓▓▒
▒▓▒▓▒ ▄▒▒▒▒▒▒▌ ░▓▒▓▒▓▒       ▀▀▀▓▒▓▒░▒▒▀▄        ▒░▒▓▒       ░▒░▓▒
▓░▓▒▒▓▒▒▒░▒░▀  ░░▒░░   ░░░░░        ▀░▒░▒▒       ░░░░░       ░░░░░
░▀█░▀          ▀░▀░░   ▀░▀░▀ ░░▀░░   ▀░░░▄       ░░░▀░       ▀░▀░░
  ▀               ▀     ▀     ▀  ▀   ▒▀ ▀░       ▀ ▀          ▀  ▀
▓ ▓▄▀          ▄▓ ▄▓   ▓ ▄▓▄ ▓▄ ▓▄   ▓▄▄▓▓       █ ▄▓        ▓▄ ▓▄
▒▄▒▒▒          ▒▒▄░▒  ▓▒▒▒▒▓ ▄▒▓▒▄  ▄▄▒▒▒▒       ▓▓▓▒▒       ▄▒▓▒▄
░░░░░          ░░░▒░▄░░░░░░▀ ▀▄░░░░░░░▄▀      ▄▒▒▒░░▄      ░░░░░
```

**Portable Execution Substrate for Transformer Inference**
*A learning-first Rust substrate for GGUF inference: parse, dequantize, and run
transformer forward passes from scratch — with a numpy conformance oracle to prove it.*

---

## What This Is

PESTI is a Rust workspace for understanding LLM inference internals, built the
way you'd build it if you wanted to *contribute back* to llama.cpp / candle /
burn rather than just call them:

- **GGUF parsing + pure-Rust dequantization** of all K-family quantizations
  (Q2_K → Q8_0), verified byte-exact against a reference.
- **A CPU transformer forward pass** (RMSNorm → GQA attention → RoPE → SwiGLU
  FFN, all 24 layers + output head) that is **numerically conformant** with an
  independent pure-numpy reference.
- **A CUDA dispatch path** (`cudarc`-based) for GPU kernels, feature-gated so
  the crate compiles and the parser/dequant layer runs without a GPU.
- **A self-contained GGUF-embedded tokenizer** — no external `tokenizer.json`
  downloads; a Qwen2 GGUF carries its complete tokenizer.

The headline capability is **conformance**: every forward-pass fix is verified
against `conformance-corpus/ref_forward.py` (an independent numpy oracle), not
just "it compiles" or "it runs."

---

## Quick Start

```bash
# Clone the workspace
git clone https://github.com/crombojambo/pesti.git
cd pesti

# Build the library (CPU path; no GPU required)
cargo build -p pesti-runner --lib

# Run the library test suite (62 unit tests: dequant, tokenizer, kvcache, ...)
cargo test -p pesti-runner --features cuda --lib

# CPU end-to-end generation on the conformance model (coherent text)
cargo run -p pesti-runner --release --features cuda \
  --example cpu_e2e_generate \
  -- conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 48

# Dump all 24 layers' hidden states + logits (the verified full-model forward
# pass; prints per-layer norms, top-8 tokens, argmax). This is the example the
# conformance check below diffs against the numpy oracle.
cargo run -p pesti-runner --release --features cuda \
  --example dump_all_layers \
  -- conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Reproduce the 24-layer conformance check (Rust vs numpy) — see below
```

> **Note on `--features cuda`:** the `transformer` module (and therefore the
> forward-pass examples) is gated behind the `cuda` feature in `src/lib.rs`.
> The *library* compiles without it (parser + dequant + CPU primitives), but
> running the transformer examples requires `--features cuda` even when they
> execute on the CPU path.

---

## Current Status (honest)

### ✅ Verified working
| Capability | Evidence |
|------------|----------|
| K-family dequantization (Q2_K → Q8_0) | Byte-exact vs reference; covered by lib unit tests |
| **24-layer CPU forward pass** | **Conformant vs numpy oracle** — max per-layer Δ 7.6e-5, correlation 1.000000 (see [Conformance](#conformance)) |
| Coherent CPU text generation | `"The capital of France is"` → `"Paris. It is the largest city in Europe..."` |
| Self-contained GGUF tokenizer | Fox-sentence encodes to `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]`, matches HF reference |
| Library test suite | **62/62** lib unit tests pass (`--features cuda --lib`) |
| llama.cpp FFI runner | ~218 tok/s on TinyLlama-1.1B, consistent across Q3_K_M–Q8_0 (see `pesti-runner/README.md`) |

### ⚠️ Frontier (not yet done — tracked in `ROADMAP.md`)
- **Real end-to-end decode tok/s** on the GPU path. CPU path measured at
  ~100 tok/s on Qwen2.5-0.5B (Week 14, `docs/history/WEEK_14_RESULTS.md`);
  the Week 12/13 throughput numbers below are *projections from synthetic
  micro-benchmarks*, **not** measured transformer decode.
- **GPU end-to-end path** (Week 17, in progress) — the `cudarc` dispatch
  path exists, failed GPU matmuls now fall back to CPU GEMM with a
  `gpu_fallback_count()` counter (no more silent zeroed buffers), and
  per-layer GPU capture tooling is in place. Remaining: per-layer oracle
  diff, divergence fixes, zero-fallback assertion, GPU decode tok/s.
- **KV-cache updates during autoregressive generation** (paged attention).
- FP8 quantization, multi-GPU scaling.

---

## Conformance

The full forward pass is verified against an independent pure-numpy float32
reference on `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Qwen2.5-0.5B, 24 layers,
hidden 896). Full methodology + reproduce commands live in
[`conformance-corpus/CONFORMANCE.md`](conformance-corpus/CONFORMANCE.md).

**Result (2026-08-22, Q4_K_M, 10-token "fox" prompt, last position):**

| Check | Rust (pesti) | numpy (ref) | Verdict |
|-------|--------------|-------------|---------|
| All 24 layer norms | e.g. L0 3.8732 … L23 50.5135 | L0 3.8731 … L23 50.5135 | max Δ 1e-4 ✅ |
| Pre-head norm | 298.7678 | 298.7678 | Δ 0 ✅ |
| Top-8 token ids | `[220, 1416, 3555, 2585, 1096, 576, 758, 715]` | identical | ✅ |
| Argmax | 220 | 220 | ✅ |
| **Full-vector (896 d × 24 L)** | — | — | max Δ 7.6e-5, corr 1.000000 ✅ |
| **Full logits (151,936 d)** | — | — | max Δ 7.0e-5, corr 1.000000 ✅ |

**VERDICT: PASS** — pesti's full 24-layer forward pass is numerically
conformant with the independent numpy reference to within f32 accumulation
order. The sub-1e-3 deltas are the expected difference between pesti's Rust
Q4_K dequant/accumulation order and gguf's numpy order.

The wiring that was historically buggy is now conformant end-to-end:
QKV **bias** applied, **SwiGLU** = `silu(gate) * up` (sigmoid-based SiLU, not
silu²), **RoPE** at the true token position, and per-head **GQA**
(`n_head=14`, `n_head_kv=2`).

Reproduce:
```bash
MODEL=conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf

# numpy oracle
python3 conformance-corpus/ref_forward.py "$MODEL" > /tmp/ref_all_layers.txt

# Rust dumper (same output grammar)
cargo build -p pesti-runner --release --features cuda --example dump_all_layers
./target/release/examples/dump_all_layers "$MODEL" > /tmp/rust_all_layers.txt

# head-8 / norm / top-8 / argmax diff
python3 conformance-corpus/compare_all_layers.py \
  /tmp/rust_all_layers.txt /tmp/ref_all_layers.txt --tol=1e-3

# gold-standard full-vector diff (all 896 dims × 24 layers + full logits)
python3 conformance-corpus/probe_all_layers.py "$MODEL" --out /tmp/probe_all_layers
./target/release/examples/dump_all_layers --dump /tmp/rust_probe "$MODEL"
python3 conformance-corpus/compare_full_vectors.py \
  /tmp/rust_probe /tmp/probe_all_layers --tol=1e-3
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     pesti-runner                             │
├─────────────────────────────────────────────────────────────┤
│  GGUF v3 parser + pure-Rust K-family dequantization         │
│  (Q2_K → Q8_0, byte-exact)                                  │
│              ↓                                               │
│  Self-contained GGUF-embedded tokenizer (BPE + Qwen2 regex) │
│              ↓                                               │
│  Transformer forward pass (CPU path — verified)             │
│    RMSNorm → GQA attention → RoPE → SwiGLU FFN × 24 layers  │
│              ↓                                               │
│  Output head → logits                                        │
│                                                              │
│  CUDA dispatch path (feature-gated, --features cuda)        │
│    cudarc kernels: GEMM, attention, softmax                 │
└─────────────────────────────────────────────────────────────┘
```

**Feature flags** (`pesti-runner/Cargo.toml`):
- `default` — empty (CPU parser + dequant + primitives)
- `cuda` — enables `cudarc` + the `transformer` forward-pass module
- `rust-tokenizer` — optional pure-Rust Qwen2 BPE tokenizer (`qwen2-bpe` crate)

> The **MistralRs** backend is a *tokenizer* source (it builds the real
> `tokenizers::Tokenizer` from GGUF-embedded arrays), not a separate inference
> engine. There is no `mistralrs` inference feature — `mistralrs-core` is a
> hard dependency used for tokenization only.

---

## Performance

### Measured
| Metric | Value | Source |
|--------|-------|--------|
| CPU forward-pass correctness | max per-layer Δ 7.6e-5 vs numpy | `compare_full_vectors.py` |
| llama.cpp FFI runner (TinyLlama-1.1B, CPU) | ~218 tok/s, <3% variance across Q3_K_M–Q8_0 | `pesti-runner/README.md` |

### Projected (synthetic micro-benchmarks — **not** real decode)
The Week 12/13 numbers below come from isolated kernel micro-benchmarks and
`backend.sync()` proxy timing. They do **not** represent end-to-end
transformer decode throughput. Week 14 replaces these with real measurement.

| Phase | Optimization | Projected |
|-------|--------------|-----------|
| Baseline | CPU-only inference | ~35 tok/s |
| Phase 1 | FP16 KV cache + paged allocation | ~42 tok/s |
| Phase 2 | Fused QKV+attention+output kernel | ~52-60 tok/s |
| Phase 3 | Batched parallelism + warp-level GEMM | ~88 tok/s |
| Phase 4.1 | Flash attention (shared-memory tiling) | ~105 tok/s |
| Phase 4.3 | WGMMA tensor-core GEMM | ~315 tok/s |

---

## Development Workflow

```bash
# Build the library (CPU)
cargo build -p pesti-runner --lib

# Build with CUDA (forward pass + GPU kernels)
cargo build -p pesti-runner --features cuda

# Run the library test suite
cargo test -p pesti-runner --features cuda --lib

# Full test suite (lib + integration tests)
cargo test -p pesti-runner --features cuda --no-fail-fast

# Format the whole crate (rustfmt, edition 2024)
cargo fmt -p pesti-runner

# Conformance tooling (numpy oracle + Rust dumper + diffs)
python3 conformance-corpus/ref_forward.py conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

The conformance loop is the core workflow: change a kernel or op → run
`dump_all_layers` → diff against `ref_forward.py` → confirm the per-layer
deltas stay within f32 accumulation order before committing.

---

## Project Structure

```
pesti/
├── pesti-runner/          # Main crate: parser, dequant, transformer, CUDA dispatch
│   ├── src/
│   │   ├── transformer/   #   forward pass (model, layer, rms_norm, linear, kv_cache, tokenizer)
│   │   ├── kernel/        #   CUDA kernels (PTX via include_str!) + dispatch
│   │   ├── dequantize.rs  #   pure-Rust K-family dequantization
│   │   └── gguf_weight_loader.rs
│   └── examples/          #   cpu_e2e_generate, dump_all_layers, ...
├── pesti-gguf/            # GGUF parser crate (workspace member)
├── pesti-safetensors/     # Safetensors crate (workspace member)
├── llm-plug-in/           # llama.cpp FFI runner (workspace member)
├── pesti-gguf-cli/        # GGUF CLI (workspace member)
├── crates/qwen2-bpe/      # Optional pure-Rust Qwen2 BPE tokenizer
├── conformance-corpus/    # numpy oracle + diff tools + canonical Q4_K_M model
│   ├── ref_forward.py     #   24-layer numpy reference
│   ├── probe_all_layers.py
│   ├── compare_all_layers.py
│   ├── compare_full_vectors.py
│   └── CONFORMANCE.md
└── examples-disabled/     # quarantined broken examples (git-tracked, not deleted)
```

> `pesti-conformance` is a **standalone** crate excluded from the workspace
> (`Cargo.toml` `exclude` list), not a workspace member — `cargo test
> --package pesti-conformance` will not resolve from the root.

---

## Hardware Requirements

### CPU-only (parser + dequant + forward pass)
- Any modern x86_64 or ARM64
- RAM: 8GB+
- No GPU required

### CUDA dispatch path
- NVIDIA GPU, CUDA 12.5+
- Tested: **RTX 4070 Ti SUPER** (sm_8.9, Ada) — primary dev hardware;
  **RTX 5060 Ti** (sm_12.0, Blackwell) in a dual-GPU setup

---

## Roadmap

For the full milestone-by-milestone plan (including engineering lessons and
known gaps), see [`ROADMAP.md`](ROADMAP.md).

### Completed
- [x] GGUF v3 parsing + byte-exact K-family dequantization
- [x] CPU transformer forward pass, **24-layer conformant vs numpy**
- [x] Self-contained GGUF-embedded tokenizer
- [x] CUDA dispatch path (feature-gated)
- [x] Conformance tooling (numpy oracle + full-vector diffs)

### Next (frontier)
- [ ] GPU end-to-end correctness + decode tok/s (Week 17, in progress)
- [ ] Measured llama.cpp baseline on same model/prompt/hardware (Week 14 remainder)
- [ ] KV-cache updates during autoregressive generation (paged attention)
- [ ] FP8 quantization, multi-GPU scaling
- [ ] Contribute back to llama.cpp / candle / burn

---

## License

**AGPL-3.0-or-later**
*Open-source, copyleft, designed for learning and contribution.*

---

*Last updated: August 25, 2026 (Week 13/14 reconciled; Week 17 GPU e2e correctness in progress)*
*This README will change as I learn more. If it looks perfect, it's lying.*
