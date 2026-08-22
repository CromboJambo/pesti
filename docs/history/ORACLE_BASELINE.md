# Oracle Baseline — llama.cpp reference for coherence diff

**Purpose:** Known-good token sequence to diff against PESTI output. The forward-pass
coherence bug (input-independent constant token) is now measurable, not a mystery.

**Oracle build:** llama.cpp `bb4caa7` (2026-08-21), built from source against system
CUDA 13.3, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`.
- Binary: `/tmp/llama.cpp/build/bin/llama-server` (also `llama-cli`)
- Note: `/tmp` is ephemeral — rebuild with:
  ```
  cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build build --target llama-cli llama-server -j20
  ```
- The pre-existing `~/.local/bin/llama-cli` and the LM Studio CUDA12 backend are both
  broken (missing `libggml-cuda.so` / `libcudart.so.12`) — do NOT use them.

**Model:** `conformance-corpus/qwen2.5-0.5b-instruct-q8_0.gguf` (638 MiB, Q8_0, 24 layers,
n_head=14, n_head_kv=2, embd=896, ffn=4864, ctx=32768, rope_base=1e6, rms_eps=1e-6)

**Exact run config (matches pesti `coherence_check.rs`):**
- prompt: `The quick brown fox jumps over the lazy dog.`  (raw, NO chat template, NO BOS)
- prompt token IDs: `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]` (10 tokens)
- temperature 0.0, top_k 40, top_p 0.9, seed 42, n_predict 32, ignore_eos

**How to reproduce (server on :8090):**
```
curl -s http://127.0.0.1:8090/completion -d '{
  "prompt":"The quick brown fox jumps over the lazy dog.",
  "temperature":0.0,"top_k":40,"top_p":0.9,"n_predict":32,
  "seed":42,"ignore_eos":true,"logprobs":true}'
```
Token IDs are under `completion_probabilities[].id`. Verified deterministic (2 runs identical).

## Oracle GEN_TOKEN_IDS (32 tokens) — THE REFERENCE
```
[2585, 1657, 14201, 525, 1052, 304, 2790, 304, 279, 2701, 8500, 315, 4357, 30,
 715, 16, 13, 576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 624, 17,
 13, 576, 3974, 13876]
```
Decoded (coherent):
```
 How Many legs are there in total in the following sequence of events? \n1. The quick brown
```

## PESTI output (buggy, pre-fix state)
First token: **127338** (input-independent constant, same on CPU and GPU).
First 5: `[127338, 145690, 15338, 80234, 9834]` — word salad, identical CPU vs GPU.

## Diff at a glance
| pos | oracle | pesti |
|-----|--------|-------|
| 0   | 2585 (" How") | 127338 (garbage) |
| 1   | 1657 (" Many") | 145690 |
| 2   | 14201 (" legs") | 15338 |

**Interpretation:** pesti's logits ignore the input entirely (constant first token), so the
bug is upstream of the CPU/GPU attention split — embedding lookup, position encoding, or
output-head layout are the prime suspects. The identical first-5 tokens across CPU+GPU
confirm it is not in the CUDA kernels.
