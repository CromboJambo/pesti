# PESTI Build Session Spec: GPU Decode Measurement & Bottleneck Analysis

## Context (for fresh agent)

PESTI is a Rust LLM inference engine with CUDA backend. The CPU path is fully correct and numerically verified against numpy oracle. The GPU path has just completed end-to-end correctness work (Week 17):

- Per-layer GPU vs numpy oracle diffing: PASS (f16 tensor-core rounding, not bugs)
- Long-sequence attention tests (seq=64 to 4096): PASS with numerical conformance checks
- Autoregressive KV cache tests: PASS (10/10 token match across all steps)
- Zero GPU fallbacks on full forward pass

**What's missing:** Real tok/s measurements and bottleneck analysis. All previous numbers were projections or CPU-only.

## Session Goals

### Goal 1: Measure GPU Decode Tok/s (Week 14 deliverable)
Run the existing `examples/week14_e2e_decode.rs` benchmark on the GPU path with a real model.

**Steps:**
1. Build release binary: `cargo build --release --features cuda`
2. Run decode benchmark on Qwen2.5-0.5B-Instruct-Q4_K_M (same model used for CPU baseline)
3. Measure tokens/second for autoregressive generation (not just prefill)
4. Compare against existing CPU baseline: ~100 tok/s (CPU path, Qwen2.5-0.5B)

**Expected output:** Real tok/s number for GPU decode path

### Goal 2: Run llama.cpp Baseline on Same Model/Hardware
Get a measured (not estimated) llama.cpp baseline for comparison.

**Steps:**
1. Ensure llama.cpp is built (or build it): `cd /home/crombo/projects/llama.cpp && make`
2. Run same model with same prompt: `./main -m conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "The quick brown fox jumps over the lazy dog." -n 64`
3. Record tok/s from llama.cpp's built-in benchmark output

**Expected output:** Measured llama.cpp tok/s on identical hardware/model/prompt

### Goal 3: Profile & Identify Top-3 Bottlenecks
Use manual sync-timing profiling (Week 13 approach) to identify where time is spent.

**Steps:**
1. Instrument the GPU decode path with timing around major operations:
   - Weight loading/dequantization per layer
   - Attention kernel execution
   - FFN layer execution
   - H2D/D2H memory transfers
   - KV cache updates
2. Run with seq_len=64 (decode phase only, not prefill)
3. Identify top 3 time consumers

**Expected output:** Top-3 bottleneck statement with percentages

## Technical Details

### Hardware
- GPUs: RTX 4070 Ti SUPER + RTX 5060 Ti (32GB VRAM total)
- Both GPUs shared with other processes (Unsloth llama-server resident)
- Use `PESTI_KV_MAX_SEQ` to cap KV allocation if needed
- Expect fallback counter to be non-zero under contention

### Model
- Primary: `conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf` (same as CPU baseline)
- Fallback: Any Q4_K_M quantized model that fits in available VRAM

### Key Files
- Benchmark example: `pesti-runner/examples/week14_e2e_decode.rs`
- GPU dispatch: `pesti-runner/src/kernel/dispatch.rs`
- CUDA runtime: `pesti-runner/src/cuda_runtime.rs`
- KV cache: `pesti-runner/src/kernel/kvcache.rs`

### Known Issues to Watch For
- OOM when both GPUs occupied by other processes (env issue, not code regression)
- `cuStreamSynchronize` is unreliable under `CU_CTX_SCHED_AUTO` — use event-based sync (`cuda_shim::stream_synchronize`)
- GPU vs CPU divergence grows smoothly with depth (f16 accumulation rounding) — expected behavior

## Deliverables for This Session

1. Real tok/s measurement for GPU decode path (write to `docs/history/WEEK_14_RESULTS.md` or new file)
2. Measured llama.cpp baseline on same model/hardware/prompt
3. Top-3 bottleneck statement with evidence from profiling
4. Updated ROADMAP.md reflecting actual measurements vs projections

## Success Criteria

- [ ] GPU decode tok/s measured and documented (real number, not projection)
- [ ] llama.cpp baseline measured on identical conditions
- [ ] Clear statement of top 3 bottlenecks with supporting data
- [ ] Evidence-based roadmap updates (no more "estimated ~1.4×")

## Notes for Future Sessions

After this measurement sprint, the next logical step is Phase 5: Slow-Friend Substrate (G1-G5 gates). The drift signal probing tooling already exists from Week 17 work (`capture_per_layer`, `dump_all_layers_gpu.rs`, numpy-oracle diff) and can be repurposed for G1.

The GPU path is now correct — it's time to measure, optimize, and then explore the Slow-Friend architecture direction.