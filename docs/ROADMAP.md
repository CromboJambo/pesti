# PESTI Roadmap — Evidence-Based (September 2026)

## Current State

PESTI has a working GPU inference path for Qwen2.5-0.5B-Instruct with:
- Fused attention kernel passing numerical conformance vs llama.cpp
- KV cache autoregressive validation suite
- Real tokenizer integration (qwen2-bpe crate, 50k vocab)
- Long sequence support verified to seq=4096

**Known broken:** Several examples don't compile after recent API changes. pesti-safetensors has 4 failing tests (Q4_K/Q5_K/Q6_K dequant + config extraction).

## Completed Milestones

### GPU Kernel Integration
- [x] CUDA dispatch system with LayerDispatch routing
- [x] Fused attention kernel via candle_bridge
- [x] Numerical stability fixes for long sequences (softmax overflow)
- [x] KV cache conformance validation suite
- [x] Weight conversion and decode-step profiling examples

### Tokenizer
- [x] Real qwen2-bpe tokenizer integrated from GGUF metadata
- [x] Fallback tokenizer for non-GGUF models
- [x] Feature flag `rust-tokenizer` for optional BPE support

## Remaining Work

### Week 18: Stabilize and Measure (✅ Complete)
- [x] Fix broken examples (`tokenize.rs`, `encode_tokens.rs`, `decode_tokens.rs`) — commit 1651549
- [x] Establish CPU vs GPU throughput benchmarks (tok/s) with real measurements
- [x] Document VRAM usage characteristics (docs/benchmarks/VRAM-USAGE.md)
- [x] Decide on pesti-safetensors failing tests (accept as known issues; mostly pass)

### Week 19: Optimization Pass
- Profile attention kernels under nvprof/nsys
- Measure GEMM vs naive implementation speedup
- Identify memory bandwidth bottlenecks
- Document optimization opportunities with concrete numbers

## Key Lessons Learned

### CUDA Gotchas
- **cuStreamSynchronize is a no-op** under CU_CTX_SCHED_AUTO — always sync via cuda_shim::stream_synchronize (event-based) or context_synchronize for legacy-default-stream work
- Stack array sizes in kernels are bounds, not dynamic allocation — `MAX_SEQ = 512` as stack size but indexing past it causes silent corruption
- Dynamic GPU memory allocation (`cudaMalloc`) is cheap; don't over-optimize by avoiding it

### Development Process
- Examples rot quickly after API changes — treat example compilation as part of the test suite
- Conformance testing must be done incrementally: kernel → layer → model, not end-to-end only
- Numerical stability issues (softmax overflow) only manifest at long sequences — test with seq=4096+

### Architecture Decisions Validated
- Candle bridge approach works for GPU inference without writing raw CUDA kernels
- KV cache validation via autoregressive generation catches subtle bugs that unit tests miss
- Real tokenizer integration is straightforward once GGUF metadata parsing works

## Future Considerations (Unvalidated)

These are potential directions not yet explored:
- Batched generation for parallel prompts
- KV cache quantization (Q4_K)
- TMA descriptors for async prefetching
- Structured logging for dispatch decisions

Each should be validated with a spike before committing to implementation.

---
*Updated: September 8, 2026 — based on git history and test results, not planning documents*
