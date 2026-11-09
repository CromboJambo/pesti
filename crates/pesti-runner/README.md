# pesti-runner - High-Performance LLM Inference Engine

Portable Execution Substrate for Transformer Inference with consistent performance across quantization levels. Pure Rust with CUDA dispatch via cudarc.

## Performance Characteristics

**Measured on Qwen2.5-0.5B-Instruct-Q4_K_M (RTX 3070 Ti):**

| Runner | Speed (tok/s) | Notes |
|--------|---------------|-------|
| pesti-runner (F16 compute, Phase 1+2b) | **307.68** | cuBLAS Hgemm, trait-based dispatch |
| llama.cpp (baseline) | 504.04 | Reference implementation |

Phase 1 target of 100 tok/s achieved. Remaining ~1.6x gap is Phase 4 optimization target.

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    pesti-runner                              │
├─────────────────────────────────────────────────────────────┤
│  Rust Application Layer                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ GGUF v3 Parser + K-family Dequantization            │   │
│  │ - Q2_K through Q8_0 (pure Rust, byte-exact)         │   │
│  │ - Self-contained tokenizer from GGUF-embedded data   │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Transformer Forward Pass (24 layers)                │   │
│  │ - RMSNorm → GQA Attention → RoPE → SwiGLU FFN      │   │
│  │ - KV cache with autoregressive generation           │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ CUDA Dispatch Layer (cudarc)                        │   │
│  │ - GEMM via cuBLAS Hgemm (F16 compute)              │   │
│  │ - Fused attention kernel (QKV+attn+output)          │   │
│  │ - Non-matmul kernels: SwiGLU, RMSNorm, RoPE, Softmax│   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### The Problem: FFI Overhead and Precision Loss

Naive GPU inference wraps C libraries with per-token FFI calls:
```rust
for token in tokens {
    // ❌ One FFI crossing per token, F32→F16 conversion overhead
    llama_decode(&mut ctx, 1);
}
```

### The Solution: Trait-Based Dispatch with F16 Compute

PESTI Runner uses a trait-based dispatch layer that minimizes FFI crossings and performs true F16 compute on the GPU:
```rust
// ✅ Build once, run many decode steps
let layer = new_linear_layer(&weights, device)?;
for token in generated_tokens {
    let logits = model.forward_step(token, &layer)?;  // F16 throughout
}
```

**Result:** True half-precision GPU compute with automatic CPU fallback for environments without CUDA.

## Benchmark Configuration

- **Model**: Qwen2.5-0.5B-Instruct-Q4_K_M
- **Hardware**: RTX 3070 Ti (sm_86)
- **Backend**: cudarc with cuBLAS Hgemm
- **Compute Precision**: F16 on device, F32 accumulation

## Usage Example

```rust
use pesti_runner::transformer::model::load_model;
use pesti_runner::kernel::dispatch::new_linear_layer;

let model_path = "/path/to/model.Q4_K_M.gguf";
let (model, tokenizer) = load_model(model_path)?;

// Build dispatch layer once
let device = Device::cuda_device(0);
let linear = new_linear_layer(&model.layers[0].w1, &device)?;

// Run inference
let tokens = tokenizer.encode("Explain quantum computing...")?;
let response = model.generate(tokens, &linear, 500)?;
println!("{}", tokenizer.decode(&response));
```

## Performance Insights

### Why F16 Compute Matters

1. **Memory bandwidth**: F16 tensors are half the size of F32, reducing H2D/D2H transfer costs
2. **Tensor core utilization**: cuBLAS Hgemm uses FP16 tensor cores for ~2x throughput vs SGEMM
3. **Numerical equivalence**: Attention scores and logits match F32 reference to within f32 accumulation order

### Phase 1 Results (Week 25)

| Metric | Before F16 | After F16 | Improvement |
|--------|------------|-----------|-------------|
| Decode tok/s | 81.78 | 307.68 | **3.76x** |
| vs llama.cpp baseline | 6x slower | 1.6x slower | Gap reduced 73% |

## Running Benchmarks

```bash
# Build with CUDA support
cargo build -p pesti-runner --features cuda --release

# Run end-to-end GPU generation
cargo run -p pesti-runner --features cuda --release \
  --example gpu_e2e_generate \
  -- conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 48

# Run CPU-only path (no GPU required)
cargo run -p pesti-runner --release \
  --example cpu_e2e_generate \
  -- conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf "The capital of France is" 48
```

## Comparison with Alternatives

| Runner | Speed (tok/s) | Notes |
|--------|---------------|-------|
| llama.cpp (CPU, naive) | ~70 | Per-token FFI overhead |
| Python bindings | ~50-60 | High overhead, GIL contention |
| pesti-runner (CPU path) | ~218 | Chunked batching, no FFI |
| **pesti-runner (GPU, F16)** | **307.68** | cuBLAS Hgemm, trait dispatch |
| llama.cpp (GPU baseline) | 504.04 | Reference for comparison |

## Known Limitations

1. **Phase 3 complete**: Non-matmul GPU kernels (SwiGLU, RMSNorm, RoPE, Softmax) implemented and integrated
2. **RoPTX version compatibility**: RoPE computed in pure Rust to avoid PTX version issues across drivers
3. **Remaining gap**: ~1.6x slower than llama.cpp GPU baseline; Phase 4 will optimize kernel fusion and memory layout

## Resources

- **Source**: [`pesti-runner/src/`](./src/)
- **CUDA Kernels**: [`pesti-runner/src/kernel/`](./src/kernel/)
- **Transformer Implementation**: [`pesti-runner/src/transformer/`](./src/transformer/)
- **Conformance Corpus**: [`conformance-corpus/`](../conformance-corpus/)

## License

AGPL-3.0-or-later (see root `LICENSE`)

---

*Built with ❤️ by PESTI Contributors*
*Last Updated: October 2, 2026 — Week 25, F16 compute validated at 307.68 tok/s*
