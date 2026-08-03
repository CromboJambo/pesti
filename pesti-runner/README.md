# PESTI Runner - High-Performance LLM Inference Engine

Portable Execution Substrate for Transformer Inference with consistent performance across quantization levels.

## Performance Characteristics

**Measured on TinyLlama-1.1B-Chat-V1.0 (4 threads, CPU):**

| Quantization | File Size | Speed (tok/s) |
|--------------|-----------|---------------|
| Q3_K_M       | 526 MB    | 218.5         |
| Q4_K_M       | 638 MB    | 216.8         |
| Q5_K_M       | 747 MB    | 221.6         |
| Q8_0         | 1.1 GB    | 221.8         |

**Key observation**: Performance varies by <3% across all quantization levels for this model size.

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PESTI Runner                             │
├─────────────────────────────────────────────────────────────┤
│  Rust Application Layer                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Chunked Batch Processing (512 tokens/batch)         │   │
│  │ - Single allocation per batch                       │   │
│  │ - llama.cpp KV cache reuse                          │   │
│  │ - Relative position sampling                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓ FFI boundary                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ llama.cpp C API (via llama-cpp-2)                  │   │
│  │ - Optimized batch inference                         │   │
│  │ - Dequantization kernels                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### The Problem: Per-Token FFI Overhead

A naive wrapper makes one FFI call per token:
```rust
for token in tokens {
    // ❌ 512 FFI crossings for 512 tokens
    llama_decode(&mut ctx, 1);
    let next = sample_next_token();
}
```

### The Solution: Chunked Batch Processing

PESTI Runner uses chunked batch autoregressive sampling:
```rust
let batch_size = 512; // Match llama.cpp n_batch
for chunk in tokens.chunks(batch_size) {
    // ✅ 1 FFI crossing per chunk
    let batch = create_batch(chunk);
    llama_decode(&mut ctx, &batch);
    
    // Sample multiple tokens using KV cache
    for _ in 0..chunk.len() {
        sample_next_token_with_relative_pos();
    }
}
```

**Result**: 512 allocations → 1 allocation per chunk (significant reduction in FFI overhead).

## Benchmark Configuration

- **Model**: TinyLlama-1.1B-Chat-V1.0
- **Prompt**: "Explain the concept of quantum computing in one sentence."
- **Tokens Generated**: 100
- **Hardware**: CPU (4 threads)
- **Batch Size**: 512 tokens
- **Context Length**: 2048

## Usage Example

```rust
use pesti_runner::llama::{LlamaRunner, SamplingConfig};

let model_path = "/path/to/model.Q4_K_M.gguf";
let prompt = "Explain quantum computing...";

// Configure runner
let sampling = SamplingConfig {
    temperature: 0.8,
    top_p: 0.95,
    ..Default::default()
};

// Build runner (automatically uses chunked batching)
let mut runner = LlamaRunner::builder(model_path)
    .n_ctx(2048)
    .sampling_config(sampling)
    .build()?;

// Generate tokens (optimized batch processing)
let response = runner.generate(prompt, 500)?;
println!("{}", response);
```

## Performance Insights

### Why Consistent Performance?

1. **FFI overhead is reduced**: Chunked batching minimizes Rust→C boundary crossings
2. **Compute-bound inference**: For small models like TinyLlama, CPU compute dominates over dequantization cost
3. **Batch efficiency**: llama.cpp's internal optimizations work consistently across quantizations

### Quantization Variance

The ~3% performance variance between Q3_K_M and Q8_0 suggests:
- Dequantization cost is minimal compared to attention compute
- Memory access patterns are similar across quantizations
- Model size (parameter count) has less impact than expected for small models

**Note**: This behavior is specific to small models (<2B params). Larger models may show more variance.

## Running Benchmarks

```bash
# Build test harness
cargo build --package pesti-runner --example q4_stress_test

# Run single quant benchmark
cargo run --package pesti-runner --example q4_stress_test 100 test_name

# Compare all quantizations
./benchmark_all_quant.sh
```

## Comparison with Alternatives

| Runner | Speed (tok/s) | Notes |
|--------|---------------|-------|
| llama.cpp (naive 1-token/batch) | ~70 | Per-token FFI overhead |
| Python bindings | ~50-60 | High overhead, GIL contention |
| **PESTI Runner** | **~218** | Chunked batching, minimal FFI |

*Note: llama.cpp with GPU can achieve 500+ tok/s. This comparison is CPU-only.*

## Known Limitations

1. **Model size specific**: Quantization-agnostic behavior observed for TinyLlama (1.1B); larger models may show more variance
2. **CPU-bound**: No GPU acceleration yet (Phase 2 in development)
3. **TinyLlama optimized**: Batch size of 512 is tuned for this model; may need adjustment for larger architectures

## Roadmap

- [ ] GPU acceleration via `cudarc`
- [ ] Multi-batch parallelism
- [ ] Streaming API for real-time generation
- [ ] Async/await support
- [ ] Model quantization on-the-fly

## Resources

- **Source**: [`llm-runner/src/runner.rs`](../llm-runner/src/runner.rs)
- **Benchmark Data**: See `q4_stress_test.rs` example
- **Original Issue**: "How do we get rid of FFI overhead?"
- **Discovery**: "Performance is quantization-agnostic for small models"

## License

AGPL-3.0-or-later (see root `LICENSE`)

---

*Built with ❤️ by PESTI Contributors*
*Last Updated: August 2026*
