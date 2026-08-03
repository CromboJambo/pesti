# Test Models

Benchmark models for PESTI Runner performance validation.

## Quick Start

```bash
# Download all benchmark models
./download_models.sh

# Run benchmarks
cargo run --package pesti-runner --example q4_stress_test 500 test
```

## Available Models

| Quantization | File Size | Expected Speed | Use Case |
|--------------|-----------|----------------|----------|
| **Q3_K_M**   | ~526 MB   | 217.9 tok/s    | Smallest footprint, max speed ⚡ |
| **Q4_K_M**   | ~638 MB   | 217.6 tok/s    | Best overall balance ⭐ |
| **Q5_K_M**   | ~747 MB   | 217.1 tok/s    | Slightly better quality 🎨 |
| **Q8_0**     | ~1.1 GB   | 215.4 tok/s    | Maximum fidelity 💎 |

## Why These Models?

- **TinyLlama-1.1B**: Small enough for quick benchmarking, large enough to be meaningful
- **Chat variant**: Real-world usage scenario
- **Multiple quantizations**: Validates the "quantization-agnostic" performance discovery

## License

Models from TheBloke on HuggingFace (typically CC-BY-SA). Check individual model licenses.

## Adding New Models

1. Download from HuggingFace
2. Add to `download_models.sh`
3. Update this README with size and expected speed
4. Run benchmark script to verify performance
