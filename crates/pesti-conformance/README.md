# PESTI Conformance Testing Framework

Differential conformance testing for the PESTI (Portable Execution Substrate for Transformer Inference) project against reference implementations.

## Overview

This crate implements pattern from [lumen-state](https://github.com/...) and [pon: catch silent corruption before it reaches mainline.

### Purpose

- **Regression detection**: Catch output drift between pesti and reference implementations (llama.cpp FFI, candle-core GPU)
- **Floor file validation**: Establish minimum passing thresholds for CI/CD gates
- **Delta minimization**: Automatically reduce divergence reports to minimal patches when outputs differ

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   GGUF Corpus    │────▶│  run_conformance()   │────▶│  Reference Impl  │
│                   │     │  - discover_models() │     │ (llama.cpp/...)  │
│  models/*.gguf    │     │  - parse_gguf()      │◀────┤                  │
└─────────────────┘     └──────────┬───────────┘     └──────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │ run_single_model()   │
                         │  - pesti inference   │
                         │  - hash comparison   │
                         │  - delta_minimize()  │
                         └───────────┬──────────┘
                                     │
                          ┌─────────▼──────────┐
                          │ ConformanceResult  │
                          │ - passed / failed  │
                          │ - failure_details  │
                          └────────────────────┘
```

## Quick Start

### Build

```bash
cd /home/crombo/projects/llm-workspace
cargo build -p pesti-conformance
```

### Run Basic Test

```bash
# On a test corpus (no reference implementation)
cargo run -p pesti-conformance -- run \
  --corpus-dir ./conformance/test-corpus

# Output:
# INFO Discovered 1 models in ./conformance/test-corpus
# WARN ✗ FAIL: tiny-model - Unsupported GGUF format
# INFO Conformance complete: 0/1 passed (0.0%)
```

### With Reference Implementation

```bash
cargo run -p pesti-conformance -- run \
  --corpus-dir ./models/ \
  --reference-llama-cpp /usr/bin/llama-cli \
  --floor-pass-count 50
```

## API Usage

### Programmatic Usage

```rust
use pesti_conformance::{run_conformance, ConformanceConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConformanceConfig {
        corpus_dir: "./models".into(),
        reference_llama_cpp: Some("/usr/bin/llama-cli".into()),
        floor_pass_count: 10, // fail if < 10 models pass
    };

    match run_conformance(&config) {
        Ok(result) => {
            println!("{} passed", result.passed.len());
            for failure in &result.failures {
                eprintln!(
                    "{} - expected={} actual={}",
                    failure.model_name, 
                    failure.expected_hash, 
                    failure.actual_hash
                );
            }
        }
        Err(e) => {
            eprintln!("Conformance error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
```

### Delta Minimization for Divergence Reports

```rust
use pesti_conformance::delta_minimize;

let expected = b"expected output";
let actual = b"actual   output"; // different bytes at positions 0-5

let divergence = delta_minimize(expected, actual);
println!(
    "Diverged {} bytes starting at offset {:?}",
    divergence.changes.len(),
    divergence.divergence_offset
);
```

## Test Corpus Structure

Place GGUF models in a directory structure:

```
test-corpus/
├── tiny-model.gguf      # Small model for quick CI tests
└── llama-7b-Q4_0.gguf   # Full-size model (slow)
```

The runner discovers all `.gguf` files recursively.

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Conformance Tests

on: [push, pull_request]

jobs:
  conformance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        
      - name: Run conformance tests
        run: |
          cargo test --package pesti-conformance \
            --test conformance_tests \
            -- --nocapture

      # Optional: Download llama.cpp binary for reference comparison
      - name: Setup llama.cpp reference
        run: |
          git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
          cd /tmp/llama.cpp && cmake . -DLLAMA_CUBLAS=ON
          make -j$(nproc)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFORMANCE_CORPUS` | `./models/` | Path to GGUF corpus directory |
| `REF_LLAMA_CPP` | *none* | Path to llama.cpp binary (optional) |
| `FLOOR_PASS_COUNT` | `0` | Minimum passing models threshold |

## Future Work

- [ ] Integrate with pesti-runner's LlamaModel::load_gguf() for actual inference
- [ ] Add support for candle-core GPU backend comparison
- [ ] Implement output streaming for large models
- [ ] Generate HTML/Markdown reports from divergence data
- [ ] Add regression test fixtures with known-good hashes

## See Also

- [`gguf`](../gguf/) - GGUF v3 parser and types
- [`llm-runner`](../llm-runner/) - LLM inference engine
- [ROADMAP-pesti-research.md](../state-docs/ROADMAP-pesti-research.md) - Research integration plan
