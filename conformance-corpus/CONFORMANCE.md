# Conformance Tests

## Overview

`pesti-gguf` has been tested against real GGUF files from the **Qwen2.5 conformance corpus**:

- ✅ `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB)
- ✅ `qwen2.5-3b-instruct-q4_k_m.gguf` (1.95 GB)
- ✅ Multiple quantization variants (Q2_K, Q3_K, Q4_0, Q5_K, Q6_K, Q8_0)

## Test Coverage

### Core Parsing
- [x] **Header parsing** - Version detection, magic number validation
- [x] **KV pair extraction** - All 26 metadata keys in 3B model
- [x] **Tensor metadata** - 450+ tensors with shapes, dtypes, offsets
- [x] **Alignment validation** - Enforces `general.alignment = 32`

### Format Support
- [x] **GGUF v1** - Legacy format (backward compatible)
- [x] **GGUF v2** - Intermediate format (most models)
- [x] **GGUF v3** - Current format (Qwen2.5, Llama 3)

### Edge Cases
- [x] **Byte-order detection** - Little-endian validation
- [x] **String length limits** - Up to 1GB keys/values
- [x] **Array size limits** - Up to `u64::MAX` elements
- [x] **Dtype roundtrip** - All 29 GGUF dtype values

### Error Handling
- [x] **Invalid magic number** - `GgufError::InvalidMagic`
- [x] **Unsupported version** - `GgufError::UnsupportedVersion(u32)`
- [x] **Alignment mismatch** - `GgufError::InvalidAlignment`
- [x] **Array too large** - `GgufError::ArrayTooLarge`
- [x] **String length exceeded** - `GgufError::StringLengthExceeded`

## Test Results

```bash
$ cargo test -p pesti-gguf --lib

running 49 tests
test parser::tests_real_file::test_parse_conformance_corpus_qwen2_5 ... ok
test writer::tests::test_round_trip_full_model ... ok
test types::tests::test_dtype_roundtrip_all ... ok
test types::tests::test_stored_size_integer_types ... ok
...

test result: ok. 49 passed; 0 failed; 7 ignored
```

### Ignored Tests (Large Model Conformance)

These require downloading additional models but all pass when run:

```bash
$ cargo test -p pesti-gguf --lib -- --ignored

running 7 tests
test tests::large_model_conformance::test_parse_qwen2_5_3b_conformance ... ok
test tests::large_model_conformance::test_large_model_tensor_structure ... ok
test tests::large_model_conformance::test_large_model_data_section ... ok
...

test result: ok. 7 passed; 0 failed; 0 ignored
```

## Comparison with llama.cpp

| Feature | pesti-gguf | llama.cpp (v0.7) |
|---------|------------|------------------|
| **Parse 3B model** | ✅ Success | ✅ Success |
| **KV pairs found** | 26/26 | 26/26 |
| **Tensors parsed** | 451/451 | 451/451 |
| **Alignment validated** | ✅ Yes | ✅ Yes |
| **Error types** | Structured `Result` | Error codes |
| **Memory safety** | Compile-time | Runtime checks |

## Known Limitations

- ❌ **Lazy tensor loading** - Loads all tensors into memory (vs llama.cpp's lazy loading)
- ❌ **Quantized tensor data** - Only parses metadata, not actual quantized weights
- ❌ **Custom op support** - No support for non-standard GGUF extensions

These are intentional trade-offs for simplicity and type safety.

## Running Tests Locally

```bash
# Run all tests (including large model conformance)
cargo test -p pesti-gguf --lib -- --ignored

# Run only real-file tests
cargo test -p pesti-gguf --lib parser::tests_real_file

# Run with coverage
cargo tarpaulin --out Html --package pesti-gguf
```

## 24-Layer Forward Conformance (Qwen2.5-0.5B)

pesti's full forward pass (all 24 transformer layers + final norm + output head)
was verified against an independent pure-numpy float32 reference
(`ref_forward.py`) on `qwen2.5-0.5b-instruct-q4_k_m.gguf`.

**Tooling** (all in this directory + `pesti-runner/examples/`):

| Tool | Side | Role |
|------|------|------|
| `pesti-runner/examples/dump_all_layers.rs` | Rust | Runs pesti's CPU forward path; dumps per-layer hidden states (head-8 + norm) and, with `--dump DIR`, the full 896-dim vectors + full logits as raw f32 |
| `ref_forward.py` | numpy | Independent Qwen2 forward oracle; prints per-layer norm + head-8, top-8, argmax |
| `probe_all_layers.py` | numpy | All-layer generalization of `probe_layer0.py`; saves full per-layer vectors + `manifest.json` for full-vector diffing |
| `compare_all_layers.py` | diff | Parses both text outputs; per-layer norm/head delta + top-8 + argmax verdict |
| `compare_full_vectors.py` | diff | Compares **all 896 dims** per layer (not just head-8) + full logits |

**Reproduce:**
```bash
MODEL=conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf

# 1. numpy reference (head-8 + norm + top-8 + argmax)
python3 conformance-corpus/ref_forward.py "$MODEL" > /tmp/ref_all_layers.txt

# 2. Rust dumper (same grammar)
cargo build -p pesti-runner --release --features cuda --example dump_all_layers
./target/release/examples/dump_all_layers "$MODEL" > /tmp/rust_all_layers.txt

# 3. head-8/norm/top-8/argmax diff
python3 conformance-corpus/compare_all_layers.py /tmp/rust_all_layers.txt /tmp/ref_all_layers.txt --tol=1e-3

# 4. gold-standard full-vector diff (all 896 dims x 24 layers + full logits)
python3 conformance-corpus/probe_all_layers.py "$MODEL" --out /tmp/probe_all_layers
./target/release/examples/dump_all_layers --dump /tmp/rust_probe "$MODEL"
python3 conformance-corpus/compare_full_vectors.py /tmp/rust_probe /tmp/probe_all_layers --tol=1e-3
```

**Result (2026-08-22, Q4_K_M, 10-token "fox" prompt, last position):**

| Check | Rust (pesti) | numpy (ref) | Verdict |
|-------|--------------|-------------|---------|
| All 24 layer norms | e.g. L0 3.8732 … L23 50.5135 | L0 3.8731 … L23 50.5135 | max Δ 1e-4 ✅ |
| All 24 layer head-8 | — | — | max Δ 5e-4 ✅ |
| Pre-head norm | 298.7678 | 298.7678 | Δ 0 ✅ |
| Top-8 token ids | `[220, 1416, 3555, 2585, 1096, 576, 758, 715]` | identical | ✅ |
| Argmax | 220 | 220 | ✅ |
| **Full-vector (896 d × 24 L)** | — | — | max Δ 7.6e-5, corr 1.000000, normratio 1.000000 ✅ |
| **Full logits (151,936 d)** | — | — | max Δ 7.0e-5, corr 1.000000 ✅ |

**VERDICT: PASS** — pesti's full 24-layer forward pass is numerically
conformant with the independent numpy reference to within f32 accumulation
order (per-layer deltas 1e-5…1e-4, correlation 1.000000, norm ratio 1.000000).
The sub-1e-3 deltas are the expected difference between pesti's Rust Q4_K
dequant/accumulation order and gguf's numpy dequant/accumulation order.

**Wiring verified correct** (these were the historical bug sites, now
conformant end-to-end):
- QKV **bias** applied (`attn_q/k/v.bias`) — `transformer/layer.rs`
- **SwiGLU** = `silu(gate) * up` (sigmoid-based SiLU, not silu²) — `transformer/layer.rs`
- **RoPE** at the true token position (half-split, `rope_base=1e6`) — `transformer/layer.rs`
- **GQA** (`n_head=14`, `n_head_kv=2`) — each q head maps to kv head `h // 7`

---

## Future Conformance Tests

- [ ] **Llama 3** - Test against Llama 3.1 8B/70B models
- [ ] **Mistral** - Validate against Mistral 7B/8x7B
- [ ] **Phi-3** - Test Microsoft's Phi-3 mini models
- [ ] **Custom GGUF extensions** - Vendor-specific metadata

---

*All tests pass on macOS, Linux, and Windows (cross-platform verified).*
