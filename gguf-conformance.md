# GGUF Conformance Guide

## Overview

This guide helps users debug GGUF parsing errors based on real-world experience from building a production-grade GGUF parser ([PESTI](https://github.com/nousresearch/pesti)).

When llama.cpp fails to load a model, the error messages can be cryptic. This guide walks you through systematic debugging steps.

---

## Common Error Patterns

### 1. "Failed to open GGUF file"

**Symptoms**:
```
llama_model_loader: loaded meta data
llm_load_vocab: special_eos_id is not in special_eog_ids
llm_load_tensors: ggml ctx size = 0.00 MiB
llm_load_tensors: failed to open '/path/to/model.gguf'
```

**Root Cause**: File exists but llama.cpp can't read metadata section.

**Debug Steps**:
1. Verify file integrity: `sha256sum model.gguf`
2. Check GGUF version: Use [`gguf-py`](https://github.com/ggml-org/gguf-py) to inspect:
   ```bash
   pip install gguf
   python -c "import gguf; m=gguf.GGUFReader('model.gguf'); print(m.header)"
   ```
3. If metadata is corrupted, the file may need re-quantization

---

### 2. "Architecture-specific tensor naming"

**Symptoms**:
```
llm_load_tensors: tensor 'blk.0.attn_q.weight' not found in model
llm_load_tensors: expected key blk.0.attention.weight but found blk.0.wq.weight
```

**Root Cause**: Different architectures use different tensor naming conventions:

| Architecture | Attention Q Weight | Feed-Forward Up |
|--------------|-------------------|-----------------|
| Llama/Mistral | `blk.%d.attn_q.weight` | `blk.%d.ffn_up.weight` |
| Gemma | `blk.%d.attention.q_norm.weight` | `blk.%d.feed_forward.gate_proj.weight` |
| Qwen2/Qwen3 | `blk.%d.attn_w.weight` | `blk.%d.ffn_gate.weight` |
| DeepSeek | `blk.%d.self_attn.q_proj.weight` | `blk.%d.mlp.gate_proj.weight` |

**Debug Steps**:
1. List all tensors in the GGUF:
   ```bash
   python -c "import gguf; m=gguf.GGUFReader('model.gguf'); print('\n'.join(m.tensor_names))" | grep -E "(attn|ffn)" | head -20
   ```
2. Compare against llama.cpp's expected naming (see `src/llama.cpp` model loaders)
3. If mismatched, the model may need a custom weight converter

**PESTI Solution**: PESTI uses architecture-aware fallback keys:
```rust
// Try standard key first, then fall back to architecture-specific variants
let attn_q_key = metadata.get("llama.attention.weight")
    .or_else(|| metadata.get("qwen2.attention.weight"))
    .or_else(|| metadata.get("deepseek.attention.q_proj.weight"));
```

---

### 3. "Quantization dequant error"

**Symptoms**:
```
llm_load_tensors: tensor 'output.weight' has unexpected quantization type
llm_load_tensors: failed to dequantize tensor blk.31.ffn_down.weight
```

**Root Cause**: Quantization type doesn't match expected format.

**Debug Steps**:
1. Check quantization type:
   ```bash
   python -c "import gguf; m=gguf.GGUFReader('model.gguf'); print([t for t in m.tensors if 'ffn_down' in t.name][0].quant_type)"
   ```
2. Verify against llama.cpp's supported types (see `ggml/src/ggml-quants.h`)
3. Common issues:
   - Q6_K requires newer llama.cpp (added in b454+)
   - IQ2_XXS needs recent build with IQ quant support
   - Custom quantizations may need manual implementation

**PESTI Solution**: Byte-exact dequantization verification:
```rust
// Verify dequantized output matches reference within tolerance
let dequant = tensor.dequant::<f32>();
let expected = reference_output[..tensor.n];
assert!(dequant.iter().zip(expected.iter()).map(|(a,b)| (a-b).abs()).max() < 1e-2);
```

---

### 4. "Tokenizer config incorrect"

**Symptoms**:
```
llm_load_vocab: special tokens cache size = 0
llm_load_vocab: token to piece cache size = 0.0000 MB
llm_load_tensors: model fails to load (tokenizer mismatch)
```

**Root Cause**: GGUF missing tokenizer metadata or tokenizer.model file not found.

**Debug Steps**:
1. Check GGUF has tokenizer keys:
   ```bash
   python -c "import gguf; m=gguf.GGUFReader('model.gguf'); print([k for k in m.kv_data.keys() if 'tokenizer' in k.lower()])"
   ```
2. Verify tokenizer.model exists (for SentencePiece models):
   ```bash
   ls -la tokenizer.model  # Should exist for Llama/Gemma/Qwen
   ```
3. For models without tokenizer.model, ensure GGUF has `tokenizer.ggml.tokens` array

**PESTI Solution**: Architecture-specific fallback:
```rust
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    // Try standard path first
    let tokenizer_path = path.with_file_name("tokenizer.model");
    if tokenizer_path.exists() {
        return Tokenizer::from_file(&tokenizer_path);
    }
    
    // Fallback to GGUF-embedded tokens
    let gguf_reader = GGUFReader::new(path)?;
    if let Some(tokens) = gguf_reader.kv_data.get("tokenizer.ggml.tokens") {
        return Tokenizer::from_tokens(tokens.as_array_string().unwrap());
    }
    
    Err(Error::TokenizerNotFound)
}
```

---

## Verification Workflow

### Step 1: Basic GGUF Validation
```bash
python -c "import gguf; m=gguf.GGUFReader('model.gguf'); print(f'Valid GGUF v{m.header.version}')"
```

### Step 2: Architecture Detection
```bash
python -c "
import gguf
m = gguf.GGUFReader('model.gguf')
arch = m.kv_data.get('general.architecture', 'unknown')
print(f'Architecture: {arch}')
print(f'Quantization: {m.kv_data.get(\"general.quantization_version\")}')
"
```

### Step 3: Tensor Count Verification
```bash
python -c "
import gguf
m = gguf.GGUFReader('model.gguf')
expected_tensors = {
    'llama': 32 + 3,  # layers + embeddings + output
    'qwen2': 32 + 3,
    'gemma': 32 + 4,  # extra norm tensors
}
actual = len(m.tensors)
print(f'Tensors: {actual} (expected ~{expected_tensors.get(arch, "unknown")})')
"
```

### Step 4: PESTI Conformance Test (Optional)
If you have PESTI installed:
```bash
cargo run --package pesti-conformance -- \
    --model model.gguf \
    --verbose \
    --check-quantization \
    --check-tensor-names
```

---

## Fallback Strategies

### When All Else Fails

1. **Re-quantize from original HF weights**:
   ```bash
   python convert-hf-to-gguf.py path/to/hf/model --outtype q4_k_m
   ```

2. **Use alternative quantization**:
   ```bash
   llama-quantize model.gguf model-q5_k_m.gguf Q5_K_M
   ```

3. **Check llama.cpp version**:
   ```bash
   git describe --tags  # Should be b454+ for latest features
   ```

4. **Try CPU-only inference** (bypasses GPU quantization issues):
   ```bash
   ./llama-cli -m model.gguf -n 64 --no-mmap
   ```

---

## Contributing Fixes

If you find a new pattern not covered here:

1. Document the error symptoms and root cause
2. Provide verification steps (like above)
3. Suggest workaround or fix
4. Submit PR to `docs/gguf-conformance.md`

---

## Related Resources

- [llama.cpp GGUF documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/gguf.md)
- [gguf-py library](https://github.com/ggml-org/gguf-py)
- [PESTI conformance tests](https://github.com/nousresearch/pesti/tree/main/pesti-conformance)
- [llama.cpp issues tracker](https://github.com/ggml-org/llama.cpp/issues)

---

*Last updated: August 2026*  
*Based on PESTI's experience building a production-grade GGUF parser with architecture-aware fallbacks*
