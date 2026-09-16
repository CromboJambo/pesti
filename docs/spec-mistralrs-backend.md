# Mistral.rs Backend Integration Spec

## Goal
Wire up mistral.rs (via candle-core) as a production-grade GPU inference backend for PESTI, providing:
- Working KV cache during autoregressive generation
- Production-tested generation loop with sampling
- Multiple model architecture support
- Benchmark reference point for custom PTX kernel development

## Architecture Overview

```
PESTI API Surface (stable, user-facing)
    │
    ├── dispatch.rs → Backend selection logic
    │       ├── Custom PTX kernels (learning path)
    │       └── MistralRs backend (production path) ← NEW
    │
    ├── kernel/mistralrs_backend.rs (adapter layer - expand stubs)
    │       ├── GemmKernel trait impl → candle tensor ops
    │       └── AttentionKernel trait impl → candle attention ops
    │
    └── transformer/ (model definitions - unchanged)
```

## Implementation Steps

### Phase 1: Wire Up Existing Adapter Stubs (Week 1)

**File: `pesti-runner/src/kernel/mistralrs_backend.rs`**

Replace stub implementations with actual candle tensor operations:

```rust
// GEMM: Replace NotAvailable stub with real matmul
fn matmul_mistralrs(&self, alpha, a, b, beta, c, m, n, k) {
    // Convert DeviceBuffer<f16> to candle Tensor
    let a_tensor = f16_to_tensor(a.as_slice(), &[m, k], None)?;
    let b_tensor = f16_to_tensor(b.as_slice(), &[k, n], None)?;
    
    // Perform matmul on GPU via candle
    let result = a_tensor.matmul(&b_tensor)?;
    
    // Convert back to DeviceBuffer<f32>
    let c_data = tensor_to_f32(&result)?;
    c.write(c_data);
    Ok(())
}

// Attention: Replace NotAvailable stub with candle SDPA
fn forward_mistralrs(&self, query, key_cache, value_cache, mask, config) {
    // Use candle's sdpa implementation
    let q_tensor = f16_to_tensor(query.as_slice(), &query_shape, None)?;
    let k_tensor = f16_to_tensor(key_cache.data(), &key_shape, None)?;
    let v_tensor = f16_to_tensor(value_cache.data(), &value_shape, None)?;
    
    let scale = 1.0 / (config.head_dim as f32).sqrt();
    let result = candle_bridge::sdpa(&q_tensor, &k_tensor, &v_tensor, scale)?;
    
    Ok(DeviceBuffer::from_vec(tensor_to_f32(&result)?))
}
```

**File: `pesti-runner/src/kernel/dispatch.rs`**

Update backend selection to prefer mistral.rs when available:

```rust
// In DispatchContext::new() or similar initialization
let use_mistralrs = cfg!(feature = "mistralrs") 
    && crate::kernel::candle_bridge::bridge_is_cuda();

if use_mistralrs {
    self.gemm_kernel = Some(Box::new(MistralRsGemmKernel::try_new(GemmArch::Cuda).unwrap()));
    self.attention_kernel = Some(Box::new(MistralRsAttentionKernel::try_new(AttentionArch::Cuda).unwrap()));
}
```

### Phase 2: KV Cache Integration (Week 1-2)

**File: `pesti-runner/src/kernel/kvcache.rs`**

Implement proper KV cache updates during autoregressive generation:

```rust
pub fn update(&mut self, seq_pos: usize, key: &[f16], value: &[f16]) {
    // Write new key/value at sequence position
    let offset = seq_pos * self.head_dim;
    self.key_cache[offset..offset+self.head_dim].copy_from_slice(key);
    self.value_cache[offset..offset+self.head_dim].copy_from_slice(value);
}

pub fn get(&self, seq_pos: usize) -> (&[f16], &[f16]) {
    let offset = seq_pos * self.head_dim;
    (&self.key_cache[offset..offset+self.head_dim], 
     &self.value_cache[offset..offset+self.head_dim])
}
```

### Phase 3: Generation Loop (Week 2)

**File: `pesti-runner/src/generation.rs` (new)**

Implement production-grade generation loop using mistral.rs patterns:

```rust
pub struct Generator {
    model: Transformer,
    tokenizer: Tokenizer,
    kv_cache: Kvcache,
    sampling_params: SamplingParams,
}

impl Generator {
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> String {
        let tokens = self.tokenizer.encode(prompt);
        let mut output = Vec::new();
        
        for i in 0..max_tokens {
            // Forward pass through model
            let logits = self.model.forward(&tokens, &mut self.kv_cache)?;
            
            // Sample next token using mistral.rs-style sampling
            let next_token = sample(logits.last(), &self.sampling_params);
            output.push(next_token);
            
            // Update sequence for next iteration
            tokens.push(next_token);
        }
        
        self.tokenizer.decode(&output)
    }
}
```

### Phase 4: Feature Flags and Build System (Week 1)

**File: `pesti-runner/Cargo.toml`**

```toml
[features]
default = []
cuda = ["cudarc"]
mistralrs = ["dep:mistralrs-core", "dep:candle-core", "dep:candle-nn"]
all-backends = ["cuda", "mistralrs"]
```

**File: `scripts/setup.sh`**

Add mistral.rs backend build target:

```bash
build_mistralrs() {
    echo "Building with mistral.rs backend..."
    cargo build --package pesti-runner --features cuda,mistralrs
}
```

### Phase 5: Conformance Testing (Week 2-3)

**File: `pesti-runner/tests/mistralrs_conformance.rs` (new)**

Verify mistral.rs backend produces identical results to custom kernels:

```rust
#[test]
fn test_mistralrs_matches_custom_kernels() {
    let prompt = "The capital of France is";
    
    // Run with custom PTX kernels
    let result_custom = run_with_backends(["custom"], prompt);
    
    // Run with mistral.rs backend
    let result_mistral = run_with_backends(["mistralrs"], prompt);
    
    // Results should be identical (within floating point tolerance)
    assert_approx_equal(result_custom, result_mistral, 1e-4);
}
```

## API Surface Design

### Backend Selection

```rust
// User-facing API for backend selection
let config = InferenceConfig {
    backend: Backend::MistralRs,  // or Backend::CustomPtx
    device: Device::Cuda(0),
    kv_cache_size: 4096,
};

let engine = InferenceEngine::new(config)?;
```

### Generation API

```rust
let generator = Generator::new(engine);
let result = generator.generate("Hello world", GenerateParams {
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
});
println!("{}", result.text);
```

## Benchmark Targets

| Metric | Current (Custom PTX) | Target (Mistral.rs Backend) |
|--------|---------------------|----------------------------|
| GPU decode tok/s | 0.52-0.60 | ≥15 tok/s (Qwen2-7B on RTX 3070 Ti) |
| KV cache support | 🚧 Frontier | ✅ Working |
| Generation loop | Custom, unverified | ✅ Production-tested |

## Success Criteria

1. **Functional**: Mistral.rs backend produces coherent text output matching custom kernel results within numerical tolerance
2. **Performance**: Achieves ≥10x speedup over current custom PTX kernels on GPU decode
3. **Integration**: Seamless switching between backends via feature flags/config
4. **Testing**: Conformance tests pass comparing both backends on same input

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Candle version incompatibility | Pin exact candle-core/candle-nn versions in Cargo.toml |
| KV cache API changes | Abstract behind PESTI's Kvcache trait |
| Performance regression vs custom kernels | Keep both backends; benchmark thoroughly |
| Build complexity increases | Use feature flags to keep builds clean |

## Dependencies

- `mistralrs-core` v0.8.1 (already in Cargo.toml)
- `candle-core` v0.10.2 (already in Cargo.toml)
- `candle-nn` v0.10.2 (already in Cargo.toml)
- `half` crate for f16 conversions (already dependency)

## Files to Modify/Create

### Modify
- `pesti-runner/src/kernel/mistralrs_backend.rs` - Expand stubs to real implementations
- `pesti-runner/src/kernel/dispatch.rs` - Update backend selection logic
- `pesti-runner/Cargo.toml` - Add mistralrs feature flag
- `scripts/setup.sh` - Add mistral.rs build target

### Create
- `pesti-runner/src/generation.rs` - Production generation loop
- `pesti-runner/tests/mistralrs_conformance.rs` - Backend comparison tests
- `docs/BENCHMARKS_MISTRALRS.md` - Performance comparison documentation

## Estimated Effort

- Phase 1 (Wire stubs): 2-3 days
- Phase 2 (KV cache): 2-3 days  
- Phase 3 (Generation loop): 3-4 days
- Phase 4 (Feature flags): 1 day
- Phase 5 (Testing): 2-3 days

**Total: ~2 weeks for full integration with production-quality results**
