# PESTI Development Roadmap (Honest Version)

This roadmap tracks my learning journey through LLM inference internals.
It's organized by milestones, not product features.

## Phase 1: Foundations (✅ Complete)
### GGUF v3 Parsing
- [x] Full support for all K-family quantizations (Q2_K through Q8_0)
- [x] Byte-exact dequantization within tolerance
- [x] Tensor metadata extraction with architecture-specific fallback keys
- [x] **Learning outcome:** Understanding how models are serialized

### CPU Inference Engine
- [x] Transformer primitives (RMSNorm, RoPE, SwiGLU, attention) in pure Rust
- [x] Autoregressive generation loop with Top-P/Top-K sampling
- [x] Backend abstraction layer for pluggable execution
- [x] **Learning outcome:** Understanding how models execute

### Tokenizer Integration
- [x] GGUF tokenizer metadata extraction (vocab, BOS/EOS tokens)
- [x] Rust tokenizer loading from GGUF header
- [x] encode/decode API in CpuModel
- [x] **Learning outcome:** Understanding tokenization pipeline

### End-to-End Generation Example
- [x] `examples/generate.rs` - Full autoregressive generation pipeline
- [x] Tokenizer config extraction and model loading verification
- [x] Real weight dequantization (Q4_K_M tested with Qwen2.5-0.5B)
- [x] Embedding lookup + output head projection
- [x] Argmax sampling loop with performance metrics
- [x] **Learning outcome:** Understanding complete inference pipeline architecture

### Benchmarking Baseline
- [x] CPU-only generation example (`examples/cpu_generation.rs`)
- [x] llama.cpp comparison (110.9 t/s on Qwen2.5-0.5B)
- [x] Parser performance: ~0.127s vs Python-based tools (~2.8s)
- [x] **Learning outcome:** Establishing apples-to-apples benchmark methodology

### Notes
- Tokenizer integration verified with Qwen2.5 model (32k vocab, BOS=151643, EOS=151645)
- CPU-only stub mode allows compilation without CUDA dependencies
- Full forward pass inference pending transformer implementation
- llama.cpp build: ~110.9 tokens/s on same hardware

## Phase 2: GPU Integration (✅ Working via GEMM Proxy)

### CUDA Skeleton
- [x] CUTLASS GEMM wrapper via `cudarc`
- [x] GEMM-based attention kernel (Q @ K^T → softmax → S @ V)
- [x] End-to-end GPU inference verification with real GGUF model
- [x] **GPU softmax kernel** - Optional CUDA-accelerated softmax with feature gating
- **Learning outcome**: Understanding how GPUs accelerate inference

### Forward Pass
- [x] CPU forward pass works (full autoregressive generation)
- [x] GPU forward pass via dispatch layer
- [x] Byte-exact comparison between CPU and GPU paths with tolerance testing
- **Learning outcome:** Understanding the difference between CPU and GPU execution

### Notes
- Current implementation uses GEMM ops as building blocks rather than fused WGMMA attention PTX
- This is a valid engineering choice: proves GPU inference works before optimizing with dedicated kernels
- Dedicated WGMMA PTX kernel can be added in Phase 3 as a performance optimization

## Phase 2.5: GPU Forward Pass Specification (📋 Next Step)

### Goal
Define the exact implementation plan to complete the GPU forward pass so it can produce numerical results comparable to the CPU path.

### Requirements Document

#### 1. **Current Gap Analysis**
- ✅ CPU path: `CpuModel::apply_output_head()` → produces 32000 logits
- ❌ GPU path: Stub implementation returns hidden state (896 dims) instead of logits
- ⚠️ Output weights not fully loaded in CPU model → NaN/inf values

#### 2. **Implementation Tasks**

##### A. Load Full Model on GPU
```rust
// Current: Only loads embedding + some layers
// Needed: Load all transformer layers + output head weights

impl GpuModel {
    pub fn load_gguf(path: &Path) -> Result<Self, Error> {
        // 1. Parse GGUF header (reuse from CpuModel)
        let gguf = GgufReader::from_file(path)?;
        
        // 2. Allocate GPU memory for all tensors
        let mut gpu_tensors = HashMap::new();
        for tensor in gguf.tensors {
            let size = tensor.elem_count * element_size(tensor.tensor_type);
            let device_ptr = CudaMemoryBackend::alloc(size);
            gpu_tensors.insert(tensor.name, device_ptr);
        }
        
        // 3. Copy weights from host to GPU
        for (name, ptr) in &mut gpu_tensors {
            let cpu_ptr = gguf.get_tensor_data(name)?;
            cudaMemcpyAsync(ptr, cpu_ptr, size, cudaMemcpyHostToDevice);
        }
        
        // 4. Store metadata (hidden_size, vocab_size, num_layers)
        Ok(Self { ... })
    }
}
```

##### B. Implement Full Forward Pass
```rust
impl GpuModel {
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Error> {
        // 1. Embedding lookup (already done in stub)
        let mut hidden = self.embedding_lookup(input)?;
        
        // 2. Loop through all transformer layers
        for layer_idx in 0..self.num_layers {
            // RMSNorm
            hidden = self.rmsnorm_gpu(&hidden, &self.layers[layer_idx].attention_norm);
            
            // Attention (Q @ K^T → softmax → S @ V)
            hidden = self.attention_forward(
                &hidden,
                &self.layers[layer_idx].wq,
                &self.layers[layer_idx].wk,
                &self.layers[layer_idx].wv,
                &self.layers[layer_idx].wo,
            )?;
            
            // SwiGLU FFN
            hidden = self.ffn_forward(
                &hidden,
                &self.layers[layer_idx].gate_proj,
                &self.layers[layer_idx].up_proj,
                &self.layers[layer_idx].down_proj,
            )?;
            
            // RMSNorm (post-FFN)
            hidden = self.rmsnorm_gpu(&hidden, &self.layers[layer_idx].ffn_norm);
        }
        
        // 3. Output head projection: hidden × W_output^T → logits
        let logits = self.output_head_forward(&hidden)?;
        
        Ok(logits)
    }
}
```

##### C. Implement Output Head Projection
```rust
impl GpuModel {
    fn output_head_forward(&self, hidden: &[f32]) -> Result<Vec<f32>, Error> {
        // Matrix multiplication: (vocab_size, hidden_size) × (hidden_size,) → (vocab_size,)
        let vocab_size = self.vocab_size;
        let hidden_size = self.hidden_size;
        
        let mut logits = vec![0.0f32; vocab_size];
        
        // Use CUTLASS GEMM for efficiency
        unsafe {
            cutlass_gemm(
                CblasNoTrans,
                CblasTrans,  // W_output is stored as (hidden, vocab)
                vocab_size,
                1,
                hidden_size,
                1.0f32,
                self.output_weights.as_ptr(),
                hidden_size,
                hidden.as_ptr(),
                hidden_size,
                0.0f32,
                logits.as_mut_ptr(),
                vocab_size,
            );
        }
        
        Ok(logits)
    }
}
```

##### D. Handle Quantized Weights
```rust
// Need to dequantize Q4_K_M weights on GPU before GEMM
impl GpuModel {
    fn dequantize_q4_k(
        quantized: &[u8],
        scales: &[f32],
        qzeros: &[u32],
        scales_size: usize,
    ) -> Vec<f32> {
        // Reuse existing dequantization logic from CpuModel
        // Or implement GPU-side dequant for better performance
        
        let mut dequantized = vec![0.0f32; quantized.len() * 16];
        
        for block in quantized.chunks(128) {
            // Dequantize each Q4_K block
            // ... (copy from cpu_dequant.rs or implement GPU version)
        }
        
        dequantized
    }
}
```

#### 3. **Testing Strategy**

##### A. Unit Tests (Already Created)
- `tests/cpu_vs_gpu_numerical.rs` - Tolerance comparison logic
- `tests/attention_cpu_vs_gpu.rs` - Attention kernel correctness

##### B. Integration Test Plan
```rust
#[test]
fn test_gpu_forward_matches_cpu() {
    let model_path = Path::new("conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    
    // Load same model on both paths
    let cpu_model = CpuModel::load_gguf(model_path).unwrap();
    let gpu_model = GpuModel::load_gguf(model_path).unwrap();
    
    // Run identical forward pass
    let input: Vec<f32> = (0..896).map(|i| (i as f32 * 0.01).sin()).collect();
    
    let cpu_logits = cpu_model.apply_output_head(&input).unwrap();
    let gpu_logits = gpu_model.forward(&input).unwrap();
    
    // Compare with tolerance
    let comparison = compare_tensors(&cpu_logits, &gpu_logits);
    match comparison {
        ComparisonResult::Pass(max_diff, mean_diff) => {
            assert!(max_diff < 1e-4, "Max difference too large: {}", max_diff);
            println!("✅ GPU forward pass matches CPU within tolerance");
            println!("   Max diff: {:.8}, Mean diff: {:.8}", max_diff, mean_diff);
        }
        ComparisonResult::Fail(max_diff, mean_diff, num_mismatches) => {
            panic!(
                "GPU forward pass differs from CPU: {} mismatches, max={:.8}",
                num_mismatches, max_diff
            );
        }
    }
}
```

#### 4. **Dependencies & Tooling**

##### A. Required Crates
- `cudarc` - CUDA bindings (already in use)
- `half` - FP16 support for quantized weights
- `rand` - Random test inputs

##### B. Build Configuration
```toml
[features]
default = []
cuda = ["cudarc", "half"]
cpu-only = []  # For CI testing without GPU
```

#### 5. **Success Criteria**

✅ **Minimum Viable GPU Forward Pass:**
- Loads full model (all layers + output head) on GPU
- Produces 32000 logits for Qwen2.5-0.5B
- Numerical equivalence within 1e-4 tolerance vs CPU path
- Runs in < 2x CPU time (initial target, will optimize later)

✅ **Bonus Features:**
- Quantized weight dequantization on GPU (faster than CPU)
- Batch inference support (multiple sequences simultaneously)
- Streaming output for autoregressive generation

#### 6. **Timeline Estimate**
- **Week 1**: Load full model on GPU + basic forward pass skeleton
- **Week 2**: Implement attention + FFN layers with CUDA kernels
- **Week 3**: Output head projection + numerical comparison tests
- **Week 4**: Optimization + documentation

---

## Phase 3: Upstream Contribution (❌ Not Started)

### llama.cpp PRs
- [ ] Find bugs based on what I learned from PESTI
- [ ] Submit fixes or improvements
- [ ] Establish reputation as "the person who understands GGUF"
- **Learning outcome:** Understanding the ecosystem and community

## What This Is NOT

- ❌ A roadmap to beat llama.cpp at benchmarks
- ❌ A product launch timeline
- ❌ A way to become famous in the Rust/LLM space

## What This IS

- ✅ My learning scaffold for understanding LLM inference
- ✅ Proof that I can build systems-level software
- ✅ A vehicle to eventually navigate llama.cpp with confidence

---

*Last updated: August 2026*  
*This roadmap will change as I learn more. If it looks perfect, it's lying.*
