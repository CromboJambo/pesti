# GPU Acceleration - Quick Start

**Goal**: Get GPU working in < 4 hours

---

## Step 1: Verify GPU Setup (5 minutes)

```bash
cd /home/crombo/projects/pesti

# Check CUDA is available
cargo run --package cuda-oxide --example device_info 2>&1 | grep -E "(✅|❌)"

# Expected output:
# ✅ CUDA driver initialized
# ✅ Device count: 2
# ✅ NVIDIA GeForce RTX 4070 detected
```

---

## Step 2: Implement WGMMA Kernel Launch (2 hours)

**File**: `pesti-runner/src/kernel/attention.rs`

**Location**: Lines ~354-402, in the `CudaAttentionKernel::forward()` method

**Current code** (placeholder):
```rust
fn forward(...) -> Result<DeviceBuffer<f32>, AttentionError> {
    // TODO: Launch WGMMA kernel
    Ok(DeviceBuffer::<f32>::zeros(out_len))
}
```

**Replace with**:
```rust
fn forward(...) -> Result<DeviceBuffer<f32>, AttentionError> {
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let cache_seq_len = key_cache.seq_len();
    let query_seq_len = query.len() / (num_heads * head_dim);
    
    // Output: attention scores [query_seq_len, num_heads, cache_seq_len]
    let out_len = query_seq_len * num_heads * cache_seq_len;
    let output = DeviceBuffer::<f32>::zeros(out_len);
    
    let scale = config.scale();
    
    // Launch WGMMA kernel
    match self.arch {
        AttentionArch::Wgmma => {
            // Configure grid dimensions: 64x64 tiles
            let grid_dim = (
                (query_seq_len + 63) / 64, // Ceiling division
                num_heads
            );
            
            // Launch kernel
            unsafe {
                self.function.launch(
                    &[
                        query_seq_len as i32,
                        cache_seq_len as i32,
                        num_heads as i32,
                        head_dim as i32,
                        scale.to_bits() as f32,
                    ],
                    grid_dim,
                    vec![
                        query.device_ptr(),
                        key_cache.device_ptr(),
                        value_cache.device_ptr(),
                        output.device_ptr(),
                    ],
                    8 * 1024, // 8 KiB shared memory
                )?;
            }
            
            // Sync to ensure completion
            cuda_core::stream::Stream::default().synchronize()?;
        },
        AttentionArch::Tcgen05 => {
            // Similar but with tcgen05-specific instructions
            todo!("Implement tcgen05 kernel launch")
        },
    }
    
    Ok(output)
}
```

---

## Step 3: Test Single Token (1 hour)

**File**: Create `pesti-runner/examples/test_single_token.rs`

```rust
use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use pesti_runner::kernel::InferenceEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model
    let runner = LlamaRunner::builder("/path/to/model.gguf")
        .n_ctx(2048)
        .build()?;
    
    // Enable GPU dispatch (if available)
    let engine = InferenceEngine::new();
    if engine.gpu_available() {
        println!("✅ GPU detected: {}", engine.full_device_info());
    } else {
        println!("⚠️ No GPU available, running on CPU");
    }
    
    // Single token test
    let prompt = "The";
    let result = runner.generate(prompt, &SamplingConfig::greedy())?;
    
    println!("Generated: {}", result.text);
    println!("Time: {:.3}s", result.eval_ms / 1000.0);
    
    Ok(())
}
```

**Run**:
```bash
cargo run --package pesti-runner --example test_single_token --release
```

**Expected**: Should complete without errors, may be slower than CPU initially

---

## Step 4: Benchmark (1 hour)

**File**: `pesti-runner/examples/benchmark_gpu.rs`

```rust
use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runner = LlamaRunner::builder("/home/crombo/projects/pesti/test_models/tinyllama-q4.gguf")
        .n_ctx(2048)
        .build()?;
    
    let prompt = "Explain quantum computing in one sentence.";
    let tokens_to_generate = 500;
    
    // Measure time
    let start = Instant::now();
    let result = runner.generate(prompt, &SamplingConfig::greedy())?;
    let elapsed = start.elapsed();
    
    let tok_per_sec = result.generated_tokens as f64 / elapsed.as_secs_f64();
    
    println!("Generated {} tokens in {:.2}s", result.generated_tokens, elapsed.as_secs_f64());
    println!("Speed: {:.1} tok/s", tok_per_sec);
    
    Ok(())
}
```

**Run**:
```bash
cargo run --package pesti-runner --example benchmark_gpu --release
```

**Expected**: Should show speedup if GPU is working (target: 650+ tok/s)

---

## Step 5: Verify Correctness (30 minutes)

Compare GPU vs CPU outputs:

```bash
# Run twice, once with GPU, once with CPU
cargo run --package pesti-runner --example benchmark_gpu --release > gpu_output.txt
# Disable GPU in config
cargo run --package pesti-runner --example benchmark_gpu --release > cpu_output.txt

# Compare
diff -u cpu_output.txt gpu_output.txt
```

**Expected**: Should match within 1e-2 tolerance (same tokens generated)

---

## Troubleshooting

### Issue: "CUDA driver not initialized"
**Fix**: Check nvidia-smi works, verify CUDA libs installed

```bash
nvidia-smi
# If this fails, install CUDA toolkit
```

### Issue: "Kernel launch failed"
**Fix**: Check PTX module loaded correctly

```bash
cargo test --package pesti-runner kernel::attention::tests
# Should show all tests passing
```

### Issue: "Output mismatch with CPU"
**Fix**: Check numerical precision, increase tolerance to 1e-1

```rust
// In attention.rs, add debug output
println!("GPU scale: {}, CPU scale: {}", gpu_scale, cpu_scale);
```

---

## Success Criteria

✅ **Minimum Viable**:
- Single token generates without error
- Output matches CPU within 1e-2 tolerance
- Measured speedup > 2x

🎯 **Target**:
- Multi-token generation (500+ tokens)
- Speed: 650+ tok/s (3x improvement)
- Memory bandwidth > 50% utilization

---

## Next Steps After Success

1. **Optimize KV cache**: Keep on GPU, avoid host transfers
2. **Add batch processing**: Process multiple sequences simultaneously
3. **Profile each layer**: Identify remaining bottlenecks
4. **Test all quantizations**: Q3, Q4, Q5, Q8

---

## Resources

- **Full plan**: `GPU-ACCELERATION-PLAN.md`
- **Kernel template**: `GPU_KERNEL_LAUNCH_PLAN.md`
- **PTX reference**: `pesti-runner/src/kernel/ptx/attention_wgmma.ptx`

**Estimated total time**: 4 hours to first GPU inference
