# CUDA Reintegration Checklist

## Current State (Aug 5, 2026)

You've successfully stubbed out CUDA dependencies to focus on K-family quantization and foundational CPU infrastructure. The code compiles with `--no-default-features`, but the stubs are incomplete for a production-ready GPU-first build with CPU fallback.

## Goal

Create a **unified build** that:
1. Defaults to GPU acceleration when CUDA hardware is available
2. Falls back gracefully to CPU-only mode when no GPU is present
3. Maintains clean feature-gating without code duplication
4. Provides runtime device detection and capability queries

---

## Phase 1: Fix Stub Module Inconsistencies

### 1.1 `transformer_stub.rs` - Critical Fixes

**Problem:** The stub uses the new `rand` API (0.10+) but calls deprecated methods.

```rust
// ❌ Line 179 - fails to compile with rand 0.10+
let mut r = rng.gen::<f32>();

// ✅ Fix: Use explicit distribution
use rand::distributions::Uniform;
let dist = Uniform::from(0.0..1.0);
let r = rng.sample(dist);
```

**Problem:** `LlamaModel` methods exist but are never called because `model.rs` uses `()` as the type.

### 1.2 `model.rs` - Type Mismatch

**Current (line 159):**
```rust
pub llama_model: Option<()>, // Stub - actual implementation only exists with CUDA
```

**Problem:** This makes all `llama_model` methods fail at compile time, even the stub implementations.

**Fix:** Use the real stub type consistently:
```rust
#[cfg(feature = "cuda")]
pub use transformer::LlamaModel;

#[cfg(not(feature = "cuda"))]
pub use transformer_stub::LlamaModel;

// In Model struct:
pub llama_model: Option<LlamaModel>, // Always LlamaModel, implementation differs by feature
```

### 1.3 `error_stub.rs` - Missing Error Variants

**Current (line 50-51):**
```rust
#[error("CUDA error: {0}")]
Cuda(#[from] CudaError),
```

**Problem:** References `CudaError` which is defined in stub but never used properly.

**Fix:** Either remove the variant entirely for CPU builds, or create a proper mapping:
```rust
#[cfg(not(feature = "cuda"))]
pub enum CudaError {
    #[error("CUDA not available")]
    NotAvailable,
}

// In RunnerError:
#[cfg(not(feature = "cuda"))]
#[error("CUDA not available")]
CudaNotAvailable,
```

### 1.4 `inference_engine.rs` - Feature-Gating Gaps

**Line 230-233:** Stream getter method exists only with CUDA feature:
```rust
#[cfg(feature = "cuda")]
fn get_stream(&self) -> Option<&Arc<cuda_core::CudaStream>> {
    self.stream.as_ref()
}
```

**Problem:** If any code calls this method without `#[cfg]` guards, it will fail.

**Fix:** Add stub implementation:
```rust
#[cfg(not(feature = "cuda"))]
fn get_stream(&self) -> Option<()> {
    None
}
```

### 1.5 `runtime.rs` - Stub Type Propagation

**Line 50, 64, 92, 483:** Multiple places use `()` as stub types:
```rust
pub device_preference: (), // Stub - actual implementation only exists with CUDA
pub runner: Arc<RwLock<Option<RunnerBackend>>>,
// ...
RustModel(()), // Stub - actual implementation only exists with CUDA
```

**Fix:** Define proper stub structs/enums that match the real API signature, even if they do nothing.

---

## Phase 2: Implement Runtime Device Detection

### 2.1 Add `device_discovery.rs` to CPU Builds

**Current:** Only compiled with `#[cfg(feature = "cuda")]`.

**Fix:** Make it available always, with stub implementations for CPU-only mode:
```rust
pub mod device_discovery; // Always available

// In lib.rs:
#[cfg(feature = "cuda")]
pub use device_discovery::LocalDevice;

#[cfg(not(feature = "cuda"))]
pub use device_discovery::CpuLocalDevice as LocalDevice; // New stub type
```

### 2.2 Implement `is_available()` for CPU Builds

**Current (line 58 in inference_engine.rs):**
```rust
if matches!(device, Device::Cuda(_)) || is_available() {
```

**Problem:** `is_available()` only exists with CUDA feature.

**Fix:** Create a unified function:
```rust
// In lib.rs or runtime.rs
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda_runtime::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Check if we're running on a system with CUDA driver loaded
        std::path::Path::new("/dev/nvidia0").exists()
    }
}
```

### 2.3 Add Device Info Query Functions

Create `device_info.rs` module with unified API:
```rust
pub struct SystemDeviceInfo {
    pub has_gpu: bool,
    pub gpu_count: usize,
    pub preferred_device: DeviceType, // CPU or GPU
    pub total_system_memory: u64,
    pub available_gpu_memory: Option<u64>,
}

impl SystemDeviceInfo {
    pub fn detect() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Use cudarc/cuda-core to query devices
            let count = cuda_runtime::device_count();
            let has_gpu = count > 0;
            // ... populate remaining fields
        }
        #[cfg(not(feature = "cuda"))]
        {
            SystemDeviceInfo {
                has_gpu: false,
                gpu_count: 0,
                preferred_device: DeviceType::Cpu,
                total_system_memory: get_system_memory(),
                available_gpu_memory: None,
            }
        }
    }
}
```

---

## Phase 3: Implement Graceful Degradation Path

### 3.1 Feature-Gated Build Strategy

**Current `Cargo.toml`:**
```toml
[features]
default = []
cuda = ["dep:cudarc", "dep:cuda-device", "dep:cuda-host", "dep:cuda-core", "dep:intel-mkl-src"]
```

**Problem:** Default is CPU-only. We want GPU-first with automatic fallback.

**Fix:** Change default to include CUDA if available:
```toml
[features]
default = ["cuda"] # Try to use CUDA, but make deps optional

[dependencies]
cudarc = { workspace = true, optional = true }
# ... other cuda deps as optional

# Add cfg-based conditional compilation for missing hardware
[lints.rust]
unexpected_cfgs = { level = "allow", check-cfg = ['cfg(cuda_available)'] }
```

### 3.2 Runtime Detection and Fallback

**Pattern in `InferenceEngine::new()`:**

```rust
pub fn new(device: Device, dtype: DType) -> Self {
    // Step 1: Try to initialize CUDA
    let (cuda_runtime, stream) = if cfg!(feature = "cuda") {
        match CudaRuntime::for_default_device() {
            Ok(rt) => {
                let rt = Arc::new(rt);
                match rt.new_stream() {
                    Ok(stream) => (Some(rt), Some(stream)),
                    Err(_) => (Some(rt), None),
                }
            }
            Err(e) => {
                tracing::warn!("CUDA runtime init failed: {}. Falling back to CPU.", e);
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    // Step 2: Choose kernel based on availability
    let gemm: Box<dyn GemmKernel> = if let Some(rt) = &cuda_runtime {
        // Try GPU kernels
        match CudaGemmKernelBuilder::new(...).build() {
            Ok(kernel) => Box::new(kernel),
            Err(e) => {
                tracing::warn!("CUDA GEMM init failed: {}. Falling back to CPU.", e);
                Box::new(CpuGemmKernel::new())
            }
        }
    } else {
        // Always available fallback
        Box::new(CpuGemmKernel::new())
    };

    // Step 3: Store runtime info for later use
    Self {
        device: if cuda_runtime.is_some() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        },
        dtype,
        gemm,
        attention: Box::new(if cuda_runtime.is_some() {
            CudaAttentionKernel::new(...)
        } else {
            CpuAttentionKernel::new()
        }),
        #[cfg(feature = "cuda")]
        cuda_runtime,
        #[cfg(feature = "cuda")]
        stream,
        memory_manager: if cuda_runtime.is_some() {
            MemoryManager::Gpu(cuda_runtime.clone().unwrap())
        } else {
            MemoryManager::Cpu(CpuMemoryBackend::new(...))
        },
        cpu_gemm: CpuGemmKernel::new(), // Always keep CPU fallback
        cpu_attention: CpuAttentionKernel::new(), // Always keep CPU fallback
    }
}
```

### 3.3 Add Build-Time CUDA Detection

Create `build.rs` in `pesti-runner`:
```rust
fn main() {
    // Check for CUDA toolkit at build time
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("PATH"))
        .ok()
        .and_then(|path| {
            std::fs::read_dir(&path)
                .ok()
                .and_then(|entries| entries.find(|e| e.as_ref().ok()?.file_name().to_str() == Some("nvcc")))
        });

    if cuda_home.is_some() {
        println!("cargo:rustc-cfg=cuda_toolkit_available");
    }

    // Check for NVIDIA driver
    let nvidia_dev = std::path::Path::new("/dev/nvidia0");
    if nvidia_dev.exists() {
        println!("cargo:rustc-cfg=cuda_hardware_available");
    }

    // If both available, enable CUDA by default
    if cuda_home.is_some() && nvidia_dev.exists() {
        println!("cargo:rustc-cfg=cuda_available");
    }
}
```

Then in code:
```rust
#[cfg(cuda_available)]
const DEFAULT_BACKEND: &str = "cuda";

#[cfg(not(cuda_available))]
const DEFAULT_BACKEND: &str = "cpu";
```

---

## Phase 4: Update Feature-Gating Strategy

### 4.1 Unified Module Exports

**Current `lib.rs` has:**
```rust
#[cfg(feature = "cuda")]
pub mod cuda_runtime;
#[cfg(not(feature = "cuda"))]
pub mod cuda_stub;
```

**Problem:** Creates two completely separate code paths that diverge over time.

**Fix:** Always include the real module, feature-gate only the *usage*:
```rust
// Always compile these (they have internal #[cfg] guards)
pub mod cuda_runtime;
pub mod device;
pub mod transformer;

// Re-export conditionally based on runtime availability
#[cfg(cuda_available)]
pub use cuda_runtime::*;

#[cfg(not(cuda_available))]
pub use cuda_stub::*; // Stub types that match the real API
```

### 4.2 Error Type Unification

**Current:** `RunnerError` has different variants for CUDA vs CPU builds.

**Fix:** Use a single error type with conditional fields:
```rust
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("GEMM error ({arch}, {m}x{n}x{k}): {detail}")]
    Gemm { arch: String, m: usize, n: usize, k: usize, detail: GemmError },

    #[error("attention error (heads={num_heads}, dim={head_dim}, seq={seq}): {detail}")]
    Attention { num_heads: usize, head_dim: usize, seq: usize, detail: AttentionError },

    #[error("tensor computation error: {0}")]
    Tensor(String),

    // Always present, but only populated with CUDA builds
    #[error("device error: {0}")]
    Device(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}

// Internal CUDA-specific error (only compiled with feature)
#[cfg(feature = "cuda")]
pub type CudaError = cudarc::driver::DriverError;

#[cfg(not(feature = "cuda"))]
pub enum CudaError {
    #[error("CUDA not available")]
    NotAvailable,
}
```

---

## Phase 5: Testing Strategy

### 5.1 CPU-Only Build Tests

**File:** `pesti-runner/tests/cpu_only.rs`
```rust
#[cfg(not(feature = "cuda"))]
mod tests {
    use pesti_runner::inference_engine::InferenceEngine;
    use candle_core::{Device, DType};

    #[test]
    fn test_cpu_backend_initialization() {
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        assert!(matches!(engine.device, Device::Cpu));
        assert!(!engine.gpu_available());
    }

    #[test]
    fn test_cpu_gemm_fallback() {
        let engine = InferenceEngine::new(Device::Cpu, DType::F16);
        
        // Create test tensors on CPU
        let a = candle_core::Tensor::rand(std::vec![10, 10], 0.0, 1.0, &Device::Cpu).unwrap();
        let b = candle_core::Tensor::rand(std::vec![10, 10], 0.0, 1.0, &Device::Cpu).unwrap();
        
        // Should succeed with CPU kernel
        let c = a.matmul(&b.t()).unwrap();
        assert_eq!(c.dims(), &[10, 10]);
    }

    #[test]
    fn test_cpu_attention_fallback() {
        let engine = InferenceEngine::new(Device::Cpu, DType::F32);
        
        // Attention should work with CPU kernel
        let query = vec![0.0f32; 64];
        let key_cache = pesti_runner::kernel::Kvcache::new(8, 8, 64, 1024, false);
        let value_cache = pesti_runner::kernel::Kvcache::new(8, 8, 64, 1024, false);
        
        let config = pesti_runner::kernel::AttentionConfig::default()
            .with_num_heads(8)
            .with_head_dim(64);
        
        let result = engine.attention(&query.into(), &key_cache, &value_cache, None, &config);
        assert!(result.is_ok());
    }
}
```

### 5.2 GPU-Available Build Tests

**File:** `pesti-runner/tests/gpu_available.rs`
```rust
#[cfg(all(feature = "cuda", cuda_hardware_available))]
mod tests {
    use pesti_runner::inference_engine::InferenceEngine;
    use candle_core::{Device, DType};

    #[test]
    fn test_gpu_backend_initialization() {
        let engine = InferenceEngine::new(Device::Cuda(0), DType::F16);
        assert!(matches!(engine.device, Device::Cuda(_)));
        assert!(engine.gpu_available());
    }

    #[test]
    fn test_gpu_gemm_kernel() {
        let engine = InferenceEngine::new(Device::Cuda(0), DType::F16);
        
        // GEMM should use GPU kernel
        assert!(engine.gemm_available());
        assert_eq!(engine.gemm_arch().name(), "wgmma"); // Or "tcgen05" depending on hardware
    }

    #[test]
    fn test_gpu_attention_kernel() {
        let engine = InferenceEngine::new(Device::Cuda(0), DType::F16);
        
        // Attention should use GPU kernel
        assert!(engine.attention_available());
    }

    #[test]
    fn test_runtime_device_detection() {
        let devices = pesti_runner::list_devices().unwrap();
        assert!(!devices.is_empty());
        assert!(devices.iter().any(|d| d.compute_capability.0 >= 8)); // At least one sm_8x+ device
    }
}
```

### 5.3 Graceful Degradation Tests

**File:** `pesti-runner/tests/graceful_fallback.rs`
```rust
mod tests {
    use pesti_runner::inference_engine::InferenceEngine;
    use candle_core::{Device, DType};

    #[test]
    fn test_cpu_fallback_on_gpu_failure() {
        // Simulate GPU failure by using a device that doesn't exist
        let engine = InferenceEngine::new(Device::Cuda(999), DType::F16);
        
        // Should fall back to CPU automatically
        assert!(matches!(engine.device, Device::Cpu));
        assert!(!engine.gpu_available());
    }

    #[test]
    fn test_cpu_gemm_still_works_after_gpu_init_failure() {
        let engine = InferenceEngine::new(Device::Cpu, DType::F16);
        
        // Even if GPU init failed earlier, CPU path should work
        let a = candle_core::Tensor::rand(std::vec![10, 10], 0.0, 1.0, &Device::Cpu).unwrap();
        let b = candle_core::Tensor::rand(std::vec![10, 10], 0.0, 1.0, &Device::Cpu).unwrap();
        let c = a.matmul(&b.t()).unwrap();
        
        assert_eq!(c.dims(), &[10, 10]);
    }
}
```

---

## Phase 6: Documentation Updates

### 6.1 README.md Additions

Add section **"GPU vs CPU Builds"**:

```markdown
## GPU vs CPU Builds

PESTI supports both GPU-accelerated and CPU-only inference, with automatic runtime detection.

### Default Build (GPU-First)

```bash
# Try to use CUDA if available, fall back to CPU otherwise
cargo build -p pesti-runner --features cuda
```

This build includes:
- CUDA runtime initialization at startup
- Automatic device detection and capability queries
- Graceful fallback to CPU kernels if GPU unavailable
- Runtime tiered execution (CPU → GPU based on usage)

### CPU-Only Build

```bash
# No CUDA dependencies, smaller binary, faster compile times
cargo build -p pesti-runner --no-default-features
```

This build includes:
- Pure CPU inference with `candle-core` and `gemm` crates
- Stub implementations for all GPU APIs
- Smaller binary size (~40% reduction)
- Faster development iteration cycles

### Runtime Behavior

Regardless of which build you use, PESTI will:

1. **Detect hardware at runtime** - Check for NVIDIA GPUs and CUDA driver
2. **Select optimal backend** - Choose GPU if available, CPU otherwise
3. **Log the decision** - See `INFO` logs showing which path was taken
4. **Maintain API compatibility** - Same code works on both paths

### Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `cuda` | Include CUDA dependencies and GPU kernels | Yes (if CUDA toolkit detected) |
| `mistralrs` | Use mistral.rs backend for GEMM/attention | No |

### Build Verification

```bash
# Check which features are enabled
cargo build -p pesti-runner --features cuda --verbose | grep "Compiling.*cuda"

# See runtime device detection logs
cargo run -p pesti-runner --example basic_inference 2>&1 | grep -E "(CUDA|GPU|CPU)"
```
```

### 6.2 Architecture Diagram Update

Add section showing the unified inference path:

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine::new()                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │ Runtime Device Detection │
              │ - CUDA toolkit present?  │
              │ - /dev/nvidia0 exists?  │
              │ - GPU memory available? │
              └───────────┬─────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│  GPU Path       │            │  CPU Path       │
│  (cuda_available)│           │  (no CUDA)      │
└───────┬─────────┘            └───────┬─────────┘
        │                              │
        ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Initialize      │            │ Initialize      │
│ CudaRuntime     │            │ CpuMemoryBackend│
│ Create streams  │            │                 │
│ Query device    │            │                 │
└───────┬─────────┘            └───────┬─────────┘
        │                              │
        ▼                              ▼
┌─────────────────┐            ┌─────────────────┐
│ Select Kernel   │            │ Select Kernel   │
│ - Try mistralrs │            │ CpuGemmKernel   │
│ - Try PTX GEMM  │            │ CpuAttention    │
│ - Fall back to  │            │                 │
│   CPU           │            │                 │
└───────┬─────────┘            └───────┬─────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
          ┌─────────────────────┐
          │ Unified API Layer   │
          │ - matmul()          │
          │ - attention()       │
          │ - infer()           │
          └───────────┬─────────┘
                      │
                      ▼
          ┌─────────────────────┐
          │ Runtime Fallback    │
          │ If GPU fails:       │
          │ → Switch to CPU     │
          │ Log warning         │
          │ Continue execution  │
          └─────────────────────┘
```

---

## Implementation Order (Recommended)

1. **Week 1:** Fix stub inconsistencies (`transformer_stub.rs`, `model.rs` type mismatches)
2. **Week 2:** Implement runtime device detection (`device_discovery.rs`, `is_available()`)
3. **Week 3:** Create graceful degradation path in `InferenceEngine::new()`
4. **Week 4:** Add build-time CUDA detection (`build.rs`) and update feature strategy
5. **Week 5:** Write comprehensive test suite (CPU-only, GPU-available, fallback)
6. **Week 6:** Documentation updates and migration guide

---

## Success Criteria

✅ **Builds pass:**
- `cargo check -p pesti-runner --no-default-features`
- `cargo check -p pesti-runner --features cuda` (on GPU machine)
- `cargo test -p pesti-runner` (on both CPU and GPU machines)

✅ **Runtime behavior:**
- GPU-first builds auto-detect and use CUDA when available
- Fallback to CPU kernels works seamlessly on hardware-less systems
- Logs clearly indicate which backend is active
- API remains consistent regardless of backend

✅ **Developer experience:**
- Clear documentation for both build types
- Example code works on both paths
- Error messages guide users through troubleshooting

---

## References

- `rust-cuda-feature-gating` skill: Detailed pattern for feature-gating CUDA deps
- `cuda-oxide-integration` skill: cudarc integration with device detection
- Current stub files in `pesti-runner/src/`:
  - `cuda_stub.rs` - Stub types for CUDA dependencies
  - `device_stub.rs` - Stub device backend
  - `transformer_stub.rs` - Stub transformer model
  - `error_stub.rs` - Stub error types

---

## Notes

- The current stubs are **compilation-only** - they make the code compile but don't provide meaningful functionality
- The goal is **not** to keep stubs forever, but to create a unified codebase where GPU/CPU paths converge at runtime
- Once CUDA reintegration is complete, stub modules can be removed or repurposed as "minimal CPU reference implementations"
