# Foundational Fixes Progress

**Status:** Aug 5, 2026 - Partially complete (2/7 fixed)

## ✅ Completed

### 1. `model.rs` - Type Mismatch
**Problem:** `llama_model: Option<()>` prevented all method calls  
**Fix:** Changed to feature-gated `LlamaModel` type  
**Files modified:** `pesti-runner/src/model.rs`

```rust
// Before:
pub llama_model: Option<()>,

// After:
#[cfg(feature = "cuda")]
pub llama_model: crate::transformer::LlamaModel,
#[cfg(not(feature = "cuda"))]
pub llama_model: crate::transformer_stub::LlamaModel,
```

### 2. `transformer_stub.rs` - Rand API Compatibility  
**Problem:** Used deprecated `rng.gen::<f32>()` from rand <0.10  
**Fix:** Added explicit distribution import and usage  
**Files modified:** `pesti-runner/src/transformer_stub.rs`

```rust
// Before:
let mut r = rng.gen::<f32>();

// After:
use rand::distributions::Uniform;
let dist = Uniform::from(0.0..1.0);
let mut r = rng.sample(dist);
```

## ❌ Remaining Work (5/7)

### 3. `error.rs` vs `error_stub.rs` - Divergent Error Types
**Status:** ❌ Not started  
**Impact:** High - prevents unified error handling  
**Files to fix:** `pesti-runner/src/error.rs`, `error_stub.rs`

```rust
// Current: Different types for CUDA/CPU builds
// Goal: Unified RunnerError with conditional fields
```

### 4. `inference_engine.rs` - Feature-Gating Gaps
**Status:** ❌ Not started  
**Impact:** High - prevents runtime detection  
**Files to fix:** `pesti-runner/src/inference_engine.rs`

Methods needing stubs:
- `get_stream()` (line 230)
- `list_devices()` (line 431)
- `full_device_info()` logic (line 250-263)

### 5. `runtime.rs` - Stub Type Propagation
**Status:** ❌ Not started  
**Impact:** Medium - config/serialization issues  
**Files to fix:** `pesti-runner/src/runtime.rs`

Locations with `()`:
- Line 50: `device_preference: ()`
- Line 92: `RunnerBackend::RustModel(())`
- Line 483: `device_preference()` returns `()`

### 6. `device_discovery.rs` - Missing CPU Path
**Status:** ❌ Not started  
**Impact:** Medium - no device detection without CUDA  
**Files to fix:** `pesti-runner/src/device_discovery.rs`, `lib.rs`

Currently only compiled with `#[cfg(feature = "cuda")]`.

### 7. `kernel/memory_stub.rs` - Missing MemoryManager Variant
**Status:** ❌ Not started  
**Impact:** Medium - can't switch backends at runtime  
**Files to fix:** `pesti-runner/src/kernel/memory_stub.rs`, `memory.rs`

Missing the `Gpu` variant in stub version.

## Next Steps

### Immediate (Blocker Fixes)
1. ✅ **Done:** Fix model.rs type mismatch
2. ✅ **Done:** Fix transformer_stub.rs rand API
3. ⏳ **Next:** Unify error types between CUDA/CPU builds

### Medium Term (Runtime Detection)
4. Fix inference_engine feature-gating gaps
5. Fix runtime.rs stub type propagation  
6. Make device_discovery always available

### Long Term (Backend Switching)
7. Add MemoryManager Gpu variant to stubs

## Verification

**Current status:** `cargo check -p pesti-runner --no-default-features` still has ~20 errors, but down from 41+ before these fixes.

**Next verification target:** Get CPU-only build to compile cleanly (even if some methods don't do much).
