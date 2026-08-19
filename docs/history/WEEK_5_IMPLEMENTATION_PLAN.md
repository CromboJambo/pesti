# Week 5 Flash Attention Implementation Plan 🚀

**Date**: August 13, 2026  
**Status**: ✅ **Infrastructure Ready** | ⏳ **Compiling Real PTX**

---

## 🎯 Current State

### ✅ What Works
1. **Kernel infrastructure**: Loads, launches, passes parameters correctly
2. **PTX loading**: Successfully loads from `flash_attention_kernel.ptx` in 447µs
3. **Function resolution**: Mangled name `_Z22flash_attention_kernelfPK6__halfS1_S1_Pfiiii` resolved
4. **Numerical conformance**: Byte-exact determinism verified
5. **Baseline performance**: ~95 tok/s on Qwen2.5-0.5B

### ❌ What's Missing
The PTX kernel body is a **stub** - it just loads/stores one value per block, no real computation!

---

## 🚀 Implementation Plan

### Step 1: ✅ Clone flash-attention Repository
**Status**: DONE ✓

```bash
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
# Location: /home/crombo/flash-attention
```

### Step 2: ⏳ Compile Real PTX Kernel
**Status**: IN PROGRESS

Two options:

#### Option A: Compile Individual CUDA File (Quick)
```bash
cd /home/crombo/flash-attention/csrc/flash_attn/src
nvcc -arch=sm_89 -ptx flash_fwd_hdim128_fp16_sm80.cu \
    -o ~/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx
```

**Pros**: Fast, simple  
**Cons**: Need CUDA headers, may need full build system

#### Option B: Full Build System (Robust)
```bash
cd /home/crombo/flash-attention
pip install torch ninja packaging
python setup.py install --cpp_ext --cuda_ext
# PTX will be in build/*/lib/*.ptx
```

**Pros**: Handles dependencies, proven build  
**Cons**: Takes longer (~5-10 min)

### Step 3: ⏳ Replace Stub PTX
**Status**: PENDING

Copy compiled PTX to:
`~/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx`

### Step 4: ⏳ Verify Numerical Output
**Status**: PENDING

Run conformance test:
```bash
cargo run --package pesti-runner --features cuda,flash-attention \
    --example test_numerical_conformance
```

Expected results:
- ✅ Byte-exact determinism (same as before)
- ✅ Logits within 1e-5 relative error of llama.cpp

### Step 5: ⏳ Benchmark Real Speedup
**Status**: PENDING

Measure tok/s on different models:
- Qwen2.5-0.5B: Expect +1-3% speedup
- Qwen2.5-3B: Expect +3-4x speedup (big win!)
- Llama 3.1 8B: Expect +4-5x speedup

---

## 📋 Scripts Created

### 1. `scripts/clone_flash_attention.sh`
Clones the official flash-attention repository.

**Status**: ✅ DONE  
**Output**: `/home/crombo/flash-attention`

### 2. `scripts/compile_flash_ptx.sh`
Attempts to compile individual CUDA file to PTX.

**Status**: ⏳ READY TO RUN  
**Command**: `./scripts/compile_flash_ptx.sh`

### 3. `scripts/build_flash_attention.sh`
Full build with Python setup.py.

**Status**: ⏳ READY TO RUN  
**Command**: `./scripts/build_flash_attention.sh`

---

## 🎯 Next Immediate Action

**Run the full build:**

```bash
cd /home/crombo/projects/pesti
./scripts/build_flash_attention.sh
```

This will:
1. Install torch/ninja dependencies
2. Build CUDA extension with proper headers
3. Generate PTX files in build directory
4. We'll then find and copy the best PTX to our project

**Estimated time**: 5-10 minutes

---

## 🏆 Success Criteria

### Minimum Viable Product (MVP)
- [ ] PTX compiles without errors
- [ ] Kernel loads successfully
- [ ] Numerical output matches llama.cpp (< 1e-5 error)
- [ ] Same or better throughput than baseline (~95 tok/s)

### Full Success
- [ ] +3% speedup on Qwen2.5-0.5B
- [ ] +3x speedup on Qwen2.5-3B
- [ ] +4x speedup on Llama 3.1 8B
- [ ] Byte-exact or near-identical output

---

## 📊 Expected Performance

| Model | Baseline (GEMM) | Expected Flash Attention | Speedup |
|-------|-----------------|-------------------------|---------|
| Qwen2.5-0.5B | 95 tok/s | 98-100 tok/s | +3% |
| Qwen2.5-3B | ~25 tok/s | ~75-85 tok/s | +3x |
| Llama 3.1 8B | ~15 tok/s | ~60-75 tok/s | +4x |

**Key insight**: GPU advantage scales with model size! Small models show minimal speedup because CPU dequantization dominates. Large models will see dramatic improvements.

---

## 🚀 Ready to Go!

The infrastructure is solid. We just need to compile the real PTX code and replace the stub. Then we'll have **real Flash Attention acceleration** on your RTX 4070 Ti SUPER!

**Next command**: `./scripts/build_flash_attention.sh`

Let's get that speedup! 🎉
