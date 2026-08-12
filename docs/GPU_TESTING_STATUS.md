# GPU Benchmark Status (updated)

## What Works Now

The custom CUDA kernel layer in `pesti-runner` has a **real, verified GEMM kernel**:

- **`src/kernel/ptx/gemm_mma_sync.ptx`** — classic warp-level tensor cores
  (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`), one warp per 16x8
  output tile, target sm_80 (forward-compatible JIT to sm_89/sm_120).
- **Numerically verified** on both GPUs (4070 Ti SUPER sm_8.9, 5060 Ti sm_12.0):
  - 40x24x32 odd-dim boundary: 0/960 element errors vs CPU reference
  - 1024x1024x1024: 0/1,048,576 errors, max abs error 0.0
  - 1024^3 GEMM: ~0.45 ms on 5060 Ti (naive single-thread CPU ref: 47 s)
- **InferenceEngine reports honestly**: `GPU available: true`,
  `Backend: GPU (cuda:... | NVIDIA GeForce RTX ...)`, `GEMM arch: Mma`.

## Key Fixes (vs earlier broken state)

1. **CUDA context ordering** — `cuMemGetInfo_v2` was called before
   `CudaContext::new` (needs a current context). Context is now created first.
2. **Device ordinal routing** — engine now uses the requested `Device::Cuda(ordinal)`
   (via `location()`), not hardcoded device 0.
3. **CPU engine no longer tries CUDA init** — gated on `Device::Cuda(_)` only,
   not NVML `is_available()`.
4. **`backend_description()` / `gpu_available()`** report the truth via a
   `gpu_gemm` flag set only when a CUDA kernel actually built.
5. **candle-core/cuda feature wired** into pesti's `cuda` feature — without it,
   `Device::cuda_if_available()` silently returned `Device::Cpu`.
6. **Arch model corrected** (was backwards):
   - `wgmma` = Hopper sm_90a ONLY (ptxas rejects it on sm_120a)
   - `tcgen05` = datacenter Blackwell sm_100a/sm_103a ONLY
   - `mma.sync` = every tensor-core GPU incl. consumer Blackwell (RTX 50)
   - Capability gates in `cuda_runtime.rs` updated to match.
7. **Kernel launch fixed** — was passing block_dims=(m,n,k) and 3 args to an
   8-param kernel, and passing device pointers as param *values* (driver
   dereferenced them on host -> segfault). Now grid=(n/8, m/16), block=(32,1,1),
   params are host-side values of the exact kernel-param types.
8. **PTX must be ASCII-only** — the driver's embedded ptxas rejects non-ASCII
   in comments ("Unexpected non-ASCII character") even though standalone ptxas
   13.3 accepts it.

## Fragment Layout (mma.m16n8k16 f16, per PTX ISA 9.7.15.5)

g = lane/4 (groupID), t = lane%4, c = 2t:

```
A: reg0={A[g][c],A[g][c+1]} reg1={A[g+8][c],A[g+8][c+1]}
   reg2={A[g][c+8],A[g][c+9]} reg3={A[g+8][c+8],A[g+8][c+9]}
B: reg0={B[2t][g],B[2t+1][g]} reg1={B[2t+8][g],B[2t+9][g]}
   (B pairs are k-consecutive at same n -> stride n*2 bytes in memory!)
D: d0=D[g][c] d1=D[g][c+1] d2=D[g+8][c] d3=D[g+8][c+1]
```

Gotchas hit: A pairs are column-adjacent (stride 2 bytes); B pairs are
k-adjacent (stride n*2 bytes); B needs `tile_col*8` column offset; predicates
must not be reused across fragments with different bounds.

## How to Run

```bash
export HF_HOME=/home/crombo/projects/llm-workspace/llmstudio_models
cargo run --package pesti-runner --features cuda --example gemm_mma_verify
cargo run --package pesti-runner --features cuda --example simple_gpu_verify
```

## What Still Does NOT Work

- `attention` kernel: PTX files (`attention_wgmma.ptx` etc.) are still
  pseudocode stubs; attention falls back to CPU.
- Old `gemm_wgmma_real.ptx` / `gemm_tcgen05_real.ptx` are stubs, not real PTX
  (wgmma/tcgen05 don't exist on consumer GPUs anyway).
- `pesti-conformance` has pre-existing compile errors (private `Linear.weight`
  field) unrelated to this work.
- llama.cpp FFI path (LlamaRunner) is CPU-only unless built with
  `--features llama-cpp-2/cuda` (11-min NVCC build; benchmark example
  `llama_gpu_vs_cpu` shows ~82 tok/s CPU vs ~311 tok/s GPU for
  qwen2.5-0.5b-q4_k_m).
