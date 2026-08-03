//! GPU kernel primitives for LLM inference.
//!
//! Provides GEMM (matrix multiply) and attention kernels targeting NVIDIA tensor cores.
//! Supports two architectures:
//! - WGMMA (sm_120, consumer Blackwell: RTX 5060 Ti / 5090)
//! - tcgen05 (sm_100, datacenter Blackwell: B200)
//!
//! ## Architecture
//!
//! ```text
//! kernel/
//!   mod.rs          - module root, re-exports
//!   device_buf.rs   - DeviceBuffer<T> (host Vec / device ptr abstraction)
//!   gemm.rs         - GEMM trait, config, types, CPU fallback
//!   builder.rs      - PTX builder, kernel registration, launch config
//!   tma_descriptor.rs - Blackwell TMA global cache read descriptor (speculative bit layout)
//!   tma_bridge.rs   - Bridge to cuda-oxide 128-byte TmaDescriptor + host-side creation
//!   kvcache.rs      - KV cache with TMA descriptor support
//!   attention.rs    - Attention kernel trait, config, CPU fallback
//! ```
//!
//! ## Build Pipeline
//!
//! 1. Kernel source written with cuda-oxide `#[kernel]` attribute
//! 2. `cargo oxide build` compiles to PTX
//! 3. `GemmBuilder` loads PTX and produces launchable `GemmKernel`
//! 4. `InferenceEngine` uses `GemmKernel` and `AttentionKernel` for computation
//!
//! ## KV Cache
//!
//! KV cache uses TMA descriptors for async GMEM→SMEM copies during attention.
//! Layout: `[num_heads * head_dim, max_seq]` contiguous per layer (sequence
//! dimension is contiguous for efficient TMA transfers).
//!
//! ```text
//! Kvcache::new(num_heads, head_dim, max_seq, on_device)
//!   → append(key, value)
//!   → tma_descriptor(gmem_addr, is_key, head_idx, box_y)
//!   → KvcacheSlice → to_tma_descriptor()
//! ```
//!
//! ## Attention
//!
//! Scaled dot-product attention: `softmax(Q @ K^T / sqrt(head_dim)) @ V`
//! - Prefill: full KV cache loaded via TMA, all positions processed
//! - Decode: single position (box_y=1), append new KV, compute attention over cache
//!
//! ```text
//! AttentionKernel::forward(query, key_cache, value_cache, mask, config)
//!   → AttentionSlice from Kvcache
//!   → per-head TMA descriptor wiring
//!   → output [query_seq_len x (num_heads * head_dim)] f32
//! ```
//!
//! ## Design Decisions
//!
//! - `DeviceBuffer<T>` abstracts over host Vec and device pointer - no cuda-oxide
//!   dependency in the core trait, keeping the crate compileable without GPU toolchain
//! - GEMM trait uses f16 inputs with f32 accumulation - matches LLM inference patterns
//! - AttentionKernel mirrors GemmKernel pattern (Send + Sync, CPU fallback)
//! - PTX builder accepts pre-compiled blobs - cuda-oxide kernels compiled via `cargo oxide`
//! - CPU fallback (`CpuGemmKernel`, `CpuAttentionKernel`) enables testing without GPU hardware
//! - TMA descriptor is a 128-bit (4 u32) struct matching Blackwell hardware layout
//! - KV cache stores K and V in a single contiguous buffer with V offset by head_stride * max_seq
//! - tcgen05: K must be divisible by 64 (tile constraint), 128-thread blocks, 128x128x16 tiles

pub mod attention;
pub mod builder;
pub mod candle_bridge;
pub mod device_buf;
pub mod dispatch;
pub mod gemm;
pub mod gemm_cutlass;
pub mod kvcache;
pub mod memory;
pub mod mistralrs_backend;
pub mod rope;
pub mod tma_bridge;
pub mod tma_descriptor;

pub use attention::{
    AttentionArch, AttentionConfig, AttentionError, AttentionKernel, AttentionSlice,
    CpuAttentionKernel, CudaAttentionKernel, CudaAttentionKernelBuilder,
};
pub use builder::{GemmBuilder, KernelFromPtx, PtxSource};
pub use device_buf::{DeviceBuffer, DeviceBufferError, HostBuffer};
pub use gemm::{CpuGemmKernel, CudaGemmKernel, CudaGemmKernelBuilder, GemmArch, GemmConfig, GemmError, GemmKernel};
pub use kvcache::{KvError, Kvcache, KvcacheSlice};
pub use memory::{CpuMemoryBackend, CudaMemoryBackend, MemoryBackend, MemoryError, MemoryManager, RawHandle};
pub use tma_bridge::HostTmaDescriptor;
pub use tma_descriptor::TmaDescriptor;
pub use dispatch::{
    AttentionDispatch, DispatchContext, DispatchError, FeedForwardDispatch, LayerDispatch,
    LinearDispatch, RmsNormDispatch,
};
