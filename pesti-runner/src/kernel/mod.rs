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
//!   softmax.rs      - Softmax kernel trait, CPU/CUDA implementations
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
//!
#[cfg(feature = "cuda")]
pub mod attention;
#[cfg(not(feature = "cuda"))]
pub mod attention_stub;
#[cfg(feature = "cuda")]
pub mod builder;
#[cfg(not(feature = "cuda"))]
pub mod builder_stub;
pub mod candle_bridge;
pub mod device_buf;
pub mod dispatch;
#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod gemm_cutlass;
#[cfg(not(feature = "cuda"))]
pub mod gemm_stub;
#[cfg(feature = "cuda")]
pub mod kvcache;
#[cfg(not(feature = "cuda"))]
pub mod kvcache_stub;
#[cfg(feature = "cuda")]
pub mod memory;
#[cfg(not(feature = "cuda"))]
pub mod memory_stub;
#[cfg(feature = "cuda")]
pub mod mistralrs_backend;
#[cfg(feature = "cuda")]
pub mod rope;
#[cfg(feature = "cuda")]
pub mod softmax;
#[cfg(feature = "cuda")]
pub mod tma_bridge;
#[cfg(feature = "cuda")]
pub mod tma_descriptor;

#[cfg(feature = "cuda")]
pub use attention::{
    AttentionArch, AttentionConfig, AttentionError, AttentionKernel, AttentionSlice,
    CpuAttentionKernel, CudaAttentionKernel, CudaAttentionKernelBuilder, GemmBasedAttentionKernel,
};
#[cfg(not(feature = "cuda"))]
pub use attention_stub::{
    AttentionArch, AttentionConfig, AttentionError, AttentionKernel, AttentionSlice,
    CpuAttentionKernel,
};
#[cfg(feature = "cuda")]
pub use builder::{GemmBuilder, KernelFromPtx, PtxSource};
#[cfg(not(feature = "cuda"))]
pub use builder_stub::GemmBuilder;
pub use device_buf::{DeviceBuffer, DeviceBufferError, HostBuffer};
#[cfg(feature = "cuda")]
pub use dispatch::{
    AttentionDispatch, DispatchContext, DispatchError, FeedForwardDispatch, LayerDispatch,
    LinearDispatch, RmsNormDispatch,
};
#[cfg(not(feature = "cuda"))]
pub use dispatch::{
    AttentionDispatch, DispatchContext, DispatchError, FeedForwardDispatch, LayerDispatch,
    LinearDispatch, RmsNormDispatch,
};
#[cfg(feature = "cuda")]
pub use gemm::{CpuGemmKernel, GemmArch, GemmConfig, GemmError, GemmKernel};
#[cfg(feature = "cuda")]
pub use gemm::{CudaGemmKernel, CudaGemmKernelBuilder};
#[cfg(not(feature = "cuda"))]
pub use gemm_stub::{CpuGemmKernel, GemmArch, GemmConfig, GemmError, GemmKernel};
#[cfg(feature = "cuda")]
pub use kvcache::{KvError, Kvcache, KvcacheSlice};
#[cfg(not(feature = "cuda"))]
pub use kvcache_stub::{KvError, Kvcache, KvcacheSlice, TmaDescriptor};
#[cfg(feature = "cuda")]
pub use memory::{
    CpuMemoryBackend, CudaMemoryBackend, MemoryBackend, MemoryError, MemoryManager, RawHandle,
};
#[cfg(not(feature = "cuda"))]
pub use memory_stub::{CpuMemoryBackend, MemoryBackend, MemoryError, MemoryManager, RawHandle};
#[cfg(feature = "cuda")]
pub use tma_bridge::HostTmaDescriptor;
#[cfg(feature = "cuda")]
pub use tma_descriptor::TmaDescriptor;
#[cfg(feature = "cuda")]
pub use softmax::{SoftmaxError, SoftmaxKernel, SoftmaxKernelBuilder};
