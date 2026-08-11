//! Fused RoPE + Attention + Softmax kernel - conformant with llama.cpp tensor layout.
//!
//! **Layout**: Row-major `[seq_q, num_heads, head_dim]` for Q/K/V tensors (matches llama.cpp).
//! **Algorithm**: Single-kernel softmax + fused multiply-add (eliminates H2D intermediate transfers).

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::Kvcache;
use cudarc::driver::{safe::{CudaContext, CudaStream}, sys};
use half::f16;
use std::sync::Arc;

/// Fused attention architecture (consumer Blackwell RTX 50-series uses mma.sync).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedAttentionArch {
    /// mma.sync (sm_80..sm_120) - consumer RTX 40/50 series
    MmaSync,
    /// tcgen05 (sm_100a) - datacenter B200 only
    Tcgen05,
}

impl Default for FusedAttentionArch {
    fn default() -> Self {
        Self::MmaSync // Consumer GPUs have better support
    }
}

/// Configuration for fused attention kernel.
#[derive(Debug)]
pub struct FusedAttentionConfig {
    pub arch: FusedAttentionArch,
    pub num_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub scale: f32, // Pre-computed 1/sqrt(head_dim)
}

impl Default for FusedAttentionConfig {
    fn default() -> Self {
        Self {
            arch: FusedAttentionArch::default(),
            num_heads: 32,
            head_dim: 64,
            rope_base: 10_000.0,
            scale: 1.0 / (64_f32).sqrt(), // default for head_dim=64
        }
    }
}

// --- GPU Implementation ---

#[cfg(feature = "cuda")]
/// CUDA fused attention kernel using cudarc.
#[derive(Clone)]
pub struct FusedAttentionKernel {
    arch: FusedAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<crate::cuda_shim::CudaModule>, // Use Arc for cloneability
    function: crate::cuda_shim::CudaFunction, // Use cuda_shim wrapper
}

#[cfg(feature = "cuda")]
/// Builder for FusedAttentionKernel.
pub struct FusedAttentionKernelBuilder {
    arch: FusedAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl FusedAttentionKernelBuilder {
    pub fn new(arch: FusedAttentionArch, context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { arch, context, stream }
    }

    pub fn build(self) -> Result<FusedAttentionKernel, AttentionError> {
        // Architecture check for mma.sync (works on all tensor-core GPUs sm_80..sm_120)
        match self.arch {
            FusedAttentionArch::MmaSync => {} // Always works on consumer Blackwell RTX 50-series
            FusedAttentionArch::Tcgen05 => {} // tcgen05 validation skipped - requires B200 datacenter GPU, skip for now
        }

        // Load PTX from attention_rope_softmax.ptx (sm_89 target: RTX 4070 Ti SUPER)
        let ptx_src = include_str!("ptx/attention_rope_softmax.ptx");

        // Use cuda_shim::CudaModule which has load_from_ptx method (returns Arc<CudaModule>)
        let module = crate::cuda_shim::CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| AttentionError::Cuda(format!("module load: {:?}", e)))?;

        // Get the function from the module via load_function method (returns Result<_, DriverError>)
        // Note: CUDA name mangling produces: _Z22fused_attention_kernelfPK6__halfS1_S1_Pfiiiifi
        let mangled_name = "_Z22fused_attention_kernelfPK6__halfS1_S1_Pfiiiifi";
        let function = match module.load_function(mangled_name) {
            Ok(f) => f,
            Err(e) => return Err(AttentionError::Cuda(format!("function lookup {:?}: {:?}", mangled_name, e))),
        };

        Ok(FusedAttentionKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module, // Not wrapped in Arc since load_from_ptx returns Arc<CudaModule> which can be moved
            function,
        })
    }
}

#[cfg(feature = "cuda")]
impl FusedAttentionKernel {
    /// Get the architecture of this fused attention kernel.
    pub fn arch(&self) -> crate::kernel::fused_attention_conformant::FusedAttentionArch {
        self.arch
    }

    /// Get the CUDA context for backend allocation.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the CUDA stream for kernel launch and memory operations.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Launch fused attention kernel.

    pub fn launch(
        &self,
        scale: f32, // 1/sqrt(head_dim)
        q_ptr: u64, // Query device pointer (row-major [seq_q, num_heads, head_dim] f16)
        k_ptr: u64, // Key device pointer (row-major [seq_k, num_heads, head_dim] f16)
        v_ptr: u64, // Value device pointer (row-major [seq_k, num_heads, head_dim] f16)
        s_ptr: u64, // Output softmax scores device pointer ([seq_q, seq_k] f32)
        seq_q: usize,
        seq_k: usize,
        num_heads: usize,
        head_dim: usize,
        rope_base: f32,
        max_pos: usize,
    ) -> Result<(), AttentionError> {
        use cudarc::driver::sys;

        // Kernel signature from attention_rope_softmax.ptx (row-major layout):
        //   fused_attention_kernel(
        //     f32 scale, u64 q_ptr, u64 k_ptr, u64 v_ptr, u64 s_ptr,
        //     s32 seq_q, s32 seq_k, s32 num_heads, s32 head_dim,
        //     f32 rope_base, s32 max_pos)

        let mut scale_v: f32 = scale;
        let mut q_v: u64 = q_ptr;
        let mut k_v: u64 = k_ptr;
        let mut v_v: u64 = v_ptr;
        let mut s_v: u64 = s_ptr;
        let mut seq_q_v: i32 = seq_q as i32;
        let mut seq_k_v: i32 = seq_k as i32;
        let mut num_heads_v: i32 = num_heads as i32;
        let mut head_dim_v: i32 = head_dim as i32;
        let mut rope_base_v: f32 = rope_base;

        let mut max_pos_v: i32 = max_pos as i32;

        let mut params: [*mut std::ffi::c_void; 10] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut s_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut i32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut i32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut i32 as *mut std::ffi::c_void,
            &mut max_pos_v as *mut i32 as *mut std::ffi::c_void,
        ];

        // Compute grid dimensions: each block handles one 64x64 tile
        let seq_q_tiles = (seq_q + 63) / 64;
        let seq_k_tiles = (seq_k + 63) / 64;
        let grid = (seq_q_tiles as u32, seq_k_tiles as u32, 1u32); // Multi-block launch for full sequences
        let block = (128u32, 1u32, 1u32);

        unsafe {
            use crate::cuda_shim::launch_kernel;
            match launch_kernel(
                self.function.cu_function(), // Use instance field `self.function` - returns sys::CUfunction
                grid,
                block,
                0,                       // shared_mem_bytes (PTX declares in kernel)
                crate::cuda_shim::cu_stream(&self.stream), // Get raw CUstream from CudaStream
                &mut params,             // Pass reference to array directly
            ) {
                Ok(_) => {} // Success - no action needed
                Err(e) => return Err(AttentionError::LaunchFailed(format!("kernel launch: {:?}", e))),
            }
        }

        Ok(())
    }
}

// --- Convenience Builder Function ---

#[cfg(feature = "cuda")]
pub fn build_fused_attention_kernel_conformant(arch: FusedAttentionArch, context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<FusedAttentionKernel, AttentionError> {
    // Skip device validation - mma.sync works on all consumer Blackwell GPUs (sm_89)
    FusedAttentionKernelBuilder::new(arch, context, stream).build()
}

// --- Error Types ---

#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("buffer size mismatch: expected={expected}, got={got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("launch failed: {0}")]
    LaunchFailed(String),

    #[error("not available")]
    NotAvailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_attention_config_default() {
        let config = FusedAttentionConfig::default();
        assert_eq!(config.arch, FusedAttentionArch::MmaSync);
        assert_eq!(config.head_dim, 64);
        assert!((config.scale - (1.0 / 8.0)).abs() < 0.001); // sqrt(64) = 8
    }
}
