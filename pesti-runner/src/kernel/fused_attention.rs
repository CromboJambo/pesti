//! Fused attention kernel: RoPE + Q@K^T + softmax in one launch.
//!
//! This replaces the 2-GEMM CPU-softmax path with a single fused kernel,
//! eliminating intermediate H2D transfers and precision loss from f32→f16 conversions.

use crate::cuda_runtime::{CudaRuntime, IntoResult};
use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::Kvcache;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// Fused attention architecture selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedAttentionArch {
    /// Consumer Blackwell (RTX 40-series, RTX 50-series) using mma.sync
    MmaSync,
    /// Datacenter B200 using tcgen05
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
    /// Architecture (mma.sync or tcgen05)
    pub arch: FusedAttentionArch,
    /// Whether to use TMA for async transfers (tcgen05 only)
    pub use_tma: bool,
}

impl Default for FusedAttentionConfig {
    fn default() -> Self {
        Self {
            arch: FusedAttentionArch::default(),
            use_tma: false, // mma.sync doesn't use TMA
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
    module: Arc<CudaModule>,
    function: CudaFunction,
}

#[cfg(feature = "cuda")]
/// Builder for FusedAttentionKernel that handles PTX loading.
pub struct FusedAttentionKernelBuilder {
    arch: FusedAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_info: crate::cuda_runtime::CudaDeviceInfo,
}

#[cfg(feature = "cuda")]
impl FusedAttentionKernelBuilder {
    pub fn new(
        arch: FusedAttentionArch,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        device_info: crate::cuda_runtime::CudaDeviceInfo,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
            device_info,
        }
    }

    /// Build the kernel by loading PTX module and resolving function.
    pub fn build(self) -> Result<FusedAttentionKernel, AttentionError> {
        // Pre-flight architecture check for mma.sync (sm_80..sm_120)
        match self.arch {
            FusedAttentionArch::MmaSync => {
                // mma.sync works on all tensor-core GPUs, no gate needed
            }
            FusedAttentionArch::Tcgen05 if !self.device_info.supports_tcgen05() => {
                return Err(AttentionError::UnsupportedArch(format!(
                    "tcgen05 requires sm_100a (datacenter Blackwell), but device is sm_{}.{}",
                    self.device_info.compute_capability.0, self.device_info.compute_capability.1
                )));
            }
        }

        // Select PTX based on architecture
        let ptx_src = match self.arch {
            FusedAttentionArch::MmaSync => include_str!("ptx/attention_rope_softmax.ptx"),
            FusedAttentionArch::Tcgen05 => include_str!("ptx/attention_tcgen05.ptx"),
        };

        // Load module from PTX source
        let module = CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| AttentionError::Cuda(format!("module load failed: {e:?}")))?;

        // Resolve kernel function
        let kernel_name = match self.arch {
            FusedAttentionArch::MmaSync => "fused_attention_kernel",
            FusedAttentionArch::Tcgen05 => "attention_tcgen05_kernel",
        };
        let function = module
            .load_function(kernel_name)
            .map_err(|e| AttentionError::Cuda(format!("function load failed: {e:?}")))?;

        Ok(FusedAttentionKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

#[cfg(feature = "cuda")]
impl FusedAttentionKernel {
    /// Launch the fused attention kernel.
    pub fn launch(
        &self,
        scale: f32, // 1/sqrt(head_dim)
        q_ptr: u64, // Query device pointer
        k_ptr: u64, // Key device pointer  
        v_ptr: u64, // Value device pointer
        s_ptr: u64, // Output softmax scores device pointer
        seq_q: usize,
        seq_k: usize,
        num_heads: usize,
        head_dim: usize,
        rope_base: f32,
        max_pos: usize,
    ) -> Result<(), AttentionError> {
        use cudarc::driver::sys;

        // Kernel signature from attention_rope_softmax.ptx:
        //   fused_attention_kernel(f32 scale, u64 q_ptr, u64 k_ptr, u64 v_ptr,
        //                          u64 s_ptr, s32 seq_q, s32 seq_k, s32 num_heads,
        //                          s32 head_dim, f32 rope_base, s32 max_pos)
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

        let params: [*mut std::ffi::c_void; 12] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut s_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut i32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut i32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut i32 as *mut std::ffi::c_void,
            &mut rope_base_v as *mut f32 as *mut std::ffi::c_void,
            &mut max_pos_v as *mut i32 as *mut std::ffi::c_void,
        ];

        // Kernel config: 128 threads per block (4 warps), 64x64 tiles
        let grid = (1u32, 1u32, 1u32); // Single launch for simplicity
        let block = (128u32, 1u32, 1u32);

        unsafe {
            use crate::cuda_shim::launch_kernel;
            launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0, // shared_mem_bytes (PTX declares in kernel)
                self.stream.cu_stream(),
                &mut params,
            )
            .map_err(|e| AttentionError::LaunchFailed(format!("kernel launch failed: {e:?}")))?;
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
/// Helper to create fused attention kernel from inference engine context.
pub fn build_fused_attention_kernel(
    arch: FusedAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_info: crate::cuda_runtime::CudaDeviceInfo,
) -> Result<FusedAttentionKernel, AttentionError> {
    let builder = FusedAttentionKernelBuilder::new(arch, context, stream, device_info);
    builder.build()
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

    #[error("PTX compilation failed: {0}")]
    PtxCompile(String),

    #[error("not available")]
    NotAvailable,

    #[error("invalid dimensions: num_heads={num_heads}, head_dim={head_dim}, seq_q={seq_q}, seq_k={seq_k}")]
    InvalidDimensions {
        num_heads: usize,
        head_dim: usize,
        seq_q: usize,
        seq_k: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_attention_arch() {
        assert_eq!(FusedAttentionArch::MmaSync as u8, 0); // Placeholder for enum value
        assert_eq!(FusedAttentionArch::Tcgen05 as u8, 1);
    }

    #[test]
    fn test_fused_attention_config_default() {
        let config = FusedAttentionConfig::default();
        assert_eq!(config.arch, FusedAttentionArch::MmaSync);
        assert!(!config.use_tma);
    }
}
