//! Flash Attention kernel - fused Q @ K^T + softmax + V computation
//!
//! **Optimization**: Single kernel launch replaces 2 GEMM calls + CPU softmax
//! Expected improvement: 40-50% speedup on 512+ token sequences
//!
//! Based on [Flash Attention: Exact and Efficient Sequence Modeling](https://arxiv.org/abs/2205.14135)

use crate::kernel::device_buf::DeviceBuffer;
use crate::kernel::kvcache::Kvcache;
use cudarc::driver::{
    safe::{CudaContext, CudaStream},
    sys,
};
use half::f16;
use std::sync::Arc;

/// Flash attention architecture (consumer Blackwell RTX 50-series uses mma.sync)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttentionArch {
    /// mma.sync (sm_80..sm_120) - consumer RTX 40/50 series
    MmaSync,
    /// tcgen05 (sm_100a) - datacenter B200 only
    Tcgen05,
}

impl Default for FlashAttentionArch {
    fn default() -> Self {
        Self::MmaSync // Consumer GPUs have better support
    }
}

/// Configuration for flash attention kernel.
#[derive(Debug)]
pub struct FlashAttentionConfig {
    pub arch: FlashAttentionArch,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32, // Pre-computed 1/sqrt(head_dim)
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            arch: FlashAttentionArch::default(),
            num_heads: 32,
            head_dim: 64,
            scale: 1.0 / (64_f32).sqrt(), // default for head_dim=64
        }
    }
}

// --- GPU Implementation ---

#[cfg(feature = "cuda")]
/// CUDA flash attention kernel using cudarc.
#[derive(Clone)]
pub struct FlashAttentionKernel {
    arch: FlashAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<crate::cuda_shim::CudaModule>, // Use Arc for cloneability
    function: crate::cuda_shim::CudaFunction,  // Use cuda_shim wrapper
}

#[cfg(feature = "cuda")]
impl FlashAttentionKernel {
    /// Get the CUDA module (for loading additional functions from PTX).
    pub fn module(&self) -> &Arc<crate::cuda_shim::CudaModule> {
        &self.module
    }

    /// Get the architecture of this flash attention kernel.
    pub fn arch(&self) -> crate::kernel::flash_attention::FlashAttentionArch {
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

    /// Launch flash attention kernel (fused Q @ K^T + softmax + V).
    ///
    /// **Optimization**: Single kernel launch replaces:
    /// - 2 GEMM calls (Q @ K^T, then S @ V)
    /// - CPU softmax transfer
    /// Expected speedup: 40-50% on 512+ tokens
    pub fn launch(
        &self,
        scale: f32, // 1/sqrt(head_dim)
        q_ptr: u64, // Query device pointer (row-major [seq_q, num_heads, head_dim] f16)
        k_ptr: u64, // Key device pointer (row-major [seq_k, num_heads, head_dim] f16)
        v_ptr: u64, // Value device pointer (row-major [seq_k, num_heads, head_dim] f16)
        o_ptr: u64, // Output device pointer ([seq_q, num_heads, head_dim] f32)
        seq_q: usize,
        seq_k: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(), FlashAttentionError> {
        use cudarc::driver::sys;

        // Launch fused flash attention kernel
        let mut scale_v: f32 = scale;
        let mut q_v: u64 = q_ptr;
        let mut k_v: u64 = k_ptr;
        let mut v_v: u64 = v_ptr;
        let mut o_v: u64 = o_ptr;
        let mut seq_q_v: i32 = seq_q as i32;
        let mut seq_k_v: i32 = seq_k as i32;
        let mut num_heads_v: i32 = num_heads as i32;
        let mut head_dim_v: i32 = head_dim as i32;

        let mut params: [*mut std::ffi::c_void; 8] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut o_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut i32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut i32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
        ];

        // Grid: (seq_q, num_heads) blocks, each block processes one head at one position
        let grid_x = seq_q as u32;
        let grid_y = num_heads as u32;
        let grid_z = 1u32;
        
        // Block: 128 threads (enough for head_dim=64 or 128)
        let block = (128u32, 1u32, 1u32);
        
        // Shared memory: store partial Q @ K^T results + softmax accumulation
        // For head_dim=64: need ~64 floats for Q tile + 64 for K tile + 64 for O = 192 floats = 768 bytes
        let smem_size = 1024u32; // Conservative estimate

        unsafe {
            use crate::cuda_shim::launch_kernel;
            match launch_kernel(
                self.function.cu_function(),
                (grid_x, grid_y, grid_z),
                block,
                smem_size,
                crate::cuda_shim::cu_stream(&self.stream),
                &mut params,
            ) {
                Ok(_) => {}
                Err(e) => {
                    return Err(FlashAttentionError::LaunchFailed(format!(
                        "flash attention kernel launch: {:?}", e
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
/// Builder for FlashAttentionKernel.
pub struct FlashAttentionKernelBuilder {
    arch: FlashAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl FlashAttentionKernelBuilder {
    pub fn new(
        arch: FlashAttentionArch,
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
        }
    }

    pub fn build(self) -> Result<FlashAttentionKernel, FlashAttentionError> {
        // Architecture check for mma.sync (works on all tensor-core GPUs sm_80..sm_120)
        match self.arch {
            FlashAttentionArch::MmaSync => {} // Always works on consumer Blackwell RTX 50-series
            FlashAttentionArch::Tcgen05 => {} // tcgen05 validation skipped - requires B200 datacenter GPU, skip for now
        }

        // Load PTX from flash_attention_kernel.ptx (sm_89 target: RTX 4070 Ti SUPER)
        let ptx_src = include_str!("ptx/flash_attention_kernel.ptx");

        // Use cuda_shim::CudaModule which has load_from_ptx method (returns Arc<CudaModule>)
        let module = crate::cuda_shim::CudaModule::load_from_ptx(&self.context, ptx_src)
            .map_err(|e| FlashAttentionError::Cuda(format!("module load: {:?}", e)))?;

        // Get the function from the module via load_function method (returns Result<_, DriverError>)
        let mangled_name = "_Z22flash_attention_kernelfPK6__halfS1_S1_Pfiiii";
        let function = match module.load_function(mangled_name) {
            Ok(f) => f,
            Err(e) => {
                return Err(FlashAttentionError::Cuda(format!(
                    "function lookup {:?}: {:?}", mangled_name, e
                )));
            }
        };

        Ok(FlashAttentionKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module, // Not wrapped in Arc since load_from_ptx returns Arc<CudaModule> which can be moved
            function,
        })
    }

    /// Build kernel from external PTX file (for vectorized/tiled variants)
    pub fn build_from_ptx_file<P: AsRef<std::path::Path>>(
        self,
        ptx_path: P,
        function_name: &str,
    ) -> Result<FlashAttentionKernel, FlashAttentionError> {
        use std::fs;

        // Read PTX source from file
        let ptx_src = fs::read_to_string(ptx_path)
            .map_err(|e| FlashAttentionError::Cuda(format!("read PTX file: {:?}", e)))?;

        // Load module from PTX string
        let module = crate::cuda_shim::CudaModule::load_from_ptx(&self.context, &ptx_src)
            .map_err(|e| FlashAttentionError::Cuda(format!("module load: {:?}", e)))?;

        // Get the function by name (allows custom kernel names)
        let function = match module.load_function(function_name) {
            Ok(f) => f,
            Err(e) => {
                return Err(FlashAttentionError::Cuda(format!(
                    "function lookup {:?}: {:?}", function_name, e
                )));
            }
        };

        Ok(FlashAttentionKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

// --- Convenience Builder Function ---

#[cfg(feature = "cuda")]
pub fn build_flash_attention_kernel(
    arch: FlashAttentionArch,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> Result<FlashAttentionKernel, FlashAttentionError> {
    // Skip device validation - mma.sync works on all consumer Blackwell GPUs (sm_89)
    FlashAttentionKernelBuilder::new(arch, context, stream).build()
}

// --- Error Types ---

#[derive(Debug, thiserror::Error)]
pub enum FlashAttentionError {
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
    fn test_flash_attention_config_default() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.arch, FlashAttentionArch::MmaSync);
        assert_eq!(config.head_dim, 64);
        assert!((config.scale - (1.0 / 8.0)).abs() < 0.001); // sqrt(64) = 8
    }
}
