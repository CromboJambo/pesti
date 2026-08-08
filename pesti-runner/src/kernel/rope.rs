//! GPU-accelerated RoPE (Rotary Positional Embeddings) kernel.
//!
//! Applies rotary embeddings to Q and K tensors in-place on the device:
//!   q_m' = q_m * cos(pos * theta) - q_{m+head_dim/2} * sin(pos * theta)
//!   k_m' = k_m * cos(pos * theta) - k_{m+head_dim/2} * sin(pos * theta)

use crate::cuda_runtime::CudaDeviceInfo;
use crate::kernel::attention::{AttentionArch, AttentionError};
use crate::kernel::device_buf::DeviceBuffer;
use cuda_core::CudaFunction;
use half::f16;
use std::sync::Arc;

/// GPU RoPE kernel for Blackwell tensor cores.
pub struct CudaRopeKernel {
    arch: AttentionArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    module: Arc<cuda_core::CudaModule>,
    function: CudaFunction,
}

/// Builder for CudaRopeKernel that handles PTX loading.
pub struct CudaRopeKernelBuilder {
    arch: AttentionArch,
    context: Arc<cuda_core::CudaContext>,
    stream: Arc<cuda_core::CudaStream>,
    device_info: CudaDeviceInfo,
}

impl CudaRopeKernelBuilder {
    pub fn new(
        arch: AttentionArch,
        context: Arc<cuda_core::CudaContext>,
        stream: Arc<cuda_core::CudaStream>,
        device_info: CudaDeviceInfo,
    ) -> Self {
        Self {
            arch,
            context,
            stream,
            device_info,
        }
    }

    /// Build the RoPE kernel by loading PTX module.
    pub fn build(self) -> Result<CudaRopeKernel, AttentionError> {
        // Load PTX source
        let ptx_src = match self.arch {
            AttentionArch::Wgmma => include_str!("ptx/attention_rope.ptx"),
            AttentionArch::Tcgen05 | AttentionArch::Cpu => {
                return Err(AttentionError::UnsupportedArch(
                    "RoPE currently only supports WGMMA".to_string(),
                ));
            }
        };

        // Load module from PTX
        let module = self
            .context
            .load_module_from_ptx_src(ptx_src)
            .map_err(|e| AttentionError::Cuda(format!("RoPE module load failed: {}", e)))?;

        // Resolve kernel function
        let function = module
            .load_function("attention_rope_kernel")
            .map_err(|e| AttentionError::Cuda(format!("RoPE function load failed: {}", e)))?;

        Ok(CudaRopeKernel {
            arch: self.arch,
            context: self.context,
            stream: self.stream,
            module,
            function,
        })
    }
}

impl CudaRopeKernel {
    /// Apply RoPE to Q and K tensors in-place on GPU.
    pub fn apply(
        &self,
        q: &mut DeviceBuffer<f16>,
        k: &mut DeviceBuffer<f16>,
        num_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        start_pos: usize,
    ) -> Result<(), AttentionError> {
        if !self.is_available() {
            return Err(AttentionError::NotAvailable);
        }

        // Validate dimensions
        if num_heads == 0 || head_dim == 0 || seq_q == 0 || seq_k == 0 {
            return Err(AttentionError::InvalidDimensions {
                num_heads,
                head_dim,
                seq_len: seq_q.max(seq_k),
            });
        }

        // Get device pointers
        let q_ptr = q.device_ptr();
        let k_ptr = k.device_ptr();

        unsafe {
            // Build kernel parameters
            let seq_q_val = seq_q as u32;
            let seq_k_val = seq_k as u32;
            let num_heads_val = num_heads as u32;
            let head_dim_val = head_dim as u32;
            let start_pos_val = start_pos as u32;

            // Convert base to f32 bits
            let rope_base_val = 10000.0f32.to_bits();

            let mut kernel_params: [*mut std::ffi::c_void; 8] = [
                &(q_ptr as u64) as *const u64 as *mut std::ffi::c_void,
                &(k_ptr as u64) as *const u64 as *mut std::ffi::c_void,
                &num_heads_val as *const u32 as *mut std::ffi::c_void,
                &seq_q_val as *const u32 as *mut std::ffi::c_void,
                &seq_k_val as *const u32 as *mut std::ffi::c_void,
                &head_dim_val as *const u32 as *mut std::ffi::c_void,
                &rope_base_val as *const u32 as *mut std::ffi::c_void,
                &start_pos_val as *const u32 as *mut std::ffi::c_void,
            ];

            // Launch configuration: one block per (head, pos) pair
            let grid_x = ((seq_q as u32 + 127) / 128).min(num_heads as u32);
            let grid_y = 1;
            let block_size = 128;

            cuda_core::launch_kernel(
                self.function.cu_function(),
                (grid_x, grid_y, 1),
                (block_size, 1, 1),
                0, // shared memory
                self.stream.cu_stream(),
                &mut kernel_params,
            )
            .map_err(|e| AttentionError::LaunchFailed(format!("RoPE launch failed: {}", e)))?;

            // Synchronize
            self.stream
                .synchronize()
                .map_err(|e| AttentionError::LaunchFailed(format!("RoPE sync failed: {}", e)))?;
        }

        Ok(())
    }

    pub fn context(&self) -> &Arc<cuda_core::CudaContext> {
        &self.context
    }

    pub fn stream(&self) -> &Arc<cuda_core::CudaStream> {
        &self.stream
    }

    pub fn is_available(&self) -> bool {
        unsafe { !self.function.cu_function().is_null() }
    }
}
