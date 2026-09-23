//! Fused GEMM + SwiGLU activation kernel for transformer FFN layers.
//!
//! Computes: y = SwiGLU(x @ W^T) where SwiGLU(z) = silu(z[:half]) * z[half:]
//! in a single kernel launch, eliminating the intermediate tensor write
//! between GEMM and activation kernels.
//!
//! Input layout: x is [batch, embed_dim] row-major F32 on device
//!              W is [out_features*2, embed_dim] projection weights on device (F16)
//! Output: y is [batch, out_features] F32 on device

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    safe::{CudaContext, CudaStream},
    sys,
};

/// Configuration for fused GEMM + SwiGLU kernel.
#[derive(Debug)]
pub struct FusedGemmSwigluConfig {
    pub embed_dim: usize,
    pub out_features: usize,
    pub batch_size: usize,
}

impl Default for FusedGemmSwigluConfig {
    fn default() -> Self {
        let embed_dim = 4096;
        let out_features = 11008; // Typical FFN intermediate size (2.75x embed)
        Self {
            embed_dim,
            out_features,
            batch_size: 1,
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA fused GEMM + SwiGLU kernel.
#[derive(Clone)]
pub struct FusedGemmSwigluKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl FusedGemmSwigluKernel {
    /// Load and compile the fused GEMM+SwiGLU kernel.
    pub fn load(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, String> {
        let ptx_source = include_str!("ptx/fused_gemm_swiGLU.ptx");

        let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_source)
            .map_err(|e| format!("module load failed: {:?}", e))?;

        let function = module
            .load_function("fused_gemm_swiglu_kernel")
            .map_err(|e| format!("function lookup failed: {:?}", e))?;

        Ok(Self {
            context,
            stream,
            module,
            function,
        })
    }

    /// Launch fused GEMM + SwiGLU kernel.
    pub fn launch(
        &self,
        x: u64,
        w: u64,
        y: u64,
        embed_dim: usize,
        out_features: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        let mut x_v: u64 = x;
        let mut w_v: u64 = w;
        let mut y_v: u64 = y;
        let mut embed_dim_v: i32 = embed_dim as i32;
        let mut out_features_v: i32 = out_features as i32;
        let mut batch_size_v: i32 = batch_size as i32;

        let mut params = [
            &mut x_v as *mut u64 as *mut std::ffi::c_void,
            &mut w_v as *mut u64 as *mut std::ffi::c_void,
            &mut y_v as *mut u64 as *mut std::ffi::c_void,
            &mut embed_dim_v as *mut i32 as *mut std::ffi::c_void,
            &mut out_features_v as *mut i32 as *mut std::ffi::c_void,
            &mut batch_size_v as *mut i32 as *mut std::ffi::c_void,
        ];

        let grid = (batch_size as u32, 1u32, 1u32);
        let block = ((out_features / 4).min(1024) as u32, 1u32, 1u32);

        unsafe {
            crate::cuda_shim::launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0u32,
                crate::cuda_shim::cu_stream(&self.stream),
                &mut params,
            )
            .map_err(|e| format!("kernel launch failed: {:?}", e))?;
        }

        Ok(())
    }
}