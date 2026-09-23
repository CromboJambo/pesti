//! Fused RMSNorm + QKV projection kernel for transformer attention layers.
//!
//! Combines RMSNorm normalization with all three attention projections (Q, K, V)
//! into a single kernel launch. Computes:
//!   norm = RMSNorm(x)
//!   q = norm @ Wq^T
//!   k = norm @ Wk^T  
//!   v = norm @ Wv^T
//! Output is concatenated as [batch, 3*embed_dim] on device.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{safe::{CudaContext, CudaStream}, sys};

/// Configuration for fused RMSNorm + QKV kernel.
#[derive(Debug)]
pub struct FusedRMSNormQKVConfig {
    pub embed_dim: usize,
    pub batch_size: usize,
}

impl Default for FusedRMSNormQKVConfig {
    fn default() -> Self {
        Self {
            embed_dim: 4096,
            batch_size: 1,
        }
    }
}

#[cfg(feature = "cuda")]
/// CUDA fused RMSNorm + QKV projection kernel.
#[derive(Clone)]
pub struct FusedRMSNormQKVKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl FusedRMSNormQKVKernel {
    /// Load and compile the fused RMSNorm+QKV kernel.
    pub fn load(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, String> {
        let ptx_source = include_str!("ptx/fused_rmsnorm_qkv.ptx");

        let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_source)
            .map_err(|e| format!("module load failed: {:?}", e))?;

        let function = module
            .load_function("fused_rmsnorm_qkv_kernel")
            .map_err(|e| format!("function lookup failed: {:?}", e))?;

        Ok(Self {
            context,
            stream,
            module,
            function,
        })
    }

    /// Launch fused RMSNorm + QKV kernel.
    /// 
    /// # Arguments
    /// * `x` - input tensor [batch, embed_dim] F32 on device
    /// * `weight` - RMSNorm scale [embed_dim] F32 on device
    /// * `wq` - query projection weights [embed_dim, embed_dim] F16 on device
    /// * `wk` - key projection weights [embed_dim, embed_dim] F16 on device
    /// * `wv` - value projection weights [embed_dim, embed_dim] F16 on device
    /// * `qkv_out` - output tensor [batch, 3*embed_dim] F16 on device
    pub fn launch(
        &self,
        x: u64,
        weight: u64,
        wq: u64,
        wk: u64,
        wv: u64,
        qkv_out: u64,
        embed_dim: usize,
        batch_size: usize,
    ) -> Result<(), String> {
        let mut x_v: u64 = x;
        let mut weight_v: u64 = weight;
        let mut wq_v: u64 = wq;
        let mut wk_v: u64 = wk;
        let mut wv_v: u64 = wv;
        let mut qkv_out_v: u64 = qkv_out;
        let mut embed_dim_v: i32 = embed_dim as i32;
        let mut batch_size_v: i32 = batch_size as i32;

        let mut params = [
            &mut x_v as *mut u64 as *mut std::ffi::c_void,
            &mut weight_v as *mut u64 as *mut std::ffi::c_void,
            &mut wq_v as *mut u64 as *mut std::ffi::c_void,
            &mut wk_v as *mut u64 as *mut std::ffi::c_void,
            &mut wv_v as *mut u64 as *mut std::ffi::c_void,
            &mut qkv_out_v as *mut u64 as *mut std::ffi::c_void,
            &mut embed_dim_v as *mut i32 as *mut std::ffi::c_void,
            &mut batch_size_v as *mut i32 as *mut std::ffi::c_void,
        ];

        let grid = (batch_size as u32, 1u32, 1u32);
        let block = ((embed_dim / 4).min(1024) as u32, 1u32, 1u32);

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