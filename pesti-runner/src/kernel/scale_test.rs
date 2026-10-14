//! Simple scale kernel for testing CUDA infrastructure.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    safe::{CudaContext, CudaStream},
};

/// Simple scale kernel: y[i] = x[i] * 2.0
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct ScaleKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: std::sync::Arc<crate::cuda_shim::CudaModule>,
    function: crate::cuda_shim::CudaFunction,
}

#[cfg(feature = "cuda")]
impl ScaleKernel {
    pub fn load(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, String> {
        let ptx_source = include_str!("ptx/scale.ptx");

        let module = crate::cuda_shim::CudaModule::load_from_ptx(&context, ptx_source)
            .map_err(|e| format!("module load failed: {:?}", e))?;

        let function = module
            .load_function("scale_kernel")
            .map_err(|e| format!("function lookup failed: {:?}", e))?;

        Ok(Self {
            context,
            stream,
            module,
            function,
        })
    }

    pub fn launch(&self, x: u64, y: u64, n: usize) -> Result<(), String> {
        let grid = (((n + 255) / 256) as u32, 1u32, 1u32);
        let block = (256u32, 1u32, 1u32);

        let mut x_v: u64 = x;
        let mut y_v: u64 = y;
        let mut n_v: i32 = n as i32;
        let mut params = [
            &mut x_v as *mut u64 as *mut std::ffi::c_void,
            &mut y_v as *mut u64 as *mut std::ffi::c_void,
            &mut n_v as *mut i32 as *mut std::ffi::c_void,
        ];

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