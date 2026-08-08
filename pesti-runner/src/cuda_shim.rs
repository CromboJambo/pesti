//! CUDA shim: provides types that mirror cuda-oxide's API but backed by cudarc.
//!
//! This enables a mechanical migration from cuda-oxide (nightly-only) to cudarc (stable-compatible).
//! The shim provides a PTX loading path using `cuModuleLoadData` and kernel launch via `cuLaunchKernel`.

use cudarc::driver::result::{self, DriverError};
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::driver::sys;
use std::ffi::CStr;
use std::sync::Arc;

/// Thin wrapper around a loaded CUDA module (PTX/cubin).
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { result::module::unload(self.cu_module) });
    }
}

impl CudaModule {
    /// Load a module from a PTX source string.
    pub fn load_from_ptx(ctx: &Arc<CudaContext>, ptx: &str) -> Result<Arc<Self>, DriverError> {
        ctx.bind_to_thread()?;
        let cstring = std::ffi::CString::new(ptx)
            .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
        let cu_module =
            unsafe { result::module::load_data(cstring.as_ptr() as *const std::ffi::c_void) }?;
        Ok(Arc::new(Self {
            cu_module,
            ctx: ctx.clone(),
        }))
    }

    /// Load a kernel function from this module.
    pub fn load_function(&self, name: &str) -> Result<CudaFunction, DriverError> {
        let cname = std::ffi::CString::new(name).unwrap();
        let cu_function = unsafe { result::module::get_function(self.cu_module, cname) }?;
        Ok(CudaFunction {
            cu_function,
            _module: (),
        })
    }
}

/// Thin wrapper around a CUDA function handle.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    #[allow(dead_code)]
    pub(crate) _module: (),
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {
    /// Get the raw CUfunction handle.
    pub fn cu_function(&self) -> sys::CUfunction {
        self.cu_function
    }
}

/// Launch a CUDA kernel (mirrors cuda-oxide's launch_kernel).
pub unsafe fn launch_kernel(
    function: sys::CUfunction,
    grid_dims: (u32, u32, u32),
    block_dims: (u32, u32, u32),
    shared_mem: u32,
    stream: sys::CUstream,
    params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    result::launch_kernel(function, grid_dims, block_dims, shared_mem, stream, params)
}

/// Get the raw CUstream from a CudaStream.
pub fn cu_stream(stream: &Arc<CudaStream>) -> sys::CUstream {
    stream.cu_stream()
}

/// Trait extension for CUresult to provide `.result()` method.
pub trait IntoResult {
    fn result(self) -> Result<(), DriverError>;
}

impl IntoResult for sys::CUresult {
    fn result(self) -> Result<(), DriverError> {
        match self {
            sys::CUresult::CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}
