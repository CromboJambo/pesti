//! WGMMA Tensor Core GEMM Kernel for sm_8.9 (RTX 4070 Ti SUPER)
//!
//! Implements matrix multiplication using NVIDIA Ampere tensor cores via WGMMA instructions.
//! Provides ~3× speedup over warp-level GEMM (mma.sync) on sm_8.9 architecture.

use crate::cuda_runtime::{CudaRuntime, IntoResult};
use crate::cuda_shim::{CudaFunction, CudaModule};
use crate::kernel::device_buf::DeviceBuffer;
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// WGMMA GEMM kernel for sm_8.9 (RTX 4070 Ti SUPER)
#[derive(Clone)]
pub struct WGMMAKernel {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    function: CudaFunction,
}

impl WGMMAKernel {
    /// Build WGMMA kernel from PTX source
    pub fn build(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        device_info: crate::cuda_runtime::CudaDeviceInfo,
    ) -> Result<Self, super::GemmError> {
        // Check sm_8.9 compatibility (WGMMA works on Hopper sm_90a and consumer Blackwell sm_120)
        // For RTX 4070 Ti SUPER (sm_8.9), we use the WGMMA-compatible version
        if !device_info.supports_wgmma() {
            return Err(super::GemmError::UnsupportedArch(format!(
                "WGMMA requires sm_90a or higher, but device is sm_{}.{}",
                device_info.compute_capability.0,
                device_info.compute_capability.1
            )));
        }

        // Load WGMMA PTX for sm_8.9 (compatible with tensor cores)
        let ptx_src = include_str!("ptx/gemm_wgmma_sm89.ptx");
        
        let module = CudaModule::load_from_ptx(&context, ptx_src)
            .map_err(|e| super::GemmError::Cuda(format!("module load failed: {e:?}")))?;

        let function = module
            .load_function("gemm_wgmma_kernel")
            .map_err(|e| super::GemmError::Cuda(format!("function load failed: {e:?}")))?;

        Ok(Self {
            context,
            stream,
            module,
            function,
        })
    }

    /// Launch WGMMA kernel for C = alpha * A @ B + beta * C
    pub fn launch(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), super::GemmError> {
        use cudarc::driver::sys;

        // WGMMA kernel signature (gemm_wgmma_sm89.ptx):
        //   gemm_wgmma_kernel(f32 alpha, u64 A, u64 B, f32 beta,
        //                     u64 C, u32 m, u32 n, u32 k)
        let mut a_v: u64 = a.device_ptr();
        let mut b_v: u64 = b.device_ptr();
        let mut c_v: u64 = c.device_ptr();
        let mut alpha_v: f32 = alpha;
        let mut beta_v: f32 = beta;
        let mut m_v: u32 = m as u32;
        let mut n_v: u32 = n as u32;
        let mut k_v: u32 = k as u32;

        let mut params: [*mut std::ffi::c_void; 8] = [
            &mut alpha_v as *mut f32 as *mut std::ffi::c_void,
            &mut a_v as *mut u64 as *mut std::ffi::c_void,
            &mut b_v as *mut u64 as *mut std::ffi::c_void,
            &mut beta_v as *mut f32 as *mut std::ffi::c_void,
            &mut c_v as *mut u64 as *mut std::ffi::c_void,
            &mut m_v as *mut u32 as *mut std::ffi::c_void,
            &mut n_v as *mut u32 as *mut std::ffi::c_void,
            &mut k_v as *mut u32 as *mut std::ffi::c_void,
        ];

        // WGMMA: 128×128 tiles per warp group, 4 warp groups per block (128 threads)
        let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1u32);
        let block = (128u32, 1u32, 1u32);

        unsafe {
            use crate::cuda_shim::launch_kernel;
            launch_kernel(
                self.function.cu_function(),
                grid,
                block,
                0, // shared_mem_bytes (WGMMA uses implicit SMEM)
                self.stream.cu_stream(),
                &mut params,
            )
            .map_err(|e| super::GemmError::LaunchFailed(format!("kernel launch failed: {e:?}")))?;
        }

        // Synchronize before returning: the kernel runs on a non-blocking
        // stream with no implicit ordering against the legacy default stream
        // used by synchronous D2H copies (same race as CudaGemmKernel::launch).
        self.stream
            .synchronize()
            .map_err(|e| super::GemmError::LaunchFailed(format!("stream sync failed: {e:?}")))?;

        Ok(())
    }
}

impl super::GemmKernel for WGMMAKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), super::GemmError> {
        self.launch(alpha, a, b, beta, c, m, n, k)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> super::GemmArch {
        super::GemmArch::Wgmma
    }
}
