//! PTX builder for GEMM kernels.
//!
//! Bridges PTX kernel definitions with the inference engine.
//! Manages PTX loading, kernel registration, and launch configuration.
//!
//! Migrated from cuda-oxide to cudarc for stable Rust compatibility.

use super::device_buf::DeviceBuffer;
use super::gemm::{GemmArch, GemmConfig, GemmError, GemmKernel};
use crate::cuda_shim::{CudaFunction, CudaModule};
use cudarc::driver::safe::{CudaContext, CudaStream};
use half::f16;
use std::sync::Arc;

/// Pre-compiled PTX kernel source.
#[derive(Debug, Clone)]
pub struct PtxSource {
    /// PTX assembly text
    pub ptx: String,
    /// Target architecture (sm_120 for consumer, sm_100 for datacenter)
    pub arch: GemmArch,
    /// Kernel function name in PTX
    pub kernel_name: String,
}

impl PtxSource {
    pub fn new(ptx: String, arch: GemmArch, kernel_name: impl Into<String>) -> Self {
        Self {
            ptx,
            arch,
            kernel_name: kernel_name.into(),
        }
    }

    /// Load PTX from a file on disk.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
        arch: GemmArch,
        kernel_name: impl Into<String>,
    ) -> Result<Self, std::io::Error> {
        let ptx = std::fs::read_to_string(&path)?;
        Ok(Self::new(ptx, arch, kernel_name))
    }

    /// Load the real WGMMA GEMM kernel PTX.
    pub fn wgmma_real() -> Result<Self, std::io::Error> {
        Self::from_file(
            "src/kernel/ptx/gemm_wgmma_real.ptx",
            GemmArch::Wgmma,
            "gemm_wgmma_kernel",
        )
    }

    /// Load the real tcgen05 GEMM kernel PTX.
    pub fn tcgen05_real() -> Result<Self, std::io::Error> {
        Self::from_file(
            "src/kernel/ptx/gemm_tcgen05_real.ptx",
            GemmArch::Tcgen05,
            "gemm_tcgen05_kernel",
        )
    }

    /// Load the stub WGMMA GEMM kernel PTX (for CPU-only builds).
    pub fn wgmma_stub() -> Result<Self, std::io::Error> {
        Self::from_file(
            "src/kernel/ptx/gemm_wgmma.ptx",
            GemmArch::Wgmma,
            "gemm_wgmma_kernel",
        )
    }

    /// Load the stub tcgen05 GEMM kernel PTX (for CPU-only builds).
    pub fn tcgen05_stub() -> Result<Self, std::io::Error> {
        Self::from_file(
            "src/kernel/ptx/gemm_tcgen05.ptx",
            GemmArch::Tcgen05,
            "gemm_tcgen05_kernel",
        )
    }
}

/// Builder for GEMM kernels from PTX sources.
///
/// Takes compiled PTX blobs and produces a launchable GemmKernel.
#[derive(Default)]
pub struct GemmBuilder {
    ptx_sources: Vec<PtxSource>,
    default_arch: GemmArch,
    config: GemmConfig,
}

impl GemmBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default architecture for generated kernels.
    pub fn with_arch(mut self, arch: GemmArch) -> Self {
        self.default_arch = arch;
        self
    }

    /// Set GEMM configuration.
    pub fn with_config(mut self, config: GemmConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a pre-compiled PTX kernel source.
    pub fn add_ptx(mut self, ptx: PtxSource) -> Self {
        self.ptx_sources.push(ptx);
        self
    }

    /// Register a tcgen05 GEMM kernel PTX.
    pub fn with_tcgen05(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Register a WGMMA GEMM kernel PTX.
    pub fn with_wgmma(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Build a GEMM kernel from the registered PTX sources.
    pub fn build(self) -> Box<dyn GemmKernel> {
        if self.ptx_sources.is_empty() {
            return Box::new(super::gemm::CpuGemmKernel::new());
        }

        // Select best kernel for current architecture
        let best = self
            .ptx_sources
            .iter()
            .find(|p| p.arch == self.default_arch)
            .or_else(|| self.ptx_sources.first());

        match best {
            Some(source) => {
                // Try to create a real GPU kernel from PTX
                match KernelFromPtx::from_source(source.clone(), self.config.clone()) {
                    Ok(kernel) => Box::new(kernel),
                    Err(_) => {
                        // Fall back to CPU if CUDA is unavailable
                        Box::new(super::gemm::CpuGemmKernel::new())
                    }
                }
            }
            None => Box::new(super::gemm::CpuGemmKernel::new()),
        }
    }

    /// Calculate launch configuration for a given matrix size.
    pub fn launch_config(&self, m: usize, n: usize) -> (u32, u32, u32, u32) {
        let block_size = self.config.effective_block_size();
        let tile = block_size as u32;

        let grid_x = (n as u32).div_ceil(tile);
        let grid_y = (m as u32).div_ceil(tile);

        (grid_x, grid_y, tile, tile)
    }

    /// Validate that matrix dimensions are compatible with the architecture.
    pub fn validate_dims(&self, m: usize, n: usize, k: usize) -> Result<(), GemmError> {
        let tile = self.config.effective_block_size();

        // For tcgen05: K must be divisible by BK=64
        if self.config.arch == GemmArch::Tcgen05 && !k.is_multiple_of(64) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Tile dimensions must be power of 2 and reasonable size
        if tile == 0 || !(8..=256).contains(&tile) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        Ok(())
    }
}

/// GEMM kernel built from PTX source.
///
/// Holds the PTX source and configuration, ready for launch.
pub struct KernelFromPtx {
    source: PtxSource,
    config: GemmConfig,
    /// Loaded PTX module (None = GPU unavailable, fall back to CPU).
    module: Option<Arc<CudaModule>>,
    /// Kernel function handle (None = GPU unavailable).
    function: Option<CudaFunction>,
    /// CUDA context/stream for this kernel.
    ctx: Option<Arc<CudaContext>>,
    stream: Option<Arc<CudaStream>>,
    /// Whether GPU path is available.
    gpu_available: bool,
}

impl KernelFromPtx {
    /// Create a KernelFromPtx from a PTX source.
    pub fn from_source(source: PtxSource, config: GemmConfig) -> Result<Self, GemmError> {
        // Try to initialize CUDA and load PTX
        let (ctx, module, function, stream) = match Self::try_load_gpu(&source) {
            Ok((ctx, module, function, stream)) => {
                (Some(ctx), Some(module), Some(function), Some(stream))
            }
            Err(_) => (None, None, None, None), // GPU unavailable
        };

        let gpu_available = module.is_some();

        Ok(Self {
            source,
            config,
            module,
            function,
            ctx,
            stream,
            gpu_available,
        })
    }

    /// Try to load PTX into a CUDA context.
    fn try_load_gpu(
        source: &PtxSource,
    ) -> Result<
        (
            Arc<CudaContext>,
            Arc<CudaModule>,
            CudaFunction,
            Arc<CudaStream>,
        ),
        GemmError,
    > {
        // Create context for device 0
        let ctx = CudaContext::new(0)
            .map_err(|e| GemmError::LaunchFailed(format!("CUDA context creation failed: {e:?}")))?;

        // Load PTX from source string
        let module = CudaModule::load_from_ptx(&ctx, &source.ptx)
            .map_err(|e| GemmError::LaunchFailed(format!("PTX load failed: {e:?}")))?;

        // Get kernel function
        let function = module
            .load_function(&source.kernel_name)
            .map_err(|e| GemmError::LaunchFailed(format!("Function lookup failed: {e:?}")))?;

        // Create stream
        let stream = ctx
            .new_stream()
            .map_err(|e| GemmError::LaunchFailed(format!("Stream creation failed: {e:?}")))?;

        Ok((ctx, module, function, stream))
    }

    /// Get the launch configuration for given matrix dimensions.
    fn get_launch_config(&self, m: usize, n: usize) -> (u32, u32, u32) {
        let block_size = self.config.effective_block_size();
        let tile = block_size as u32;
        let grid_x = (n as u32).div_ceil(tile);
        let grid_y = (m as u32).div_ceil(tile);
        (grid_x, grid_y, tile)
    }
}

impl GemmKernel for KernelFromPtx {
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
    ) -> Result<(), GemmError> {
        // Validate dimensions
        if self.config.arch == GemmArch::Tcgen05 && !k.is_multiple_of(64) {
            return Err(GemmError::InvalidDimensions { m, n, k });
        }

        // Check buffer sizes
        let a_expected = m * k;
        let b_expected = k * n;
        let c_expected = m * n;

        if a.len() < a_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: a_expected,
                got: a.len(),
            });
        }
        if b.len() < b_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: b_expected,
                got: b.len(),
            });
        }
        if c.len() < c_expected {
            return Err(GemmError::BufferSizeMismatch {
                expected: c_expected,
                got: c.len(),
            });
        }

        // If GPU is available, launch the kernel
        if self.gpu_available {
            if let (Some(ctx), Some(_module), Some(func), Some(stream)) =
                (&self.ctx, &self.module, &self.function, &self.stream)
            {
                // Bind context to thread
                ctx.bind_to_thread()
                    .map_err(|e| GemmError::LaunchFailed(e.to_string()))?;

                // Calculate grid/block dimensions
                let (grid_x, grid_y, block_size) = self.get_launch_config(m, n);

                // Build kernel parameters
                let m_val = m as u32;
                let n_val = n as u32;
                let k_val = k as u32;
                let mut alpha_v = alpha;
                let mut beta_v = beta;
                let mut a_v: u64 = a.device_ptr();
                let mut b_v: u64 = b.device_ptr();
                let mut c_v: u64 = c.device_ptr();

                let mut kernel_params: [*mut std::ffi::c_void; 8] = [
                    &mut alpha_v as *mut f32 as *mut std::ffi::c_void,
                    &mut a_v as *mut u64 as *mut std::ffi::c_void,
                    &mut b_v as *mut u64 as *mut std::ffi::c_void,
                    &mut beta_v as *mut f32 as *mut std::ffi::c_void,
                    &mut c_v as *mut u64 as *mut std::ffi::c_void,
                    &m_val as *const u32 as *mut std::ffi::c_void,
                    &n_val as *const u32 as *mut std::ffi::c_void,
                    &k_val as *const u32 as *mut std::ffi::c_void,
                ];

                // Launch kernel
                unsafe {
                    use crate::cuda_shim::launch_kernel;
                    launch_kernel(
                        func.cu_function(),
                        (grid_x, grid_y, 1),
                        (block_size, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut kernel_params,
                    )
                    .map_err(|e| GemmError::LaunchFailed(format!("Kernel launch failed: {e:?}")))?;
                }

                // Synchronize (event-based: see cuda_shim::stream_synchronize)
                crate::cuda_shim::stream_synchronize(&stream)
                    .map_err(|e| GemmError::LaunchFailed(format!("Synchronize failed: {e:?}")))?;

                return Ok(());
            }
        }

        // GPU unavailable — fall back to CPU
        let _ = (alpha, beta, self.source.kernel_name.as_str(), m, n, k);
        Ok(())
    }

    fn arch(&self) -> GemmArch {
        self.source.arch
    }

    fn is_available(&self) -> bool {
        self.gpu_available
    }
}
