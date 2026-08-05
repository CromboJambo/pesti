//! PTX builder for GEMM kernels.
//!
//! Bridges cuda-oxide kernel definitions with the inference engine.
//! Manages PTX compilation, kernel loading, and launch configuration.
//!
//! Build pipeline:
//! 1. Kernel source (cuda-oxide #[kernel] attribute)
//! 2. `cargo oxide build` → PTX binary
//! 3. GemmBuilder loads PTX and configures launches
//!
//! The builder supports both WGMMA (sm_120) and tcgen05 (sm_100) architectures.

use super::device_buf::DeviceBuffer;
use super::gemm::{GemmArch, GemmConfig, GemmError, GemmKernel};
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
    ///
    /// tcgen05 uses 128x128x16 tiles with tensor memory (TMEM).
    /// Recommended for sm_100+ (Blackwell B200 / RTX 5090).
    pub fn with_tcgen05(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Register a WGMMA GEMM kernel PTX.
    ///
    /// WGMMA uses 64x64x64 tiles with shared memory.
    /// Recommended for sm_120 (consumer Blackwell RTX 5060 Ti / 5090).
    pub fn with_wgmma(self, ptx: PtxSource) -> Self {
        self.add_ptx(ptx)
    }

    /// Build a GEMM kernel from the registered PTX sources.
    ///
    /// Returns a CpuGemmKernel if no PTX sources are registered (CPU fallback).
    /// Returns the best-matching GPU kernel if PTX sources are available.
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
/// Wraps a cuda-oxide LoadedModule and CudaFunction for actual kernel execution.
pub struct KernelFromPtx {
    source: PtxSource,
    config: GemmConfig,
    /// Loaded PTX module (None = GPU unavailable, fall back to CPU).
    module: Option<Arc<cuda_core::CudaModule>>,
    /// Kernel function handle (None = GPU unavailable).
    function: Option<cuda_core::CudaFunction>,
    /// CUDA context/stream for this kernel.
    ctx: Option<Arc<cuda_core::CudaContext>>,
    stream: Option<Arc<cuda_core::CudaStream>>,
    /// Whether GPU path is available.
    gpu_available: bool,
}

impl KernelFromPtx {
    /// Create a KernelFromPtx from a PTX source.
    ///
    /// Attempts to load the PTX into a CUDA context. If CUDA is unavailable,
    /// falls back gracefully by setting gpu_available = false.
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
            Arc<cuda_core::CudaContext>,
            Arc<cuda_core::CudaModule>,
            cuda_core::CudaFunction,
            Arc<cuda_core::CudaStream>,
        ),
        GemmError,
    > {
        // Initialize CUDA
        unsafe {
            cuda_core::init(0).map_err(|_| GemmError::LaunchFailed("CUDA init failed".into()))?;
        };

        // Create context for device 0
        let ctx = cuda_core::CudaContext::new(0)
            .map_err(|e| GemmError::LaunchFailed(format!("CUDA context creation failed: {e}")))?;

        // Load PTX from source string
        let module = ctx
            .load_module_from_ptx_src(&source.ptx)
            .map_err(|e| GemmError::LaunchFailed(format!("PTX load failed: {e}")))?;

        // Get kernel function
        let function = module
            .load_function(&source.kernel_name)
            .map_err(|e| GemmError::LaunchFailed(format!("Function lookup failed: {e}")))?;

        // Create stream
        let stream = ctx
            .new_stream()
            .map_err(|e| GemmError::LaunchFailed(format!("Stream creation failed: {e}")))?;

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

                // Convert device buffers to cuda-core format
                let dev_a = if a.is_backed() {
                    let buf = unsafe {
                        cuda_core::DeviceBuffer::from_raw_parts(
                            a.device_ptr() as cuda_core::sys::CUdeviceptr,
                            a.len(),
                            (*stream).clone(),
                        )
                    };
                    Arc::new(buf)
                } else {
                    let buf = cuda_core::DeviceBuffer::from_host(
                        stream,
                        a.as_slice().ok_or_else(|| GemmError::BufferSizeMismatch {
                            expected: a_expected,
                            got: 0,
                        })?,
                    )
                    .map_err(|e| GemmError::LaunchFailed(format!("H2D alloc failed: {e}")))?;
                    Arc::new(buf)
                };

                let dev_b = if b.is_backed() {
                    let buf = unsafe {
                        cuda_core::DeviceBuffer::from_raw_parts(
                            b.device_ptr() as cuda_core::sys::CUdeviceptr,
                            b.len(),
                            (*stream).clone(),
                        )
                    };
                    Arc::new(buf)
                } else {
                    let buf = cuda_core::DeviceBuffer::from_host(
                        stream,
                        b.as_slice().ok_or_else(|| GemmError::BufferSizeMismatch {
                            expected: b_expected,
                            got: 0,
                        })?,
                    )
                    .map_err(|e| GemmError::LaunchFailed(format!("H2D alloc failed: {e}")))?;
                    Arc::new(buf)
                };

                let mut dev_c = if c.is_backed() {
                    let buf = unsafe {
                        cuda_core::DeviceBuffer::from_raw_parts(
                            c.device_ptr() as cuda_core::sys::CUdeviceptr,
                            c.len(),
                            (*stream).clone(),
                        )
                    };
                    Arc::new(buf)
                } else {
                    let buf = cuda_core::DeviceBuffer::from_host(
                        stream,
                        c.as_mut_slice()
                            .ok_or_else(|| GemmError::BufferSizeMismatch {
                                expected: c_expected,
                                got: 0,
                            })?,
                    )
                    .map_err(|e| GemmError::LaunchFailed(format!("H2D alloc failed: {e}")))?;
                    Arc::new(buf)
                };

                // Calculate grid/block dimensions
                let (grid_x, grid_y, block_size) = self.get_launch_config(m, n);

                // Build kernel parameters (pointers to buffers)
                // Note: alpha/beta are passed as scalar values, not pointers
                let m_val = m as u32;
                let n_val = n as u32;
                let k_val = k as u32;

                let mut kernel_params: Vec<*mut std::ffi::c_void> = vec![
                    &alpha as *const f32 as *mut std::ffi::c_void,
                    &dev_a as *const _ as *mut std::ffi::c_void,
                    &dev_b as *const _ as *mut std::ffi::c_void,
                    &beta as *const f32 as *mut std::ffi::c_void,
                    &mut dev_c as *mut _ as *mut std::ffi::c_void,
                    &m_val as *const u32 as *mut std::ffi::c_void,
                    &n_val as *const u32 as *mut std::ffi::c_void,
                    &k_val as *const u32 as *mut std::ffi::c_void,
                ];

                // Launch kernel
                unsafe {
                    cuda_core::launch_kernel(
                        func.cu_function(),
                        (grid_x, grid_y, 1),
                        (block_size, 1, 1),
                        0,
                        stream.cu_stream(),
                        &mut kernel_params,
                    )
                    .map_err(|e| GemmError::LaunchFailed(format!("Kernel launch failed: {e}")))?;
                }

                // Synchronize
                stream
                    .synchronize()
                    .map_err(|e| GemmError::LaunchFailed(format!("Synchronize failed: {e}")))?;

                // Copy result back if c was a host buffer
                if !c.is_backed() {
                    if let Ok(host_c) = dev_c.to_host_vec(stream) {
                        if let Some(slice) = c.as_mut_slice() {
                            for (i, val) in host_c.iter().enumerate().take(slice.len()) {
                                slice[i] = *val;
                            }
                        }
                    }
                }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults() {
        let builder = GemmBuilder::new();
        let kernel = builder.build();
        assert!(kernel.is_available());
    }

    #[test]
    fn builder_with_arch() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Wgmma);
        let kernel = builder.build();
        assert_eq!(kernel.arch(), GemmArch::Wgmma);
    }

    #[test]
    fn builder_with_config() {
        let config = GemmConfig::default().with_block_size(256);
        let builder = GemmBuilder::new().with_config(config);
        let (gx, gy, bx, by) = builder.launch_config(512, 512);
        assert_eq!(gx, 2);
        assert_eq!(gy, 2);
        assert_eq!(bx, 256);
        assert_eq!(by, 256);
    }

    #[test]
    fn builder_launch_config_tcgen05() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        let (gx, gy, bx, by) = builder.launch_config(256, 256);
        assert_eq!(gx, 2);
        assert_eq!(gy, 2);
        assert_eq!(bx, 128);
        assert_eq!(by, 128);
    }

    #[test]
    fn builder_launch_config_non_aligned() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        let (gx, gy, _, _) = builder.launch_config(200, 300);
        assert_eq!(gx, 3); // (300 + 128 - 1) / 128 = 3
        assert_eq!(gy, 2); // (200 + 128 - 1) / 128 = 2
    }

    #[test]
    fn builder_validate_dims_tcgen05_k_not_divisible() {
        let builder = GemmBuilder::new().with_arch(GemmArch::Tcgen05);
        assert!(builder.validate_dims(128, 128, 63).is_err());
        assert!(builder.validate_dims(128, 128, 64).is_ok());
        assert!(builder.validate_dims(128, 128, 128).is_ok());
    }

    #[test]
    fn builder_validate_dims_wgmma_any_k() {
        // Wgmma has no K divisibility constraint — only block_size matters
        let config = GemmConfig::default()
            .with_arch(GemmArch::Wgmma)
            .with_block_size(64);
        let builder = GemmBuilder::new().with_config(config);
        assert!(builder.validate_dims(64, 64, 1).is_ok());
        assert!(builder.validate_dims(64, 64, 17).is_ok());
        assert!(builder.validate_dims(64, 64, 64).is_ok());
    }

    #[test]
    fn builder_validate_dims_invalid_tile() {
        let config = GemmConfig::default().with_block_size(0);
        let builder = GemmBuilder::new().with_config(config);
        // block_size=0 falls back to arch default (128 for Tcgen05) which is valid
        assert!(builder.validate_dims(64, 64, 64).is_ok());

        let config = GemmConfig::default().with_block_size(512);
        let builder = GemmBuilder::new().with_config(config);
        assert!(builder.validate_dims(64, 64, 64).is_err());
    }

    #[test]
    fn ptx_source_new() {
        let ptx = PtxSource::new("version 8.0".to_string(), GemmArch::Tcgen05, "tcgen05_gemm");
        assert_eq!(ptx.arch, GemmArch::Tcgen05);
        assert_eq!(ptx.kernel_name, "tcgen05_gemm");
        assert_eq!(ptx.ptx, "version 8.0");
    }

    #[test]
    fn kernel_from_ptx_arch() {
        let ptx = PtxSource::new("ptx".into(), GemmArch::Wgmma, String::from("wgmma"));
        let config = GemmConfig::default();
        let kernel = KernelFromPtx {
            source: ptx,
            config,
            module: None,
            function: None,
            ctx: None,
            stream: None,
            gpu_available: false,
        };
        assert_eq!(kernel.arch(), GemmArch::Wgmma);
    }
}
