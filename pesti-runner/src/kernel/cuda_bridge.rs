//! cuBLAS-based F16 GEMM bridge for PESTI.
//! Thin wrapper over cudarc's cuBLAS bindings, using hgemm (F16 matmul).
//! Uses cudarc's safe API: CudaBlas + Gemm trait with CudaSlice memory management.

use half::f16;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

/// CUDA bridge that manages cuBLAS handle and device context.
pub struct CudaBridge {
    #[cfg(feature = "cuda")]
    blas: Arc<CudaBlas>,
    #[cfg(feature = "cuda")]
    stream: Arc<CudaStream>,
}

impl CudaBridge {
    /// Create a new CUDA bridge with cuBLAS handle.
    pub fn new() -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {:?}", e))?;
            let stream = ctx.default_stream();
            let blas = Arc::new(
                CudaBlas::new(stream.clone())
                    .map_err(|e| format!("cuBLAS create failed: {}", e))?,
            );
            Ok(Self { blas, stream })
        }

        #[cfg(not(feature = "cuda"))]
        Err("CUDA feature not enabled".to_string())
    }

    /// Execute F16 GEMM: y = x @ W^T where x is [m,k], W is [n,k] -> y is [m,n]
    pub fn gemm_f16(
        &self,
        x: &[f16],
        weights: &[f16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::sys;

            let stream = &self.stream;

            // Allocate device memory using cudarc's safe slice API
            let mut x_dev = unsafe { stream.alloc(x.len()) }
                .map_err(|e| format!("cudaMalloc X failed: {:?}", e))?;
            let mut w_dev = unsafe { stream.alloc(weights.len()) }
                .map_err(|e| format!("cudaMalloc W failed: {:?}", e))?;
            let mut y_dev = unsafe { stream.alloc(m * n) }
                .map_err(|e| format!("cudaMalloc Y failed: {:?}", e))?;

            // Copy input to device using cudarc's safe copy API
            stream
                .memcpy_htod(x, &mut x_dev)
                .map_err(|e| format!("cudaMemcpy H2D X failed: {:?}", e))?;
            stream
                .memcpy_htod(weights, &mut w_dev)
                .map_err(|e| format!("cudaMemcpy H2D W failed: {:?}", e))?;

            // Compute C = X @ W^T where X is [m,k], W is [n,k] -> C is [m,n]
            let alpha = f16::from_f32(1.0);
            let beta = f16::from_f32(0.0);

            unsafe {
                self.blas.gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: n as i32,
                        ldb: m as i32,
                        beta,
                        ldc: n as i32,
                    },
                    &w_dev,
                    &x_dev,
                    &mut y_dev,
                );
            }

            // Synchronize
            stream
                .synchronize()
                .map_err(|e| format!("streamSync failed: {:?}", e))?;

            // Copy result back to host using cudarc's safe copy API
            let mut y_host = vec![f16::from_f32(0.0); m * n];
            stream
                .memcpy_dtoh(&y_dev, &mut y_host)
                .map_err(|e| format!("cudaMemcpy D2H Y failed: {:?}", e))?;

            // Free device memory (drop releases CudaSlice)
            drop(x_dev);
            drop(w_dev);
            drop(y_dev);

            // Convert F16 results to F32
            Ok(y_host.iter().map(|v| v.to_f32()).collect())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA feature not enabled".to_string())
        }
    }

    /// Execute GEMM with pre-uploaded device weights (persistent buffer).
    pub fn gemm_f16_with_device_weights(
        &self,
        x: &[f16],
        w_dev: &cudarc::driver::CudaSlice<f16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> crate::error::Result<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::sys;

            let stream = &self.stream;

            // Allocate device memory for input and output only (weights already on GPU)
            let mut x_dev = unsafe { stream.alloc(x.len()) }
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc X failed: {:?}", e)))?;
            let mut y_dev = unsafe { stream.alloc(m * n) }
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc Y failed: {:?}", e)))?;

            // Copy input to device
            stream
                .memcpy_htod(x, &mut x_dev)
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMemcpy H2D X failed: {:?}", e)))?;

            // Compute C = X @ W^T where X is [m,k], W is [n,k] -> C is [m,n]
            let alpha = f16::from_f32(1.0);
            let beta = f16::from_f32(0.0);

            unsafe {
                self.blas.gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_N,
                        transb: sys::cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: n as i32,
                        ldb: m as i32,
                        beta,
                        ldc: n as i32,
                    },
                    w_dev,
                    &x_dev,
                    &mut y_dev,
                );
            }

            // Synchronize
            stream
                .synchronize()
                .map_err(|e| crate::error::RunnerError::Kernel(format!("streamSync failed: {:?}", e)))?;

            // Copy result back to host
            let mut y_host = vec![f16::from_f32(0.0); m * n];
            stream
                .memcpy_dtoh(&y_dev, &mut y_host)
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMemcpy D2H Y failed: {:?}", e)))?;

            // Free device memory
            drop(x_dev);
            drop(y_dev);

            // Convert F16 results to F32
            Ok(y_host.iter().map(|v| v.to_f32()).collect())
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(crate::error::RunnerError::Kernel("CUDA feature not enabled".to_string()))
        }
    }
}

/// Global weight cache: maps host pointer address -> device buffer.
/// Weights are uploaded on first GEMM call and cached for reuse across forward passes.
static WEIGHT_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, cudarc::driver::CudaSlice<f16>>>> = std::sync::OnceLock::new();

fn get_weight_cache() -> &'static std::sync::Mutex<std::collections::HashMap<usize, cudarc::driver::CudaSlice<f16>>> {
    WEIGHT_CACHE.get_or_init(|| {
        std::sync::Mutex::new(std::collections::HashMap::new())
    })
}

/// Free function wrapper for GEMM (f16 input/output, returns f32).
/// Uses persistent device weight buffers cached by host pointer address.
pub fn gemm_f16(
    x: &[half::f16],
    weights: &[half::f16],
    m: usize,
    n: usize,
    k: usize,
) -> crate::error::Result<Vec<f32>> {
    static BRIDGE: std::sync::OnceLock<CudaBridge> = std::sync::OnceLock::new();

    let bridge = BRIDGE.get_or_init(|| match CudaBridge::new() {
        Ok(b) => b,
        Err(e) => panic!("CUDA bridge init failed: {}", e),
    });

    // Lazy weight caching: upload on first use, reuse across forward passes.
    // Key by pointer address (weights are static after model load).
    let weight_key = weights.as_ptr() as usize;
    
    // Check if already uploaded
    let dev_weights = {
        let cache = get_weight_cache();
        let mut guard = cache.lock().unwrap();
        if let Some(buf) = guard.get(&weight_key) {
            buf.clone()
        } else {
            drop(guard);
            // Upload to GPU
            let stream = &bridge.stream;
            let mut buf = unsafe { stream.alloc(weights.len()) }
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc weight failed: {:?}", e)))?;
            stream.memcpy_htod(weights, &mut buf)
                .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMemcpy H2D weight failed: {:?}", e)))?;
            
            // Insert into cache
            let mut guard = get_weight_cache().lock().unwrap();
            guard.insert(weight_key, buf.clone());
            buf
        }
    };

    bridge.gemm_f16_with_device_weights(x, &dev_weights, m, n, k)
}

/// Get a reference to the shared CUDA stream for weight uploads.
#[cfg(feature = "cuda")]
pub fn get_stream() -> std::sync::Arc<CudaStream> {
    static BRIDGE: std::sync::OnceLock<CudaBridge> = std::sync::OnceLock::new();
    let bridge = BRIDGE.get_or_init(|| match CudaBridge::new() {
        Ok(b) => b,
        Err(e) => panic!("CUDA bridge init failed for stream: {}", e),
    });
    bridge.stream.clone()
}

/// Check if CUDA bridge is available.
pub fn cuda_bridge_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        CudaContext::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}