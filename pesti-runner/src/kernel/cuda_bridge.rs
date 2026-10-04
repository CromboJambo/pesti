//! cuBLAS-based F16 GEMM bridge for PESTI.
//! Thin wrapper over cudarc's cuBLAS bindings, using hgemm (F16 matmul).
//! Uses cudarc's safe API: CudaBlas + Gemm trait with CudaSlice memory management.

use half::f16;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// GPU weight cache: maps (ptr, len) → cached device buffer.
/// Avoids redundant H2D transfers of the same weight buffer across decode steps.
struct WeightCache {
    buffers: HashMap<CacheKey, cudarc::driver::CudaSlice<half::f16>>,
}

#[derive(Debug)]
struct CacheKey {
    ptr: usize,
    len: usize,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
        self.len.hash(state);
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.len == other.len
    }
}

static WEIGHT_CACHE: OnceLock<Mutex<WeightCache>> = OnceLock::new();

fn weight_cache() -> &'static Mutex<WeightCache> {
    WEIGHT_CACHE.get_or_init(|| Mutex::new(WeightCache { buffers: HashMap::new() }))
}

/// Get or upload cached GPU buffer for a weight slice.
fn get_or_upload_weights(weights: &[half::f16]) -> Result<cudarc::driver::CudaSlice<half::f16>, String> {
    let key = CacheKey { ptr: weights.as_ptr() as usize, len: weights.len() };
    let cache = weight_cache();
    let mut guard = cache.lock().unwrap();

    if let Some(buf) = guard.buffers.get(&key).cloned() {
        return Ok(buf);
    }

    // Not cached — upload to GPU
    drop(guard);
    let stream = get_stream();
    let mut w_dev = unsafe { stream.alloc(weights.len()) }.map_err(|e| format!("cudaMalloc W failed: {:?}", e))?;
    stream.memcpy_htod(weights, &mut w_dev).map_err(|e| format!("cudaMemcpy H2D W failed: {:?}", e))?;

    // Insert into cache
    let mut guard = cache.lock().unwrap();
    guard.buffers.insert(key, w_dev.clone());
    Ok(w_dev)
}

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

            // Don't sync here — caller will sync once at end of forward pass
            // stream.synchronize() removed to eliminate per-GEMM serialization

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

            // Copy result back to host (no sync — caller will sync once at end of forward pass)
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

/// Free function wrapper for GEMM (f16 input/output, returns f32).
/// Uses cuBLAS algorithm selection with shape-based caching for tall-skinny LLM shapes.
/// GPU weight tensors are cached across calls to avoid redundant H2D transfers.
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

    // Use persistent GPU weight buffer — upload once, reuse across all decode steps
    let w_dev = get_or_upload_weights(weights)?;
    bridge.gemm_f16_with_device_weights(x, &w_dev, m, n, k)
}

/// Free function wrapper for GEMM with persistent GPU weight buffer.
/// Weights are uploaded once and reused across forward passes (eliminates per-call H2D transfer).
pub fn gemm_f16_with_persistent_weights(
    x: &[half::f16],
    w_dev: &cudarc::driver::CudaSlice<half::f16>,
    m: usize,
    n: usize,
    k: usize,
) -> crate::error::Result<Vec<f32>> {
    static BRIDGE: std::sync::OnceLock<CudaBridge> = std::sync::OnceLock::new();

    let bridge = BRIDGE.get_or_init(|| match CudaBridge::new() {
        Ok(b) => b,
        Err(e) => panic!("CUDA bridge init failed for persistent weights: {}", e),
    });

    bridge.gemm_f16_with_device_weights(x, w_dev, m, n, k)
}

/// cuBLASLt-based GEMM with shape-optimized algorithm selection.
/// Uses cublasLtMatmul via cudarc's safe API for optimal kernel dispatch.
/// This is the Week 28 CUTLASS integration path — cuBLASLt uses CUTLASS kernels internally.
pub fn gemm_f16_cublaslt(
    x: &[half::f16],
    w_dev: &cudarc::driver::CudaSlice<half::f16>,
    m: usize,
    n: usize,
    k: usize,
) -> crate::error::Result<Vec<f32>> {
    use cudarc::cublaslt::{result, sys};

    static BRIDGE: std::sync::OnceLock<CudaBridge> = std::sync::OnceLock::new();
    let bridge = BRIDGE.get_or_init(|| match CudaBridge::new() {
        Ok(b) => b,
        Err(e) => panic!("CUDA bridge init failed for cublasLt: {}", e),
    });

    // Create cublasLt handle per-call (not thread-safe across calls in cudarc's safe API)
    let handle = result::create_handle()
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLt handle creation failed: {}", e)))?;

    // Allocate device memory for input and output
    let stream = &bridge.stream;
    let mut x_dev = unsafe { stream.alloc(x.len()) }
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc X failed: {:?}", e)))?;
    let mut y_dev: cudarc::driver::CudaSlice<half::f16> = unsafe { stream.alloc(m * n) }
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc Y failed: {:?}", e)))?;

    // Copy input to device
    stream.memcpy_htod(x, &mut x_dev).map_err(|e| {
        crate::error::RunnerError::Kernel(format!("cudaMemcpy H2D X failed: {:?}", e))
    })?;

    // Create matrix layouts via cudarc's safe API (handles all attributes internally)
    let a_layout = result::create_matrix_layout(
        sys::cudaDataType_t::CUDA_R_16F, n as u64, k as u64, n as i64,
    ).map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatrixLayoutCreate A failed: {}", e)))?;
    let b_layout = result::create_matrix_layout(
        sys::cudaDataType_t::CUDA_R_16F, k as u64, m as u64, k as i64,
    ).map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatrixLayoutCreate B failed: {}", e)))?;
    let c_layout = result::create_matrix_layout(
        sys::cudaDataType_t::CUDA_R_16F, n as u64, m as u64, n as i64,
    ).map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatrixLayoutCreate C failed: {}", e)))?;

    // Create matmul descriptor with F16 compute type
    let matmul_desc = result::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType_t::CUDA_R_32F,
    ).map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatmulDescCreate failed: {}", e)))?;

    // Set up preference with 4MB workspace limit
    let pref = result::create_matmul_pref()
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatmulPreferenceCreate failed: {}", e)))?;
    let workspace_size: usize = 4 * 1024 * 1024;

    // Get optimal algorithm for this shape via heuristic search (requires unsafe block)
    let algo = unsafe { result::get_matmul_algo_heuristic(
        handle, matmul_desc, a_layout, b_layout, c_layout, c_layout, pref,
    ) }.map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatmulAlgoGetHeuristic failed: {}", e)))?;

    // Allocate workspace for the chosen algorithm (as u8 slice)
    let mut ws_dev: cudarc::driver::CudaSlice<u8> = unsafe { stream.alloc(workspace_size) }
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc workspace failed: {:?}", e)))?;

    // Run cublasLtMatmul with the selected algorithm
    let alpha_f32: f32 = 1.0;
    let beta_f32: f32 = 0.0;
    unsafe {
        let (x_ptr_u64, _sync_x) = x_dev.device_ptr(stream);
        let (w_ptr_u64, _sync_w) = w_dev.device_ptr(stream);
        let (y_ptr_u64, _sync_y) = y_dev.device_ptr_mut(stream);
        let (ws_ptr_u64, _sync_ws) = ws_dev.device_ptr_mut(stream);

        // Convert u64 device pointers to raw pointers for cublasLt API
        let x_ptr: *const half::f16 = (x_ptr_u64 as usize) as *const half::f16;
        let w_ptr: *const half::f16 = (w_ptr_u64 as usize) as *const half::f16;
        let y_ptr: *mut half::f16 = (y_ptr_u64 as usize) as *mut half::f16;
        let ws_ptr: *mut u8 = (ws_ptr_u64 as usize) as *mut u8;

        result::matmul(
            handle,
            matmul_desc,
            (&alpha_f32) as *const _ as *const _,
            x_ptr as *const _,
            w_ptr as *const _,
            b_layout,
            (&beta_f32) as *const _ as *const _,
            a_layout,
            y_ptr as *mut _,
            c_layout,
            y_ptr as *mut _,
            c_layout,
            (&algo.algo) as *const _,
            ws_ptr as *mut _,
            workspace_size,
            stream.cu_stream() as *mut _,
        ).map_err(|e| crate::error::RunnerError::Kernel(format!("cublasLtMatmul failed: {}", e)))?;
    }

    // Synchronize
    stream.synchronize()
        .map_err(|e| crate::error::RunnerError::Kernel(format!("streamSync failed: {:?}", e)))?;

    // Copy result back to host
    let mut y_host = vec![half::f16::from_f32(0.0); m * n];
    stream.memcpy_dtoh(&y_dev, &mut y_host).map_err(|e| {
        crate::error::RunnerError::Kernel(format!("cudaMemcpy D2H Y failed: {:?}", e))
    })?;

    // Cleanup layouts and descriptors
    unsafe {
        result::destroy_matrix_layout(a_layout).ok();
        result::destroy_matrix_layout(b_layout).ok();
        result::destroy_matrix_layout(c_layout).ok();
        result::destroy_matmul_desc(matmul_desc).ok();
        result::destroy_matmul_pref(pref).ok();
        result::destroy_handle(handle).ok();
    }

    // Free device memory
    drop(x_dev);
    drop(y_dev);
    drop(ws_dev);

    // Convert F16 results to F32
    Ok(y_host.iter().map(|v| v.to_f32()).collect())
}

/// Upload weights to GPU once for persistent caching. Returns the device buffer handle.
pub fn upload_weights_to_gpu(
    weights: &[half::f16],
) -> crate::error::Result<cudarc::driver::CudaSlice<half::f16>> {
    static BRIDGE: std::sync::OnceLock<CudaBridge> = std::sync::OnceLock::new();

    let bridge = BRIDGE.get_or_init(|| match CudaBridge::new() {
        Ok(b) => b,
        Err(e) => panic!("CUDA bridge init failed for upload: {}", e),
    });

    let stream = &bridge.stream;
    let mut w_dev = unsafe { stream.alloc(weights.len()) }
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMalloc W failed: {:?}", e)))?;
    stream
        .memcpy_htod(weights, &mut w_dev)
        .map_err(|e| crate::error::RunnerError::Kernel(format!("cudaMemcpy H2D W failed: {:?}", e)))?;

    Ok(w_dev)
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
