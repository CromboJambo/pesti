//! CUDA cuBLAS GEMM bridge using cudarc's result API.
//!
//! Uses cudarc's result::hgemm which calls cublasHgemm directly via libloading,
//! avoiding the gemm_ex parameter issues entirely.

use std::sync::OnceLock;

static BLAS: OnceLock<cudarc::cublas::CudaBlas> = OnceLock::new();

fn get_blas() -> &'static cudarc::cublas::CudaBlas {
    BLAS.get_or_init(|| {
        let stream = Default::default();
        cudarc::cublas::CudaBlas::new(stream.clone()).expect("cuda_bridge: CudaBlas::new failed")
    })
}

/// Perform F16 matrix multiply using cuBLAS via cudarc's result API.
///
/// Computes C = A @ B where:
///   - A is [m x k] row-major F16 (on device)
///   - B is [k x n] row-major F16 (on device, transposed to column-major for cuBLAS)
///   - C is [m x n] row-major F32 (on device)
pub fn gemm_f16f32(
    m: i32, k: i32, n: i32,
    a_dev: *const u8, b_t_dev: *const u8, c_dev: *mut f32,
) -> Result<(), String> {
    let blas = get_blas();

    // Cast to half::f16 slices for cudarc's safe API
    let a_slice = unsafe { std::slice::from_raw_parts(a_dev as *const half::f16, m * k) };
    let b_t_slice = unsafe { std::slice::from_raw_parts(b_t_dev as *const half::f16, n * k) };
    let c_slice = unsafe { std::slice::from_raw_parts_mut(c_dev as *mut f32, m * n) };

    // For row-major data with cuBLAS (column-major), swap operands:
    // C^T = B^T @ A^T => compute B^T(A^T) and transpose result layout
    // We pass B_t as "A" (already transposed), A as "B"
    let cfg = cudarc::cublas::GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,  // swapped for row-major output
        n: m,
        k: k,
        alpha: half::f16::from_f32(1.0),
        lda: n,
        ldb: k,
        beta: half::f16::from_f32(0.0),
        ldc: n,
    };

    unsafe {
        blas.gemm(cfg, b_t_slice, a_slice, c_slice)
            .map_err(|e| format!("cuda_bridge::gemm_f16f32: gemm failed: {:?}", e))
    }
}

/// Synchronize CUDA stream via event-based synchronization.
pub fn stream_synchronize() -> Result<(), String> {
    cudarc::driver::stream_synchronize(Default::default())
        .map_err(|e| format!("cuda_bridge::stream_sync failed: {:?}", e))
}

/// Free device memory.
pub fn free(ptr: *mut u8) -> Result<(), String> {
    if ptr.is_null() { return Ok(()); }
    cudarc::driver::free(ptr).map_err(|e| format!("cuda_bridge::free failed: {:?}", e))
}

/// Allocate device memory.
pub fn alloc(size: usize) -> Result<*mut u8, String> {
    let mut ptr = std::ptr::null_mut();
    cudarc::driver::alloc(&mut ptr, size).map_err(|e| format!("cuda_bridge::alloc failed: {:?}", e))?;
    Ok(ptr)
}

/// Copy data to device.
pub fn memcpy_h2d(dst: *mut u8, src: &[u8]) -> Result<(), String> {
    let size = src.len();
    cudarc::driver::memcpy_h2d(dst, src.as_ptr(), size).map_err(|e| format!("cuda_bridge::memcpy_h2d failed: {:?}", e))
}

/// Copy data from device.
pub fn memcpy_d2h(dst: &mut [u8], src: *const u8) -> Result<(), String> {
    let size = dst.len();
    cudarc::driver::memcpy_d2h(dst.as_mut_ptr(), src, size).map_err(|e| format!("cuda_bridge::memcpy_d2h failed: {:?}", e))
}

/// Allocate pinned host memory for faster transfers.
pub fn alloc_pinned(size: usize) -> Result<*mut u8, String> {
    let mut ptr = std::ptr::null_mut();
    cudarc::driver::alloc_host(&mut ptr, size).map_err(|e| format!("cuda_bridge::alloc_pinned failed: {:?}", e))?;
    Ok(ptr)
}

/// Free pinned host memory.
pub fn free_pinned(ptr: *mut u8) -> Result<(), String> {
    if ptr.is_null() { return Ok(()); }
    cudarc::driver::free_host(ptr).map_err(|e| format!("cuda_bridge::free_pinned failed: {:?}", e))
}

/// Query device attributes.
pub fn device_attribute(attr: u32, dev: i32) -> Result<i32, String> {
    let mut val = 0;
    cudarc::driver::device_get_attribute(&mut val, attr, dev).map_err(|e| format!("cuda_bridge::device_attr failed: {:?}", e))?;
    Ok(val)
}

/// Get the number of CUDA devices.
pub fn device_count() -> Result<i32, String> {
    let mut count = 0;
    cudarc::driver::device_get_count(&mut count).map_err(|e| format!("cuda_bridge::device_count failed: {:?}", e))?;
    Ok(count)
}

/// Set the active CUDA device.
pub fn set_device(dev: i32) -> Result<(), String> {
    cudarc::driver::set_device(dev).map_err(|e| format!("cuda_bridge::set_device failed: {:?}", e))
}

/// Convert f32 values to F16 and upload to device.
pub fn convert_f32_to_f16_device(src: &[f32]) -> Result<*mut u8, String> {
    let size_bytes = src.len() * 2;
    let mut dev_ptr = std::ptr::null_mut();
    cudarc::driver::alloc(&mut dev_ptr, size_bytes).map_err(|e| format!("convert: alloc failed: {:?}", e))?;

    // Convert f32 -> f16 in place using half crate
    let halfs: Vec<half::f16> = src.iter().map(|v| half::f16::from_f32(*v)).collect();
    let raw_bytes: &[u8] = unsafe { std::slice::from_raw_parts(halfs.as_ptr() as *const u8, size_bytes) };

    cudarc::driver::memcpy_h2d(dev_ptr, raw_bytes.as_ptr(), size_bytes)
        .map_err(|e| format!("convert: memcpy failed: {:?}", e))?;

    Ok(dev_ptr)
}

/// Convert device F16 values back to f32 on host.
pub fn convert_f16_to_f32_host(src_dev: *const u8, count: usize) -> Result<Vec<f32>, String> {
    let size_bytes = count * 2;
    let mut host_buf = vec![0u8; size_bytes];

    cudarc::driver::memcpy_d2h(host_buf.as_mut_ptr(), src_dev, size_bytes)
        .map_err(|e| format!("convert back: memcpy failed: {:?}", e))?;

    // Convert f16 -> f32
    let halfs: &[half::f16] = unsafe { std::slice::from_raw_parts(host_buf.as_ptr() as *const half::f16, count) };
    let result: Vec<f32> = halfs.iter().map(|h| h.to_f32()).collect();

    Ok(result)
}