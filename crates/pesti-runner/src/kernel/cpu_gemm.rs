//! Optimized CPU GEMM with tiling and blocking for cache efficiency.
//!
//! Replaces the naive O(m*n*k) loop with blocked/tiled computation that
//! improves L1/L2 cache hit rates significantly on large matrices.

use crate::error::{GemmError, Result};
use crate::kernel::gemm::{CpuGemmKernel as NaiveCpuGemmKernel, GemmArch, GemmKernel};
use half::f16;

/// Tile sizes optimized for typical L1/L2 cache hierarchies.
const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 32;

/// Optimized CPU GEMM kernel using blocked/tiled computation.
pub struct CpuGemmKernel {
    naive: NaiveCpuGemmKernel,
}

impl CpuGemmKernel {
    pub fn new() -> Self {
        Self {
            naive: NaiveCpuGemmKernel::new(),
        }
    }

    /// Blocked/tiled GEMM for cache efficiency.
    /// C = alpha * A @ B + beta * C
    /// A: m x k row-major, B: k x n row-major, C: m x n row-major
    fn gemm_blocked(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        // Initialize C if needed
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            for val in c.iter_mut() {
                *val *= beta;
            }
        }

        // Blocked GEMM: iterate over tiles of C
        for mm in (0..m).step_by(TILE_M) {
            let m_end = (mm + TILE_M).min(m);
            for nn in (0..n).step_by(TILE_N) {
                let n_end = (nn + TILE_N).min(n);

                // For each C tile, compute contribution from all K tiles
                for kk in (0..k).step_by(TILE_K) {
                    let k_end = (kk + TILE_K).min(k);

                    // Compute this tile of C
                    for i in mm..m_end {
                        for j in nn..n_end {
                            let mut sum = 0.0f32;
                            for l in kk..k_end {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            c[i * n + j] = alpha * sum + beta * c[i * n + j];
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for CpuGemmKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmKernel for CpuGemmKernel {
    fn matmul(
        &self,
        alpha: f32,
        a: &crate::kernel::device_buf::DeviceBuffer<f16>,
        b: &crate::kernel::device_buf::DeviceBuffer<f16>,
        beta: f32,
        c: &mut crate::kernel::device_buf::DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // Convert f16 to f32 once (not per-multiply)
        let a_f32: Vec<f32> = a.as_slice()
            .ok_or(GemmError::BufferSizeMismatch { expected: m * k, got: 0 })?
            .iter().map(|x| x.to_f32()).collect();

        let b_f32: Vec<f32> = b.as_slice()
            .ok_or(GemmError::BufferSizeMismatch { expected: k * n, got: 0 })?
            .iter().map(|x| x.to_f32()).collect();

        let c_host = c.as_mut_slice()
            .ok_or(GemmError::BufferSizeMismatch { expected: m * n, got: 0 })?;

        // Use optimized blocked path for larger matrices, naive for small ones
        if m * n < 1024 {
            return self.naive.matmul(alpha, a, b, beta, c, m, n, k);
        }

        self.gemm_blocked(&a_f32, &b_f32, c_host, m, n, k, alpha, beta)
    }

    fn is_available(&self) -> bool {
        true
    }

    fn arch(&self) -> GemmArch {
        GemmArch::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_gemm_kernel_available() {
        let kernel = CpuGemmKernel::new();
        assert!(kernel.is_available());
    }

    #[test]
    fn test_small_matrix_falls_back_to_naive() {
        // Small matrices should use naive path (no tiling overhead)
        let kernel = CpuGemmKernel::new();
        let a = crate::kernel::device_buf::DeviceBuffer::from_host(vec![
            f16::from_f32(1.0), f16::from_f32(2.0),
            f16::from_f32(3.0), f16::from_f32(4.0),
        ]);
        let b = crate::kernel::device_buf::DeviceBuffer::from_host(vec![
            f16::from_f32(1.0), f16::from_f32(0.0),
            f16::from_f32(0.0), f16::from_f32(1.0),
        ]);
        let mut c = crate::kernel::device_buf::DeviceBuffer::zeros(4);

        kernel.matmul(1.0, &a, &b, 0.0, &mut c, 2, 2, 2).unwrap();

        let c_host = c.to_host();
        assert!((c_host[0] - 1.0).abs() < 0.01);
        assert!((c_host[1] - 2.0).abs() < 0.01);
        assert!((c_host[2] - 3.0).abs() < 0.01);
        assert!((c_host[3] - 4.0).abs() < 0.01);
    }
}
