//! Linear layer with device-resident weights (Phase 2b).
//!
//! Key change from Phase 2a: weight matrices are uploaded to GPU ONCE at model
//! load time via `DeviceTensor`, not re-uploaded on every forward pass. This
//! eliminates the dominant transfer bottleneck in transformer inference.
//!
//! The layer now manages three memory locations for its weight matrix:
//! - `cpu_data`: always present, f32 (model loading / CPU fallback)
//! - `gpu_buffer`: uploaded once as f16 for GEMM efficiency (optional)
//! - Transfers are explicit and cached

#![allow(clippy::redundant_closure)]

use crate::kernel::candle_bridge;
use crate::kernel::GemmKernel;
use crate::kernel::device_buf::DeviceBuffer;
use crate::tile_dequant::{self, QuantDtype};
use half::f16;
use std::sync::{Arc, OnceLock};

/// Linear layer with optional device-resident weights.
pub struct Linear {
    /// Host-side f32 weight data (always present for compatibility).
    pub weight: Vec<f32>,
    /// Optional GPU buffer holding the transposed weight matrix in f16 format.
    /// Uploaded once at load time, reused across all forward passes.
    gpu_weight: Option<Arc<DeviceBuffer<f16>>>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
    /// Raw quantized weight bytes for fused dequant+GEMM on CPU.
    /// When present and no GPU kernel is set, forward() uses the
    /// quantized path to reduce memory bandwidth by 4-8x.
    quant_data: Option<(Vec<u8>, QuantDtype)>,
    gemm_kernel: Option<Arc<dyn GemmKernel + Send + Sync>>,
    /// Cached candle-core weight tensor for optimized CPU GEMM.
    /// Lazily initialized on first CPU forward pass to avoid re-creating
    /// the tensor from the weight buffer every time.
    cpu_weight_tensor: OnceLock<candle_core::Tensor>,
}

impl Linear {
    /// Create a linear layer from CPU weight data.
    pub fn new(
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            weight,
            gpu_weight: None,
            bias,
            in_features,
            out_features,
            quant_data: None,
            gemm_kernel: None,
            cpu_weight_tensor: OnceLock::new(),
        }
    }

    /// Set a GPU GEMM kernel for this layer. Subsequent forward() calls will
    /// use the GPU path when available.
    pub fn set_gemm_kernel(&mut self, kernel: Arc<dyn GemmKernel + Send + Sync>) {
        self.gemm_kernel = Some(kernel);
    }

    /// Clear the GPU GEMM kernel (revert to CPU matmul).
    pub fn clear_gemm_kernel(&mut self) {
        self.gemm_kernel = None;
    }

    /// Set raw quantized weight bytes for fused dequant+GEMM on CPU.
    /// When set and no GPU kernel is active, forward() uses the quantized path
    /// which reduces memory bandwidth by 4-8x compared to f32 GEMM.
    pub fn set_quantized_data(&mut self, raw_bytes: Vec<u8>, dtype: QuantDtype) {
        self.quant_data = Some((raw_bytes, dtype));
    }

    /// Check if this layer has quantized weight data available.
    pub fn is_quantized(&self) -> bool {
        self.quant_data.is_some()
    }

    /// Upload weight matrix to GPU as f16 transposed layout for GEMM efficiency.
    /// Must be called after set_gemm_kernel() and before the first GPU forward pass.
    pub fn upload_weights_to_gpu(&mut self) {
        let k = self.in_features;
        let n = self.out_features;

        // Transpose weights: W is [out, in] row-major; GEMM needs B as [k, n].
        // Convert to f16 for GPU GEMM efficiency.
        let w = &self.weight;
        let mut w_t = Vec::with_capacity(k * n);
        for i in 0..k {
            for j in 0..n {
                w_t.push(f16::from_f32(w[j * k + i]));
            }
        }

        self.gpu_weight = Some(Arc::new(DeviceBuffer::from_host(w_t)));
    }

    /// Check if weights are resident on GPU.
    pub fn weights_on_gpu(&self) -> bool {
        self.gpu_weight.is_some()
    }

    /// Build from f16 weight bytes (legacy constructor).
    pub fn from_f16_weight(weight_f16: &[u8], bias: Option<Vec<f32>>) -> Self {
        let elements = weight_f16.len() / 2;
        let weight: Vec<f32> = weight_f16
            .chunks_exact(2)
            .map(|chunk| f16_to_f32(chunk))
            .collect();
        let (in_features, out_features) = if elements > 0 { (1, elements) } else { (0, 0) };
        Self {
            weight,
            gpu_weight: None,
            bias,
            in_features,
            out_features,
            quant_data: None,
            gemm_kernel: None,
            cpu_weight_tensor: OnceLock::new(),
        }
    }

    /// Build from f32 weight bytes with explicit shape.
    pub fn from_f32_weight_with_dims(
        weight_f32: &[u8],
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let weight: Vec<f32> = weight_f32
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self {
            weight,
            gpu_weight: None,
            bias,
            in_features,
            out_features,
            quant_data: None,
            gemm_kernel: None,
            cpu_weight_tensor: OnceLock::new(),
        }
    }

    /// Build from f32 weight bytes (legacy).
    pub fn from_f32_weight(weight_f32: &[u8], bias: Option<Vec<f32>>) -> Self {
        let elements = weight_f32.len() / 4;
        let weight: Vec<f32> = weight_f32
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let (in_features, out_features) = if elements > 0 { (1, elements) } else { (0, 0) };
        Self {
            weight,
            gpu_weight: None,
            bias,
            in_features,
            out_features,
            quant_data: None,
            gemm_kernel: None,
            cpu_weight_tensor: OnceLock::new(),
        }
    }

    /// Build from f32 weight bytes with explicit shape.
    pub fn from_f32_weight_with_shape(
        weight_f32: &[u8],
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        let weight: Vec<f32> = weight_f32
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self {
            weight,
            gpu_weight: None,
            bias,
            in_features,
            out_features,
            quant_data: None,
            gemm_kernel: None,
            cpu_weight_tensor: OnceLock::new(),
        }
    }

    /// Forward pass: y = x @ W^T + bias.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        // Try GPU GEMM path if kernel is available and weights are on device.
        #[cfg(feature = "cuda")]
        if let Some(ref gemm) = self.gemm_kernel
            && gemm.is_available()
            && self.gpu_weight.is_some()
        {
            return self.forward_gpu(gemm, x, m, k, n);
        }

        // Quantized CPU path: fused dequant+GEMM when raw quantized bytes available.
        // Reduces memory bandwidth by 4-8x vs f32 for Q4_K/Q8_0 models.
        if let Some((ref raw_bytes, ref dtype)) = self.quant_data {
            return self.forward_quantized_cpu(x, batch_size, raw_bytes, dtype);
        }

        // CPU fallback: hand-written matmul with rayon parallelism.
        self.forward_cpu(x, batch_size)
    }

    /// GPU GEMM path using device-resident weights.
    #[cfg(feature = "cuda")]
    fn forward_gpu(
        &self,
        gemm: &Arc<dyn GemmKernel + Send + Sync>,
        x: &[f32],
        m: usize,
        _k: usize,
        n: usize,
    ) -> Vec<f32> {
        // Convert input to F16 for GPU GEMM.
        let x_f16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
        let a = DeviceBuffer::from_host(x_f16);

        // Use pre-uploaded GPU weight buffer (transposed, f16).
        let b = self.gpu_weight.as_ref().expect("GPU weights not uploaded");

        // Allocate output buffer on device.
        let mut c = DeviceBuffer::zeros(m * n);

        // Launch GPU GEMM: C = alpha * A @ B + beta * C
        gemm.matmul(1.0, &a, b, 0.0, &mut c, m, n, self.in_features)
            .expect("GPU GEMM matmul failed");

        // Transfer result back to host.
        let mut output = c.to_host();

        // Apply bias if present (on CPU for simplicity).
        if let Some(ref bias) = self.bias {
            for b_idx in 0..m {
                for o in 0..n {
                    output[b_idx * n + o] += bias[o];
                }
            }
        }

        output
    }

    /// CPU fallback: optimized matmul using candle-core's GEMM (SIMD + rayon).
    /// Fused dequantize+GEMM for quantized weights on CPU.
    /// Dequantizes weight blocks on-the-fly during matrix multiplication to
    /// minimize memory bandwidth usage. Supports Q4_K, Q8_0, Q4_0, Q5_K, Q6_K.
    fn forward_quantized_cpu(
        &self,
        x: &[f32],
        batch_size: usize,
        raw_bytes: &[u8],
        dtype: &QuantDtype,
    ) -> Vec<f32> {
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        // Safety check: input must be [batch_size * in_features] for standard matmul.
        // Embedding lookups pass token IDs (length=batch_size), so fall back to f32 path.
        if x.len() < m * k {
            return self.forward_cpu(x, batch_size);
        }

        let mut output = vec![0.0f32; m * n];

        // Fused dequantize+GEMM: for each output feature, dequantize weight row
        // on-the-fly and compute dot product with input.
        for b in 0..m {
            let x_row = &x[b * k..(b + 1) * k];
            
            // Process output features in parallel where possible
            for o in 0..n {
                let mut acc = 0.0f32;
                
                // Dequantize and accumulate in tiles to stay cache-friendly
                let tile_size = 512;
                for tile_start in (0..k).step_by(tile_size) {
                    let tile_end = (tile_start + tile_size).min(k);
                    let tile_len = tile_end - tile_start;
                    
                    // Dequantize this tile of the weight row
                    let w_tile = match self.dequantize_weight_tile(raw_bytes, dtype, o, tile_start, tile_len) {
                        Ok(w) => w,
                        Err(_) => continue, // Skip tile on dequant error
                    };
                    
                    // Dot product for this tile
                    for (i, &w) in w_tile.iter().enumerate() {
                        acc += x_row[tile_start + i] * w;
                    }
                }
                
                output[b * n + o] = acc;
            }
        }

        // Apply bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        output
    }

    /// Dequantize a tile of weights from a specific row at a specific offset.
    fn dequantize_weight_tile(
        &self,
        raw_bytes: &[u8],
        dtype: &QuantDtype,
        row: usize,
        start_elem: usize,
        len: usize,
    ) -> crate::error::Result<Vec<f32>> {
        // Calculate byte offset for this row and element position
        let bytes_per_elem = match dtype {
            QuantDtype::Q4_0 => 18.0 / 32.0,   // 18 bytes per 32 elements
            QuantDtype::Q4_1 => 20.0 / 32.0,
            QuantDtype::Q8_0 => 34.0 / 32.0,
            QuantDtype::Q4_K => 28.0 / 16.0,   // 28 bytes per 16 elements
            QuantDtype::Q5_K => 36.0 / 16.0,
            QuantDtype::Q6_K => 42.0 / 16.0,
        };
        
        let row_bytes = (self.in_features as f32 * bytes_per_elem).ceil() as usize;
        let row_offset = row * row_bytes;
        
        if row_offset >= raw_bytes.len() {
            return Ok(vec![0.0f32; len]);
        }
        
        // For block-quantized formats, we need to align to block boundaries
        let block_size = match dtype {
            QuantDtype::Q4_0 | QuantDtype::Q4_1 | QuantDtype::Q8_0 => 32,
            QuantDtype::Q4_K | QuantDtype::Q5_K | QuantDtype::Q6_K => 16,
        };
        
        // Dequantize from block-aligned start
        let aligned_start = (start_elem / block_size) * block_size;
        let dequant_len = len + (start_elem - aligned_start);
        
        let row_data = &raw_bytes[row_offset..(row_offset + row_bytes).min(raw_bytes.len())];
        
        match dtype {
            QuantDtype::Q4_0 => tile_dequant::dequantize_q4_0_tile(row_data, aligned_start, dequant_len)
                .map(|v| v[start_elem - aligned_start..].to_vec()),
            QuantDtype::Q8_0 => tile_dequant::dequantize_q8_0_tile(row_data, aligned_start, dequant_len)
                .map(|v| v[start_elem - aligned_start..].to_vec()),
            QuantDtype::Q4_K => tile_dequant::dequantize_q4_k_tile(row_data, aligned_start, dequant_len)
                .map(|v| v[start_elem - aligned_start..].to_vec()),
            QuantDtype::Q5_K => tile_dequant::dequantize_q5_k_tile(row_data, aligned_start, dequant_len)
                .map(|v| v[start_elem - aligned_start..].to_vec()),
            QuantDtype::Q6_K => tile_dequant::dequantize_q6_k_tile(row_data, aligned_start, dequant_len)
                .map(|v| v[start_elem - aligned_start..].to_vec()),
            QuantDtype::Q4_1 => Err(crate::error::RunnerError::Internal("Q4_1 not yet supported in quantized GEMM".into())),
        }
    }

    fn forward_cpu(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let m = batch_size;
        let k = self.in_features;
        let n = self.out_features;

        // Fast path for very small matrices where tensor overhead dominates.
        if m * k * n < 1024 {
            return self.forward_cpu_naive(x, batch_size);
        }

        // Get or lazily create the cached weight tensor [k, n] (transposed).
        let w_t = match self.cpu_weight_tensor.get() {
            Some(t) => t.clone(),
            None => {
                // Transpose weights: W is [out, in] row-major; GEMM needs B as [k, n].
                let w = &self.weight;
                let mut w_t_data = Vec::with_capacity(k * n);
                for i in 0..k {
                    for j in 0..n {
                        w_t_data.push(w[j * k + i]);
                    }
                }
                match candle_core::Tensor::from_vec(w_t_data, (k, n), &candle_core::Device::Cpu) {
                    Ok(tensor) => self.cpu_weight_tensor.get_or_init(|| tensor.clone()).clone(),
                    Err(e) => {
                        eprintln!("Warning: failed to create CPU weight tensor: {e}; falling back to naive matmul");
                        return self.forward_cpu_naive(x, batch_size);
                    }
                }
            }
        };

        // Use candle-core's optimized GEMM (SIMD + rayon parallelism).
        match candle_bridge::cpu_gemm_f32(x, m, k, &w_t, self.bias.as_deref()) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Warning: candle CPU GEMM failed: {e}; falling back to naive matmul");
                self.forward_cpu_naive(x, batch_size)
            }
        }
    }

    /// Original naive matmul (kept as fallback for small matrices and error recovery).
    fn forward_cpu_naive(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; batch_size * self.out_features];

        use rayon::prelude::*;

        let k = self.in_features;
        let n = self.out_features;

        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(b, out_row)| {
                let start_idx = b * k;
                let end_idx = std::cmp::min((b + 1) * k, x.len());
                let x_row = &x[start_idx..end_idx];

                for o in 0..n {
                    let w_row = &self.weight[o * k..(o + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..std::cmp::min(k, x_row.len()) {
                        acc += x_row[i] * w_row[i];
                    }
                    out_row[o] = acc;
                }
            });

        // Apply bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    output[b * self.out_features + o] += bias[o];
                }
            }
        }

        output
    }

    /// Matrix multiply: C = A @ B^T, where A is [m x k] and B is [n x k].
    pub fn matmul_transpose_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[j * k + l];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }
}

/// Convert half-float bytes to f32.
fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let f32_bits = (sign << 31) | (frac << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        let f32_exp = (exp - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_vs_scalar() {
        let weight = vec![
            1.0, 2.0, 3.0, // row 0: W[0][0]=1, W[0][1]=2, W[0][2]=3
            4.0, 5.0, 6.0, // row 1: W[1][0]=4, W[1][1]=5, W[1][2]=6
        ];
        let bias = Some(vec![0.1, 0.2]);

        let linear = Linear::new(weight, bias, 3, 2);

        // Input: 2x3 matrix
        let x = vec![
            1.0, 2.0, 3.0, // batch 0
            4.0, 5.0, 6.0, // batch 1
        ];

        let output = linear.forward(&x, 2);

        // Expected: C[b,o] = sum_i(x[b,i] * W[o,i]) + bias[o]
        // C[0,0] = 1*1 + 2*2 + 3*3 + 0.1 = 14.1
        // C[0,1] = 1*4 + 2*5 + 3*6 + 0.2 = 32.2
        // C[1,0] = 4*1 + 5*2 + 6*3 + 0.1 = 32.1
        // C[1,1] = 4*4 + 5*5 + 6*6 + 0.2 = 77.2

        let expected = [14.1, 32.2, 32.1, 77.2];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let linear = Linear::new(weight, None, 2, 3);

        let x = vec![1.0, 2.0];

        let output = linear.forward(&x, 1);

        // Expected: C[0,o] = sum_i(x[i] * W[o,i])
        // C[0,0] = 1*1 + 2*2 = 5
        // C[0,1] = 1*3 + 2*4 = 11
        // C[0,2] = 1*5 + 2*6 = 17

        let expected = [5.0, 11.0, 17.0];

        for (i, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_gpu_weight_upload() {
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut linear = Linear::new(weight.clone(), None, 2, 3);

        // Before upload
        assert!(!linear.weights_on_gpu());

        // Upload to GPU
        linear.upload_weights_to_gpu();
        assert!(linear.weights_on_gpu());
    }
}

#[cfg(test)]
mod bench_gemm {
    use super::*;
    use crate::kernel::candle_bridge;
    use std::time::Instant;

    #[test]
    fn bench_gemm_comparison() {
        // Representative sizes for Qwen2.5-0.5B: dim=896, hidden=14336 (FFN)
        let k = 896;   // in_features
        let n = 14336; // out_features (FFN up projection)
        let m = 1;     // batch size 1 (decode)

        // Create random weights [n, k] and input [m, k]
        let weight: Vec<f32> = (0..(n * k)).map(|i| ((i as f32 % 97.0) - 48.0) / 100.0).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i as f32 % 53.0) - 26.0) / 100.0).collect();

        // Transpose weight to [k, n] for GEMM
        let mut w_t_data = Vec::with_capacity(k * n);
        for i in 0..k {
            for j in 0..n {
                w_t_data.push(weight[j * k + i]);
            }
        }

        // Benchmark naive matmul (10 iterations)
        let t_naive_start = Instant::now();
        for _ in 0..10 {
            let mut output = vec![0.0f32; m * n];
            for o in 0..n {
                let w_row = &weight[o * k..(o + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[i] * w_row[i];
                }
                output[o] = acc;
            }
            std::hint::black_box(&output);
        }
        let naive_time = t_naive_start.elapsed().as_secs_f64() / 10.0;

        // Benchmark candle-core GEMM (10 iterations)
        let w_t = candle_bridge::get_or_cache_cpu_weight(&w_t_data, k, n).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10)); // ensure cache warm
        
        let t_candle_start = Instant::now();
        for _ in 0..10 {
            let result = candle_bridge::cpu_gemm_f32(&x, m, k, &w_t, None).unwrap();
            assert_eq!(result.len(), m * n);
            std::hint::black_box(&result);
        }
        let candle_time = t_candle_start.elapsed().as_secs_f64() / 10.0;

        println!("=== GEMM Benchmark (m={}, k={}, n={}) ===", m, k, n);
        println!("Naive matmul:   {:.3} ms/op ({:.2} GFLOPS)", 
            naive_time * 1000.0,
            (2.0 * m as f64 * k as f64 * n as f64) / naive_time / 1e9);
        println!("Candle GEMM:    {:.3} ms/op ({:.2} GFLOPS)", 
            candle_time * 1000.0,
            (2.0 * m as f64 * k as f64 * n as f64) / candle_time / 1e9);
        println!("Speedup:        {:.2}x", naive_time / candle_time);

        // Verify correctness: both should produce similar results
        let naive_result = {
            let mut output = vec![0.0f32; m * n];
            for o in 0..n {
                let w_row = &weight[o * k..(o + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += x[i] * w_row[i];
                }
                output[o] = acc;
            }
            output
        };
        let candle_result = candle_bridge::cpu_gemm_f32(&x, m, k, &w_t, None).unwrap();

        let max_diff = naive_result.iter().zip(candle_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff between implementations: {:.6e}", max_diff);
        assert!(max_diff < 1e-4, "Results differ too much: {}", max_diff);
    }
}
