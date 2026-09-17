//! Conformance test: fused decode attention kernel vs CPU reference.
//!
//! Verifies that the single-kernel softmax + weighted-sum approach produces
//! numerically identical results to the existing CPU attention loop in layer.rs.

#[cfg(feature = "cuda")]
mod fused_decode_attention_conformance {
    use cudarc::driver::{
        CuResult,
        safe::{CudaContext, CudaStream},
    };
    use std::sync::Arc;

    // CPU reference: softmax(q @ K^T / sqrt(d)) @ V
    fn cpu_reference(
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute attention scores
        let mut scores = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let mut score = 0.0f32;
            for j in 0..head_dim {
                score += q[j] * k_cache[pos * head_dim + j];
            }
            scores.push(score * scale);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        for s in exp_scores.iter_mut() {
            *s /= sum_exp;
        }

        // Weighted sum of V
        let mut output = vec![0.0f32; head_dim];
        for pos in 0..seq_len {
            for j in 0..head_dim {
                output[j] += exp_scores[pos] * v_cache[pos * head_dim + j];
            }
        }

        output
    }

    #[test]
    fn test_fused_decode_attention_conformance() {
        let seq_len = 64;
        let head_dim = 64;

        // Generate deterministic test data (no RNG to avoid syntax issues)
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let k_cache: Vec<f32> = (0..(seq_len * head_dim))
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.05 - 2.5)
            .collect();
        let v_cache: Vec<f32> = (0..(seq_len * head_dim))
            .map(|i| ((i * 13 + 5) % 100) as f32 * 0.04 - 2.0)
            .collect();

        // Compute CPU reference
        let cpu_result = cpu_reference(&q, &k_cache, &v_cache, seq_len, head_dim);

        // GPU path: allocate device memory, launch kernel, copy back
        let context = Arc::new(CudaContext::current().unwrap());
        let stream = Arc::new(CudaStream::default_stream());

        // Allocate device buffers
        let q_dev = unsafe {
            let mut ptr = std::ptr::null_mut();
            CuResult::cuMemAlloc(&mut ptr, (head_dim * 4) as u64).unwrap();
            ptr
        };
        let k_dev = unsafe {
            let mut ptr = std::ptr::null_mut();
            CuResult::cuMemAlloc(&mut ptr, (seq_len * head_dim * 4) as u64).unwrap();
            ptr
        };
        let v_dev = unsafe {
            let mut ptr = std::ptr::null_mut();
            CuResult::cuMemAlloc(&mut ptr, (seq_len * head_dim * 4) as u64).unwrap();
            ptr
        };
        let out_dev = unsafe {
            let mut ptr = std::ptr::null_mut();
            CuResult::cuMemAlloc(&mut ptr, (head_dim * 4) as u64).unwrap();
            ptr
        };

        // Copy data to device
        unsafe {
            CuResult::cuMemcpyHtoD(q_dev, q.as_ptr() as _, (head_dim * 4) as u64).unwrap();
            CuResult::cuMemcpyHtoD(
                k_dev,
                k_cache.as_ptr() as _,
                (seq_len * head_dim * 4) as u64,
            )
            .unwrap();
            CuResult::cuMemcpyHtoD(
                v_dev,
                v_cache.as_ptr() as _,
                (seq_len * head_dim * 4) as u64,
            )
            .unwrap();
        }

        // Build and launch fused attention kernel
        let kernel =
            pesti_runner::kernel::fused_decode_attention::build_fused_decode_attention_kernel(
                context.clone(),
                stream.clone(),
            )
            .expect("failed to build fused decode attention kernel");

        kernel
            .launch(
                q_dev as u64,
                k_dev as u64,
                v_dev as u64,
                1.0 / (head_dim as f32).sqrt(),
                seq_len,
                head_dim,
                out_dev as u64,
            )
            .expect("kernel launch failed");

        // Synchronize and copy result back
        unsafe {
            CuResult::cuStreamSynchronize(stream.as_raw()).unwrap();
            let mut gpu_result = vec![0.0f32; head_dim];
            CuResult::cuMemcpyDtoH(gpu_result.as_mut_ptr() as _, out_dev, (head_dim * 4) as u64)
                .unwrap();

            // Free device memory
            CuResult::cuMemFree(q_dev).unwrap();
            CuResult::cuMemFree(k_dev).unwrap();
            CuResult::cuMemFree(v_dev).unwrap();
            CuResult::cuMemFree(out_dev).unwrap();

            // Compare results
            let max_error = gpu_result
                .iter()
                .zip(cpu_result.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);

            println!("Max absolute error: {}", max_error);
            assert!(
                max_error < 1e-4,
                "GPU/CPU mismatch: max error {} exceeds tolerance 1e-4",
                max_error
            );
        }
    }
}
