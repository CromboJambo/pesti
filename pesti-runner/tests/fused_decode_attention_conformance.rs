//! Conformance test: fused decode attention kernel vs CPU reference.
//!
//! Verifies that the single-kernel softmax + weighted-sum approach produces
//! numerically identical results to the existing CPU attention loop in layer.rs.

#[cfg(feature = "cuda")]
mod fused_decode_attention_conformance {
    use pesti_runner::cuda_runtime::{
        allocate_device_memory, copy_device_to_host, copy_host_to_device, free_device_memory,
        CudaRuntime,
    };
    use pesti_runner::kernel::fused_decode_attention::build_fused_decode_attention_kernel;

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

        // GPU path: create runtime and stream
        let rt = CudaRuntime::for_default_device().expect("failed to init CUDA runtime");
        let stream = rt.new_stream().expect("failed to create stream");

        // Allocate device memory using pesti's cuda_runtime helpers
        let q_size = head_dim * 4;
        let kv_size = seq_len * head_dim * 4;

        let q_dev = allocate_device_memory(q_size).expect("failed to allocate q on device");
        let k_dev = allocate_device_memory(kv_size).expect("failed to allocate k on device");
        let v_dev = allocate_device_memory(kv_size).expect("failed to allocate v on device");
        let out_dev = allocate_device_memory(q_size).expect("failed to allocate output on device");

        // Copy data to device using pesti's wrappers (synchronous)
        copy_host_to_device(
            q_dev,
            q.as_ptr() as *const u8,
            q_size,
        ).expect("H2D copy for q failed");
        copy_host_to_device(
            k_dev,
            k_cache.as_ptr() as *const u8,
            kv_size,
        ).expect("H2D copy for k failed");
        copy_host_to_device(
            v_dev,
            v_cache.as_ptr() as *const u8,
            kv_size,
        ).expect("H2D copy for v failed");

        // Build and launch fused attention kernel
        let kernel = build_fused_decode_attention_kernel(rt.context().clone(), stream.clone())
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
        pesti_runner::cuda_shim::stream_synchronize(&stream).unwrap();

        let mut gpu_result = vec![0.0f32; head_dim];
        copy_device_to_host(
            gpu_result.as_mut_ptr() as *mut u8,
            out_dev,
            q_size,
        ).expect("D2H copy failed");

        // Free device memory
        free_device_memory(q_dev).unwrap();
        free_device_memory(k_dev).unwrap();
        free_device_memory(v_dev).unwrap();
        free_device_memory(out_dev).unwrap();

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