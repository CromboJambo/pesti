//! cuBLASLt GEMM integration tests for PESTI CUTLASS path.
//! Verifies correctness and benchmarks performance vs plain cuBLAS.

use half::f16;

// CPU reference implementation (F32)
fn cpu_gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[test]
fn test_cublaslt_gemm_correctness_small() {
    // Small 4x4 matmul: verify cuBLASLt produces correct results
    let m = 4;
    let n = 4;
    let k = 4;

    // Create simple test matrices with small integer values
    let mut a_vals = Vec::new();
    for i in 0..(m * k) {
        a_vals.push(f16::from_f32((i as f32) % 5.0));
    }

    let mut b_vals = Vec::new();
    for j in 0..(k * n) {
        b_vals.push(f16::from_f32(((j as f32) * 2.0) % 7.0));
    }

    // Run cuBLASLt GEMM via pesti-runner's CUDA bridge
    use pesti_runner::kernel::cuda_bridge::{gemm_f16_cublaslt, upload_weights_to_gpu};

    let w_dev = upload_weights_to_gpu(&b_vals).expect("upload weights failed");
    let result = gemm_f16_cublaslt(&a_vals, &w_dev, m, n, k).expect("cublasLt GEMM failed");

    // Verify output shape
    assert_eq!(result.len(), m * n, "output shape mismatch");

    // Compute CPU reference
    let a_f32: Vec<f32> = a_vals.iter().map(|v| v.to_f32()).collect();
    let b_f32: Vec<f32> = b_vals.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_gemm_f32(&a_f32, &b_f32, m, n, k);

    // Compare with F16 tolerance (0.1)
    for i in 0..(m * n) {
        let diff = (result[i] - expected[i]).abs();
        assert!(
            diff < 0.15,
            "element {} differs: cublasLt={:.4}, expected={:.4}, diff={:.4}",
            i,
            result[i],
            expected[i],
            diff
        );
    }

    println!("✓ cuBLASLt GEMM correctness test passed (small 4x4)");
}

#[test]
fn test_cublaslt_vs_cublas_performance() {
    // Benchmark cuBLASLt vs plain cuBLAS for LLM decode shapes
    use pesti_runner::kernel::cuda_bridge::{gemm_f16, gemm_f16_cublaslt, upload_weights_to_gpu};
    use std::time::Instant;

    let shapes = [
        (1, 4096, 4096), // LLM decode: single token, large hidden dim
        (1, 2048, 2048), // Smaller LLM
        (32, 512, 1024), // Batched smaller ops
    ];

    for &(m, n, k) in &shapes {
        println!("Benchmarking shape m={}, n={}, k={}", m, n, k);

        // Create random test data (deterministic seed)
        let mut rng = Vec::new();
        for i in 0..(m * k) {
            rng.push(f16::from_f32((i as f32) % 3.0 - 1.5));
        }

        let mut w_vals = Vec::new();
        for j in 0..(k * n) {
            w_vals.push(f16::from_f32(((j as f32) * 1.7) % 5.0 - 2.5));
        }

        // Upload weights once for both methods
        let w_dev_cublas = upload_weights_to_gpu(&w_vals).expect("upload failed");
        let w_dev_lt = upload_weights_to_gpu(&w_vals).expect("upload failed");

        // Warmup runs
        let _ = gemm_f16(&rng, &w_vals, m, n, k);
        let _ = gemm_f16_cublaslt(&rng, &w_dev_lt, m, n, k);

        // Benchmark plain cuBLAS (re-uploads weights each call)
        let start = Instant::now();
        for _ in 0..5 {
            let _ = gemm_f16(&rng, &w_vals, m, n, k);
        }
        let cublas_time = start.elapsed().as_secs_f64() / 5.0;

        // Benchmark cuBLASLt with persistent weights
        let start = Instant::now();
        for _ in 0..5 {
            let _ = gemm_f16_cublaslt(&rng, &w_dev_lt, m, n, k);
        }
        let lt_time = start.elapsed().as_secs_f64() / 5.0;

        println!(
            "  cuBLAS: {:.3}ms | cuBLASLt: {:.3}ms | speedup: {:.2}x",
            cublas_time * 1000.0,
            lt_time * 1000.0,
            if lt_time > 0.0 {
                cublas_time / lt_time
            } else {
                0.0
            }
        );

        // Cleanup device buffers
        drop(w_dev_cublas);
        drop(w_dev_lt);
    }
}

#[test]
fn test_cublaslt_gemm_large_shape() {
    // Test with LLM-scale dimensions to verify numerical stability
    let m = 1;
    let n = 4096;
    let k = 4096;

    use pesti_runner::kernel::cuda_bridge::{gemm_f16_cublaslt, upload_weights_to_gpu};

    // Create test data with varied values
    let mut a_vals = Vec::new();
    for i in 0..(m * k) {
        a_vals.push(f16::from_f32(((i as f32) * 0.7) % 4.0 - 2.0));
    }

    let mut w_vals = Vec::new();
    for j in 0..(k * n) {
        w_vals.push(f16::from_f32(((j as f32) * 1.3) % 6.0 - 3.0));
    }

    let w_dev = upload_weights_to_gpu(&w_vals).expect("upload failed");
    let result = gemm_f16_cublaslt(&a_vals, &w_dev, m, n, k).expect("large shape GEMM failed");

    // Verify no NaN or Inf values (numerical stability check)
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "non-finite value at index {}", i);
    }

    // Compute CPU reference on smaller subset to verify correctness
    let subset_size = 64;
    let a_f32: Vec<f32> = a_vals.iter().map(|v| v.to_f32()).collect();
    let w_f32: Vec<f32> = w_vals.iter().map(|v| v.to_f32()).collect();

    // CPU reference for first subset_size columns only (full would be too slow)
    let expected_subset = cpu_gemm_f32(&a_f32, &w_f32[..k * subset_size], m, subset_size, k);

    // Compare subset
    for i in 0..subset_size {
        let diff = (result[i] - expected_subset[i]).abs();
        assert!(
            diff < 1.0,
            "large shape element {} differs: cublasLt={:.4}, expected={:.4}, diff={:.4}",
            i,
            result[i],
            expected_subset[i],
            diff
        );
    }

    println!(
        "✓ cuBLASLt large shape test passed (m={}, n={}, k={})",
        m, n, k
    );
}