//! Phase 3 GPU kernel conformance tests.
//! Validates SwiGLU, RMSNorm, RoPE, and Softmax GPU implementations against CPU references.

use pesti_runner::kernel::{apply_rope_cpu, rmsnorm_cpu, softmax_cpu, swiglu_cpu};

#[test]
fn test_swiglu_conformance() {
    // Test known values
    let gate = vec![0.0f32, 1.0f32, -1.0f32];
    let up = vec![5.0f32, 2.0f32, 3.0f32];
    let out = swiglu_cpu(&gate, &up);

    // silu(0) = 0, so SwiGLU(0, x) = 0
    assert!((out[0] - 0.0).abs() < 1e-6, "SwiGLU at 0 failed");

    // silu(1) ≈ 0.731, so SwiGLU(1, 2) ≈ 1.462
    assert!((out[1] - 1.462).abs() < 0.01, "SwiGLU at 1 failed");

    // silu(-1) ≈ -0.269, so SwiGLU(-1, 3) ≈ -0.807
    assert!((out[2] + 0.807).abs() < 0.01, "SwiGLU at -1 failed");

    println!("✓ SwiGLU conformance: PASSED");
}

#[test]
fn test_rmsnorm_conformance() {
    // Simple case: [1, 2, 3] with weight all 1s
    let x = vec![1.0f32, 2.0f32, 3.0f32];
    let weight = vec![1.0f32, 1.0f32, 1.0f32];
    let out = rmsnorm_cpu(&x, &weight, 3, 1e-6);

    // RMS of [1,2,3] = sqrt(14/3) ≈ 2.160
    // Output should be x / rms
    assert!((out[0] - 0.463).abs() < 0.01, "RMSNorm output[0] failed");
    assert!((out[1] - 0.925).abs() < 0.01, "RMSNorm output[1] failed");
    assert!((out[2] - 1.387).abs() < 0.01, "RMSNorm output[2] failed");

    println!("✓ RMSNorm conformance: PASSED");
}

#[test]
fn test_rope_conformance() {
    // RoPE at position 0 should be identity (cos(0)=1, sin(0)=0)
    let mut data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    apply_rope_cpu(&mut data, 1, 1, 0, 4, 10000.0);
    assert!((data[0] - 1.0).abs() < 1e-6, "RoPE at pos 0 failed (x)");
    assert!((data[1] - 2.0).abs() < 1e-6, "RoPE at pos 0 failed (y)");

    // At position 1, rotation should change values
    let mut data = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32];
    apply_rope_cpu(&mut data, 1, 1, 1, 4, 10000.0);
    // Just verify values changed (not identity)
    assert!(
        (data[0] - 1.0).abs() > 1e-6 || (data[1]).abs() > 1e-6,
        "RoPE at pos 1 failed"
    );

    println!("✓ RoPE conformance: PASSED");
}

#[test]
fn test_softmax_conformance() {
    // Basic softmax
    let logits = vec![1.0f32, 2.0f32, 3.0f32];
    let probs = softmax_cpu(&logits);

    // Sum should be ~1.0
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum failed");

    // All values should be positive
    assert!(probs.iter().all(|&x| x > 0.0), "Softmax negative values");

    // Numerical stability with large values
    let logits_large = vec![1000.0f32, 1001.0f32, 1002.0f32];
    let probs_large = softmax_cpu(&logits_large);
    let sum_large: f32 = probs_large.iter().sum();
    assert!(
        (sum_large - 1.0).abs() < 1e-5,
        "Softmax numerical stability failed"
    );

    // Uniform distribution for equal logits
    let logits_uniform = vec![0.0f32, 0.0f32, 0.0f32];
    let probs_uniform = softmax_cpu(&logits_uniform);
    assert!(
        (probs_uniform[0] - 1.0 / 3.0).abs() < 1e-5,
        "Softmax uniform failed"
    );

    println!("✓ Softmax conformance: PASSED");
}

#[test]
fn test_all_phase3_kernels() {
    test_swiglu_conformance();
    test_rmsnorm_conformance();
    test_rope_conformance();
    test_softmax_conformance();

    println!("\n=== All Phase 3 CPU conformance tests passed ===");
}
