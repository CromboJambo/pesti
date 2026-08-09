//! Test demonstrating softmax kernel works in the forward pass.
//!
//! This test verifies that the SoftmaxKernel trait works correctly with CPU backend.

#[cfg(test)]
mod tests {
    // Import directly from the module (not through kernel re-export which is feature-gated)
    use pesti_runner::kernel::softmax::{SoftmaxKernel, SoftmaxKernelBuilder};

    #[test]
    fn test_cpu_softmax_basic() {
        // CPU path should always work
        let softmax_kernel = SoftmaxKernelBuilder::cpu();

        let logits = vec![2.0, 1.0, 0.0, -1.0, -2.0];
        let probs = softmax_kernel
            .forward(&logits)
            .expect("softmax should succeed");

        // Verify sum is ~1.0
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax probabilities should sum to ~1.0, got {}",
            sum
        );

        // Verify all values are positive
        assert!(
            probs.iter().all(|&p| p > 0.0),
            "All softmax values should be positive"
        );

        // Verify numerical stability with large values
        let large_logits = vec![1000.0, 1001.0, 1002.0];
        let stable_probs = softmax_kernel
            .forward(&large_logits)
            .expect("softmax should handle large values");
        let stable_sum: f32 = stable_probs.iter().sum();
        assert!(
            (stable_sum - 1.0).abs() < 1e-5,
            "Stable softmax should sum to ~1.0, got {}",
            stable_sum
        );

        println!("✅ CPU softmax test passed!");
    }

    #[test]
    fn test_cpu_softmax_uniform() {
        // Uniform distribution for [0, 0, 0]
        let softmax_kernel = SoftmaxKernelBuilder::cpu();
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax_kernel
            .forward(&logits)
            .expect("softmax should succeed");

        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-5);
        assert!((probs[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((probs[2] - 1.0 / 3.0).abs() < 1e-5);

        println!("✅ CPU softmax uniform test passed!");
    }
}
