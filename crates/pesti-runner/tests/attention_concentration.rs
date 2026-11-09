//! Attention concentration kernel integration test.
//! Verifies the kernel computes correct concentration scores on synthetic data.

#[cfg(feature = "cuda")]
mod tests {
    use std::sync::Arc;

    #[test]
    fn test_attention_concentration_kernel() {
        // Initialize CUDA runtime and stream
        let runtime = pesti_runner::cuda_runtime::CudaRuntime::new(0)
            .expect("failed to init cuda runtime");
        let stream = runtime.new_stream().expect("failed to create stream");

        // Test config: batch=1, seq_len=4, heads=4
        let seq_len = 4usize;
        let num_heads = 4usize;
        let batch_size = 1usize;

        // Create synthetic attention weights that sum to 1.0 per row (softmax-style)
        // Head 0: concentrated on position 0 (expect high concentration)
        // Head 1: uniform distribution (expect low concentration)
        // Head 2: medium concentration
        // Head 3: single position (expect max concentration)
        let mut weights = vec![0.0f32; batch_size * seq_len * num_heads];

        // Head 0: concentrated on pos 0
        weights[0] = 0.95;
        weights[1] = 0.025;
        weights[2] = 0.015;
        weights[3] = 0.01;

        // Head 1: uniform
        for i in 4..8 {
            weights[i] = 0.25;
        }

        // Head 2: medium concentration
        weights[8] = 0.6;
        weights[9] = 0.2;
        weights[10] = 0.15;
        weights[11] = 0.05;

        // Head 3: single position
        weights[12] = 1.0;

        // Allocate device memory for input and output
        let weights_buf = pesti_runner::cuda_runtime::allocate_device_memory(
            weights.len() * std::mem::size_of::<f32>(),
        )
        .expect("failed to alloc weights");
        let concentrations_buf = pesti_runner::cuda_runtime::allocate_device_memory(
            num_heads * std::mem::size_of::<f32>(),
        )
        .expect("failed to alloc concentrations");

        // Copy input to device
        pesti_runner::cuda_runtime::copy_host_to_device(weights_buf, &weights).expect("h2d failed");

        // Build and launch kernel
        let kernel = pesti_runner::kernel::attention_concentration::build_attention_concentration_kernel(
            Arc::clone(runtime.context()),
            Arc::clone(&stream),
        )
        .expect("failed to build attention concentration kernel");

        kernel.forward(
            weights_buf,
            concentrations_buf,
            seq_len,
            num_heads,
            batch_size,
        )
        .expect("kernel launch failed");

        // Read back results
        let mut concentrations = vec![0.0f32; num_heads];
        pesti_runner::cuda_runtime::copy_device_to_host(
            &mut concentrations,
            concentrations_buf,
        )
        .expect("d2h failed");

        // Verify concentration scores
        assert!(concentrations[0] > 0.9, "Head 0 should be highly concentrated (>0.9), got {}", concentrations[0]);
        assert!(concentrations[1] < 0.3, "Head 1 should be diffuse (<0.3), got {}", concentrations[1]);
        assert!(concentrations[2] > 0.4 && concentrations[2] < 0.7, "Head 2 should be medium concentration (0.4-0.7), got {}", concentrations[2]);
        assert!(concentrations[3] > 0.99, "Head 3 should be max concentrated (>0.99), got {}", concentrations[3]);

        println!("✅ Attention concentration kernel test passed!");
        println!("  Head 0 (concentrated): {:.4}", concentrations[0]);
        println!("  Head 1 (uniform):      {:.4}", concentrations[1]);
        println!("  Head 2 (medium):       {:.4}", concentrations[2]);
        println!("  Head 3 (single pos):   {:.4}", concentrations[3]);

        // Cleanup
        pesti_runner::cuda_runtime::free_device_memory(weights_buf).expect("free weights failed");
        pesti_runner::cuda_runtime::free_device_memory(concentrations_buf).expect("free concentrations failed");
    }
}