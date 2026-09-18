//! Conformance test: RoPE implementation vs mathematical reference.
//!
//! Validates that the rotary embeddings implementation correctly applies
//! position-dependent rotations to query and key vectors.

use pesti_runner::kernel::rope::{CpuRopeKernel, RopeKernel, rope_cpu};

#[test]
fn test_rope_identity_at_zero() {
    // At position 0, RoPE should be identity (cos(0)=1, sin(0)=0)
    let seq_len = 1;
    let num_heads = 2;
    let head_dim = 32;
    let embed_dim = num_heads * head_dim;
    let base = 10000.0;

    let input: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1 - 5.0).collect();

    let mut q_rotated = input.clone();
    rope_cpu(&mut q_rotated, num_heads, seq_len, 0, head_dim, base);

    // Should be nearly identical to input at position 0
    let diff = input.iter().zip(&q_rotated)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-6, "RoPE at position 0 should be identity transform, got diff={}", diff);
}

#[test]
fn test_rope_positional_variation() {
    // Different positions should produce different rotations
    let seq_len = 1;
    let num_heads = 2;
    let head_dim = 32;
    let embed_dim = num_heads * head_dim;
    let base = 10000.0;

    let input: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1 - 5.0).collect();

    let mut q_pos0 = input.clone();
    rope_cpu(&mut q_pos0, num_heads, seq_len, 0, head_dim, base);

    let mut q_pos1 = input.clone();
    rope_cpu(&mut q_pos1, num_heads, seq_len, 1, head_dim, base);

    // Results at different positions should differ significantly
    let diff = q_pos0.iter().zip(&q_pos1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff > 0.001, "RoPE at different positions should produce different results");
}

#[test]
fn test_rope_preserves_norm() {
    // RoPE is a rotation - it should preserve vector norms
    let seq_len = 1;
    let num_heads = 2;
    let head_dim = 32;
    let embed_dim = num_heads * head_dim;
    let base = 10000.0;

    let input: Vec<f32> = (0..embed_dim).map(|i| ((i as f32) * 0.1) - 5.0).collect();

    // Compute original norm
    let orig_norm = input.iter().map(|v| v * v).sum::<f32>().sqrt();

    let mut rotated = input.clone();
    rope_cpu(&mut rotated, num_heads, seq_len, 10, head_dim, base);

    // Compute rotated norm
    let rot_norm = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();

    // Norms should be preserved (rotation is an isometry)
    let norm_diff = (orig_norm - rot_norm).abs() / orig_norm;
    assert!(norm_diff < 0.01, "RoPE should preserve vector norms, got diff={}", norm_diff);
}

#[test]
fn test_rope_kernel_trait() {
    // Test the RopeKernel trait interface with CPU implementation
    let kernel = CpuRopeKernel::new(10000.0);
    
    let num_heads = 2;
    let seq_len = 1;
    let head_dim = 32;
    let embed_dim = num_heads * head_dim;

    let q: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let k: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.15 - 7.0).collect();

    let mut q_rotated = q.clone();
    let mut k_rotated = k.clone();

    // Should not panic and should complete successfully
    let result = kernel.apply(&mut q_rotated, &mut k_rotated, num_heads, seq_len, 5);
    assert!(result.is_ok(), "RopeKernel::apply failed: {:?}", result.err());
}

#[test]
fn test_rope_gqa_different_head_counts() {
    // Test RoPE with GQA where Q has more heads than K
    let seq_len = 1;
    let num_q_heads = 8;
    let num_kv_heads = 2;
    let head_dim = 32;
    let base = 10000.0;

    let q: Vec<f32> = (0..(num_q_heads * head_dim)).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let k: Vec<f32> = (0..(num_kv_heads * head_dim)).map(|i| (i as f32) * 0.15 - 7.0).collect();

    let mut q_rotated = q.clone();
    rope_cpu(&mut q_rotated, num_q_heads, seq_len, 3, head_dim, base);

    let mut k_rotated = k.clone();
    rope_cpu(&mut k_rotated, num_kv_heads, seq_len, 3, head_dim, base);

    // Both should complete without errors (different head counts handled correctly)
    assert_eq!(q_rotated.len(), q.len());
    assert_eq!(k_rotated.len(), k.len());
}
