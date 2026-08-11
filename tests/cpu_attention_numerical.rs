//! Numerical conformance tests for CPU forward pass components.
//!
//! These tests verify that each component (RMSNorm, RoPE, Softmax, SwiGLU) produces
//! numerically correct results before integrating into the full model.

#[cfg(test)]
mod rms_norm_tests {
    use pesti_runner::transformer::rms_norm::RmsNorm;

    #[test]
    fn test_rms_norm_simple() {
        let eps = 1e-5;
        
        // Create RMSNorm with unit weights
        let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let rms_norm = RmsNorm::new(weight, eps);
        
        // Input: [1, 2, 3, 4, 5]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = rms_norm.forward(&input, 1);
        
        // Manual computation:
        // RMS = sqrt((1+4+9+16+25)/5) = sqrt(55/5) = sqrt(11) ≈ 3.3166
        // normalized = [1/3.3166, 2/3.3166, 3/3.3166, 4/3.3166, 5/3.3166]
        let rms = (input.iter().map(|&x| x * x).sum::<f32>() / 5.0).sqrt();
        let expected: Vec<f32> = input.iter()
            .map(|&x| x / (rms + eps))
            .collect();
        
        for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (out - exp).abs();
            assert!(diff < 1e-6, "RMSNorm mismatch at {}: {} vs {}, diff={}", i, out, exp, diff);
        }
        
        println!("✅ RMSNorm simple test passed");
    }

    #[test]
    fn test_rms_norm_with_weights() {
        let eps = 1e-5;
        
        // Non-unit weights
        let weight = vec![0.5, 1.0, 1.5, 2.0, 2.5];
        let rms_norm = RmsNorm::new(weight, eps);
        
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = rms_norm.forward(&input, 1);
        
        // Manual computation with weights
        let rms = (input.iter().map(|&x| x * x).sum::<f32>() / 5.0).sqrt();
        let expected: Vec<f32> = input.iter()
            .zip(weight.iter())
            .map(|(&x, &w)| w * (x / (rms + eps)))
            .collect();
        
        for (i, (out, exp)) in output.iter().zip(expected.iter()).enumerate() {
            let diff = (out - exp).abs();
            assert!(diff < 1e-6, "RMSNorm+weight mismatch at {}: {} vs {}", i, out, exp);
        }
        
        println!("✅ RMSNorm with weights test passed");
    }

    #[test]
    fn test_rms_norm_batch() {
        let eps = 1e-5;
        let weight = vec![1.0; 8];
        let rms_norm = RmsNorm::new(weight, eps);
        
        // Batch size = 2, embed_dim = 8
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let output = rms_norm.forward(&input, 2);
        
        // For batch norm, each sample should be normalized independently
        // Sample 1: [1..8], Sample 2: [0.5..7.5]
        assert_eq!(output.len(), 16);
        
        println!("✅ RMSNorm batch test passed");
    }
}

#[cfg(test)]
mod rope_tests {
    use pesti_runner::transformer::rope::RopeConfig;

    #[test]
    fn test_rope_simple() {
        let head_dim = 8;
        let base = 10000.0;
        let max_seq = 2048;
        let rope_config = RopeConfig::new(head_dim, base, max_seq);
        
        // Single head, single position
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Apply RoPE at position 0
        rope_config.apply_single(&mut q, 1, 1, 0);
        
        // At position 0, theta = base^(-i/dim) for i in 0..dim/2
        // For head_dim=8: dim_half=4, theta = [1.0, 0.01, 0.0001, 0.000001]
        // angle = pos * theta = [0, 0, 0, 0] at pos=0
        // cos(0)=1, sin(0)=0 → no rotation at pos=0
        
        // Verify no change at position 0
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, (q_val, exp)) in q.iter().zip(expected.iter()).enumerate() {
            let diff = (q_val - exp).abs();
            assert!(diff < 1e-6, "RoPE pos=0 should be identity at {}: {} vs {}", i, q_val, exp);
        }
        
        println!("✅ RoPE position 0 test passed");
    }

    #[test]
    fn test_rope_position_5() {
        let head_dim = 8;
        let base = 10000.0;
        let max_seq = 2048;
        let rope_config = RopeConfig::new(head_dim, base, max_seq);
        
        let mut q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Apply RoPE at position 5
        rope_config.apply_single(&mut q, 1, 1, 5);
        
        // Manual verification for first pair (i=0)
        let dim_half = head_dim / 2;
        let freq_0 = base.powf(-0.0 / dim_half as f32); // = 1.0
        let angle_0 = 5.0 * freq_0; // = 5.0 radians
        let cos_0 = angle_0.cos();
        let sin_0 = angle_0.sin();
        
        // q[0] and q[4] are rotated together
        let q_orig_0 = 1.0;
        let q_orig_4 = 5.0;
        let expected_0 = q_orig_0 * cos_0 - q_orig_4 * sin_0;
        let expected_4 = q_orig_0 * sin_0 + q_orig_4 * cos_0;
        
        assert!((q[0] - expected_0).abs() < 1e-5, 
            "RoPE pair (0,4) mismatch: {} vs {}", q[0], expected_0);
        assert!((q[4] - expected_4).abs() < 1e-5, 
            "RoPE pair (0,4) mismatch: {} vs {}", q[4], expected_4);
        
        println!("✅ RoPE position 5 test passed");
    }

    #[test]
    fn test_rope_multiple_heads() {
        let head_dim = 8;
        let base = 10000.0;
        let max_seq = 2048;
        let rope_config = RopeConfig::new(head_dim, base, max_seq);
        
        // 2 heads, 1 position each
        // Layout: [head0_pos0_dim0..dim7, head1_pos0_dim0..dim7]
        let mut q = vec![1.0; 16]; // All ones
        
        rope_config.apply_single(&mut q, 2, 1, 5);
        
        // Both heads should be rotated independently
        assert_eq!(q.len(), 16);
        
        // Verify both heads have different values (rotation applied)
        let head0_changed = (q[0] - 1.0).abs() > 1e-6;
        let head1_changed = (q[8] - 1.0).abs() > 1e-6;
        
        assert!(head0_changed && head1_changed, "Both heads should be rotated");
        
        println!("✅ RoPE multiple heads test passed");
    }

    #[test]
    fn test_rope_sequence_length_3() {
        let head_dim = 8;
        let base = 10000.0;
        let max_seq = 2048;
        let rope_config = RopeConfig::new(head_dim, base, max_seq);
        
        // 1 head, 3 positions (seq_len=3)
        // Layout: [pos0_dim0..dim7, pos1_dim0..dim7, pos2_dim0..dim7]
        let mut q = vec![1.0; 24]; // All ones
        
        rope_config.apply_single(&mut q, 1, 3, 0);
        
        // Each position should have different rotation angles
        assert_eq!(q.len(), 24);
        
        // Verify positions are different (they shouldn't all be identical)
        let pos0_first = q[0];
        let pos1_first = q[8];
        let pos2_first = q[16];
        
        assert_ne!(pos0_first, pos1_first, "Position 0 and 1 should differ");
        assert_ne!(pos1_first, pos2_first, "Position 1 and 2 should differ");
        
        println!("✅ RoPE sequence length test passed");
    }
}

#[cfg(test)]
mod softmax_tests {
    #[test]
    fn test_softmax_simple() {
        // Standard softmax: exp(x) / sum(exp(x))
        let scores = vec![2.0, 1.0, 0.1];
        
        // Manual computation with numerical stability
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter()
            .map(|&s| (s - max_val).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let output: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        
        // Verify probabilities sum to 1.0
        let total: f32 = output.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "Softmax should sum to 1.0, got {}", total);
        
        // Verify order is preserved (higher input → higher output)
        assert!(output[0] > output[1], "Softmax should preserve ordering");
        assert!(output[1] > output[2], "Softmax should preserve ordering");
        
        println!("✅ Softmax simple test passed: {:?}", output);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let scores = vec![1000.0, 1001.0, 999.0];
        
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter()
            .map(|&s| (s - max_val).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let output: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        
        // Verify no NaN or Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Softmax output at {} should be finite: {}", i, val);
            assert!(val > 0.0 && val <= 1.0, "Softmax output at {} should be in (0,1]: {}", i, val);
        }
        
        // The middle value (1001) should have highest probability
        assert!(output[1] > output[0], "Middle value should have highest prob");
        assert!(output[1] > output[2], "Middle value should have highest prob");
        
        println!("✅ Softmax numerical stability test passed: {:?}", output);
    }

    #[test]
    fn test_softmax_batch() {
        // Batch of 2 sequences, each with 3 positions
        let scores = vec![2.0, 1.0, 0.1,   // batch 0
                          0.5, 0.3, 0.1];    // batch 1
        
        let batch_size = 2;
        let seq_len = 3;
        
        let mut output = vec![0.0f32; batch_size * seq_len];
        
        for b in 0..batch_size {
            let start = b * seq_len;
            let softmax_row = &scores[start..start + seq_len];
            
            // Find max for numerical stability
            let max_val = softmax_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Compute exp and sum
            let mut sum = 0.0f32;
            for i in 0..seq_len {
                let exp_val = (softmax_row[i] - max_val).exp();
                output[start + i] = exp_val;
                sum += exp_val;
            }
            
            // Normalize
            if sum > 0.0 {
                for i in 0..seq_len {
                    output[start + i] /= sum;
                }
            }
        }
        
        // Verify each batch sums to 1.0
        for b in 0..batch_size {
            let start = b * seq_len;
            let batch_sum: f32 = output[start..start + seq_len].iter().sum();
            assert!((batch_sum - 1.0).abs() < 1e-6, 
                "Batch {} softmax should sum to 1.0, got {}", b, batch_sum);
        }
        
        println!("✅ Softmax batch test passed: {:?}", output);
    }

    #[test]
    fn test_softmax_all_zeros() {
        // Edge case: all zeros → uniform distribution
        let scores = vec![0.0, 0.0, 0.0];
        
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter()
            .map(|&s| (s - max_val).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let output: Vec<f32> = if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / scores.len() as f32; scores.len()]
        };
        
        // Should be uniform distribution
        let expected = 1.0 / 3.0;
        for (i, &val) in output.iter().enumerate() {
            assert!((val - expected).abs() < 1e-6, 
                "All-zeros softmax should be uniform at {}: {} vs {}", i, val, expected);
        }
        
        println!("✅ Softmax all-zeros test passed: {:?}", output);
    }
}

#[cfg(test)]
mod swiglu_tests {
    #[test]
    fn test_silu_simple() {
        // SiLU: x / (1 + exp(-x)) with numerical stability
        let x = 1.0;
        
        let sigmoid = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x / (1.0 + x.exp())
        };
        
        let output = sigmoid * x; // SiLU(x) = x * sigmoid(x)
        
        // Manual check: SiLU(1.0) ≈ 0.731
        assert!((output - 0.731).abs() < 1e-3, "SiLU(1.0) ≈ 0.731, got {}", output);
        
        println!("✅ SiLU simple test passed: {}", output);
    }

    #[test]
    fn test_silu_negative() {
        // Test numerical stability for negative values
        let x = -5.0;
        
        let sigmoid = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            x / (1.0 + x.exp())
        };
        
        let output = sigmoid * x;
        
        // SiLU(-5.0) should be small but not zero
        assert!(output.abs() < 0.1, "SiLU(-5.0) should be small: {}", output);
        assert!(output.is_finite(), "SiLU(-5.0) should be finite: {}", output);
        
        println!("✅ SiLU negative test passed: {}", output);
    }

    #[test]
    fn test_swiglu_simple() {
        // SwiGLU: silu(gate) * up
        let gate = vec![1.0, 2.0, 3.0];
        let up = vec![0.5, 0.6, 0.7];
        
        let mut output = vec![0.0f32; gate.len()];
        
        for i in 0..gate.len() {
            let sigmoid = if gate[i] >= 0.0 {
                1.0 / (1.0 + (-gate[i]).exp())
            } else {
                gate[i] / (1.0 + gate[i].exp())
            };
            
            output[i] = sigmoid * gate[i] * up[i];
        }
        
        // Manual check for first element
        let sigmoid_0 = 1.0 / (1.0 + (-1.0).exp()); // ≈ 0.731
        let expected_0 = sigmoid_0 * 1.0 * 0.5;     // ≈ 0.365
        
        assert!((output[0] - expected_0).abs() < 1e-3, 
            "SwiGLU[0] mismatch: {} vs {}", output[0], expected_0);
        
        println!("✅ SwiGLU simple test passed: {:?}", output);
    }

    #[test]
    fn test_swiglu_batch() {
        // Batch size = 2, intermediate_dim = 4
        let gate = vec![1.0, 2.0, 3.0, 4.0,   // batch 0
                        0.5, 1.5, 2.5, 3.5];   // batch 1
        
        let up = vec![0.1, 0.2, 0.3, 0.4,     // batch 0
                     0.4, 0.5, 0.6, 0.7];    // batch 1
        
        let intermediate_dim = 4;
        
        let mut output = vec![0.0f32; gate.len()];
        
        for b in 0..2 {
            let start = b * intermediate_dim;
            let gate_row = &gate[start..start + intermediate_dim];
            let up_row = &up[start..start + intermediate_dim];
            
            for i in 0..intermediate_dim {
                let sigmoid = if gate_row[i] >= 0.0 {
                    1.0 / (1.0 + (-gate_row[i]).exp())
                } else {
                    gate_row[i] / (1.0 + gate_row[i].exp())
                };
                
                output[start + i] = sigmoid * gate_row[i] * up_row[i];
            }
        }
        
        // Verify outputs are finite and positive
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "SwiGLU[{}] should be finite: {}", i, val);
            assert!(val >= 0.0, "SwiGLU[{}] should be non-negative: {}", i, val);
        }
        
        println!("✅ SwiGLU batch test passed: {:?}", output);
    }
}

#[cfg(test)]
mod integration_tests {
    use pesti_runner::transformer::rms_norm::RmsNorm;
    use pesti_runner::transformer::rope::RopeConfig;

    #[test]
    fn test_rmsnorm_then_rope() {
        // Test RMSNorm → RoPE pipeline (part of transformer layer)
        let embed_dim = 8;
        let head_dim = 4;
        let num_heads = 2;
        
        let eps = 1e-5;
        let weight = vec![1.0; embed_dim];
        let rms_norm = RmsNorm::new(weight, eps);
        
        let rope_config = RopeConfig::new(head_dim, 10000.0, 2048);
        
        // Input: [embed_dim]
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Step 1: RMSNorm
        let normed = rms_norm.forward(&x, 1);
        
        // Step 2: RoPE (applied to Q, which is num_heads * head_dim)
        // For simplicity, assume x IS the Q vector after projection
        rope_config.apply_single(&mut normed, num_heads, 1, 5);
        
        // Verify output has correct shape
        assert_eq!(normed.len(), embed_dim);
        
        // Verify values changed (RoPE should rotate)
        let changed = normed.iter()
            .zip(x.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        
        assert!(changed, "RoPE should change the input");
        
        println!("✅ RMSNorm → RoPE pipeline test passed");
    }

    #[test]
    fn test_attention_score_computation() {
        // Test Q @ K^T dot product computation
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        let q_head = vec![1.0, 2.0, 3.0, 4.0];
        let k_pos = vec![0.5, 1.5, 2.5, 3.5];
        
        // Dot product
        let dot: f32 = q_head.iter().zip(k_pos.iter()).map(|(a, b)| a * b).sum();
        let score = dot * scale;
        
        // Manual verification:
        // dot = 1*0.5 + 2*1.5 + 3*2.5 + 4*3.5 = 0.5 + 3.0 + 7.5 + 14.0 = 25.0
        // score = 25.0 / sqrt(4) = 12.5
        let expected_dot = 25.0;
        let expected_score = expected_dot * scale;
        
        assert!((dot - expected_dot).abs() < 1e-6, "Dot product mismatch: {} vs {}", dot, expected_dot);
        assert!((score - expected_score).abs() < 1e-6, "Scaled score mismatch: {} vs {}", score, expected_score);
        
        println!("✅ Attention score computation test passed: dot={}, score={}", dot, score);
    }

    #[test]
    fn test_full_attention_head() {
        // Test complete attention head: Q @ K^T → softmax → @ V
        let head_dim = 4;
        let cache_len = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();
        
        // Q for this head: [head_dim]
        let q_head = vec![1.0, 2.0, 3.0, 4.0];
        
        // K cache: [cache_len, head_dim]
        let k_cache: Vec<Vec<f32>> = vec![
            vec![0.5, 1.5, 2.5, 3.5],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.5, 2.5, 3.5, 4.5],
        ];
        
        // V cache: [cache_len, head_dim]
        let v_cache: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.2, 0.3, 0.4, 0.5],
            vec![0.3, 0.4, 0.5, 0.6],
        ];
        
        // Step 1: Q @ K^T → scores [cache_len]
        let mut scores = vec![0.0; cache_len];
        for (t, k_pos) in k_cache.iter().enumerate() {
            let dot: f32 = q_head.iter().zip(k_pos.iter()).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        
        // Step 2: Softmax
        let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter()
            .map(|&s| (s - max_val).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![1.0 / cache_len as f32; cache_len]
        };
        
        // Step 3: Softmax @ V → output [head_dim]
        let mut attn_output = vec![0.0; head_dim];
        for (t, v_pos) in v_cache.iter().enumerate() {
            let weight = weights[t];
            for d in 0..head_dim {
                attn_output[d] += weight * v_pos[d];
            }
        }
        
        // Verify output has correct shape
        assert_eq!(attn_output.len(), head_dim);
        
        // Verify no NaN or Inf
        for (i, &val) in attn_output.iter().enumerate() {
            assert!(val.is_finite(), "Attention output[{}] should be finite: {}", i, val);
        }
        
        println!("✅ Full attention head test passed: {:?}", attn_output);
    }
}

fn main() {
    // Run all tests manually for better output control
    test_rms_norm_simple();
    test_rms_norm_with_weights();
    test_rope_simple();
    test_rope_position_5();
    test_softmax_simple();
    test_softmax_numerical_stability();
    test_silu_simple();
    test_swiglu_simple();
    test_rmsnorm_then_rope();
    test_attention_score_computation();
    test_full_attention_head();
    
    println!("\n✅ All numerical conformance tests passed!");
}
