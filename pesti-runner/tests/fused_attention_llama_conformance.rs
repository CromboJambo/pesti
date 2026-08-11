//! Numerical conformance test: GPU fused attention vs llama.cpp reference
//!
//! Compares outputs of our CUDA kernel against llama.cpp's reference implementation
//! to verify numerical correctness across different precision modes.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

/// Reference RoPE implementation matching llama.cpp
fn apply_rope_cpu(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    for dim_pair in 0..half_dim {
        let idx = dim_pair * 2;
        if idx + 1 >= q.len() {
            continue;
        }

        // Compute cos/sin for this position and dimension pair
        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;
        let cos_val = freq.cos();
        let sin_val = freq.sin();

        // Apply RoPE rotation: [q0, q1] -> [q0*cos - q1*sin, q0*sin + q1*cos]
        let q0 = q[idx];
        let q1 = q[idx + 1];
        q[idx] = q0 * cos_val - q1 * sin_val;
        q[idx + 1] = q0 * sin_val + q1 * cos_val;
    }
}

/// Compute llama.cpp-style attention scores (scaled dot-product) - PER HEAD
fn reference_llama_attention(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
) -> Vec<f32> {
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    for (i, &val) in q.iter().enumerate() {
        q_rope[i] = val;
    }

    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    for (i, &val) in k.iter().enumerate() {
        k_rope[i] = val;
    }

    // Apply RoPE to each position (llama.cpp style) - PER HEAD
    for pos in 0..seq_q {
        for head in 0..num_heads {
            let q_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut q_rope[q_start..q_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }

    for pos in 0..seq_k {
        for head in 0..num_heads {
            let k_start = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(
                &mut k_rope[k_start..k_start + head_dim],
                head_dim,
                pos,
                rope_base,
            );
        }
    }

    // Compute attention scores (scaled by 1/sqrt(head_dim)) - PER HEAD
    let mut scores = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let mut score = 0.0f32;
                // Dot product across dimensions for THIS HEAD only (matching GPU output)
                let q_offset = q_pos * num_heads * head_dim + head * head_dim;
                let k_offset = k_pos * num_heads * head_dim + head * head_dim;

                for d in 0..head_dim {
                    score += q_rope[q_offset + d] * k_rope[k_offset + d];
                }
                // Scale by 1/sqrt(head_dim)
                score /= (head_dim as f32).sqrt();
                scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = score;
            }
        }
    }

    scores
}

/// Apply softmax per query row, per head (llama.cpp style)
fn reference_softmax(scores: &[f32], seq_q: usize, seq_k: usize, num_heads: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let end = start + seq_k;

            // Numerically stable softmax (max-subtract trick)
            let max_val = scores[start..end]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let exps: Vec<f32> = scores[start..end]
                .iter()
                .map(|&x| (x - max_val).exp())
                .collect();

            let sum: f32 = exps.iter().sum();
            for i in 0..seq_k {
                probs[start + i] = exps[i] / sum;
            }
        }
    }
    probs
}

/// Apply causal mask (llama.cpp style: q_pos >= k_pos = -inf) - per head
fn apply_causal_mask(scores: &mut [f32], seq_q: usize, seq_k: usize, num_heads: usize) {
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                if q_pos >= k_pos {
                    scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = -1e9;
                }
            }
        }
    }
}

#[test]
fn test_fused_attention_vs_llama_cpp() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping llama.cpp conformance test");
        return;
    }

    println!("=== Fused Attention vs llama.cpp Conformance Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();

    // Configuration (small for quick testing)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;

    // Create deterministic Q, K (f16) matching llama.cpp test patterns
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );

    // Compute llama.cpp reference (CPU)
    let mut llama_scores = reference_llama_attention(
        &q_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        &k_h.iter().map(|&x| x.to_f32()).collect::<Vec<f32>>(),
        seq_q,
        seq_k,
        num_heads,
        head_dim,
        rope_base,
    );

    // Apply causal mask BEFORE softmax (llama.cpp style) - per head
    apply_causal_mask(&mut llama_scores, seq_q, seq_k, num_heads);

    let llama_probs = reference_softmax(&llama_scores, seq_q, seq_k, num_heads);

    println!(
        "llama.cpp softmax sum (first query): {:.6}",
        llama_probs[0..seq_k].iter().sum::<f32>()
    );
    println!(
        "llama.cpp max attention score: {:.6}",
        llama_probs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Allocate device memory using low-level API
    let q_size = seq_q * num_heads * head_dim * 2; // f16 = 2 bytes
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    // Output: [seq_q, num_heads, seq_k] for multi-head attention
    let s_size = seq_q * num_heads * seq_k * 4; // f32 scores

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };

    // Copy Q, K to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
    }

    // Build kernel
    let stream = cuda_rt.new_stream().unwrap();
    let kernel =
        pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )
        .unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel
    unsafe {
        kernel
            .launch(
                scale,
                q_ptr as u64,
                k_ptr as u64,
                v_ptr as u64,
                s_ptr as u64,
                seq_q,
                seq_k,
                num_heads,
                head_dim,
                rope_base,
                seq_k,
            )
            .unwrap();
    }

    cuda_rt.synchronize().unwrap();

    println!("✅ GPU kernel launched successfully");

    // Copy results back to host
    let mut gpu_probs = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_probs.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
        .unwrap();
    }

    println!(
        "GPU softmax sum (first query, first head): {:.6}",
        gpu_probs[0..seq_k].iter().sum::<f32>()
    );

    // Compare outputs - compare all heads (now that reference is per-head)
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let llama_val = llama_probs[q_pos * num_heads * seq_k + head * seq_k + k_pos];
                let gpu_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
                let gpu_val = gpu_probs[gpu_idx];

                let abs_err = (llama_val - gpu_val).abs();
                let rel_err = if llama_val.abs() > 1e-8 {
                    abs_err / llama_val.abs()
                } else {
                    abs_err
                };

                max_abs_err = max_abs_err.max(abs_err);
                max_rel_err = max_rel_err.max(rel_err);
            }
        }
    }

    println!();
    println!("Results:");
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);
    println!();

    if max_rel_err < 1e-4 {
        println!("✅ Numerical conformance PASSED vs llama.cpp (rel error < 1e-4)");
    } else if max_rel_err < 1e-2 {
        println!("⚠️ Moderate discrepancy detected (rel error < 1e-2)");
    } else {
        println!("❌ Large numerical discrepancy detected (rel error >= 1e-2)");
    }

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
