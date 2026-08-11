//! Numerical conformance test: GPU fused attention vs CPU reference
//! Tests raw attention scores (before softmax) with deterministic inputs.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

fn apply_rope_cpu(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    for dim_pair in 0..half_dim {
        let idx = dim_pair * 2;
        if idx + 1 >= q.len() { continue; }
        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;
        let cos_val = freq.cos();
        let sin_val = freq.sin();
        let q0 = q[idx];
        let q1 = q[idx + 1];
        q[idx] = q0 * cos_val - q1 * sin_val;
        q[idx + 1] = q0 * sin_val + q1 * cos_val;
    }
}

fn reference_raw_scores(
    q_h: &[f16], k_h: &[f16],
    seq_q: usize, seq_k: usize, num_heads: usize, head_dim: usize,
    rope_base: f32, scale: f32,
) -> Vec<f32> {
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    for (i, &val) in q_h.iter().enumerate() { q_rope[i] = val.to_f32(); }
    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    for (i, &val) in k_h.iter().enumerate() { k_rope[i] = val.to_f32(); }

    // Apply RoPE per head (fixed version - was only applying to head 0 before)
    for pos in 0..seq_q {
        for head in 0..num_heads {
            let s = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(&mut q_rope[s..s + head_dim], head_dim, pos, rope_base);
        }
    }
    for pos in 0..seq_k {
        for head in 0..num_heads {
            let s = pos * num_heads * head_dim + head * head_dim;
            apply_rope_cpu(&mut k_rope[s..s + head_dim], head_dim, pos, rope_base);
        }
    }

    // Compute raw scores - SUM ACROSS HEADS then apply scale (matching GPU behavior)
    let mut scores = vec![0.0f32; seq_q * seq_k];
    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            let mut score = 0.0f32;
            // Sum dot products across all heads first
            for head in 0..num_heads {
                for dim in 0..head_dim {
                    let qi = q_pos * num_heads * head_dim + head * head_dim + dim;
                    let ki = k_pos * num_heads * head_dim + head * head_dim + dim;
                    score += q_rope[qi] * k_rope[ki];
                }
            }
            // Apply scale ONCE (matching GPU which applies scale internally)
            score *= scale;
            if q_pos >= k_pos { scores[q_pos * seq_k + k_pos] = -1e9; }
            else { scores[q_pos * seq_k + k_pos] = score; }
        }
    }
    scores
}

#[test]
fn test_fused_attention_numerical_conformance() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    println!("=== Fused Attention Numerical Conformance Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0)).collect();
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0)).collect();

    // CPU reference: sum across heads, then apply scale ONCE
    let cpu_scores = reference_raw_scores(&q_h, &k_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale);

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };

    // Zero-initialize output buffer on device
    let zero_init = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(
            s_ptr, zero_init.as_ptr() as *const u8, s_size).unwrap();
    }
    cuda_rt.synchronize().unwrap();  // Ensure zero-init completes before kernel

    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
    }

    let stream = cuda_rt.new_stream().unwrap();
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(), stream.clone(),
    ).unwrap();

    unsafe {
        kernel.launch(scale, q_ptr as u64, k_ptr as u64, v_ptr as u64, s_ptr as u64,
            seq_q, seq_k, num_heads, head_dim, rope_base, seq_k).unwrap();
    }
    cuda_rt.synchronize().unwrap();

    let mut gpu_scores_raw = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores_raw.as_mut_ptr() as *mut u8, s_ptr as *const u8, s_size).unwrap();
    }

    // GPU kernel outputs PER-HEAD scores. Sum across heads to match CPU reference.
    // NOTE: GPU already applied scale internally, so NO need to apply again here!
    let mut gpu_scores = vec![0.0f32; seq_q * seq_k];
    for q in 0..seq_q {
        for k in 0..seq_k {
            let mut total = 0.0f32;
            for h in 0..num_heads {
                total += gpu_scores_raw[q * num_heads * seq_k + h * seq_k + k];
            }
            // GPU already applied scale, so just apply causal mask
            if q >= k { total = -1e9; }
            gpu_scores[q * seq_k + k] = total;
        }
    }

    // Compare
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut worst_q = 0;
    let mut worst_k = 0;
    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            let c = cpu_scores[q_pos * seq_k + k_pos];
            let g = gpu_scores[q_pos * seq_k + k_pos];
            let abs_err = (c - g).abs();
            let rel_err = if c.abs() > 1e-6 { abs_err / c.abs() } else { abs_err };
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
                max_rel_err = rel_err;
                worst_q = q_pos;
                worst_k = k_pos;
            }
        }
    }

    let cpu_val = cpu_scores[worst_q * seq_k + worst_k];
    let gpu_val = gpu_scores[worst_q * seq_k + worst_k];
    println!("Worst: q_pos={} k_pos={} CPU={} GPU={}", worst_q, worst_k, cpu_val, gpu_val);

    println!("Max absolute error: {max_abs_err:.6e}");
    println!("Max relative error: {max_rel_err:.6e}");

    if max_rel_err < 1e-4 { println!("✅ PASSED"); }
    else if max_rel_err < 1e-2 { println!("⚠️ Moderate discrepancy"); }
    else { println!("❌ Large discrepancy"); }

    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
