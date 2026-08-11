//! Debug numerical conformance test with detailed output

use half::f16;
use pesti_runner::cuda_runtime::{CudaRuntime, allocate_device_memory, copy_host_to_device, copy_device_to_host, free_device_memory};

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
    q_h: &[f16],
    k_h: &[f16],
    v_h: &[f16],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
    scale: f32,
) -> Vec<f32> {
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    for (i, &val) in q_h.iter().enumerate() { q_rope[i] = val.to_f32(); }
    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    for (i, &val) in k_h.iter().enumerate() { k_rope[i] = val.to_f32(); }

    // Apply RoPE to Q and K
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

    // Compute attention output: softmax(Q @ K^T) @ V
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];
    
    for q_pos in 0..seq_q {
        for h in 0..num_heads {
            let q_head = &q_rope[q_pos * num_heads * head_dim + h * head_dim..][..head_dim];
            
            let mut scores = vec![0.0f32; seq_k];
            for k_pos in 0..seq_k {
                let k_head = &k_rope[k_pos * num_heads * head_dim + h * head_dim..][..head_dim];
                let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(a, b)| a * b).sum();
                scores[k_pos] = dot * scale;
                
                // Causal mask (mask future tokens: k_pos > q_pos)
                if k_pos > q_pos {
                    scores[k_pos] = -1e9;
                }
            }
            
            // Softmax
            let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let weights: Vec<f32> = if sum > 0.0 {
                exps.iter().map(|&e| e / sum).collect()
            } else {
                vec![1.0 / seq_k as f32; seq_k]
            };
            
            // Weighted sum of V → output head
            let mut attn_output = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                for k_pos in 0..seq_k {
                    let v_idx = k_pos * num_heads * head_dim + h * head_dim + d;
                    let v_val = v_h[v_idx].to_f32();
                    attn_output[d] += weights[k_pos] * v_val;
                }
            }
            
            // Write to output buffer
            for (d, &val) in attn_output.iter().enumerate() {
                output[q_pos * num_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }
    
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_rt = CudaRuntime::new(0)?;
    println!("=== Debug Numerical Conformance Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let cpu_output = reference_raw_scores(
        &q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale,
    );

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = (seq_q * num_heads * seq_k + seq_q * num_heads * head_dim) * 4;

    let q_ptr = allocate_device_memory(q_size)?;
    let k_ptr = allocate_device_memory(k_size)?;
    let v_ptr = allocate_device_memory(v_size)?;
    let s_ptr = allocate_device_memory(s_size)?;

    let zero_init = vec![0.0f32; seq_q * num_heads * head_dim];
    copy_host_to_device(s_ptr, zero_init.as_ptr() as *const u8, s_size)?;
    cuda_rt.synchronize()?;

    copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)?;
    copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)?;
    copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)?;

    let stream = cuda_rt.new_stream()?;
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(), stream.clone(),
    )?;

    kernel.launch(
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
    )?;
    cuda_rt.synchronize()?;
    
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // 4 bytes per f32
    let output_size = seq_q * num_heads * head_dim * 4;
    println!("Debug: seq_q={}, seq_k={}, num_heads={}, head_dim={}", seq_q, seq_k, num_heads, head_dim);
    println!("Debug: score_buffer_size={} bytes, output_size={} bytes", score_buffer_size, output_size);
    
    let mut gpu_output = vec![0.0f32; seq_q * num_heads * head_dim];
    
    // Output is stored after the score buffer, so copy from correct offset
    copy_device_to_host(
        gpu_output.as_mut_ptr() as *mut u8,
        (s_ptr as usize + score_buffer_size) as *const u8,
        output_size, // Copy full output portion
    )?;
    
    // Print ALL GPU output values (non-zero only)
    println!("\nFull GPU output ({} values):", gpu_output.len());
    for (i, &val) in gpu_output.iter().enumerate() {
        if val != 0.0 {
            let q = i / (num_heads * head_dim);
            let h = (i % (num_heads * head_dim)) / head_dim;
            let d = i % head_dim;
            println!("  idx[{}]: q={}, h={}, d={}, val={:.6}", i, q, h, d, val);
        }
    }

    // Print first few values for comparison
    println!("\nFirst 16 values (Q0, H0):");
    for i in 0..16 {
        let cpu_val = cpu_output[i];
        let gpu_val = gpu_output[i];
        let abs_err = (cpu_val - gpu_val).abs();
        println!(
            "  idx[{}]: CPU={:8.5}, GPU={:8.5}, err={:.6e}",
            i, cpu_val, gpu_val, abs_err
        );
    }

    // Calculate max error
    let mut max_abs_err = 0.0f32;
    for q in 0..seq_q {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let cpu_idx = q * num_heads * head_dim + h * head_dim + d;
                let gpu_idx = q * num_heads * head_dim + h * head_dim + d;
                let abs_err = (cpu_output[cpu_idx] - gpu_output[gpu_idx]).abs();
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                }
            }
        }
    }

    println!("\nMax absolute error: {:.6e}", max_abs_err);

    free_device_memory(q_ptr)?;
    free_device_memory(k_ptr)?;
    free_device_memory(v_ptr)?;
    free_device_memory(s_ptr)?;
    
    Ok(())
}
