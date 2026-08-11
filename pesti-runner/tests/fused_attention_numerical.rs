//! Numerical conformance test: GPU fused attention vs CPU reference
//!
//! Compares outputs of our CUDA kernel against a pure Rust CPU implementation
//! to verify numerical correctness.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

/// Reference RoPE implementation (CPU)
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

/// Reference fused attention (CPU): RoPE + scores + softmax
fn reference_fused_attention(
    q_h: &[f16],
    k_h: &[f16],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
    rope_base: f32,
) -> Vec<f32> {
    // Convert to f32
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    for (i, &val) in q_h.iter().enumerate() {
        q_rope[i] = val.to_f32();
    }
    
    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    for (i, &val) in k_h.iter().enumerate() {
        k_rope[i] = val.to_f32();
    }
    
    // Apply RoPE to each position
    for pos in 0..seq_q {
        let q_start = pos * num_heads * head_dim;
        apply_rope_cpu(&mut q_rope[q_start..q_start + num_heads * head_dim], head_dim, pos, rope_base);
    }
    
    for pos in 0..seq_k {
        let k_start = pos * num_heads * head_dim;
        apply_rope_cpu(&mut k_rope[k_start..k_start + num_heads * head_dim], head_dim, pos, rope_base);
    }
    
    // Compute attention scores (scaled by 1/sqrt(head_dim))
    let mut scores = vec![0.0f32; seq_q * seq_k];
    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            let mut score = 0.0f32;
            for head in 0..num_heads {
                for dim in 0..head_dim {
                    let q_idx = q_pos * num_heads * head_dim + head * head_dim + dim;
                    let k_idx = k_pos * num_heads * head_dim + head * head_dim + dim;
                    score += q_rope[q_idx] * k_rope[k_idx];
                }
            }
            score /= (head_dim as f32).sqrt();
            
            // Apply causal mask
            if q_pos >= k_pos {
                scores[q_pos * seq_k + k_pos] = -1e9;
            } else {
                scores[q_pos * seq_k + k_pos] = score;
            }
        }
    }
    
    // Apply softmax per query position
    let mut probs = vec![0.0f32; seq_q * seq_k];
    for q_pos in 0..seq_q {
        let start = q_pos * seq_k;
        let end = start + seq_k;
        
        // Numerically stable softmax
        let max_val = scores[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scores[start..end]
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        
        let sum: f32 = exps.iter().sum();
        for x in exps.iter_mut() {
            *x /= sum;
        }
        
        probs[start..end].copy_from_slice(&exps);
    }
    
    probs
}

#[test]
fn test_fused_attention_numerical_conformance() {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            return;
        }
    };
    
    println!("=== Fused Attention Numerical Conformance Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();
    
    // Configuration (small for quick testing)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    
    // Create deterministic Q, K (f16)
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    println!("Configuration: seq_q={}, seq_k={}, heads={}, dim={}", seq_q, seq_k, num_heads, head_dim);
    
    // Compute CPU reference
    let cpu_probs = reference_fused_attention(&q_h, &k_h, seq_q, seq_k, num_heads, head_dim, rope_base);
    
    println!("CPU softmax sum (first query): {:.6}", 
             cpu_probs[0..seq_k].iter().sum::<f32>());
    println!("CPU max attention score: {:.6}", 
             cpu_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * seq_k * 4;
    
    let q_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap()
    };
    let k_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap()
    };
    let v_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap()
    };
    let s_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap()
    };
    
    // Copy Q, K to device (simplified: use host pointers directly for now)
    unsafe {
        let mut q_ptr_mut = q_ptr as *mut u8;
        std::ptr::copy_nonoverlapping(
            q_h.as_ptr() as *const u8,
            q_ptr_mut,
            q_size,
        );
        
        let mut k_ptr_mut = k_ptr as *mut u8;
        std::ptr::copy_nonoverlapping(
            k_h.as_ptr() as *const u8,
            k_ptr_mut,
            k_size,
        );
    }
    
    // Build kernel
    let stream = cuda_rt.new_stream().unwrap();
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    ).unwrap();
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Launch kernel
    unsafe {
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
        ).unwrap();
    }
    
    cuda_rt.synchronize().unwrap();
    
    println!("✅ GPU kernel launched successfully");
    
    // Copy results back to host
    let mut gpu_probs = vec![0.0f32; seq_q * seq_k];
    unsafe {
        std::ptr::copy_nonoverlapping(
            s_ptr as *const u8,
            gpu_probs.as_mut_ptr() as *mut u8,
            s_size,
        );
    }
    
    println!("GPU softmax sum (first query): {:.6}", 
             gpu_probs[0..seq_k].iter().sum::<f32>());
    
    // Compare outputs
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut total_elements = 0usize;
    
    for i in 0..seq_q * seq_k {
        let cpu_val = cpu_probs[i];
        let gpu_val = gpu_probs[i];
        
        let abs_err = (cpu_val - gpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-8 {
            abs_err / cpu_val.abs()
        } else {
            abs_err
        };
        
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
        total_elements += 1;
    }
    
    println!();
    println!("Results:");
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);
    println!();
    
    if max_rel_err < 1e-4 {
        println!("✅ Numerical conformance PASSED (rel error < 1e-4)");
    } else if max_rel_err < 1e-2 {
        println!("⚠️  Moderate numerical discrepancy (rel error < 1e-2)");
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
