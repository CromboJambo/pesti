//! Numerical conformance test: Tiled attention kernel vs llama.cpp reference

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

#[test]
fn test_tiled_attention_vs_llama_cpp() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping tiled attention conformance test");
        return;
    }

    println!("=== Tiled Attention vs llama.cpp Conformance Test ===");
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

    // Compute llama.cpp reference (CPU) - compute per-head scores
    let mut llama_scores = vec![0.0f32; seq_q * num_heads * seq_k];

    // Apply RoPE to Q and K
    let q_h_f32: Vec<f32> = q_h.iter().map(|&x| x.to_f32()).collect();
    let k_h_f32: Vec<f32> = k_h.iter().map(|&x| x.to_f32()).collect();

    let mut q_rope = q_h_f32.clone();
    let mut k_rope = k_h_f32.clone();

    // Apply RoPE to each position (llama.cpp style)
    for pos in 0..seq_q {
        let q_start = pos * num_heads * head_dim;
        apply_rope_cpu(
            &mut q_rope[q_start..q_start + num_heads * head_dim],
            head_dim,
            pos,
            rope_base,
        );
    }

    for pos in 0..seq_k {
        let k_start = pos * num_heads * head_dim;
        apply_rope_cpu(
            &mut k_rope[k_start..k_start + num_heads * head_dim],
            head_dim,
            pos,
            rope_base,
        );
    }

    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            for head in 0..num_heads {
                let mut score = 0.0f32;
                // Dot product across dimensions for this specific head
                let q_offset = q_pos * num_heads * head_dim + head * head_dim;
                let k_offset = k_pos * num_heads * head_dim + head * head_dim;

                for d in 0..head_dim {
                    score += q_rope[q_offset + d] * k_rope[k_offset + d];
                }
                // Scale by 1/sqrt(head_dim)
                score /= (head_dim as f32).sqrt();
                llama_scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = score;
            }
        }
    }

    // Apply causal mask BEFORE softmax (llama.cpp style) - per query, across all k positions
    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            if q_pos >= k_pos {
                llama_scores[q_pos * num_heads * seq_k + k_pos] = -1e9; // head=0
            }
        }
    }

    let mut llama_probs = vec![0.0f32; seq_q * num_heads * seq_k];
    for qh in 0..seq_q * num_heads {
        let start = qh * seq_k;
        let end = start + seq_k;

        // Numerically stable softmax (max-subtract trick)
        let max_val = llama_scores[start..end]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = llama_scores[start..end]
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        let sum: f32 = exps.iter().sum();
        for i in 0..seq_k {
            llama_probs[start + i] = exps[i] / sum;
        }
    }

    println!(
        "llama.cpp softmax sum (first query): {:.6}",
        llama_probs[0..seq_k].iter().sum::<f32>()
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

    // Build tiled kernel - load PTX from file using build_from_ptx_file helper
    let stream = cuda_rt.new_stream().unwrap();

    let ptx_path = std::path::PathBuf::from(
        "/home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/attention_rope_softmax_tiled.ptx",
    );

    // Note: We need to manually launch both kernels since build_from_ptx_file only loads one
    // For now, let's just run the attention kernel and check if softmax is applied
    let tiled_kernel = pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder::new(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    )
    .build_from_ptx_file(ptx_path, "_Z28fused_attention_kernel_tiledfPK6__halfS1_S1_Pfiiiifi")
    .unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel - note: tiled variant has same signature as conformant
    unsafe {
        tiled_kernel
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
                seq_k, // max_pos = seq_k for causal mask
            )
            .unwrap();
    }

    cuda_rt.synchronize().unwrap();

    println!("✅ GPU kernel launched successfully");

    // Debug: print raw attention scores (before softmax) for first query, first head
    let mut gpu_raw_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_raw_scores.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        ).unwrap();
    }

    println!();
    println!("Raw attention scores (q_pos=0, head=0):");
    for k_pos in 0..std::cmp::min(5, seq_k) {
        let gpu_idx = 0 * num_heads * seq_k + k_pos;
        println!("  k_pos {}: GPU score = {:.6}", k_pos, gpu_raw_scores[gpu_idx]);
    }

    // Copy results back to host (after softmax)
    let mut gpu_probs = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_probs.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        ).unwrap();
    }

    // Debug: check if softmax is being applied
    let gpu_softmax_sum: f32 = gpu_probs[0..seq_k].iter().sum();
    println!();
    println!("GPU softmax sum (first query): {:.6}", gpu_softmax_sum);

    // Check if causal mask is working - k_pos=0 should be 0.0 for q_pos=0
    println!();
    println!("Causal mask check (q_pos=0, k_pos=0):");
    println!("  Expected: 0.0 (masked)");
    println!("  GPU value: {:.6}", gpu_probs[0]);

    // Compare outputs - compare first head only (llama.cpp style single-head test)
    let mut max_abs_err = 0.0f32;

    for q_pos in 0..seq_q {
        for k_pos in 0..seq_k {
            // GPU output is [seq_q, num_heads, seq_k], compare first head (head=0)
            let gpu_idx = q_pos * num_heads * seq_k + k_pos;
            // llama_probs has shape [seq_q * num_heads * seq_k] now (multi-head)
            let llama_val = llama_probs[q_pos * num_heads * seq_k + k_pos]; // head=0
            let gpu_val = gpu_probs[gpu_idx];

            let abs_err = (llama_val - gpu_val).abs();
            max_abs_err = max_abs_err.max(abs_err);

            if abs_err > 1e-5 {
                println!(
                    "Large error at q_pos={}, k_pos={}: llama={:.6}, GPU={:.6}, abs_err={:.6e}",
                    q_pos, k_pos, llama_val, gpu_val, abs_err
                );
            }
        }
    }

    // Debug: print softmax sums per query (per head)
    println!();
    println!("Softmax sums per query, per head (should all be ~1.0):");
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let sum: f32 = gpu_probs[start..start + seq_k].iter().sum();
            println!("  Query {}, Head {}: GPU sum = {:.6}", q_pos, head, sum);
        }
    }

    println!();
    println!("Results:");
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!();

    if max_abs_err < 1e-5 {
        println!("✅ Numerical conformance PASSED vs llama.cpp (abs error < 1e-5)");
    } else if max_abs_err < 1e-3 {
        println!("⚠️ Moderate discrepancy detected (abs error < 1e-3)");
    } else {
        println!("❌ Large numerical discrepancy detected (abs error >= 1e-3)");
    }

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
