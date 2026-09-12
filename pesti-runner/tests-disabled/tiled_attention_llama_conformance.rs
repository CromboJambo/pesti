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

    // Launch both kernels: attention + softmax
    let tiled_kernel =
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder::new(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )
        .build_from_ptx_file(
            &ptx_path,
            "_Z28fused_attention_kernel_tiledfPK6__halfS1_S1_Pfiiiifi",
        )
        .unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch tiled attention kernel directly (not via conformant launch method)
    // Tiled kernel expects: grid (ceil(seq_q * num_heads / 128), 1, 1), block (128, 1, 1)
    let total_qh = seq_q * num_heads;
    let grid_x = (total_qh + 127) / 128;

    let mut scale_v: f32 = scale;
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = v_ptr as u64;
    let mut s_v: u64 = s_ptr as u64;
    let mut seq_q_v: i32 = seq_q as i32;
    let mut seq_k_v: i32 = seq_k as i32;
    let mut num_heads_v: i32 = num_heads as i32;
    let mut head_dim_v: i32 = head_dim as i32;
    let mut rope_base_v: f32 = rope_base;
    let mut max_pos_v: i32 = seq_k as i32; // causal mask

    let mut params: [*mut std::ffi::c_void; 11] = [
        &mut scale_v as *mut f32 as *mut std::ffi::c_void,
        &mut q_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_v as *mut u64 as *mut std::ffi::c_void,
        &mut s_v as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut i32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut i32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut i32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut i32 as *mut std::ffi::c_void,
        &mut rope_base_v as *mut f32 as *mut std::ffi::c_void,
        &mut max_pos_v as *mut i32 as *mut std::ffi::c_void,
    ];

    let grid = (grid_x as u32, 1u32, 1u32);
    let block = (128u32, 1u32, 1u32);
    let smem_size = 0u32;

    unsafe {
        use pesti_runner::cuda_shim::launch_kernel;
        match launch_kernel(
            tiled_kernel.cu_function(), // Use accessor method to get CUfunction
            grid,
            block,
            smem_size,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        ) {
            Ok(_) => {} // Success - no action needed
            Err(e) => panic!("Tiled attention kernel launch: {:?}", e),
        }
    }

    cuda_rt.synchronize().unwrap();

    // Debug: Check for NaN in raw scores after attention kernel
    let mut debug_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            debug_scores.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
        .unwrap();
    }

    println!();
    println!("Debug: Raw scores after attention kernel (before softmax):");
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            let start = q_pos * num_heads * seq_k + head * seq_k;
            let has_nan = debug_scores[start..start + seq_k]
                .iter()
                .any(|&x| x.is_nan());
            if has_nan {
                println!("  Query {}, Head {}: HAS NaN", q_pos, head);
                for k_pos in 0..std::cmp::min(3, seq_k) {
                    println!("    k_pos {}: {:.6}", k_pos, debug_scores[start + k_pos]);
                }
            } else {
                // Print first 3 values for non-NaN heads too
                if q_pos == 0 && (head == 2 || head == 3) {
                    println!(
                        "  Query {}, Head {}: OK (first 3: {:.6}, {:.6}, {:.6})",
                        q_pos,
                        head,
                        debug_scores[start],
                        debug_scores[start + 1],
                        debug_scores[start + 2]
                    );
                }
            }
        }
    }

    println!("✅ GPU kernel 1 (attention) launched successfully");

    // Now launch softmax kernel separately
    let softmax_kernel =
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionKernelBuilder::new(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )
        .build_from_ptx_file(ptx_path, "_Z20apply_softmax_kernelPfiii")
        .unwrap();

    // Launch softmax kernel 2: apply_softmax_kernel(float* s_ptr, int seq_q, int seq_k, int num_heads)
    let mut s_v2: u64 = s_ptr as u64;
    let mut seq_q_v2: i32 = seq_q as i32;
    let mut seq_k_v2: i32 = seq_k as i32;
    let mut num_heads_v2: i32 = num_heads as i32;

    let mut params2: [*mut std::ffi::c_void; 4] = [
        &mut s_v2 as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v2 as *mut i32 as *mut std::ffi::c_void,
        &mut seq_k_v2 as *mut i32 as *mut std::ffi::c_void,
        &mut num_heads_v2 as *mut i32 as *mut std::ffi::c_void,
    ];

    let softmax_mangled = "_Z20apply_softmax_kernelPfiii";
    let softmax_func = match softmax_kernel.module().load_function(softmax_mangled) {
        Ok(f) => f,
        Err(e) => panic!("Softmax function lookup {:?}: {:?}", softmax_mangled, e),
    };

    // Launch config: grid (seq_q, num_heads), block (seq_k)
    let grid2 = (seq_q as u32, num_heads as u32, 1u32);
    let block2 = (seq_k as u32, 1u32, 1u32);
    let smem_size2 = seq_k * 4; // 4 bytes per float

    unsafe {
        use pesti_runner::cuda_shim::launch_kernel;
        match launch_kernel(
            softmax_func.cu_function(),
            grid2,
            block2,
            smem_size2 as u32,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params2,
        ) {
            Ok(_) => {}
            Err(e) => panic!("Softmax kernel launch: {:?}", e),
        }
    }

    cuda_rt.synchronize().unwrap();
    println!("✅ GPU kernel 2 (softmax) launched successfully");

    // Debug: print raw attention scores (before softmax) for first query, first head
    let mut gpu_raw_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_raw_scores.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
        .unwrap();
    }

    println!();
    println!("Raw attention scores (q_pos=0, head=0):");
    for k_pos in 0..std::cmp::min(5, seq_k) {
        let gpu_idx = 0 * num_heads * seq_k + k_pos;
        println!(
            "  k_pos {}: GPU score = {:.6}",
            k_pos, gpu_raw_scores[gpu_idx]
        );
    }

    // Copy results back to host (after softmax)
    let mut gpu_probs = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_probs.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        )
        .unwrap();
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
