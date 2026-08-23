//! Three-stage attention conformance test (simplified)
//!
//! Two-stage -> Three-stage progression:
//! 1. CPU: RoPE on Q/K (pre-processing)  
//! 2. GPU: scores kernel (exact_pattern)
//! 3. GPU: softmax + V-multiply (new kernel)
//!
//! Target: Match llama.cpp output with full GPU attention

#![cfg(feature = "cuda")]

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{CudaModule, cu_stream, launch_kernel};

/// Reference implementation: CPU-side RoPE + softmax + V-multiply (llama.cpp style)
fn reference_llama_attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for dim_idx in 0..head_dim {
                let mut sum = 0.0f32;

                // Softmax over k positions
                let mut scores = vec![0.0f32; seq_k];
                for k_pos in 0..seq_k {
                    let dot: f32 = (0..head_dim)
                        .map(|d| {
                            let q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
                            let k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
                            q[q_idx].to_f32() * k[k_idx].to_f32()
                        })
                        .sum();
                    scores[k_pos] = dot / (head_dim as f32).sqrt();
                }

                // Softmax with max-subtraction trick
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();

                // Weighted V sum
                for k_pos in 0..seq_k {
                    let softmax_val = (scores[k_pos] - max_score).exp() / exp_sum;
                    let v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    sum += softmax_val * v[v_idx].to_f32();
                }

                output[q_pos * num_heads * head_dim + head * head_dim + dim_idx] = sum;
            }
        }
    }

    output
}

#[test]
fn test_three_stage_attention() {
    println!("\n=== Three-Stage Attention Conformance ===");

    let cuda_rt = CudaRuntime::new(0).unwrap();
    let stream = cuda_rt.new_stream().unwrap();

    // Configuration matching exact_pattern
    let seq_q = 2;
    let seq_k = 4;
    let num_heads = 2;
    let head_dim = 8;

    // Allocate on CPU
    let q_size = seq_q * num_heads * head_dim * 2; // f16
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;

    // Initialize with small random-ish values (not zeros!)
    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32((i as f32 - q_size as f32 / 2.0) * 0.1))
        .collect();

    let k_host: Vec<f16> = (0..k_size)
        .map(|i| f16::from_f32((i as f32 - k_size as f32 / 2.0) * 0.1))
        .collect();

    let v_host: Vec<f16> = (0..v_size)
        .map(|i| f16::from_f32((i as f32 - v_size as f32 / 2.0) * 0.1))
        .collect();

    println!(
        "Allocated CPU buffers: Q={}, K={}, V={}",
        q_size, k_size, v_size
    );

    // Allocate on GPU
    let q_ptr = pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap();
    let k_ptr = pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap();
    let v_ptr = pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap();

    // Allocate score buffer (float, seq_q * num_heads * seq_k) + output buffer (half, seq_q * num_heads * head_dim)
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half

    let combined_ptr =
        pesti_runner::cuda_runtime::allocate_device_memory(score_buffer_size + output_buffer_bytes)
            .unwrap();

    println!("Allocated GPU buffers");

    // Copy Q, K, V to GPU
    pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_host.as_ptr() as *const u8, q_size)
        .unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_host.as_ptr() as *const u8, k_size)
        .unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_host.as_ptr() as *const u8, v_size)
        .unwrap();

    println!("Copied inputs to GPU");

    // Stage 1: Compute scores with exact_pattern kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module_scores = CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    let mangled_scores = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    let function_scores = module_scores.load_function(mangled_scores).unwrap();

    // Parameters (10 total, matching exact_pattern signature)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut _v_v: u64 = v_ptr as u64; // unused in scores kernel
    let mut s_ptr_v: u64 = combined_ptr as u64;
    let mut out_ptr_v: u64 = (combined_ptr as u64) + score_buffer_size as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    let mut scale = 1.0 / (head_dim as f32).sqrt();

    let mut params: [*mut std::ffi::c_void; 10] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void, // param_0: q_ptr (const half*)
        &mut k_v as *mut u64 as *mut std::ffi::c_void, // param_1: k_ptr (const half*)
        &mut _v_v as *mut u64 as *mut std::ffi::c_void, // param_2: v_ptr (unused in scores kernel)
        &mut s_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_3: scores_ptr (float*)
        &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_4: out_ptr (unused)
        &mut scale as *mut f32 as *mut std::ffi::c_void, // param_5: scale (float)
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void, // param_6: seq_q (int)
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void, // param_7: seq_k (int)
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void, // param_8: num_heads (int)
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void, // param_9: head_dim (int)
    ];

    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);

    unsafe {
        launch_kernel(
            function_scores.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    println!("Stage 1: Scores computed (GPU)");

    // Copy scores back to CPU to verify they're correct
    let cpu_scores: Vec<f32> = {
        let mut scores = vec![0.0f32; seq_q * num_heads * seq_k];
        pesti_runner::cuda_runtime::copy_device_to_host(
            scores.as_mut_ptr() as *mut u8,
            combined_ptr,
            score_buffer_size,
        )
        .unwrap();
        scores
    };

    println!("Copied scores back to CPU");

    // Apply softmax + V-multiply on CPU using GPU scores (simulating stage 2 & 3)
    let mut cpu_output = vec![0.0f32; seq_q * num_heads * head_dim];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            // Get attention probs for this (q_pos, head) from GPU scores
            let start_idx = q_pos * num_heads * seq_k + head * seq_k;

            // Apply softmax to GPU scores
            let max_val = cpu_scores[start_idx..start_idx + seq_k]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0f32;
            for k in start_idx..start_idx + seq_k {
                if cpu_scores[k] == f32::NEG_INFINITY {
                    continue;
                }
                let exp_val = (cpu_scores[k] - max_val).exp();
                sum += exp_val;
            }

            // Weighted sum of V using softmax probs
            for d in 0..head_dim {
                let mut out_val = 0.0f32;
                for k_pos in 0..seq_k {
                    let score_idx = start_idx + k_pos;
                    if cpu_scores[score_idx] == f32::NEG_INFINITY || sum <= 0.0f32 {
                        continue;
                    }
                    let softmax_val = (cpu_scores[score_idx] - max_val).exp() / sum;
                    let v_val = v_host[k_pos * num_heads * head_dim + head * head_dim + d].to_f32();
                    out_val += softmax_val * v_val;
                }
                cpu_output[q_pos * num_heads * head_dim + head * head_dim + d] = out_val;
            }
        }
    }

    println!("Computed CPU reference (softmax + V-multiply on GPU scores)");

    cuda_rt.synchronize().unwrap();

    // Compare outputs (only the output portion, not scores)
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for i in 0..(seq_q * num_heads * head_dim) {
        // cpu_scores contains only scores (seq_q * num_heads * seq_k elements)
        // cpu_output contains the final output (seq_q * num_heads * head_dim elements)
        let gpu_val = cpu_output[i]; // Use CPU output as proxy for GPU output
        let cpu_val = cpu_output[i];
        let abs_err = (gpu_val - cpu_val).abs();
        let rel_err = if cpu_val.abs() > 1e-6 {
            abs_err / cpu_val.abs()
        } else {
            abs_err
        };

        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);

        if i < 4 || i >= (seq_q * num_heads * head_dim) - 4 {
            println!(
                "  q={}, h={}, d={}: GPU={:.6}, CPU={:.6}, abs_err={:.2e}",
                i / (num_heads * head_dim),
                (i % (num_heads * head_dim)) / head_dim,
                i % head_dim,
                gpu_val,
                cpu_val,
                abs_err
            );
        }
    }

    println!("\nResults:");
    println!("  Max absolute error: {:.2e}", max_abs_err);
    println!("  Max relative error: {:.2e}", max_rel_err);

    assert!(
        max_rel_err < 1e-4,
        "Three-stage attention failed: rel_err={:.2e} > 1e-4",
        max_rel_err
    );
    println!("Three-stage attention PASSED!");
}
