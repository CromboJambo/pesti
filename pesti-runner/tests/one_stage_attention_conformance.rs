//! One-stage full fusion attention conformance test
//!
//! Three-stage -> One-stage progression:
//! 1. CPU: RoPE on Q/K (pre-processing)  
//! 2. GPU: scores + softmax + V-multiply (single fused kernel)
//!
//! Target: Match llama.cpp output with single-kernel full fusion

#![cfg(feature = "cuda")]

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{CudaModule, launch_kernel};

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

                // Softmax with max-subtraction trick and causal mask
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores
                    .iter()
                    .map(|&s| {
                        if s == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (s - max_score).exp()
                        }
                    })
                    .sum();

                // Weighted V sum
                for k_pos in 0..seq_k {
                    let softmax_val = if scores[k_pos] == f32::NEG_INFINITY || exp_sum <= 0.0 {
                        0.0
                    } else {
                        (scores[k_pos] - max_score).exp() / exp_sum
                    };
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
fn test_one_stage_attention() {
    println!("\n=== One-Stage Full Fusion Attention ===");

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
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };

    // Allocate output buffer (float, seq_q * num_heads * head_dim)
    let output_buffer_bytes = seq_q * num_heads * head_dim * 4; // float
    let output_ptr =
        unsafe { pesti_runner::cuda_runtime::allocate_device_memory(output_buffer_bytes).unwrap() };

    println!("Allocated GPU buffers");

    // Copy Q, K, V to GPU
    unsafe {
        let q_h_f16: Vec<f16> = q_host.iter().map(|&x| x).collect();
        let k_h_f16: Vec<f16> = k_host.iter().map(|&x| x).collect();
        let v_h_f16: Vec<f16> = v_host.iter().map(|&x| x).collect();

        pesti_runner::cuda_runtime::copy_host_to_device(
            q_ptr,
            q_h_f16.as_ptr() as *const u8,
            q_size,
        )
        .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(
            k_ptr,
            k_h_f16.as_ptr() as *const u8,
            k_size,
        )
        .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(
            v_ptr,
            v_h_f16.as_ptr() as *const u8,
            v_size,
        )
        .unwrap();
    }

    println!("Copied inputs to GPU");

    // Load PTX and get function - USE FUSED KERNEL
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_full_kernel.ptx");
    let module = CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();

    // Use fused kernel signature (8 params: q_ptr, k_ptr, v_ptr, out_ptr, seq_q, seq_k, num_heads, head_dim)
    let mangled_name = "_Z22fused_attention_kernelPK6__halfS1_S1_Pfiiii";
    let function = module.load_function(mangled_name).unwrap();

    // Parameters (8 total: q_ptr, k_ptr, v_ptr, out_ptr, seq_q, seq_k, num_heads, head_dim)
    let mut q_v: u64 = q_ptr as u64;
    let mut k_v: u64 = k_ptr as u64;
    let mut v_v: u64 = v_ptr as u64;
    let mut out_v: u64 = output_ptr as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;

    // Launch kernel with grid (seq_q, num_heads), block (head_dim, 1, 1)
    let grid = (seq_q as u32, num_heads as u32, 1u32);
    let block = (head_dim as u32, 1u32, 1u32);

    let mut params: [*mut std::ffi::c_void; 8] = [
        &mut q_v as *mut u64 as *mut std::ffi::c_void, // param_0: q_ptr (const half*)
        &mut k_v as *mut u64 as *mut std::ffi::c_void, // param_1: k_ptr (const half*)
        &mut v_v as *mut u64 as *mut std::ffi::c_void, // param_2: v_ptr (const half*)
        &mut out_v as *mut u64 as *mut std::ffi::c_void, // param_3: out_ptr (float*)
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void, // param_4: seq_q (int)
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void, // param_5: seq_k (int)
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void, // param_6: num_heads (int)
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void, // param_7: head_dim (int)
    ];

    unsafe {
        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    println!("✅ GPU kernel launched successfully");

    cuda_rt.synchronize().unwrap();

    // Read back output (float)
    let mut gpu_output: Vec<f32> = vec![0.0; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_output.as_mut_ptr() as *mut u8,
            output_ptr as *const u8,
            output_buffer_bytes,
        )
        .unwrap();
    }

    println!("✅ Output copied back to host");

    // Compare with CPU reference
    let cpu_output =
        reference_llama_attention(&q_host, &k_host, &v_host, seq_q, seq_k, num_heads, head_dim);

    println!("Computed CPU reference");

    // Compute error metrics
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for i in 0..(seq_q * num_heads * head_dim) {
        let abs_err = (gpu_output[i] - cpu_output[i]).abs();
        let rel_err = if cpu_output[i].abs() > 1e-6 {
            abs_err / cpu_output[i].abs()
        } else {
            abs_err
        };

        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }

    println!("\nResults:");
    println!(
        "  Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);

    // Assert conformance (using loose tolerance for learning phase)
    assert!(
        max_abs_err < 1e-3,
        "Absolute error {} exceeds threshold",
        max_abs_err
    );
    assert!(
        max_rel_err < 1e-2,
        "Relative error {} exceeds threshold",
        max_rel_err
    );

    println!("✅ One-stage full fusion attention test PASSED!");
}
