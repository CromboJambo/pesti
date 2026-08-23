//! Full numerical conformance test for the scores-only fused attention kernel
//! (`fused_attention_simple_kernel`).
//!
//! The kernel computes, for every (q_pos, k_pos, head):
//!     score = dot(q[q_pos, head], k[k_pos, head]) * scale
//! and applies a causal mask: `score = -FLT_MAX` when `k_pos > q_pos`.
//! Output is f32, laid out as `scores[q_pos, head, k_pos]`.
//!
//! This test launches the real PTX kernel on-device and compares every
//! output element against an f32 CPU reference that mirrors the kernel's
//! exact arithmetic (f16 inputs, f32 accumulation, f32 scale).

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

#[cfg(feature = "cuda")]
#[test]
fn test_single_kernel_numerical_conformance() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping numerical conformance test");
        return;
    }

    println!("=== Single-Kernel Numerical Conformance Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();

    // Configuration (small for quick testing)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;

    // Create deterministic Q, K, V (f16) matching llama.cpp test patterns
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );

    // Allocate device memory.
    // Q/K/V are f16; the scores output is f32 [seq_q, num_heads, seq_k].
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * seq_k * 4;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    // Copy Q, K, V to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)
            .unwrap();
    }

    // Load PTX and get function
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    // Get the function - use the exact mangled name from nvcc
    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_Pfiiiif";
    let function = module.load_function(mangled_name).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel using launch_kernel helper (like gemm.rs does)
    unsafe {
        // Parameters: q, k, v, scores, seq_q, seq_k, num_heads, head_dim, scale
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut scores_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;
        let mut scale_v: f32 = scale;

        // cuLaunchKernel wants pointers to host values, not device pointers directly
        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut scores_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
        ];

        // Grid: (seq_q, seq_k, num_heads). Block must be (1,1,1): the kernel has no
        // cross-thread reduction, so a single thread accumulates all head_dim terms.
        let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
        let block = (1u32, 1u32, 1u32);

        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0, // shared_mem_bytes (none for this kernel)
            cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    cuda_rt.synchronize().unwrap();

    println!("✅ Single-kernel launched successfully");

    // Copy results back to host (f32 scores)
    let mut gpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }

    println!("✅ Results copied back to host");

    // Compute the CPU reference, mirroring the kernel exactly:
    //   score = (sum_d f32(q16[d]) * f32(k16[d])) * scale
    //   causal mask: k_pos > q_pos => -FLT_MAX
    let mut cpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for k_pos in 0..seq_k {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q_h[q_pos * num_heads * head_dim + head * head_dim + d].to_f32();
                    let k_val = k_h[k_pos * num_heads * head_dim + head * head_dim + d].to_f32();
                    dot += q_val * k_val;
                }
                let score = dot * scale;
                let score = if k_pos > q_pos {
                    f32::MIN // -FLT_MAX
                } else {
                    score
                };
                cpu_scores[q_pos * num_heads * seq_k + head * seq_k + k_pos] = score;
            }
        }
    }

    // Compare outputs
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let mut masked_ok = 0usize;
    let mut unmasked = 0usize;

    for i in 0..gpu_scores.len() {
        let ref_val = cpu_scores[i];
        let gpu_val = gpu_scores[i];

        // Masked entries must be exactly -FLT_MAX on both sides.
        if ref_val == f32::MIN {
            if gpu_val == f32::MIN {
                masked_ok += 1;
            } else {
                max_abs_err = f32::MAX;
            }
            continue;
        }

        unmasked += 1;
        let abs_err = (ref_val - gpu_val).abs();
        let rel_err = if ref_val.abs() > 1e-8 {
            abs_err / ref_val.abs()
        } else {
            abs_err
        };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }

    println!();
    println!("Results:");
    println!("  Unmasked scores compared: {}", unmasked);
    println!("  Masked scores verified:   {} (all -FLT_MAX)", masked_ok);
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);
    println!();

    // ASSERTION: Conformance requires <1e-4 relative error on unmasked scores
    // and exact -FLT_MAX on masked scores.
    assert!(
        max_abs_err != f32::MAX,
        "A causally-masked score was not -FLT_MAX on device"
    );
    assert!(
        max_rel_err < 1e-4,
        "Numerical conformance FAILED: max_rel_error={:.6e} >= 1e-4 target",
        max_rel_err
    );

    println!("✅ Numerical conformance PASSED (rel error < 1e-4, causal mask exact)");

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
