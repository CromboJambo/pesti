//! Minimal test to isolate CUDA_ERROR_ILLEGAL_ADDRESS in simple kernel
//!
//! Uses the scores-only fused attention kernel
//! (`fused_attention_simple_kernel`), which computes
//! `scores[q, head, k] = dot(q, k) * scale` with a causal mask
//! (`k_pos > q_pos` is masked to `-FLT_MAX`).
//!
//! With Q, K, V all ones and `scale = 1/sqrt(head_dim)`, the expected
//! scores are fully deterministic, so this test both exercises the kernel
//! on-device and verifies the numerical result.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

#[cfg(feature = "cuda")]
#[test]
fn test_simple_kernel_minimal() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Minimal Simple Kernel Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    // Very small configuration to isolate the issue
    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4; // Small dimension

    println!(
        "Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );

    // Create simple Q, K, V (all ones)
    let q_h: Vec<f16> = vec![f16::from_f32(1.0); seq_q * num_heads * head_dim];
    let k_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];
    let v_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];

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

    // Copy to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size)
            .unwrap();
    }

    // Load PTX
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_Pfiiiif";
    let function = module.load_function(mangled_name).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch with the scores-only kernel's grid: (seq_q, seq_k, num_heads)
    unsafe {
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut scores_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;
        let mut scale_v: f32 = scale;

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

        // Grid: (seq_q, seq_k, num_heads); Block must be (1,1,1) — the kernel has
        // no cross-thread reduction, so one thread accumulates all head_dim terms.
        let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
        let block = (1u32, 1u32, 1u32);

        println!("Launching with grid={:?}, block={:?}", grid, block);

        launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    println!("✅ Kernel launched");

    // Synchronize to catch errors
    cuda_rt.synchronize().unwrap();

    println!("✅ GPU execution completed successfully!");

    // Copy results back (f32 scores)
    let mut gpu_scores = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_scores.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }

    println!("✅ Results copied back: {:?}", gpu_scores);

    // Verify: dot of head_dim=4 ones = 4.0, scaled by 1/sqrt(4) = 2.0.
    // Causal mask: score[q=0, k=1] is masked to -FLT_MAX.
    let expected_unmasked = head_dim as f32 * scale; // 4.0 * 0.5 = 2.0
    let score_k0 = gpu_scores[0]; // q=0, head=0, k=0
    let score_k1 = gpu_scores[1]; // q=0, head=0, k=1 (masked)
    assert!(
        (score_k0 - expected_unmasked).abs() < 1e-3,
        "score[k=0] expected ~{}, got {}",
        expected_unmasked,
        score_k0
    );
    assert!(
        score_k1 < 0.0,
        "score[k=1] should be causally masked (negative), got {}",
        score_k1
    );
    println!(
        "✅ score[k=0]={:.3} (expected ~{}), score[k=1]={:.3} (masked, expected negative)",
        score_k0, expected_unmasked, score_k1
    );

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
