//! Minimal kernel launch test with proper memory allocation
//!
//! Uses the scores-only fused attention kernel
//! (`fused_attention_simple_kernel`), which computes
//! `scores[q, head, k] = dot(q, k) * scale` with a causal mask.
//! This test allocates separate buffers for Q, K, V and the f32 scores
//! output, then verifies the kernel launches cleanly.

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_minimal_kernel_separate_memory() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Minimal Kernel Test (Separate Memory) ===");

    let seq_q = 1;
    let seq_k = 1;
    let num_heads = 1;
    let head_dim = 16;

    // Allocate separate memory for each buffer.
    // Q/K/V are f16 [seq, num_heads, head_dim]; scores output is f32 [seq_q, num_heads, seq_k].
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * seq_k * 4;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    // Fill Q, K with a known value so the dot product is deterministic.
    let q_h: Vec<f16> = vec![f16::from_f32(1.0); q_size / 2];
    let k_h: Vec<f16> = vec![f16::from_f32(1.0); k_size / 2];
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size)
            .unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size)
            .unwrap();
    }

    println!("✅ Allocated separate memory for Q, K, V, Out");

    // Load the scores-only attention kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_Pfiiiif";
    let function = module.load_function(mangled_name).unwrap();

    println!("✅ Loaded kernel");

    // Parameters: q, k, v, scores, seq_q, seq_k, num_heads, head_dim, scale
    let mut q_ptr_v: u64 = q_ptr as u64;
    let mut k_ptr_v: u64 = k_ptr as u64;
    let mut v_ptr_v: u64 = v_ptr as u64;
    let mut scores_ptr_v: u64 = out_ptr as u64;
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    let mut scale_v: f32 = 1.0;

    let mut params: [*mut std::ffi::c_void; 9] = [
        &mut q_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut k_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut v_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut scores_ptr_v as *mut u64 as *mut std::ffi::c_void,
        &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
        &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
        &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
        &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        &mut scale_v as *mut f32 as *mut std::ffi::c_void,
    ];

    // Grid: (seq_q, seq_k, num_heads). Block must be (1,1,1): the kernel has no
    // cross-thread reduction, so a single thread accumulates all head_dim terms.
    let stream = cuda_rt.new_stream().unwrap();
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (1u32, 1u32, 1u32);

    println!("🚀 Launching with grid={:?}, block={:?}", grid, block);

    unsafe {
        pesti_runner::cuda_shim::launch_kernel(
            function.cu_function(),
            grid,
            block,
            0,
            pesti_runner::cuda_shim::cu_stream(&stream),
            &mut params,
        )
        .unwrap();
    }

    println!("✅ Kernel launched");

    // Synchronize
    cuda_rt.synchronize().unwrap();
    println!("✅ Kernel completed successfully!");

    // Copy the single score back and sanity-check it:
    // dot(q,k) over head_dim=16 of ones = 16.0, scale=1.0, no causal mask (k=0,q=0).
    let mut score = 0.0f32;
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            &mut score as *mut f32 as *mut u8,
            out_ptr as *const u8,
            4,
        )
        .unwrap();
    }
    assert!(
        (score - 16.0).abs() < 1e-3,
        "expected score ~16.0, got {}",
        score
    );
    println!("✅ Score = {:.3} (expected ~16.0)", score);

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }

    println!("\n=== Minimal Test PASSED ===");
}
