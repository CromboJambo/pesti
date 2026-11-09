//! Minimal kernel launch test - verify basic cuLaunchKernel works
//!
//! Uses the scores-only fused attention kernel
//! (`fused_attention_simple_kernel`), which computes
//! `scores[q, head, k] = dot(q, k) * scale` with a causal mask.
//! This test only checks that the kernel loads and launches cleanly.

use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_minimal_kernel_launch() {
    let cuda_rt = CudaRuntime::new(0).unwrap();

    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Minimal Kernel Launch Test ===");

    // Scores-only kernel configuration
    let seq_q = 2;
    let seq_k = 1;
    let num_heads = 4;
    let head_dim = 16;

    // Output is f32 scores of shape [seq_q, num_heads, seq_k]
    let out_size = seq_q * num_heads * seq_k * 4; // f32

    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };

    println!("✅ Allocated {} bytes for output", out_size);

    // Load the scores-only attention kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();

    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_Pfiiiif";
    let function = module.load_function(mangled_name).unwrap();

    println!("✅ Loaded kernel: {}", mangled_name);

    // Parameters: q, k, v, scores, seq_q, seq_k, num_heads, head_dim, scale
    let mut q_ptr_v: u64 = out_ptr as u64; // Reuse same pointer for all inputs (simplified)
    let mut k_ptr_v: u64 = out_ptr as u64;
    let mut v_ptr_v: u64 = out_ptr as u64;
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

    println!(
        "🚀 Launching kernel with grid={:?}, block={:?}",
        grid, block
    );

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

    println!("✅ Kernel launched successfully!");

    // Synchronize and check for errors
    cuda_rt.synchronize().unwrap();
    println!("✅ Kernel completed (no errors)");

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }

    println!("\n=== Minimal Launch Test PASSED ===");
}
