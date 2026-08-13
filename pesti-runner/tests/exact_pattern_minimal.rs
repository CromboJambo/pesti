//! Minimal exact_pattern test - does NOT write Q,K,V data (like the simple two-buffer test)

use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_exact_pattern_no_data_write() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    println!("=== Exact Pattern Kernel - No Data Write Test ===");
    
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    
    // Allocate ONE buffer containing both scores AND output
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
    let total_size = score_buffer_size + output_buffer_bytes;
    
    let combined_ptr = pesti_runner::cuda_runtime::allocate_device_memory(total_size).unwrap();
    
    println!("✅ Allocated single buffer: {} bytes", total_size);
    
    // Load exact pattern kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();
    
    // Check mangled name
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    println!("✅ Loaded exact pattern kernel");
    
    // Parameters: q_ptr, k_ptr, v_ptr, s_ptr (scores), out_ptr, scale, seq_q, seq_k, num_heads, head_dim
    let mut scale_v: f32 = 1.0 / (head_dim as f32).sqrt();
    let mut q_ptr_v: u64 = combined_ptr as u64; // Q,K,V all point to same buffer for this test
    let mut k_ptr_v: u64 = combined_ptr as u64;
    let mut v_ptr_v: u64 = combined_ptr as u64;
    let mut s_ptr_v: u64 = combined_ptr as u64; // scores start at offset 0
    let mut out_ptr_v: u64 = (combined_ptr as u64) + score_buffer_size as u64; // output starts after scores
    
    let mut seq_q_v: u32 = seq_q as u32;
    let mut seq_k_v: u32 = seq_k as u32;
    let mut num_heads_v: u32 = num_heads as u32;
    let mut head_dim_v: u32 = head_dim as u32;
    
    let stream = cuda_rt.new_stream().unwrap();
    // Exact pattern uses different grid dimensions!
    let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
    let block = (head_dim as u32, 1u32, 1u32);
    
    println!("🚀 Launching with grid={:?}, block={:?}", grid, block);
    
    let mut params = [std::ptr::null_mut(); 10];
    params[0] = &mut q_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[1] = &mut k_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[2] = &mut v_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[3] = &mut s_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[4] = &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void;
    params[5] = &mut scale_v as *mut f32 as *mut std::ffi::c_void;
    params[6] = &mut seq_q_v as *mut u32 as *mut std::ffi::c_void;
    params[7] = &mut seq_k_v as *mut u32 as *mut std::ffi::c_void;
    params[8] = &mut num_heads_v as *mut u32 as *mut std::ffi::c_void;
    params[9] = &mut head_dim_v as *mut u32 as *mut std::ffi::c_void;
    
    // NOTE: We do NOT write Q,K,V data to device - just like the simple two-buffer test
    
    unsafe {
        pesti_runner::cuda_shim::launch_kernel(
            function.cu_function(),
            grid,
            block,
            0, // No shared memory for this kernel
            stream.cu_stream(),
            &mut params,
        )
        .unwrap();
    }
    
    println!("✅ Kernel launched");
    
    cuda_rt.synchronize().unwrap();
    println!("✅ Exact pattern execution completed (no crash)!");
    
    // Cleanup
    pesti_runner::cuda_runtime::free_device_memory(combined_ptr).unwrap();
    
    println!("\n=== Test PASSED (but reads uninitialized memory) ===");
}
