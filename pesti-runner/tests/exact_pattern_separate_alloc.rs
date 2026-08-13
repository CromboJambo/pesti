//! Exact pattern test with SEPARATE Q,K,V allocations

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_exact_pattern_separate_allocations() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    println!("=== Exact Pattern Kernel - Separate Q,K,V Allocations ===");
    
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    
    // Allocate SEPARATE buffers for Q, K, V, scores, and output
    let q_size = seq_q * num_heads * head_dim * 2; // half
    let k_size = seq_k * num_heads * head_dim * 2; // half
    let v_size = seq_k * num_heads * head_dim * 2; // half
    let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
    let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
    
    let q_ptr = pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap();
    let k_ptr = pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap();
    let v_ptr = pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap();
    let combined_ptr = 
        pesti_runner::cuda_runtime::allocate_device_memory(
            score_buffer_size + output_buffer_bytes
        ).unwrap();
    
    println!("✅ Allocated separate buffers:");
    println!("   Q: {} bytes @ {:?} (cast to u64)", q_size, q_ptr);
    println!("   K: {} bytes @ {:?} (cast to u64)", k_size, k_ptr);
    println!("   V: {} bytes @ {:?} (cast to u64)", v_size, v_ptr);
    println!("   Scores+Output: {} bytes @ {:?}", score_buffer_size + output_buffer_bytes, combined_ptr);
    
    // Load exact pattern kernel
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module =
        pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), &ptx_src).unwrap();
    
    // Check mangled name
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    println!("✅ Loaded exact pattern kernel");
    
    // Create deterministic Q, K, V data
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();
    
    let v_h: Vec<f16> = k_h.clone();
    
    // Parameters: q_ptr, k_ptr, v_ptr (pointers to separate buffers), s_ptr (scores), out_ptr, scale, seq_q, seq_k, num_heads, head_dim
    let mut scale_v: f32 = 1.0 / (head_dim as f32).sqrt();
    let mut q_ptr_v: u64 = q_ptr as u64; // Separate allocation!
    let mut k_ptr_v: u64 = k_ptr as u64; // Separate allocation!
    let mut v_ptr_v: u64 = v_ptr as u64; // Separate allocation!
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
    
    // Write Q, K, V data to device (async - synchronize after)
    unsafe {
        cudarc::driver::result::memcpy_htod_async(q_ptr as u64, &q_h, stream.cu_stream());
        cudarc::driver::result::memcpy_htod_async(k_ptr as u64, &k_h, stream.cu_stream());
        cudarc::driver::result::memcpy_htod_async(v_ptr as u64, &v_h, stream.cu_stream());
    }
    
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
    println!("✅ Exact pattern execution completed!");
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(combined_ptr).unwrap();
    }
    
    println!("\n=== Test PASSED ===");
}
