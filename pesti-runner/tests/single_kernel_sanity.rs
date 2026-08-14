//! Quick sanity check for exact-pattern scores-only kernel
//! Tests that the PTX compiles, loads, and launches without errors

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{cu_stream, launch_kernel};

#[test]
fn test_single_kernel_sanity() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping sanity check");
        return;
    }
    
    println!("=== Single-Kernel Sanity Check (Exact Pattern) ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    
    // Small configuration matching exact_pattern
    let seq_q = 2;
    let seq_k = 4;
    let num_heads = 2;
    let head_dim = 8;
    
    // Create simple Q, K, V (all zeros except a few values)
    let q_h: Vec<f16> = vec![f16::from_f32(1.0); seq_q * num_heads * head_dim];
    let k_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];
    let v_h: Vec<f16> = vec![f16::from_f32(1.0); seq_k * num_heads * head_dim];
    
    println!("Configuration: seq_q={}, seq_k={}, heads={}, dim={}", 
             seq_q, seq_k, num_heads, head_dim);
    
    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    
    // Copy to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }
    
    // Load PTX and get function - USE EXACT PATTERN KERNEL (proven working)
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_exact_pattern.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    
    // Use exact pattern mangled name (5 pointers + scale + dims)
    let mangled_name = "_Z36fused_attention_exact_pattern_kernelPK6__halfS1_S1_PfS2_fiiii";
    
    match module.load_function(mangled_name) {
        Ok(function) => {
            println!("✅ Function loaded: {}", mangled_name);
            
            // Allocate combined buffer: scores (float) + output (half)
            let score_buffer_size = seq_q * num_heads * seq_k * 4; // float
            let output_buffer_bytes = seq_q * num_heads * head_dim * 2; // half
            
            let combined_ptr = 
                pesti_runner::cuda_runtime::allocate_device_memory(
                    score_buffer_size + output_buffer_bytes
                ).unwrap();
            
            unsafe {
                let mut q_v: u64 = q_ptr as u64;
                let mut k_v: u64 = k_ptr as u64;
                let mut _v_v: u64 = v_ptr as u64; // const half*
                let mut s_ptr_v: u64 = combined_ptr as u64; // scores start at offset 0
                let mut out_ptr_v: u64 = (combined_ptr as u64) + score_buffer_size as u64; // output after scores
                let mut seq_q_v: u32 = seq_q as u32;
                let mut seq_k_v: u32 = seq_k as u32;
                let mut num_heads_v: u32 = num_heads as u32;
                let mut head_dim_v: u32 = head_dim as u32;
                let scale = 1.0 / (head_dim as f32).sqrt();

                let mut params: [*mut std::ffi::c_void; 10] = [
                    &mut q_v as *mut u64 as *mut std::ffi::c_void, // param_0: q_ptr (const half*)
                    &mut k_v as *mut u64 as *mut std::ffi::c_void, // param_1: k_ptr (const half*)
                    &mut _v_v as *mut u64 as *mut std::ffi::c_void, // param_2: v_ptr (const half* cast to u64)
                    &mut s_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_3: scores_ptr (float*)
                    &mut out_ptr_v as *mut u64 as *mut std::ffi::c_void, // param_4: out_ptr (half*)
                    &mut (scale as f32) as *mut f32 as *mut std::ffi::c_void, // param_5: scale (float)
                    &mut seq_q_v as *mut u32 as *mut std::ffi::c_void, // param_6: seq_q (int)
                    &mut seq_k_v as *mut u32 as *mut std::ffi::c_void, // param_7: seq_k (int)
                    &mut num_heads_v as *mut u32 as *mut std::ffi::c_void, // param_8: num_heads (int)
                    &mut head_dim_v as *mut u32 as *mut std::ffi::c_void, // param_9: head_dim (int)
                ];

                // Grid: (seq_q, seq_k, num_heads), Block: (head_dim, 1, 1)
                let grid = (seq_q as u32, seq_k as u32, num_heads as u32);
                let block = (head_dim as u32, 1u32, 1u32);

                launch_kernel(
                    function.cu_function(),
                    grid,
                    block,
                    0, // shared memory
                    cu_stream(&stream),
                    &mut params,
                )
                .unwrap();
            }
            
            cuda_rt.synchronize().unwrap();
            println!("✅ Kernel launched successfully!");
            
            // Copy scores back (first buffer)
            let mut gpu_scores: Vec<f32> = vec![0.0; seq_q * num_heads * seq_k];
            unsafe {
                pesti_runner::cuda_runtime::copy_device_to_host(
                    gpu_scores.as_mut_ptr() as *mut u8,
                    combined_ptr as *const u8,
                    score_buffer_size,
                ).unwrap();
            }
            
            println!("✅ Results copied back to host");
            println!("   First 4 scores: {:?}", &gpu_scores[..4]);
            
            // Cleanup combined buffer
            unsafe {
                pesti_runner::cuda_runtime::free_device_memory(combined_ptr).unwrap();
            }
        }
        Err(e) => {
            println!("❌ Function load failed: {}", e);
            println!("   Tried mangled name: {}", mangled_name);
            
            // List all functions in module (if possible)
            // For now, just fail
            panic!("Exact-pattern PTX function not found. Check nvcc compilation.");
        }
    }
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
    }
    
    println!("\n=== Sanity Check PASSED ===");
}
