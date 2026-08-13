//! Quick sanity check for single-kernel fused attention PTX
//! Tests that the PTX compiles, loads, and launches without errors

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_single_kernel_sanity() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping sanity check");
        return;
    }
    
    println!("=== Single-Kernel Sanity Check ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    
    // Small configuration
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
    let out_size = seq_q * num_heads * head_dim * 2;
    
    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(out_size).unwrap() };
    
    // Copy to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }
    
    // Load PTX and get function
    let stream = cuda_rt.new_stream().unwrap();
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    
    // Try to get the function - print all available functions for debugging
    println!("Loading kernel from PTX...");
    
    // The mangled name depends on nvcc compilation, let's try common patterns
    let mangled_name = "_Z34fused_attention_simple_kernelPK6__halfS2_S2_PS_fiii";
    
    match module.load_function(mangled_name) {
        Ok(function) => {
            println!("✅ Function loaded: {}", mangled_name);
            
            // Launch with small grid/block
            let scale = 1.0 / (head_dim as f32).sqrt();
            
            unsafe {
                function.launch(
                    &(seq_q, num_heads, 1),
                    &(128, 1, 1),
                    (
                        scale,
                        q_ptr as u64,
                        k_ptr as u64,
                        v_ptr as u64,
                        out_ptr as u64,
                        seq_q,
                        seq_k,
                        num_heads,
                        head_dim,
                    ),
                ).unwrap();
            }
            
            cuda_rt.synchronize().unwrap();
            println!("✅ Kernel launched successfully!");
            
            // Copy results back
            let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
            unsafe {
                pesti_runner::cuda_runtime::copy_device_to_host(
                    gpu_out.as_mut_ptr() as *mut u8,
                    out_ptr as *const u8,
                    out_size,
                ).unwrap();
            }
            
            println!("✅ Results copied back to host");
            println!("   First 4 output values: {:?}", &gpu_out[..4]);
        }
        Err(e) => {
            println!("❌ Function load failed: {}", e);
            println!("   Tried mangled name: {}", mangled_name);
            
            // List all functions in module (if possible)
            // For now, just fail
            panic!("Single-kernel PTX function not found. Check nvcc compilation.");
        }
    }
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
    
    println!("\n=== Sanity Check PASSED ===");
}
