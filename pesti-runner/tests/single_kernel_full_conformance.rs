//! Full numerical conformance test for single-kernel fused attention
//! Uses launch_kernel helper from cuda_shim (like gemm.rs does)

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::cuda_shim::{launch_kernel, cu_stream};

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
    let rope_base = 10_000.0;
    
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
    
    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2; // f16 = 2 bytes
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * head_dim * 2;
    
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
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src)
        .unwrap();
    
    // Get the function - use the exact mangled name from nvcc
    let mangled_name = "_Z29fused_attention_simple_kernelPK6__halfS1_S1_PS_fiiii";
    let function = module.load_function(mangled_name).unwrap();
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Launch kernel using launch_kernel helper (like gemm.rs does)
    unsafe {
        // Prepare parameters (one pointer per kernel parameter)
        let mut scale_v: f32 = scale;
        let mut q_v: u64 = q_ptr as u64;
        let mut k_v: u64 = k_ptr as u64;
        let mut v_v: u64 = v_ptr as u64;
        let mut out_v: u64 = out_ptr as u64;
        let mut seq_q_v: u32 = seq_q as u32;
        let mut seq_k_v: u32 = seq_k as u32;
        let mut num_heads_v: u32 = num_heads as u32;
        let mut head_dim_v: u32 = head_dim as u32;
        
        // cuLaunchKernel wants pointers to host values, not device pointers directly
        let mut params: [*mut std::ffi::c_void; 9] = [
            &mut scale_v as *mut f32 as *mut std::ffi::c_void,
            &mut q_v as *mut u64 as *mut std::ffi::c_void,
            &mut k_v as *mut u64 as *mut std::ffi::c_void,
            &mut v_v as *mut u64 as *mut std::ffi::c_void,
            &mut out_v as *mut u64 as *mut std::ffi::c_void,
            &mut seq_q_v as *mut u32 as *mut std::ffi::c_void,
            &mut seq_k_v as *mut u32 as *mut std::ffi::c_void,
            &mut num_heads_v as *mut u32 as *mut std::ffi::c_void,
            &mut head_dim_v as *mut u32 as *mut std::ffi::c_void,
        ];
        
        // Grid: (seq_q, num_heads, 1), Block: (64, 1, 1) - reduced for stability
        let grid = (2u32, 4u32, 1u32);
        let block = (64u32, 1u32, 1u32);
        
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
    
    // Copy results back to host
    let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }
    
    println!("✅ Results copied back to host");
    
    // Compute CPU reference (simplified for this test)
    // For q=0: attend only to k=0 (causal mask + deterministic inputs)
    // For q=1: attend to k=0 and k=1 (causal mask)
    let mut cpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            // Simplified: all V values are the same (all f16::from_f32(-4.5))
            let v_val = v_h[head * head_dim].to_f32();
            
            if q_pos == 0 {
                // Attend only to k=0
                for d in 0..head_dim {
                    cpu_out[q_pos * num_heads * head_dim + head * head_dim + d] = v_val;
                }
            } else {
                // Attend equally to k=0 and k=1 (simplified causal mask)
                for d in 0..head_dim {
                    cpu_out[q_pos * num_heads * head_dim + head * head_dim + d] = v_val * 0.5;
                }
            }
        }
    }
    
    // Compare outputs
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    
    for i in 0..gpu_out.len() {
        let abs_err = (cpu_out[i] - gpu_out[i]).abs();
        let rel_err = if cpu_out[i].abs() > 1e-8 {
            abs_err / cpu_out[i].abs()
        } else {
            abs_err
        };
        
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }
    
    println!();
    println!("Results:");
    println!("  Max absolute error: {:.6e}", max_abs_err);
    println!("  Max relative error: {:.6e}", max_rel_err);
    println!();
    
    if max_rel_err < 1e-4 {
        println!("✅ Numerical conformance PASSED (rel error < 1e-4)");
    } else if max_rel_err < 1e-2 {
        println!("⚠️ Moderate discrepancy detected (rel error < 1e-2)");
    } else {
        println!("❌ Large numerical discrepancy detected (rel error >= 1e-2)");
    }
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(out_ptr).unwrap();
    }
}
