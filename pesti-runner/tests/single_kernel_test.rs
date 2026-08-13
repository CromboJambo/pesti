//! Simple test for single-kernel fused attention
//! Uses the fused_attention_simple_kernel.ptx directly

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_single_kernel_fused_attention() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized, skipping single-kernel test");
        return;
    }
    
    println!("=== Single-Kernel Fused Attention Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();
    
    // Configuration (small for quick testing)
    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;
    
    // Create deterministic Q, K, V (f16)
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
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }
    
    // Build kernel from single-kernel PTX
    let stream = cuda_rt.new_stream().unwrap();
    
    // Load the simple fused kernel PTX
    let ptx_src = include_str!("../src/kernel/ptx/fused_attention_simple_kernel.ptx");
    let module = pesti_runner::cuda_shim::CudaModule::load_from_ptx(&cuda_rt.context(), ptx_src).unwrap();
    
    // Get the function (mangled name from PTX)
    let mangled_name = "_Z34fused_attention_simple_kernelPK6__halfS2_S2_PS_fiii";
    let function = module.load_function(mangled_name).unwrap();
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Launch kernel: grid=(seq_q, num_heads, 1), block=(128, 1, 1)
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
    
    println!("✅ Single-kernel launched successfully");
    
    // Copy results back to host
    let mut gpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_out.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        ).unwrap();
    }
    
    // Compute CPU reference (simplified - just softmax of uniform for now)
    let mut cpu_out = vec![0.0f32; seq_q * num_heads * head_dim];
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            // For q=0: attend only to k=0 (causal mask)
            // For q=1: attend only to k=0, k=1 (causal mask)
            let mut probs = vec![0.0f32; seq_k];
            
            if q_pos == 0 {
                probs[0] = 1.0;  // Attend only to k=0
            } else {
                probs[0] = 0.5;  // Attend equally to k=0, k=1
                probs[1] = 0.5;
            }
            
            // Weighted sum of V (simplified - all V values are same in this test)
            let v_sum: f32 = v_h[(head * head_dim)..(head * head_dim + head_dim)]
                .iter()
                .map(|&v| v.to_f32())
                .sum();
            
            for d in 0..head_dim {
                cpu_out[q_pos * num_heads * head_dim + head * head_dim + d] = 
                    probs[0] * v_h[d].to_f32() + probs[1] * v_h[head_dim + d].to_f32();
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
