//! Minimal sanity check with known values

use half::f16;
use pesti_runner::cuda_runtime::{CudaRuntime, allocate_device_memory, copy_host_to_device, copy_device_to_host, free_device_memory};

fn main() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    println!("=== Fused Attention Sanity Check ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    // Minimal test: 2 tokens, 1 head, dim=4
    let seq_q = 2;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 4;
    
    // Simple values for manual verification
    let q_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 0
        f16::from_f32(3.0), f16::from_f32(4.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 1
    ];
    
    let k_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 0
        f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 1
    ];
    
    let v_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 0
        f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0), // token 1
    ];

    let scale = 1.0;
    let rope_base = 10_000.0;

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    // Output size: [seq_q, num_heads, head_dim]
    let out_size = seq_q * num_heads * head_dim * 4;

    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(out_size).unwrap() };

    // Copy inputs to device
    unsafe {
        copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }

    let stream = cuda_rt.new_stream().unwrap();
    
    // Build and launch kernel
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(), stream.clone(),
    ).unwrap();

    unsafe {
        kernel.launch(scale, q_ptr as u64, k_ptr as u64, v_ptr as u64, out_ptr as u64,
            seq_q, seq_k, num_heads, head_dim, rope_base, seq_k).unwrap();
    }
    
    cuda_rt.synchronize().unwrap();

    // Read output
    let mut gpu_output = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        copy_device_to_host(
            gpu_output.as_mut_ptr() as *mut u8, out_ptr as *const u8, out_size).unwrap();
    }

    println!("GPU Output:");
    for q in 0..seq_q {
        for d in 0..head_dim {
            let idx = q * num_heads * head_dim + d;
            println!("  Token {}: dim[{}] = {}", q, d, gpu_output[idx]);
        }
    }

    // Manual calculation:
    // Q0 = [1,2,0,0], K0 = [1,0,0,0] → dot = 1.0
    // Q0 = [1,2,0,0], K1 = [0,1,0,0] → dot = 2.0
    // Causal mask: Q0 sees K0 only (k_pos <= q_pos)
    // Softmax(Q0) = softmax([1, -inf]) = [1, 0]
    // Output[Q0] = 1.0 * V0 + 0.0 * V1 = V0 = [1,0,0,0]
    
    println!("\nExpected (manual calc):");
    println!("  Token 0: dim[0] = 1.0 (from K0), others 0");
    println!("  Token 1: dim[1] = 1.0 (from K1), others 0");

    unsafe {
        free_device_memory(q_ptr).unwrap();
        free_device_memory(k_ptr).unwrap();
        free_device_memory(v_ptr).unwrap();
        free_device_memory(out_ptr).unwrap();
    }
}
