//! Test softmax with non-causal setup (seq_q >= seq_k)

use half::f16;
use pesti_runner::cuda_runtime::{CudaRuntime, allocate_device_memory, copy_host_to_device, copy_device_to_host, free_device_memory};

fn main() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    println!("=== Non-Causal Softmax Sanity Check ===");

    // seq_q=2, seq_k=1: both queries see the single key (no masking)
    let seq_q = 2;
    let seq_k = 1;
    let num_heads = 1;
    let head_dim = 2;

    // Q0 = [3, 4], Q1 = [5, 6]
    // K0 = [1, 0] (single key)
    // Scores: Q0@K0 = 3*1+4*0 = 3, Q1@K0 = 5*1+6*0 = 5
    // Softmax for each query independently: softmax([3]) = [1], softmax([5]) = [1]
    // V0 = [1, 0]
    // Output[Q0] = 1*V0 = [1, 0], Output[Q1] = 1*V0 = [1, 0]

    let q_h: Vec<f16> = vec![
        f16::from_f32(3.0), f16::from_f32(4.0), // Q0
        f16::from_f32(5.0), f16::from_f32(6.0), // Q1
    ];
    
    let k_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(0.0), // K0
    ];

    let v_h: Vec<f16> = vec![
        f16::from_f32(1.0), f16::from_f32(0.0), // V0
    ];

    let scale = 1.0;
    let rope_base = 10_000.0;

    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let out_size = seq_q * num_heads * head_dim * 4;

    let q_ptr = unsafe { allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { allocate_device_memory(v_size).unwrap() };
    let out_ptr = unsafe { allocate_device_memory(out_size).unwrap() };

    unsafe {
        copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }

    let stream = cuda_rt.new_stream().unwrap();
    
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(), stream.clone(),
    ).unwrap();

    unsafe {
        kernel.launch(scale, q_ptr as u64, k_ptr as u64, v_ptr as u64, out_ptr as u64,
            seq_q, seq_k, num_heads, head_dim, rope_base, seq_k).unwrap();
    }
    
    cuda_rt.synchronize().unwrap();

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

    println!("\nExpected: Both tokens output [1, 0] (single key with softmax=1)");

    unsafe {
        free_device_memory(q_ptr).unwrap();
        free_device_memory(k_ptr).unwrap();
        free_device_memory(v_ptr).unwrap();
        free_device_memory(out_ptr).unwrap();
    }
}
