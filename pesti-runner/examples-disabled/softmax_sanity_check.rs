//! Test softmax computation in isolation

use half::f16;
use pesti_runner::cuda_runtime::{
    CudaRuntime, allocate_device_memory, copy_device_to_host, copy_host_to_device,
    free_device_memory,
};

fn main() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    println!("=== Softmax Sanity Check ===");

    // Test case: scores [3, 4] → softmax should be [exp(3)/(exp(3)+exp(4)), exp(4)/(exp(3)+exp(4))] ≈ [0.269, 0.731]
    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 2;

    // Q and K such that Q@K^T = [3, 4] for the one token
    // Q = [a, b], K0 = [c, d], K1 = [e, f]
    // Q@K0 = a*c + b*d = 3
    // Q@K1 = a*e + b*f = 4
    // Let's use: Q = [3, 4], K0 = [1, 0], K1 = [0, 1]
    // Then: Q@K0 = 3*1 + 4*0 = 3 ✓, Q@K1 = 3*0 + 4*1 = 4 ✓

    let q_h: Vec<f16> = vec![
        f16::from_f32(3.0),
        f16::from_f32(4.0), // Q for token 0
    ];

    let k_h: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0), // K0
        f16::from_f32(0.0),
        f16::from_f32(1.0), // K1
    ];

    let v_h: Vec<f16> = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.0), // V0
        f16::from_f32(0.0),
        f16::from_f32(1.0), // V1
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

    unsafe {
        copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
        copy_host_to_device(v_ptr, v_h.as_ptr() as *const u8, v_size).unwrap();
    }

    let stream = cuda_rt.new_stream().unwrap();

    let kernel =
        pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
            pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
            cuda_rt.context().clone(),
            stream.clone(),
        )
        .unwrap();

    unsafe {
        kernel
            .launch(
                scale,
                q_ptr as u64,
                k_ptr as u64,
                v_ptr as u64,
                out_ptr as u64,
                seq_q,
                seq_k,
                num_heads,
                head_dim,
                rope_base,
                seq_k,
            )
            .unwrap();
    }

    cuda_rt.synchronize().unwrap();

    let mut gpu_output = vec![0.0f32; seq_q * num_heads * head_dim];
    unsafe {
        copy_device_to_host(
            gpu_output.as_mut_ptr() as *mut u8,
            out_ptr as *const u8,
            out_size,
        )
        .unwrap();
    }

    println!("GPU Output:");
    for d in 0..head_dim {
        let idx = d;
        println!("  dim[{}] = {}", d, gpu_output[idx]);
    }

    // Expected: softmax([3,4]) = [0.269, 0.731]
    // Output = 0.269*V0 + 0.731*V1 = [0.269, 0.731]
    println!("\nExpected: [0.269, 0.731]");

    unsafe {
        free_device_memory(q_ptr).unwrap();
        free_device_memory(k_ptr).unwrap();
        free_device_memory(v_ptr).unwrap();
        free_device_memory(out_ptr).unwrap();
    }
}
