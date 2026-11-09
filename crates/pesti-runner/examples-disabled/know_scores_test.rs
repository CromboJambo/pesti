//! Test with known scores [3, 4] and no causal masking

use half::f16;
use pesti_runner::cuda_runtime::{
    CudaRuntime, allocate_device_memory, copy_device_to_host, copy_host_to_device,
    free_device_memory,
};

fn main() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    println!("=== Known Scores Test ===");

    // seq_q=1, seq_k=2 with causal masking: q_pos=0 sees only k_pos=0
    // So scores should be [3, -inf] → softmax([3, -inf]) = [1, 0]

    let seq_q = 1;
    let seq_k = 2;
    let num_heads = 1;
    let head_dim = 2;

    // Q = [3, 4], K0 = [1, 0], K1 = [0, 1]
    // With causal mask: scores = [3, -inf]
    // Softmax([3, -inf]) = [1, 0]
    // V0 = [1, 0], V1 = [0, 1]
    // Output = 1*V0 + 0*V1 = [1, 0]

    let q_h: Vec<f16> = vec![f16::from_f32(3.0), f16::from_f32(4.0)];

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
        println!("  dim[{}] = {}", d, gpu_output[d]);
    }

    // Expected: softmax([3, -inf]) = [1, 0] → Output = [1, 0]
    println!("\nExpected: [1, 0] (causal mask applied)");

    unsafe {
        free_device_memory(q_ptr).unwrap();
        free_device_memory(k_ptr).unwrap();
        free_device_memory(v_ptr).unwrap();
        free_device_memory(out_ptr).unwrap();
    }
}
