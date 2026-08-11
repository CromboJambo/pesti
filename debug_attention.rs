//! Quick debug test to print first few values

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

fn main() {
    let cuda_rt = CudaRuntime::new(0).unwrap();
    if !cuda_rt.is_valid() {
        eprintln!("CUDA not initialized");
        return;
    }

    println!("=== Debug: GPU vs CPU Attention Values ===");
    println!("GPU: {}", cuda_rt.device_info().name);

    let seq_q = 2;
    let seq_k = 32;
    let num_heads = 4;
    let head_dim = 16;
    let rope_base = 10_000.0;

    // Create deterministic Q, K
    let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
        .map(|i| f16::from_f32((i as f32 - 50.0) / 10.0))
        .collect();

    println!("Configuration: seq_q={}, seq_k={}, heads={}, dim={}", seq_q, seq_k, num_heads, head_dim);

    // Allocate device memory
    let q_size = seq_q * num_heads * head_dim * 2;
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * num_heads * seq_k * 4;

    let q_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap() };
    let k_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap() };
    let v_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap() };
    let s_ptr = unsafe { pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap() };

    // Copy Q, K to device
    unsafe {
        pesti_runner::cuda_runtime::copy_host_to_device(q_ptr, q_h.as_ptr() as *const u8, q_size).unwrap();
        pesti_runner::cuda_runtime::copy_host_to_device(k_ptr, k_h.as_ptr() as *const u8, k_size).unwrap();
    }

    // Build kernel
    let stream = cuda_rt.new_stream().unwrap();
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    ).unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Launch kernel
    unsafe {
        kernel.launch(
            scale,
            q_ptr as u64,
            k_ptr as u64,
            v_ptr as u64,
            s_ptr as u64,
            seq_q,
            seq_k,
            num_heads,
            head_dim,
            rope_base,
            seq_k,
        ).unwrap();
    }

    cuda_rt.synchronize().unwrap();
    println!("✅ GPU kernel launched");

    // Copy results back
    let mut gpu_probs = vec![0.0f32; seq_q * num_heads * seq_k];
    unsafe {
        pesti_runner::cuda_runtime::copy_device_to_host(
            gpu_probs.as_mut_ptr() as *mut u8,
            s_ptr as *const u8,
            s_size,
        ).unwrap();
    }

    // Print first query, first head
    println!("\nFirst query (q_pos=0), first head:");
    println!("GPU outputs (first 10):");
    for k in 0..10.min(seq_k) {
        let idx = 0 * num_heads * seq_k + 0 * seq_k + k; // [q, head, k]
        println!("  k_pos={}: {:.6}", k, gpu_probs[idx]);
    }

    println!("\nGPU softmax sum (first query, first head): {}", 
        (0..seq_k).map(|k| gpu_probs[0 * num_heads * seq_k + 0 * seq_k + k]).sum::<f32>());

    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
