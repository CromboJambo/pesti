//! Simple launch verification for fused attention kernel
//!
//! Tests that the kernel builds and launches without errors.

use pesti_runner::cuda_runtime::CudaRuntime;

#[test]
fn test_fused_attention_kernel_launch() {
    // Initialize CUDA runtime
    let cuda_rt = match CudaRuntime::new(0) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("CUDA not available: {}", e);
            return;
        }
    };
    
    println!("=== Fused Attention Kernel Launch Test ===");
    println!("GPU: {}", cuda_rt.device_info().name);
    println!();
    
    // Configuration (small for quick testing)
    let seq_q = 1;
    let seq_k = 64;
    let num_heads = 4;
    let head_dim = 32;
    let rope_base = 10_000.0;
    
    // Allocate minimal device memory using low-level API
    let q_size = seq_q * num_heads * head_dim * 2; // f16 = 2 bytes
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    let s_size = seq_q * seq_k * 4; // f32 = 4 bytes
    
    let q_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap()
    };
    let k_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap()
    };
    let v_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap()
    };
    let s_ptr = unsafe {
        pesti_runner::cuda_runtime::allocate_device_memory(s_size).unwrap()
    };
    
    // Build kernel
    let stream = cuda_rt.new_stream().unwrap();
    let kernel = pesti_runner::kernel::fused_attention_conformant::build_fused_attention_kernel_conformant(
        pesti_runner::kernel::fused_attention_conformant::FusedAttentionArch::MmaSync,
        cuda_rt.context().clone(),
        stream.clone(),
    ).unwrap();
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Launch kernel with dummy pointers
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
            seq_k, // max_pos
        ).unwrap();
    }
    
    cuda_rt.synchronize().unwrap();
    
    println!("✅ Kernel launched successfully!");
    println!(
        "  Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    
    // Cleanup
    unsafe {
        pesti_runner::cuda_runtime::free_device_memory(q_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(k_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(v_ptr).unwrap();
        pesti_runner::cuda_runtime::free_device_memory(s_ptr).unwrap();
    }
}
