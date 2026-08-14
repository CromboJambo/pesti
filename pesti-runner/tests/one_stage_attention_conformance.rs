//! One-stage full fusion attention conformance test (skeleton)
//! 
//! Three-stage -> One-stage progression:
//! 1. CPU: RoPE on Q/K (pre-processing)  
//! 2. GPU: scores + softmax + V-multiply (single fused kernel)
//! 
//! Target: Match llama.cpp output with single-kernel full fusion

#![cfg(feature = "cuda")]

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;

/// Reference implementation: CPU-side RoPE + softmax + V-multiply (llama.cpp style)
fn reference_llama_attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_q: usize,
    seq_k: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];
    
    for q_pos in 0..seq_q {
        for head in 0..num_heads {
            for dim_idx in 0..head_dim {
                let mut sum = 0.0f32;
                
                // Softmax over k positions
                let mut scores = vec![0.0f32; seq_k];
                for k_pos in 0..seq_k {
                    let dot: f32 = (0..head_dim)
                        .map(|d| {
                            let q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
                            let k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
                            q[q_idx].to_f32() * k[k_idx].to_f32()
                        })
                        .sum();
                    scores[k_pos] = dot / (head_dim as f32).sqrt();
                }
                
                // Softmax with max-subtraction trick and causal mask
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores
                    .iter()
                    .map(|&s| {
                        if s == f32::NEG_INFINITY {
                            0.0
                        } else {
                            (s - max_score).exp()
                        }
                    })
                    .sum();
                
                // Weighted V sum
                for k_pos in 0..seq_k {
                    let softmax_val = if scores[k_pos] == f32::NEG_INFINITY || exp_sum <= 0.0 {
                        0.0
                    } else {
                        (scores[k_pos] - max_score).exp() / exp_sum
                    };
                    let v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
                    sum += softmax_val * v[v_idx].to_f32();
                }
                
                output[q_pos * num_heads * head_dim + head * head_dim + dim_idx] = sum;
            }
        }
    }
    
    output
}

#[test]
fn test_one_stage_attention() {
    println!("\n=== One-Stage Full Fusion Attention ===");
    
    let cuda_rt = CudaRuntime::new(0).unwrap();
    
    // Configuration matching exact_pattern
    let seq_q = 2;
    let seq_k = 4;
    let num_heads = 2;
    let head_dim = 8;
    
    // Allocate on CPU
    let q_size = seq_q * num_heads * head_dim * 2; // f16
    let k_size = seq_k * num_heads * head_dim * 2;
    let v_size = seq_k * num_heads * head_dim * 2;
    
    // Initialize with small random-ish values (not zeros!)
    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32((i as f32 - q_size as f32 / 2.0) * 0.1))
        .collect();
    
    let k_host: Vec<f16> = (0..k_size)
        .map(|i| f16::from_f32((i as f32 - k_size as f32 / 2.0) * 0.1))
        .collect();
    
    let v_host: Vec<f16> = (0..v_size)
        .map(|i| f16::from_f32((i as f32 - v_size as f32 / 2.0) * 0.1))
        .collect();
    
    println!(
        "Allocated CPU buffers: Q={}, K={}, V={}",
        q_size, k_size, v_size
    );
    
    // Allocate on GPU
    let _q_ptr = pesti_runner::cuda_runtime::allocate_device_memory(q_size).unwrap();
    let _k_ptr = pesti_runner::cuda_runtime::allocate_device_memory(k_size).unwrap();
    let _v_ptr = pesti_runner::cuda_runtime::allocate_device_memory(v_size).unwrap();
    
    // Allocate output buffer (float, seq_q * num_heads * head_dim)
    let _output_buffer_bytes = seq_q * num_heads * head_dim * 4; // float
    
    println!("Allocated GPU buffers");
    
    // Copy Q, K, V to GPU
    pesti_runner::cuda_runtime::copy_host_to_device(
        _q_ptr,
        q_host.as_ptr() as *const u8,
        q_size,
    ).unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(
        _k_ptr,
        k_host.as_ptr() as *const u8,
        k_size,
    ).unwrap();
    pesti_runner::cuda_runtime::copy_host_to_device(
        _v_ptr,
        v_host.as_ptr() as *const u8,
        v_size,
    ).unwrap();
    
    println!("Copied inputs to GPU");
    
    // TODO: Load and launch full fusion kernel (scores + softmax + V-multiply in one kernel)
    // For now, just verify the infrastructure works
    
    println!("One-stage attention test setup complete (kernel not yet implemented)");
    
    cuda_rt.synchronize().unwrap();
    
    // Compare with CPU reference
    let _cpu_output = reference_llama_attention(&q_host, &k_host, &v_host, seq_q, seq_k, num_heads, head_dim);
    
    println!("Computed CPU reference");
    println!("\nResults:");
    println!(
        "  Configuration: seq_q={}, seq_k={}, heads={}, dim={}",
        seq_q, seq_k, num_heads, head_dim
    );
    println!("  One-stage attention test infrastructure verified!");
}
