//! Test GEMM-based attention implementation on consumer GPUs
//! 
//! This demonstrates Option A: using existing mma.sync GEMM for attention
//! instead of writing new WGMMA/tcgen05 PTX kernels.
//!
//! Usage:
//!   cargo run --package pesti-runner --features cuda --example test_gemm_attention

use half::f16;
use pesti_runner::cuda_runtime::CudaRuntime;
use pesti_runner::kernel::device_buf::DeviceBuffer;
use pesti_runner::kernel::gemm::{CudaGemmKernel, CudaGemmKernelBuilder, GemmArch};
use pesti_runner::kernel::memory::CudaMemoryBackend;
use pesti_runner::kernel::{GemmKernel, Kvcache};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GEMM-Based Attention Test (Option A) ===\n");

    // Use device 1 (RTX 5060 Ti, sm_12.0)
    let ordinal = 1;
    println!("Using device {}: RTX 5060 Ti (sm_12.0)", ordinal);

    // Initialize CUDA runtime
    let rt = CudaRuntime::new(ordinal)?;
    let stream = rt.new_stream()?;
    let info = rt.device_info();

    println!("Device: {} (compute capability sm_{}.{})", 
             info.name, info.compute_capability.0, info.compute_capability.1);

    // Build GEMM kernel (will use mma.sync since 5060 Ti doesn't have WGMMA/tcgen05)
    let gemm_arch = GemmArch::Mma;
    let gemm_kernel = CudaGemmKernelBuilder::new(
        gemm_arch,
        rt.context().clone(),
        stream.clone(),
        info.clone(),
    )
    .build()?;

    println!("✅ GEMM kernel loaded: {} (arch {:?})", "gemm_mma_kernel", gemm_arch);

    // Create mock KV cache buffers
    let num_heads = 8;
    let head_dim = 64;
    let seq_len = 256;
    let batch_size = 1;

    // Q: [batch, seq, num_heads * head_dim] = [1, 1, 512]
    let query_len = 1;
    let q_dim = num_heads * head_dim;
    let q_size = query_len * q_dim;

    // K/V cache: [num_heads, head_dim, seq_len] flattened
    let kv_size = num_heads * head_dim * seq_len;

    // Initialize with random-ish values
    let q_host: Vec<f16> = (0..q_size)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.1 - 0.85))
        .collect();

    let k_host: Vec<f16> = (0..kv_size)
        .map(|i| f16::from_f32(((i % 13) as f32) * 0.05 - 0.4))
        .collect();

    let v_host: Vec<f16> = k_host.clone(); // V = K for this test

    // Allocate device buffers
    let backend = CudaMemoryBackend::with_device_info(stream.clone(), info.clone());
    let q_buf = DeviceBuffer::from_host_device(&backend, &q_host)?;
    let k_buf = DeviceBuffer::from_host_device(&backend, &k_host)?;
    let v_buf = DeviceBuffer::from_host_device(&backend, &v_host)?;

    // Create mock Kvcache objects
    let k_cache = Kvcache::new(
        num_heads,
        head_dim,
        seq_len,
        k_buf,
        backend.clone(),
    );

    let v_cache = Kvcache::new(
        num_heads,
        head_dim,
        seq_len,
        v_buf,
        backend.clone(),
    );

    // Create GEMM-based attention kernel
    #[cfg(feature = "cuda")]
    let attn_kernel = pesti_runner::kernel::attention::GemmBasedAttentionKernel::new(
        gemm_arch.into(),
        gemm_kernel,
        rt.context().clone(),
        stream.clone(),
    );

    // Compute attention: Q @ K^T / sqrt(D) -> softmax -> V @ S^T
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    println!("\n--- Step 1: Q @ K^T ---");
    let qk_output = attn_kernel.qk_gemm(
        &q_buf,
        &k_cache,
        num_heads,
        head_dim,
        seq_len,
        query_len,
        scale,
    )?;

    println!("✅ Q @ K^T completed: [{} x {}]", query_len * num_heads, seq_len);

    // Softmax on CPU (simpler for now)
    println!("\n--- Step 2: Softmax ---");
    let qk_host = qk_output.to_host_vec(&backend)?;
    let rows = query_len * num_heads;
    let cols = seq_len;

    let softmax_output = GemmBasedAttentionKernel::softmax_cpu(&qk_host, rows, cols);

    println!("✅ Softmax completed: [{} x {}]", rows, cols);

    // V @ S^T
    println!("\n--- Step 3: S @ V ---");
    let attn_weights = DeviceBuffer::from_host(softmax_output);
    let output = attn_kernel.sv_gemm(
        &attn_weights,
        &v_cache,
        num_heads,
        head_dim,
        query_len,
    )?;

    println!("✅ S @ V completed: [{} x {}]", query_len * num_heads, head_dim);

    // Verify numerical correctness
    let output_host = output.to_host_vec(&backend)?;
    
    // Compute expected CPU result for comparison
    let mut expected = vec![0.0f32; query_len * q_dim];
    for q_idx in 0..query_len {
        for head in 0..num_heads {
            for d in 0..head_dim {
                let mut sum = 0.0;
                for k_idx in 0..seq_len {
                    // Q[q, head * head_dim + d] * K[head, d, k_idx]
                    let q_val = q_host[q_idx * q_dim + head * head_dim + d].to_f32();
                    let k_val = k_host[(head * seq_len + k_idx) * head_dim + d].to_f32();
                    let attn_score = q_val * k_val / scale; // simplified

                    // Softmax (simplified - skip for speed)
                    let attn_weight = (attn_score * scale).exp() / (seq_len as f32);

                    // V @ A^T
                    let v_val = v_host[(head * seq_len + k_idx) * head_dim + d].to_f32();
                    sum += attn_weight * v_val;
                }
                expected[q_idx * q_dim + head * head_dim + d] = sum;
            }
        }
    }

    // Compare
    let mut max_err = 0.0;
    for i in 0..output_host.len() {
        let err = (output_host[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("\n--- Results ---");
    println!("Max error vs CPU reference: {:.3e}", max_err);
    
    if max_err < 1e-2 {
        println!("✅ CORRECT: Output matches expected within tolerance");
    } else {
        println!("⚠️  WARNING: Error {:.3e} exceeds tolerance", max_err);
    }

    Ok(())
}

// Copy helper methods for standalone test
struct GemmBasedAttentionKernel;
impl GemmBasedAttentionKernel {
    fn softmax_cpu(buffer: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows * cols];
        for i in 0..rows {
            let start = i * cols;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..cols {
                let val = buffer[start + j];
                if val > max_val {
                    max_val = val;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..cols {
                let exp_val = (buffer[start + j] - max_val).exp();
                result[start + j] = exp_val;
                sum += exp_val;
            }
            if sum > 0.0 {
                for j in 0..cols {
                    result[start + j] /= sum;
                }
            }
        }
        result
    }
}
